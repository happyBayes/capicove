"""Provider-neutral immutable video index and generation-set values."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

from capicove.contracts import (
    DistanceMetric,
    EmbeddingSpaceRef,
    ProcessingFingerprint,
    QueryModality,
    SearchableUnit,
    TargetType,
    VectorNormalization,
)
from capicove.retrieval.generations import (
    PinnedTextIndexGeneration,
    TextIndexGenerationManifest,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_IDENTITY_LENGTH = 500
_VISUAL_TARGETS = frozenset({TargetType.KEYFRAME, TargetType.VIDEO_SEGMENT})


class VideoGenerationSetState(StrEnum):
    """Describe the durable lifecycle of one complete video generation set."""

    BUILDING = "building"
    READY = "ready"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    CORRUPT = "corrupt"
    EVICTED = "evicted"


class VideoGenerationSetEventType(StrEnum):
    """Describe one append-only generation-set lifecycle observation."""

    REGISTERED = "registered"
    ACTIVATED = "activated"
    SUPERSEDED = "superseded"
    EVICTED = "evicted"


class VideoGenerationError(RuntimeError):
    """Report a bounded video generation or catalog failure."""


class VideoGenerationConflictError(VideoGenerationError):
    """Report immutable identity reuse for different evidence."""


class VideoGenerationTransitionError(VideoGenerationError):
    """Report an illegal or stale generation-set transition."""


@dataclass(frozen=True, slots=True)
class VisualIndexEntry:
    """Bind one immutable visual unit to its finite target vector."""

    unit: SearchableUnit
    vector: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.unit, SearchableUnit):
            raise TypeError("unit must be a SearchableUnit")
        if self.unit.unit_type not in _VISUAL_TARGETS:
            raise ValueError("unit must be a keyframe or video segment visual target")
        vector = tuple(self.vector)
        if not vector:
            raise ValueError("vector must not be empty")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in vector
        ):
            raise ValueError("vector must contain only finite numbers")
        compatible_dimensions = {
            space.dimension
            for space in self.unit.embedding_spaces
            if self.unit.unit_type in space.target_modalities
        }
        if compatible_dimensions and len(vector) not in compatible_dimensions:
            raise ValueError("vector dimension must match a unit embedding space")
        object.__setattr__(self, "vector", tuple(float(value) for value in vector))


@dataclass(frozen=True, slots=True)
class VisualCandidate:
    """Contain one bounded candidate from one pinned visual generation."""

    unit_id: str
    target_type: TargetType
    lane_id: str
    raw_score: float
    score_semantics: str
    rank: int
    provider_id: str
    generation_id: str

    def __post_init__(self) -> None:
        for field_name in (
            "unit_id",
            "lane_id",
            "score_semantics",
            "provider_id",
            "generation_id",
        ):
            _require_identity(field_name, getattr(self, field_name))
        _require_visual_target(self.target_type)
        if (
            isinstance(self.raw_score, bool)
            or not isinstance(self.raw_score, (int, float))
            or not math.isfinite(self.raw_score)
        ):
            raise ValueError("raw_score must be finite")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("rank must be a positive integer")


@dataclass(frozen=True, slots=True)
class VisualIndexGenerationManifest:
    """Describe one immutable provider visual-index generation."""

    generation_id: str
    workspace_ref: str
    corpus_refs: tuple[str, ...]
    target_type: TargetType
    searchable_unit_manifest_sha256: str
    content_sha256: str
    embedding_space: EmbeddingSpaceRef
    retriever_id: str
    index_format_version: str
    build_processing_fingerprint: ProcessingFingerprint
    expected_unit_count: int
    schema_version: int = 1
    manifest_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "generation_id",
            "workspace_ref",
            "retriever_id",
            "index_format_version",
        ):
            _require_identity(field_name, getattr(self, field_name))
        corpus_refs = _freeze_identities("corpus_ref", self.corpus_refs)
        if not corpus_refs:
            raise ValueError("corpus_refs must not be empty")
        object.__setattr__(self, "corpus_refs", tuple(sorted(corpus_refs)))
        _require_visual_target(self.target_type)
        _require_sha256(
            "searchable_unit_manifest_sha256",
            self.searchable_unit_manifest_sha256,
        )
        _require_sha256("content_sha256", self.content_sha256)
        if not isinstance(self.embedding_space, EmbeddingSpaceRef):
            raise TypeError("embedding_space must be an EmbeddingSpaceRef")
        if self.target_type not in self.embedding_space.target_modalities:
            raise ValueError("embedding space must accept the visual target")
        if not (
            QueryModality.TEXT in self.embedding_space.query_modalities
            or QueryModality.IMAGE in self.embedding_space.query_modalities
        ):
            raise ValueError("embedding space must accept a supported query modality")
        if self.embedding_space.normalization is not VectorNormalization.L2:
            raise ValueError("visual embedding space must use L2 normalization")
        if self.embedding_space.distance_metric is not DistanceMetric.INNER_PRODUCT:
            raise ValueError("visual embedding space must use inner-product distance")
        if not isinstance(self.build_processing_fingerprint, ProcessingFingerprint):
            raise TypeError("build_processing_fingerprint must be a ProcessingFingerprint")
        _require_non_negative_int("expected_unit_count", self.expected_unit_count)
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        object.__setattr__(self, "manifest_sha256", _visual_manifest_digest(self))


@dataclass(frozen=True, slots=True)
class VisualIndexValidationReport:
    """Record bounded validation evidence for one physical visual index."""

    generation_id: str
    manifest_sha256: str
    physical_index_ref: str
    content_sha256: str
    expected_unit_count: int
    observed_unit_count: int
    vector_dimension: int
    issue_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identity("generation_id", self.generation_id)
        _require_sha256("manifest_sha256", self.manifest_sha256)
        _require_identity("physical_index_ref", self.physical_index_ref)
        _require_sha256("content_sha256", self.content_sha256)
        _require_non_negative_int("expected_unit_count", self.expected_unit_count)
        _require_non_negative_int("observed_unit_count", self.observed_unit_count)
        if (
            isinstance(self.vector_dimension, bool)
            or not isinstance(self.vector_dimension, int)
            or self.vector_dimension < 1
        ):
            raise ValueError("vector_dimension must be a positive integer")
        issues = tuple(sorted(_freeze_identities("issue_code", self.issue_codes)))
        count_mismatch = self.expected_unit_count != self.observed_unit_count
        if count_mismatch != ("unit_count_mismatch" in issues):
            raise ValueError("unit_count_mismatch must reflect observed unit count")
        object.__setattr__(self, "issue_codes", issues)

    @property
    def valid(self) -> bool:
        """Return whether every required integrity check passed."""
        return not self.issue_codes


@dataclass(frozen=True, slots=True)
class PinnedVisualIndexGeneration:
    """Identify one concrete activated visual generation."""

    generation_id: str
    manifest_sha256: str
    physical_index_ref: str
    target_type: TargetType
    embedding_space: EmbeddingSpaceRef

    def __post_init__(self) -> None:
        _require_identity("generation_id", self.generation_id)
        _require_sha256("manifest_sha256", self.manifest_sha256)
        _require_identity("physical_index_ref", self.physical_index_ref)
        _require_visual_target(self.target_type)
        if not isinstance(self.embedding_space, EmbeddingSpaceRef):
            raise TypeError("embedding_space must be an EmbeddingSpaceRef")
        if self.target_type not in self.embedding_space.target_modalities:
            raise ValueError("embedding space must accept the pinned target")


@dataclass(frozen=True, slots=True)
class VideoGenerationSetManifest:
    """Describe one complete compatible video retrieval generation set."""

    set_id: str
    workspace_ref: str
    profile_ref: str
    corpus_refs: tuple[str, ...]
    unit_catalog_ref: str
    unit_manifest_sha256: str
    keyframe_visual: VisualIndexGenerationManifest
    segment_visual: VisualIndexGenerationManifest
    asr_text: TextIndexGenerationManifest
    ocr_text: TextIndexGenerationManifest | None
    fusion_profile_ref: str
    schema_version: int = 1
    manifest_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "set_id",
            "workspace_ref",
            "profile_ref",
            "unit_catalog_ref",
            "fusion_profile_ref",
        ):
            _require_identity(field_name, getattr(self, field_name))
        corpus_refs = _freeze_identities("corpus_ref", self.corpus_refs)
        if not corpus_refs:
            raise ValueError("corpus_refs must not be empty")
        object.__setattr__(self, "corpus_refs", tuple(sorted(corpus_refs)))
        _require_sha256("unit_manifest_sha256", self.unit_manifest_sha256)
        if not isinstance(self.keyframe_visual, VisualIndexGenerationManifest):
            raise TypeError("keyframe_visual must be a VisualIndexGenerationManifest")
        if not isinstance(self.segment_visual, VisualIndexGenerationManifest):
            raise TypeError("segment_visual must be a VisualIndexGenerationManifest")
        if not isinstance(self.asr_text, TextIndexGenerationManifest):
            raise TypeError("asr_text must be a TextIndexGenerationManifest")
        if self.ocr_text is not None and not isinstance(self.ocr_text, TextIndexGenerationManifest):
            raise TypeError("ocr_text must be a TextIndexGenerationManifest or None")
        if self.keyframe_visual.target_type is not TargetType.KEYFRAME:
            raise ValueError("keyframe visual member must target keyframes")
        if self.segment_visual.target_type is not TargetType.VIDEO_SEGMENT:
            raise ValueError("segment visual member must target video segments")
        if self.keyframe_visual.embedding_space != self.segment_visual.embedding_space:
            raise ValueError("visual members must use the same embedding space")
        for member in (
            self.keyframe_visual,
            self.segment_visual,
            self.asr_text,
            self.ocr_text,
        ):
            if member is None:
                continue
            if member.workspace_ref != self.workspace_ref:
                raise ValueError("generation member workspace does not match the set")
            if member.corpus_refs != self.corpus_refs:
                raise ValueError("generation member corpora do not match the set")
            if member.searchable_unit_manifest_sha256 != self.unit_manifest_sha256:
                raise ValueError("generation member unit manifest does not match the set")
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        object.__setattr__(self, "manifest_sha256", _video_set_manifest_digest(self))


@dataclass(frozen=True, slots=True)
class PinnedVideoGenerationSet:
    """Pin every physical member used for one video search request."""

    set_id: str
    manifest_sha256: str
    workspace_ref: str
    profile_ref: str
    unit_catalog_ref: str
    unit_manifest_sha256: str
    keyframe_visual: PinnedVisualIndexGeneration
    segment_visual: PinnedVisualIndexGeneration
    asr_text: PinnedTextIndexGeneration
    ocr_text: PinnedTextIndexGeneration | None
    fusion_profile_ref: str

    def __post_init__(self) -> None:
        for field_name in (
            "set_id",
            "workspace_ref",
            "profile_ref",
            "unit_catalog_ref",
            "fusion_profile_ref",
        ):
            _require_identity(field_name, getattr(self, field_name))
        _require_sha256("manifest_sha256", self.manifest_sha256)
        _require_sha256("unit_manifest_sha256", self.unit_manifest_sha256)
        if not isinstance(self.keyframe_visual, PinnedVisualIndexGeneration):
            raise TypeError("keyframe_visual must be a PinnedVisualIndexGeneration")
        if not isinstance(self.segment_visual, PinnedVisualIndexGeneration):
            raise TypeError("segment_visual must be a PinnedVisualIndexGeneration")
        if self.keyframe_visual.target_type is not TargetType.KEYFRAME:
            raise ValueError("keyframe visual pin must target keyframes")
        if self.segment_visual.target_type is not TargetType.VIDEO_SEGMENT:
            raise ValueError("segment visual pin must target video segments")
        if self.keyframe_visual.embedding_space != self.segment_visual.embedding_space:
            raise ValueError("visual pins must use the same embedding space")
        if not isinstance(self.asr_text, PinnedTextIndexGeneration):
            raise TypeError("asr_text must be a PinnedTextIndexGeneration")
        if self.ocr_text is not None and not isinstance(self.ocr_text, PinnedTextIndexGeneration):
            raise TypeError("ocr_text must be a PinnedTextIndexGeneration or None")


@dataclass(frozen=True, slots=True)
class VideoGenerationSetRecord:
    """Project one generation set's authoritative lifecycle state."""

    manifest: VideoGenerationSetManifest
    pin: PinnedVideoGenerationSet
    state: VideoGenerationSetState
    revision: int

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, VideoGenerationSetManifest):
            raise TypeError("manifest must be a VideoGenerationSetManifest")
        if not isinstance(self.pin, PinnedVideoGenerationSet):
            raise TypeError("pin must be a PinnedVideoGenerationSet")
        if not isinstance(self.state, VideoGenerationSetState):
            raise TypeError("state must be a VideoGenerationSetState")
        _require_non_negative_int("revision", self.revision)
        if self.pin.set_id != self.manifest.set_id:
            raise ValueError("pin set identity does not match manifest")
        if self.pin.manifest_sha256 != self.manifest.manifest_sha256:
            raise ValueError("pin manifest identity does not match manifest")
        if self.pin.workspace_ref != self.manifest.workspace_ref:
            raise ValueError("pin workspace identity does not match manifest")
        if self.pin.profile_ref != self.manifest.profile_ref:
            raise ValueError("pin profile identity does not match manifest")
        if self.pin.unit_catalog_ref != self.manifest.unit_catalog_ref:
            raise ValueError("pin unit catalog identity does not match manifest")
        if self.pin.unit_manifest_sha256 != self.manifest.unit_manifest_sha256:
            raise ValueError("pin unit manifest identity does not match manifest")
        pairs = (
            (self.pin.keyframe_visual, self.manifest.keyframe_visual),
            (self.pin.segment_visual, self.manifest.segment_visual),
            (self.pin.asr_text, self.manifest.asr_text),
        )
        for pinned, member in pairs:
            if (
                pinned.generation_id != member.generation_id
                or pinned.manifest_sha256 != member.manifest_sha256
            ):
                raise ValueError("pin member identity does not match manifest")
        if (self.pin.ocr_text is None) != (self.manifest.ocr_text is None):
            raise ValueError("OCR pin presence does not match manifest")
        if self.pin.ocr_text is not None and self.manifest.ocr_text is not None:
            if (
                self.pin.ocr_text.generation_id != self.manifest.ocr_text.generation_id
                or self.pin.ocr_text.manifest_sha256 != self.manifest.ocr_text.manifest_sha256
            ):
                raise ValueError("OCR pin identity does not match manifest")


@dataclass(frozen=True, slots=True)
class VideoGenerationSetEvent:
    """Capture one immutable lifecycle projection in append-only order."""

    sequence: int
    event_type: VideoGenerationSetEventType
    record: VideoGenerationSetRecord
    recorded_at: datetime

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 1
        ):
            raise ValueError("sequence must be a positive integer")
        if not isinstance(self.event_type, VideoGenerationSetEventType):
            raise TypeError("event_type must be a VideoGenerationSetEventType")
        if not isinstance(self.record, VideoGenerationSetRecord):
            raise TypeError("record must be a VideoGenerationSetRecord")
        if (
            not isinstance(self.recorded_at, datetime)
            or self.recorded_at.tzinfo is None
            or self.recorded_at.utcoffset() != timedelta(0)
        ):
            raise ValueError("recorded_at must be timezone-aware UTC")


def searchable_units_manifest_sha256(units: Iterable[SearchableUnit]) -> str:
    """Hash one ordered unique canonical searchable-unit manifest."""
    digest = hashlib.sha256(b"[")
    seen: set[str] = set()
    for unit in units:
        if not isinstance(unit, SearchableUnit):
            raise TypeError("units must contain SearchableUnit values")
        if unit.unit_id in seen:
            raise ValueError("units must not contain duplicate unit IDs")
        if seen:
            digest.update(b",")
        seen.add(unit.unit_id)
        digest.update(
            json.dumps(
                _unit_payload(unit),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        )
    digest.update(b"]")
    return digest.hexdigest()


def visual_entries_content_sha256(entries: Iterable[VisualIndexEntry]) -> str:
    """Hash ordered visual identities and their canonical float32 payload values."""
    digest = hashlib.sha256(b"[")
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, VisualIndexEntry):
            raise TypeError("entries must contain VisualIndexEntry values")
        if entry.unit.unit_id in seen:
            raise ValueError("entries must not contain duplicate unit IDs")
        if seen:
            digest.update(b",")
        seen.add(entry.unit.unit_id)
        payload = {
            "unit_id": entry.unit.unit_id,
            "target_type": entry.unit.unit_type.value,
            "vector_float32_be": [struct.pack("!f", value).hex() for value in entry.vector],
        }
        digest.update(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            ).encode()
        )
    digest.update(b"]")
    return digest.hexdigest()


def _visual_manifest_digest(manifest: VisualIndexGenerationManifest) -> str:
    return _canonical_sha256(
        {
            "generation_id": manifest.generation_id,
            "workspace_ref": manifest.workspace_ref,
            "corpus_refs": list(manifest.corpus_refs),
            "target_type": manifest.target_type.value,
            "searchable_unit_manifest_sha256": manifest.searchable_unit_manifest_sha256,
            "content_sha256": manifest.content_sha256,
            "embedding_space": _embedding_space_payload(manifest.embedding_space),
            "retriever_id": manifest.retriever_id,
            "index_format_version": manifest.index_format_version,
            "build_processing_fingerprint": manifest.build_processing_fingerprint.sha256,
            "expected_unit_count": manifest.expected_unit_count,
            "schema_version": manifest.schema_version,
        }
    )


def _video_set_manifest_digest(manifest: VideoGenerationSetManifest) -> str:
    return _canonical_sha256(
        {
            "set_id": manifest.set_id,
            "workspace_ref": manifest.workspace_ref,
            "profile_ref": manifest.profile_ref,
            "corpus_refs": list(manifest.corpus_refs),
            "unit_catalog_ref": manifest.unit_catalog_ref,
            "unit_manifest_sha256": manifest.unit_manifest_sha256,
            "keyframe_visual_manifest_sha256": manifest.keyframe_visual.manifest_sha256,
            "segment_visual_manifest_sha256": manifest.segment_visual.manifest_sha256,
            "asr_text_manifest_sha256": manifest.asr_text.manifest_sha256,
            "ocr_text_manifest_sha256": (
                None if manifest.ocr_text is None else manifest.ocr_text.manifest_sha256
            ),
            "fusion_profile_ref": manifest.fusion_profile_ref,
            "schema_version": manifest.schema_version,
        }
    )


def _unit_payload(unit: SearchableUnit) -> dict[str, object]:
    return {
        "unit_id": unit.unit_id,
        "unit_type": unit.unit_type.value,
        "source_hash": unit.source_hash,
        "artifact_ref": unit.artifact_ref,
        "evidence_ref": unit.evidence_ref,
        "representation_refs": list(unit.representation_refs),
        "processing_run_ref": unit.processing_run_ref,
        "processing_version": unit.processing_version,
        "processing_fingerprint": unit.processing_fingerprint.sha256,
        "embedding_spaces": [_embedding_space_payload(space) for space in unit.embedding_spaces],
        "timestamp_ms": unit.timestamp_ms,
        "scene_range": _range_payload(unit.scene_range),
        "time_range": _range_payload(unit.time_range),
        "text_locator": (
            None
            if unit.text_locator is None
            else {
                "origin": unit.text_locator.origin.value,
                "character_range": {
                    "start_char": unit.text_locator.character_range.start_char,
                    "end_char": unit.text_locator.character_range.end_char,
                },
                "page_number": unit.text_locator.page_number,
                "time_range": _range_payload(unit.text_locator.time_range),
            }
        ),
    }


def _embedding_space_payload(space: EmbeddingSpaceRef) -> dict[str, object]:
    return {
        "space_id": space.space_id,
        "provider_id": space.provider_id,
        "model_revision": space.model_revision,
        "preprocessing_version": space.preprocessing_version,
        "dimension": space.dimension,
        "normalization": space.normalization.value,
        "distance_metric": space.distance_metric.value,
        "query_modalities": sorted(value.value for value in space.query_modalities),
        "target_modalities": sorted(value.value for value in space.target_modalities),
        "schema_version": space.schema_version,
    }


def _range_payload(value: object) -> dict[str, int] | None:
    if value is None:
        return None
    return {"start_ms": value.start_ms, "end_ms": value.end_ms}  # type: ignore[attr-defined]


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_visual_target(value: object) -> None:
    if not isinstance(value, TargetType):
        raise TypeError("target_type must be a TargetType")
    if value not in _VISUAL_TARGETS:
        raise ValueError("target_type must be a visual target")


def _require_identity(field_name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip() or len(value) > _MAX_IDENTITY_LENGTH:
        raise ValueError(f"{field_name} must be non-blank and bounded")


def _require_sha256(field_name: str, value: object) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hash")


def _freeze_identities(field_name: str, values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} values must be an iterable of strings")
    materialized = tuple(values)
    if len(set(materialized)) != len(materialized):
        raise ValueError(f"{field_name} values must not contain duplicates")
    for value in materialized:
        _require_identity(field_name, value)
    return materialized


def _require_non_negative_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


__all__ = [
    "PinnedVideoGenerationSet",
    "PinnedVisualIndexGeneration",
    "VideoGenerationConflictError",
    "VideoGenerationError",
    "VideoGenerationSetEvent",
    "VideoGenerationSetEventType",
    "VideoGenerationSetManifest",
    "VideoGenerationSetRecord",
    "VideoGenerationSetState",
    "VideoGenerationTransitionError",
    "VisualCandidate",
    "VisualIndexEntry",
    "VisualIndexGenerationManifest",
    "VisualIndexValidationReport",
    "searchable_units_manifest_sha256",
    "visual_entries_content_sha256",
]
