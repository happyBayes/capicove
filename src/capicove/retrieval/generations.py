"""Provider-neutral immutable text index generation values."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

from capicove.contracts import (
    EmbeddingSpaceRef,
    ProcessingFingerprint,
    QueryModality,
    TargetType,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_IDENTITY_LENGTH = 500


class TextIndexGenerationState(StrEnum):
    """Describe the durable lifecycle state of one text index generation."""

    BUILDING = "building"
    READY = "ready"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    CORRUPT = "corrupt"


class TextIndexGenerationEventType(StrEnum):
    """Describe one append-only generation lifecycle observation."""

    REGISTERED = "registered"
    PHYSICAL_INDEX_CREATED = "physical_index_created"
    VALIDATION_SUCCEEDED = "validation_succeeded"
    VALIDATION_FAILED = "validation_failed"
    ACTIVATED = "activated"
    SUPERSEDED = "superseded"


class TextIndexGenerationCatalogError(RuntimeError):
    """Report a bounded authoritative generation-catalog failure."""


class TextIndexGenerationConflictError(TextIndexGenerationCatalogError):
    """Report immutable generation identity reused for different evidence."""


class TextIndexGenerationTransitionError(TextIndexGenerationCatalogError):
    """Report an illegal or stale generation lifecycle transition."""


@dataclass(frozen=True, slots=True)
class TextIndexGenerationManifest:
    """Describe one Capicove-owned immutable text index generation."""

    generation_id: str
    workspace_ref: str
    corpus_refs: tuple[str, ...]
    searchable_unit_manifest_sha256: str
    content_sha256: str
    mapping_sha256: str
    embedding_spaces: tuple[EmbeddingSpaceRef, ...]
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
        for field_name in (
            "searchable_unit_manifest_sha256",
            "content_sha256",
            "mapping_sha256",
        ):
            _require_sha256(field_name, getattr(self, field_name))
        spaces = tuple(self.embedding_spaces)
        if any(not isinstance(space, EmbeddingSpaceRef) for space in spaces):
            raise TypeError("embedding_spaces must contain EmbeddingSpaceRef values")
        if len({space.space_id for space in spaces}) != len(spaces):
            raise ValueError("embedding_spaces must not contain duplicate embedding spaces")
        if any(not space.accepts(QueryModality.TEXT, TargetType.TEXT_CHUNK) for space in spaces):
            raise ValueError("embedding spaces must accept text queries and text chunks")
        object.__setattr__(
            self,
            "embedding_spaces",
            tuple(sorted(spaces, key=lambda space: space.space_id)),
        )
        if not isinstance(self.build_processing_fingerprint, ProcessingFingerprint):
            raise TypeError("build_processing_fingerprint must be a ProcessingFingerprint")
        if (
            isinstance(self.expected_unit_count, bool)
            or not isinstance(self.expected_unit_count, int)
            or self.expected_unit_count < 0
        ):
            raise ValueError("expected_unit_count must be a non-negative integer")
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        object.__setattr__(self, "manifest_sha256", _manifest_digest(self))


@dataclass(frozen=True, slots=True)
class TextIndexValidationReport:
    """Record bounded validation evidence for one physical text index."""

    generation_id: str
    manifest_sha256: str
    physical_index_ref: str
    mapping_sha256: str
    expected_unit_count: int
    observed_unit_count: int
    issue_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identity("generation_id", self.generation_id)
        _require_sha256("manifest_sha256", self.manifest_sha256)
        _require_identity("physical_index_ref", self.physical_index_ref)
        _require_sha256("mapping_sha256", self.mapping_sha256)
        for field_name in ("expected_unit_count", "observed_unit_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        issue_codes = _freeze_identities("issue_code", self.issue_codes)
        object.__setattr__(self, "issue_codes", tuple(sorted(issue_codes)))
        count_mismatch = self.expected_unit_count != self.observed_unit_count
        if count_mismatch != ("document_count_mismatch" in issue_codes):
            raise ValueError("document_count_mismatch must reflect the observed document count")

    @property
    def valid(self) -> bool:
        """Return whether the generation passed every integrity check."""
        return not self.issue_codes


@dataclass(frozen=True, slots=True)
class PinnedTextIndexGeneration:
    """Identify one concrete generation resolved before candidate retrieval."""

    generation_id: str
    manifest_sha256: str
    physical_index_ref: str

    def __post_init__(self) -> None:
        _require_identity("generation_id", self.generation_id)
        _require_sha256("manifest_sha256", self.manifest_sha256)
        _require_identity("physical_index_ref", self.physical_index_ref)


@dataclass(frozen=True, slots=True)
class TextIndexGenerationRecord:
    """Project the current Capicove-owned state of one generation."""

    manifest: TextIndexGenerationManifest
    state: TextIndexGenerationState
    revision: int
    physical_index_ref: str | None = None
    validation: TextIndexValidationReport | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, TextIndexGenerationManifest):
            raise TypeError("manifest must be a TextIndexGenerationManifest")
        if not isinstance(self.state, TextIndexGenerationState):
            raise TypeError("state must be a TextIndexGenerationState")
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 0
        ):
            raise ValueError("revision must be a non-negative integer")
        if self.physical_index_ref is not None:
            _require_identity("physical_index_ref", self.physical_index_ref)
        if self.validation is not None:
            self._validate_report_identity(self.validation)

        if self.state is TextIndexGenerationState.BUILDING:
            if self.validation is not None:
                raise ValueError("building generation must not have validation evidence")
            return

        if self.physical_index_ref is None or self.validation is None:
            raise ValueError("validated generation requires physical and validation evidence")
        if self.state is TextIndexGenerationState.CORRUPT:
            if self.validation.valid:
                raise ValueError("corrupt generation requires failed validation evidence")
            return
        if (
            not self.validation.valid
            or self.validation.mapping_sha256 != self.manifest.mapping_sha256
        ):
            raise ValueError(f"{self.state.value} generation requires valid validation evidence")

    def _validate_report_identity(self, report: TextIndexValidationReport) -> None:
        if report.generation_id != self.manifest.generation_id:
            raise ValueError("validation generation identity does not match manifest")
        if report.manifest_sha256 != self.manifest.manifest_sha256:
            raise ValueError("validation manifest identity does not match manifest")
        if report.expected_unit_count != self.manifest.expected_unit_count:
            raise ValueError("validation expected count does not match manifest")
        if self.physical_index_ref != report.physical_index_ref:
            raise ValueError("validation physical identity does not match generation")


@dataclass(frozen=True, slots=True)
class TextIndexGenerationEvent:
    """Capture one immutable snapshot in generation lifecycle order."""

    sequence: int
    event_type: TextIndexGenerationEventType
    record: TextIndexGenerationRecord
    recorded_at: datetime

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 1
        ):
            raise ValueError("sequence must be a positive integer")
        if not isinstance(self.event_type, TextIndexGenerationEventType):
            raise TypeError("event_type must be a TextIndexGenerationEventType")
        if not isinstance(self.record, TextIndexGenerationRecord):
            raise TypeError("record must be a TextIndexGenerationRecord")
        if (
            not isinstance(self.recorded_at, datetime)
            or self.recorded_at.tzinfo is None
            or self.recorded_at.utcoffset() != timedelta(0)
        ):
            raise ValueError("recorded_at must be timezone-aware UTC")


def _manifest_digest(manifest: TextIndexGenerationManifest) -> str:
    spaces = [
        {
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
        for space in manifest.embedding_spaces
    ]
    payload = {
        "generation_id": manifest.generation_id,
        "workspace_ref": manifest.workspace_ref,
        "corpus_refs": list(manifest.corpus_refs),
        "searchable_unit_manifest_sha256": manifest.searchable_unit_manifest_sha256,
        "content_sha256": manifest.content_sha256,
        "mapping_sha256": manifest.mapping_sha256,
        "embedding_spaces": spaces,
        "retriever_id": manifest.retriever_id,
        "index_format_version": manifest.index_format_version,
        "build_processing_fingerprint": manifest.build_processing_fingerprint.sha256,
        "expected_unit_count": manifest.expected_unit_count,
        "schema_version": manifest.schema_version,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


__all__ = [
    "PinnedTextIndexGeneration",
    "TextIndexGenerationCatalogError",
    "TextIndexGenerationConflictError",
    "TextIndexGenerationEvent",
    "TextIndexGenerationEventType",
    "TextIndexGenerationManifest",
    "TextIndexGenerationRecord",
    "TextIndexGenerationState",
    "TextIndexGenerationTransitionError",
    "TextIndexValidationReport",
]
