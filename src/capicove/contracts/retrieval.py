"""Canonical contracts for transport-neutral multimodal retrieval."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from itertools import islice
from typing import TypeAlias, TypeVar

from capicove.contracts._validation import (
    require_aware_datetime,
    require_non_blank,
    require_sha256,
)


RETRIEVAL_CONTRACT_VERSION = 4
_MAX_TEXT_SNIPPET_LENGTH = 2_048


class QueryModality(StrEnum):
    """Identify the single modality supplied by a search query."""

    TEXT = "text"
    IMAGE = "image"
    VIDEO_CLIP = "video_clip"
    AUDIO_CLIP = "audio_clip"


class TargetType(StrEnum):
    """Identify a retrievable unit type."""

    IMAGE = "image"
    KEYFRAME = "keyframe"
    VIDEO_SEGMENT = "video_segment"
    TEXT_CHUNK = "text_chunk"
    AUDIO_SEGMENT = "audio_segment"


class TextOrigin(StrEnum):
    """Identify how one immutable text representation was produced."""

    DOCUMENT = "document"
    OCR = "ocr"
    ASR = "asr"


class DetailLevel(StrEnum):
    """Select the bounded presentation projection for search results."""

    COMPACT = "compact"
    FULL = "full"


class ExecutionProfile(StrEnum):
    """Select server-relative provider execution policy."""

    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"


class SourceType(StrEnum):
    """Identify an accepted media-source reference shape."""

    ARTIFACT = "artifact"
    LOCAL_PATH = "local_path"
    MOUNT = "mount"
    INLINE_BYTES = "inline_bytes"


class VectorNormalization(StrEnum):
    """Identify the normalization applied to an embedding space."""

    NONE = "none"
    L2 = "l2"


class DistanceMetric(StrEnum):
    """Identify the distance or similarity semantics of an embedding space."""

    COSINE = "cosine"
    INNER_PRODUCT = "inner_product"
    EUCLIDEAN = "euclidean"


class MatchKind(StrEnum):
    """Classify why a full search hit matched a query."""

    NEAR_DUPLICATE = "near_duplicate"
    SEMANTIC = "semantic"
    VISUAL = "visual"
    OCR = "ocr"
    ASR = "asr"
    DIRECT_SEGMENT = "direct_segment"
    ASSEMBLED_SEGMENT = "assembled_segment"
    LEXICAL = "lexical"
    AUDIO = "audio"


class SearchStatus(StrEnum):
    """Describe the canonical outcome of retrieval orchestration."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


class SearchIntent(StrEnum):
    """Select the Agent-declared reverse-image semantic intent."""

    INSTANCE = "instance"
    CONCEPT = "concept"
    BALANCED = "balanced"


class SearchFailureCode(StrEnum):
    """Identify the single primary cause of a failed search."""

    SOURCE_NOT_AUTHORIZED = "source_not_authorized"
    PERMISSION_DENIED = "permission_denied"
    CLIENT_INGRESS_DENIED = "client_ingress_denied"
    INVALID_QUERY = "invalid_query"
    STALE_CURSOR = "stale_cursor"
    SOURCE_CHANGED = "source_changed"
    INPUT_TOO_LARGE = "input_too_large"
    INVALID_MEDIA = "invalid_media"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROFILE_UNAVAILABLE = "profile_unavailable"
    CAPACITY_EXCEEDED = "capacity_exceeded"
    BACKPRESSURE = "backpressure"
    INDEX_GENERATION_UNAVAILABLE = "index_generation_unavailable"
    EMBEDDING_SPACE_INCOMPATIBLE = "embedding_space_incompatible"
    HASH_SPACE_INCOMPATIBLE = "hash_space_incompatible"
    INDEX_CORRUPT = "index_corrupt"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    INTERNAL = "internal"


class SearchWarningCode(StrEnum):
    """Identify one supplemental degraded-search observation."""

    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROFILE_UNAVAILABLE = "profile_unavailable"
    INDEX_GENERATION_UNAVAILABLE = "index_generation_unavailable"
    EMBEDDING_SPACE_INCOMPATIBLE = "embedding_space_incompatible"
    HASH_SPACE_INCOMPATIBLE = "hash_space_incompatible"
    INDEX_CORRUPT = "index_corrupt"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    RETRIEVAL_PREVIEW = "retrieval_preview"
    RETRIEVAL_LANE_UNAVAILABLE = "retrieval_lane_unavailable"
    TEXT_LEXICAL_LANE_UNAVAILABLE = "text_lexical_lane_unavailable"
    TEXT_DENSE_EMBEDDING_UNAVAILABLE = "text_dense_embedding_unavailable"
    TEXT_DENSE_LANE_UNAVAILABLE = "text_dense_lane_unavailable"
    INTERNAL = "internal"


_SEARCH_EXECUTION_LANES = frozenset({"content", "perceptual", "instance", "concept"})
_SEARCH_ACTIVATION_TIERS = frozenset({"preview", "formal", "candidate_evaluation"})
MAX_CURSOR_BYTES = 4_096
MAX_FILTER_VALUES = 256
MAX_INLINE_IMAGE_BYTES = 4 * 1024 * 1024
MAX_LANGUAGE_BYTES = 128
MAX_MEDIA_TYPE_BYTES = 255
MAX_PATH_BYTES = 4_096
MAX_REF_BYTES = 1_024
MAX_TEMPORAL_MILLISECONDS = (1 << 63) - 1
MAX_TEXT_BYTES = 16_384
_MAX_REF_BYTES = MAX_REF_BYTES
_MAX_CURSOR_BYTES = MAX_CURSOR_BYTES
_MAX_HIT_ID_BYTES = 128
_MAX_EVIDENCE_ID_BYTES = 256
_MAX_EVIDENCE_REFS = 100
_MAX_LINEAGE_REFS = 64
_MAX_LANE_SCORES = 4
_MAX_SPACES = 4
_MAX_WARNING_CODES = 16
_MAX_HITS = 100
_MAX_LANE_TEXT_BYTES = 128
_MAX_LANE_RANK = 100_000
_MAX_EMBEDDING_DIMENSION = 16_384


@dataclass(frozen=True, slots=True)
class ArtifactSource:
    """Reference an immutable artifact already known to the active node."""

    artifact_ref: str
    type: SourceType = field(init=False, default=SourceType.ARTIFACT)

    def __post_init__(self) -> None:
        _require_bounded_string("artifact_ref", self.artifact_ref, MAX_REF_BYTES)


@dataclass(frozen=True, slots=True)
class LocalPathSource:
    """Reference a node-local path subject to runtime authorization."""

    path: str
    type: SourceType = field(init=False, default=SourceType.LOCAL_PATH)

    def __post_init__(self) -> None:
        _require_bounded_string("path", self.path, MAX_PATH_BYTES)


@dataclass(frozen=True, slots=True)
class MountSource:
    """Reference a path relative to an authorized runtime mount."""

    mount_ref: str
    relative_path: str
    type: SourceType = field(init=False, default=SourceType.MOUNT)

    def __post_init__(self) -> None:
        _require_bounded_string("mount_ref", self.mount_ref, MAX_REF_BYTES)
        _require_bounded_string("relative_path", self.relative_path, MAX_PATH_BYTES)


@dataclass(frozen=True, slots=True)
class InlineBytesSource:
    """Carry bounded media bytes for transport-neutral source acceptance."""

    data: bytes
    media_type: str
    type: SourceType = field(init=False, default=SourceType.INLINE_BYTES)

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes) or not self.data:
            raise ValueError("data must contain non-empty bytes")
        if len(self.data) > MAX_INLINE_IMAGE_BYTES:
            raise ValueError(f"data must contain at most {MAX_INLINE_IMAGE_BYTES} bytes")
        _require_bounded_string("media_type", self.media_type, MAX_MEDIA_TYPE_BYTES)


MediaSource: TypeAlias = ArtifactSource | LocalPathSource | MountSource | InlineBytesSource


@dataclass(frozen=True, slots=True)
class TextQuery:
    """Search with one non-blank text query and optional language hint."""

    text: str
    language: str | None = None
    type: QueryModality = field(init=False, default=QueryModality.TEXT)

    def __post_init__(self) -> None:
        _require_bounded_string("text", self.text, MAX_TEXT_BYTES)
        if self.language is not None:
            _require_bounded_string("language", self.language, MAX_LANGUAGE_BYTES)


@dataclass(frozen=True, slots=True)
class ImageQuery:
    """Search with one image source."""

    source: MediaSource
    type: QueryModality = field(init=False, default=QueryModality.IMAGE)

    def __post_init__(self) -> None:
        _require_media_source(self.source)


@dataclass(frozen=True, slots=True)
class TemporalRange:
    """Represent a non-empty half-open interval in integer milliseconds."""

    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        _require_bounded_milliseconds("start_ms", self.start_ms)
        _require_bounded_milliseconds("end_ms", self.end_ms)
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")


@dataclass(frozen=True, slots=True)
class CharacterRange:
    """Represent a non-empty half-open range in Unicode code points."""

    start_char: int
    end_char: int

    def __post_init__(self) -> None:
        _require_non_negative_int("start_char", self.start_char)
        _require_non_negative_int("end_char", self.end_char)
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")


@dataclass(frozen=True, slots=True)
class TextChunkLocator:
    """Locate one text chunk within an immutable text representation."""

    origin: TextOrigin
    character_range: CharacterRange
    page_number: int | None = None
    time_range: TemporalRange | None = None

    def __post_init__(self) -> None:
        _require_enum("origin", self.origin, TextOrigin)
        if not isinstance(self.character_range, CharacterRange):
            raise TypeError("character_range must be a CharacterRange")
        if self.page_number is not None:
            _require_positive_int("page_number", self.page_number)
        if self.time_range is not None and not isinstance(self.time_range, TemporalRange):
            raise TypeError("time_range must be a TemporalRange")

        if self.origin is TextOrigin.DOCUMENT:
            if self.page_number is not None or self.time_range is not None:
                raise ValueError("document text forbids page and time coordinates")
            return
        if self.origin is TextOrigin.OCR:
            if (self.page_number is None) == (self.time_range is None):
                raise ValueError("OCR text requires exactly one of page_number or time_range")
            return
        if self.page_number is not None:
            raise ValueError("page_number is invalid for ASR text")
        if self.time_range is None:
            raise ValueError("time_range is required for ASR text")


@dataclass(frozen=True, slots=True)
class VideoClipQuery:
    """Search with one video source and an optional clip interval."""

    source: MediaSource
    time_range: TemporalRange | None = None
    type: QueryModality = field(init=False, default=QueryModality.VIDEO_CLIP)

    def __post_init__(self) -> None:
        _require_media_source(self.source)
        if self.time_range is not None and not isinstance(self.time_range, TemporalRange):
            raise TypeError("time_range must be a TemporalRange")


MAX_AUDIO_STREAM_INDEX = 63


@dataclass(frozen=True, slots=True)
class AudioClipQuery:
    """Search with one audio source, an optional range and an explicit stream ordinal.

    A source with several audio streams requires ``stream_index``; a query without
    a range must fit one configured window. Inline audio keeps the media ceiling.
    """

    source: MediaSource
    time_range: TemporalRange | None = None
    stream_index: int | None = None
    type: QueryModality = field(init=False, default=QueryModality.AUDIO_CLIP)

    def __post_init__(self) -> None:
        _require_media_source(self.source)
        if self.time_range is not None and not isinstance(self.time_range, TemporalRange):
            raise TypeError("time_range must be a TemporalRange")
        if self.stream_index is not None:
            if isinstance(self.stream_index, bool) or not isinstance(self.stream_index, int):
                raise TypeError("stream_index must be an integer")
            if not 0 <= self.stream_index <= MAX_AUDIO_STREAM_INDEX:
                raise ValueError(f"stream_index must be from 0 to {MAX_AUDIO_STREAM_INDEX}")


SearchQuery: TypeAlias = TextQuery | ImageQuery | VideoClipQuery | AudioClipQuery


@dataclass(frozen=True, slots=True)
class SearchFilters:
    """Contain the closed version-2 corpus, origin, and temporal predicates."""

    corpus_refs: frozenset[str] = frozenset()
    artifact_refs: frozenset[str] = frozenset()
    media_types: frozenset[str] = frozenset()
    text_origins: frozenset[TextOrigin] = frozenset()
    time_range: TemporalRange | None = None
    schema_version: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "corpus_refs",
            _freeze_bounded_strings(
                "corpus_ref", self.corpus_refs, MAX_REF_BYTES, MAX_FILTER_VALUES
            ),
        )
        object.__setattr__(
            self,
            "artifact_refs",
            _freeze_bounded_strings(
                "artifact_ref", self.artifact_refs, MAX_REF_BYTES, MAX_FILTER_VALUES
            ),
        )
        object.__setattr__(
            self,
            "media_types",
            _freeze_bounded_strings(
                "media_type", self.media_types, MAX_MEDIA_TYPE_BYTES, MAX_FILTER_VALUES
            ),
        )
        object.__setattr__(
            self,
            "text_origins",
            _freeze_enums("text_origins", self.text_origins, TextOrigin),
        )
        if self.time_range is not None and not isinstance(self.time_range, TemporalRange):
            raise TypeError("time_range must be a TemporalRange")
        _require_positive_int("schema_version", self.schema_version)


@dataclass(frozen=True, slots=True)
class SearchRequest:
    """Request retrieval without exposing provider or index implementation knobs."""

    query: SearchQuery
    targets: frozenset[TargetType]
    top_k: int = 20
    cursor: str | None = None
    detail_level: DetailLevel = DetailLevel.COMPACT
    retrieval_profile_ref: str = "local-default"
    execution_profile: ExecutionProfile = ExecutionProfile.AUTO
    data_policy_ref: str = "local-only"
    filters: SearchFilters = field(default_factory=SearchFilters)
    search_intent: SearchIntent = SearchIntent.BALANCED

    def __post_init__(self) -> None:
        if not isinstance(self.query, (TextQuery, ImageQuery, VideoClipQuery, AudioClipQuery)):
            raise TypeError("query must be a canonical SearchQuery")
        targets = _freeze_enums("targets", self.targets, TargetType)
        if not targets:
            raise ValueError("targets must not be empty")
        object.__setattr__(self, "targets", targets)
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise ValueError("top_k must be an integer from 1 to 100")
        if not 1 <= self.top_k <= 100:
            raise ValueError("top_k must be from 1 to 100")
        if self.cursor is not None:
            _require_bounded_string("cursor", self.cursor, MAX_CURSOR_BYTES)
        _require_enum("detail_level", self.detail_level, DetailLevel)
        _require_bounded_string("retrieval_profile_ref", self.retrieval_profile_ref, MAX_REF_BYTES)
        _require_enum("execution_profile", self.execution_profile, ExecutionProfile)
        _require_bounded_string("data_policy_ref", self.data_policy_ref, MAX_REF_BYTES)
        if not isinstance(self.filters, SearchFilters):
            raise TypeError("filters must be SearchFilters")
        _require_enum("search_intent", self.search_intent, SearchIntent)


@dataclass(frozen=True, slots=True)
class ProcessingFingerprint:
    """Identify every normalized input that can change a derived output."""

    sha256: str

    def __post_init__(self) -> None:
        require_sha256("sha256", self.sha256)

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, object]) -> ProcessingFingerprint:
        """Hash one JSON-compatible manifest using canonical field ordering."""
        if not isinstance(manifest, Mapping):
            raise TypeError("manifest must be a mapping")
        normalized = _canonical_json_value(manifest, "manifest")
        payload = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return cls(hashlib.sha256(payload).hexdigest())


@dataclass(frozen=True, slots=True)
class EmbeddingSpaceRef:
    """Identify one fully compatible query and target vector space."""

    space_id: str
    provider_id: str
    model_revision: str
    preprocessing_version: str
    dimension: int
    normalization: VectorNormalization
    distance_metric: DistanceMetric
    query_modalities: frozenset[QueryModality]
    target_modalities: frozenset[TargetType]
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in (
            "space_id",
            "provider_id",
            "model_revision",
            "preprocessing_version",
        ):
            _require_bounded_string(name, getattr(self, name), _MAX_REF_BYTES)
        _require_positive_int("dimension", self.dimension)
        if self.dimension > _MAX_EMBEDDING_DIMENSION:
            raise ValueError(f"dimension must be at most {_MAX_EMBEDDING_DIMENSION:,}")
        _require_enum("normalization", self.normalization, VectorNormalization)
        _require_enum("distance_metric", self.distance_metric, DistanceMetric)
        query_modalities = _freeze_enums(
            "query_modalities",
            self.query_modalities,
            QueryModality,
        )
        target_modalities = _freeze_enums(
            "target_modalities",
            self.target_modalities,
            TargetType,
        )
        if not query_modalities:
            raise ValueError("query_modalities must not be empty")
        if not target_modalities:
            raise ValueError("target_modalities must not be empty")
        object.__setattr__(self, "query_modalities", query_modalities)
        object.__setattr__(self, "target_modalities", target_modalities)
        _require_positive_int("schema_version", self.schema_version)

    def accepts(self, query: QueryModality, target: TargetType) -> bool:
        """Return whether the complete space supports a query-target pair."""
        return query in self.query_modalities and target in self.target_modalities


@dataclass(frozen=True, slots=True)
class HashSpaceRef:
    """Identify one completely compatible fixed-width perceptual-hash space."""

    space_id: str
    provider_id: str
    provider_version: str
    preprocessing_version: str
    algorithm: str
    bit_width: int
    hash_version: str
    target_modalities: frozenset[TargetType]
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in (
            "space_id",
            "provider_id",
            "provider_version",
            "preprocessing_version",
        ):
            _require_bounded_string(name, getattr(self, name), _MAX_REF_BYTES)
        _require_bounded_string("algorithm", self.algorithm, _MAX_REF_BYTES)
        if self.algorithm != "phash":
            raise ValueError("algorithm must be phash for schema version 1")
        if isinstance(self.bit_width, bool) or not isinstance(self.bit_width, int):
            raise ValueError("bit_width must be 64 for schema version 1")
        if self.bit_width != 64:
            raise ValueError("bit_width must be 64 for schema version 1")
        _require_bounded_string("hash_version", self.hash_version, _MAX_REF_BYTES)
        if self.hash_version != "phash-64-v1":
            raise ValueError("hash_version must be phash-64-v1 for schema version 1")
        target_modalities = _freeze_enums(
            "target_modalities",
            self.target_modalities,
            TargetType,
        )
        if not target_modalities:
            raise ValueError("target_modalities must not be empty")
        object.__setattr__(self, "target_modalities", target_modalities)
        _require_schema_version("schema_version", self.schema_version)


@dataclass(frozen=True, slots=True)
class IndexParametersRef:
    """Bind one generation to closed, content-addressed index parameters."""

    parameter_schema: str
    payload_ref: str
    sha256: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        _require_non_blank_string("parameter_schema", self.parameter_schema)
        _require_safe_opaque_ref("payload_ref", self.payload_ref)
        if not isinstance(self.sha256, str):
            raise TypeError("sha256 must be a string")
        require_sha256("sha256", self.sha256)
        _require_schema_version("schema_version", self.schema_version)


@dataclass(frozen=True, slots=True)
class IndexGeneration:
    """Describe one immutable lane-index generation."""

    generation_id: str
    activation_set_id: str
    workspace_id: str
    corpus_ref: str
    target_type: TargetType
    unit_manifest_sha256: str
    embedding_spaces: tuple[EmbeddingSpaceRef, ...]
    hash_spaces: tuple[HashSpaceRef, ...]
    retriever_id: str
    index_format: str
    index_parameters_ref: IndexParametersRef
    shard_refs: tuple[str, ...]
    calibration_profile_ref: str
    build_processing_fingerprint: ProcessingFingerprint
    created_at: datetime
    validation_report_ref: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in (
            "generation_id",
            "activation_set_id",
            "workspace_id",
            "corpus_ref",
            "retriever_id",
            "index_format",
            "calibration_profile_ref",
            "validation_report_ref",
        ):
            _require_safe_opaque_ref(name, getattr(self, name))
        _require_enum("target_type", self.target_type, TargetType)
        if not isinstance(self.unit_manifest_sha256, str):
            raise TypeError("unit_manifest_sha256 must be a string")
        require_sha256("unit_manifest_sha256", self.unit_manifest_sha256)

        embedding_spaces = _freeze_identity_values(
            "embedding_spaces",
            self.embedding_spaces,
            EmbeddingSpaceRef,
        )
        hash_spaces = _freeze_identity_values(
            "hash_spaces",
            self.hash_spaces,
            HashSpaceRef,
        )
        if not embedding_spaces and not hash_spaces:
            raise ValueError("generation must contain at least one embedding or hash space")
        if any(self.target_type not in space.target_modalities for space in embedding_spaces):
            raise ValueError("embedding_spaces must support the generation target_type")
        if any(self.target_type not in space.target_modalities for space in hash_spaces):
            raise ValueError("hash_spaces must support the generation target_type")
        object.__setattr__(self, "embedding_spaces", embedding_spaces)
        object.__setattr__(self, "hash_spaces", hash_spaces)

        if not isinstance(self.index_parameters_ref, IndexParametersRef):
            raise TypeError("index_parameters_ref must be an IndexParametersRef")
        shard_refs = _freeze_safe_opaque_refs("shard_ref", self.shard_refs)
        if not shard_refs:
            raise ValueError("shard_ref values must not be empty")
        object.__setattr__(self, "shard_refs", shard_refs)
        if not isinstance(self.build_processing_fingerprint, ProcessingFingerprint):
            raise TypeError("build_processing_fingerprint must be a ProcessingFingerprint")
        if not isinstance(self.created_at, datetime):
            raise TypeError("created_at must be a datetime")
        require_aware_datetime("created_at", self.created_at)
        _require_schema_version("schema_version", self.schema_version)


@dataclass(frozen=True, slots=True)
class SearchExecutionSummary:
    """Project bounded retrieval execution identity without source locators."""

    search_intent: SearchIntent
    selected_lanes: tuple[str, ...]
    provider_ids: tuple[str, ...]
    embedding_spaces: tuple[EmbeddingSpaceRef, ...]
    hash_spaces: tuple[HashSpaceRef, ...]
    index_generation_ids: tuple[str, ...]
    bundle_manifest_sha256s: tuple[str, ...]
    resolved_profile_ref: str
    execution_profile: ExecutionProfile
    activation_tier: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        _require_enum("search_intent", self.search_intent, SearchIntent)
        selected_lanes = _freeze_strings_tuple("selected_lanes", self.selected_lanes)
        if not selected_lanes:
            raise ValueError("selected_lanes must not be empty")
        if len(selected_lanes) > 4:
            raise ValueError("selected_lanes must contain at most 4 values")
        if any(lane not in _SEARCH_EXECUTION_LANES for lane in selected_lanes):
            raise ValueError("selected_lanes contains an unsupported lane")
        object.__setattr__(self, "selected_lanes", selected_lanes)

        provider_ids = _freeze_safe_opaque_refs("provider_ids", self.provider_ids)
        if len(provider_ids) > 4:
            raise ValueError("provider_ids must contain at most 4 values")
        object.__setattr__(self, "provider_ids", provider_ids)

        embedding_spaces = _freeze_identity_values(
            "embedding_spaces",
            self.embedding_spaces,
            EmbeddingSpaceRef,
        )
        hash_spaces = _freeze_identity_values(
            "hash_spaces",
            self.hash_spaces,
            HashSpaceRef,
        )
        if len(embedding_spaces) > 4:
            raise ValueError("embedding_spaces must contain at most 4 values")
        if len(hash_spaces) > 4:
            raise ValueError("hash_spaces must contain at most 4 values")
        object.__setattr__(self, "embedding_spaces", embedding_spaces)
        object.__setattr__(self, "hash_spaces", hash_spaces)

        generation_ids = _freeze_safe_opaque_refs(
            "index_generation_ids",
            self.index_generation_ids,
        )
        if len(generation_ids) > 4:
            raise ValueError("index_generation_ids must contain at most 4 values")
        object.__setattr__(self, "index_generation_ids", generation_ids)

        bundle_digests = _freeze_strings_tuple(
            "bundle_manifest_sha256s",
            self.bundle_manifest_sha256s,
        )
        if len(bundle_digests) > 2:
            raise ValueError("bundle_manifest_sha256s must contain at most 2 values")
        for digest in bundle_digests:
            require_sha256("bundle_manifest_sha256s", digest)
        object.__setattr__(self, "bundle_manifest_sha256s", bundle_digests)

        _require_safe_opaque_ref("resolved_profile_ref", self.resolved_profile_ref)
        _require_enum("execution_profile", self.execution_profile, ExecutionProfile)
        _require_non_blank_string("activation_tier", self.activation_tier)
        if self.activation_tier not in _SEARCH_ACTIVATION_TIERS:
            raise ValueError("activation_tier must be preview, formal, or candidate_evaluation")
        _require_schema_version("schema_version", self.schema_version)


@dataclass(frozen=True, slots=True)
class SearchableUnit:
    """Describe immutable metadata for one retrievable evidence unit."""

    unit_id: str
    unit_type: TargetType
    source_hash: str
    artifact_ref: str
    evidence_ref: str
    representation_refs: tuple[str, ...]
    processing_run_ref: str
    processing_version: str
    processing_fingerprint: ProcessingFingerprint
    embedding_spaces: tuple[EmbeddingSpaceRef, ...] = ()
    timestamp_ms: int | None = None
    scene_range: TemporalRange | None = None
    time_range: TemporalRange | None = None
    text_locator: TextChunkLocator | None = None

    def __post_init__(self) -> None:
        for name in ("unit_id",):
            _require_bounded_string(name, getattr(self, name), _MAX_HIT_ID_BYTES)
        _require_bounded_string("evidence_ref", self.evidence_ref, _MAX_EVIDENCE_ID_BYTES)
        for name in ("artifact_ref", "processing_run_ref", "processing_version"):
            _require_bounded_string(name, getattr(self, name), _MAX_REF_BYTES)
        _require_enum("unit_type", self.unit_type, TargetType)
        require_sha256("source_hash", self.source_hash)
        representation_refs = _freeze_bounded_strings_tuple(
            "representation_ref",
            self.representation_refs,
            _MAX_REF_BYTES,
        )
        if len(representation_refs) > _MAX_EVIDENCE_REFS:
            raise ValueError(
                f"representation_ref values must contain at most {_MAX_EVIDENCE_REFS} values"
            )
        if not representation_refs:
            raise ValueError("representation_ref values must not be empty")
        object.__setattr__(
            self,
            "representation_refs",
            representation_refs,
        )
        if not isinstance(self.processing_fingerprint, ProcessingFingerprint):
            raise TypeError("processing_fingerprint must be a ProcessingFingerprint")
        spaces = tuple(self.embedding_spaces)
        if any(not isinstance(space, EmbeddingSpaceRef) for space in spaces):
            raise TypeError("embedding_spaces must contain EmbeddingSpaceRef values")
        if len(spaces) > _MAX_SPACES:
            raise ValueError(f"embedding_spaces must contain at most {_MAX_SPACES} values")
        if len({space.space_id for space in spaces}) != len(spaces):
            raise ValueError("embedding_spaces must not repeat a space_id")
        object.__setattr__(self, "embedding_spaces", spaces)
        _validate_temporal_locator(
            self.unit_type,
            self.timestamp_ms,
            self.scene_range,
            self.time_range,
        )
        _validate_text_locator(self.unit_type, self.text_locator)


@dataclass(frozen=True, slots=True)
class LaneScore:
    """Record one lane's raw and calibrated ranking observations."""

    lane_id: str
    raw_score: float
    normalized_score: float
    score_semantics: str
    calibration_version: str
    rank: int

    def __post_init__(self) -> None:
        for name in ("lane_id",):
            _require_bounded_string(name, getattr(self, name), _MAX_LANE_TEXT_BYTES)
        for name in ("score_semantics", "calibration_version"):
            _require_bounded_string(name, getattr(self, name), _MAX_LANE_TEXT_BYTES)
        _require_finite("raw_score", self.raw_score)
        _require_unit_score("normalized_score", self.normalized_score)
        _require_positive_int("rank", self.rank)
        if self.rank > _MAX_LANE_RANK:
            raise ValueError(f"rank must be at most {_MAX_LANE_RANK}")


@dataclass(frozen=True, slots=True)
class CompactHit:
    """Return the bounded default projection of one ranked result."""

    id: str
    target_type: TargetType
    brief_score: float
    evidence_id: str
    timestamp_ms: int | None = None
    time_range: TemporalRange | None = None
    text_locator: TextChunkLocator | None = None
    snippet: str | None = None
    preview_resource_ref: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _require_bounded_string("id", self.id, _MAX_HIT_ID_BYTES)
        _require_enum("target_type", self.target_type, TargetType)
        _require_unit_score("brief_score", self.brief_score)
        _require_bounded_string("evidence_id", self.evidence_id, _MAX_EVIDENCE_ID_BYTES)
        _validate_temporal_locator(
            self.target_type,
            self.timestamp_ms,
            None,
            self.time_range,
        )
        _validate_text_locator(self.target_type, self.text_locator)
        _validate_text_snippet(self.target_type, self.snippet)
        if self.preview_resource_ref is not None:
            require_non_blank("preview_resource_ref", self.preview_resource_ref)


@dataclass(frozen=True, slots=True)
class FullHit(CompactHit):
    """Return bounded diagnostics and lineage for one ranked result."""

    artifact_ref: str | None = None
    searchable_unit_ref: str | None = None
    representation_refs: tuple[str, ...] = ()
    scene_range: TemporalRange | None = None
    match_kinds: tuple[MatchKind, ...] = ()
    lane_scores: tuple[LaneScore, ...] = ()
    embedding_spaces: tuple[EmbeddingSpaceRef, ...] = ()
    processing_fingerprint: ProcessingFingerprint | None = None
    index_generation_ids: tuple[str, ...] = ()
    provider_ids: tuple[str, ...] = ()
    execution_profile: ExecutionProfile | None = None
    warning_codes: tuple[str, ...] = ()
    lineage_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        CompactHit.__post_init__(self)
        for name in ("artifact_ref", "searchable_unit_ref"):
            value = getattr(self, name)
            if value is not None:
                _require_bounded_string(name, value, _MAX_REF_BYTES)
        representation_refs = _freeze_bounded_strings_tuple(
            "representation_ref",
            self.representation_refs,
            _MAX_REF_BYTES,
        )
        if len(representation_refs) > _MAX_EVIDENCE_REFS:
            raise ValueError(
                f"representation_ref values must contain at most {_MAX_EVIDENCE_REFS} values"
            )
        object.__setattr__(
            self,
            "representation_refs",
            representation_refs,
        )
        _validate_temporal_locator(
            self.target_type,
            self.timestamp_ms,
            self.scene_range,
            self.time_range,
        )
        match_kinds = tuple(self.match_kinds)
        if any(not isinstance(kind, MatchKind) for kind in match_kinds):
            raise TypeError("match_kinds must contain MatchKind values")
        if len(set(match_kinds)) != len(match_kinds):
            raise ValueError("match_kinds must not contain duplicates")
        if len(match_kinds) > len(MatchKind):
            raise ValueError(f"match_kinds must contain at most {len(MatchKind)} values")
        object.__setattr__(self, "match_kinds", match_kinds)
        lane_scores = tuple(self.lane_scores)
        if any(not isinstance(score, LaneScore) for score in lane_scores):
            raise TypeError("lane_scores must contain LaneScore values")
        if len(lane_scores) > _MAX_LANE_SCORES:
            raise ValueError(f"lane_scores must contain at most {_MAX_LANE_SCORES} values")
        if len({score.lane_id for score in lane_scores}) != len(lane_scores):
            raise ValueError("lane_scores must not repeat a lane_id")
        object.__setattr__(self, "lane_scores", lane_scores)
        spaces = tuple(self.embedding_spaces)
        if any(not isinstance(space, EmbeddingSpaceRef) for space in spaces):
            raise TypeError("embedding_spaces must contain EmbeddingSpaceRef values")
        if len(spaces) > _MAX_SPACES:
            raise ValueError(f"embedding_spaces must contain at most {_MAX_SPACES} values")
        if len({space.space_id for space in spaces}) != len(spaces):
            raise ValueError("embedding_spaces must not repeat a space_id")
        object.__setattr__(self, "embedding_spaces", spaces)
        if self.processing_fingerprint is not None and not isinstance(
            self.processing_fingerprint,
            ProcessingFingerprint,
        ):
            raise TypeError("processing_fingerprint must be a ProcessingFingerprint")
        for field_name, singular, maximum in (
            ("index_generation_ids", "index_generation_id", _MAX_LANE_SCORES),
            ("provider_ids", "provider_id", _MAX_LANE_SCORES),
            ("warning_codes", "warning_code", _MAX_WARNING_CODES),
            ("lineage_refs", "lineage_ref", _MAX_LINEAGE_REFS),
        ):
            values = _freeze_bounded_strings_tuple(
                singular,
                getattr(self, field_name),
                _MAX_REF_BYTES,
            )
            if len(values) > maximum:
                raise ValueError(f"{field_name} must contain at most {maximum} values")
            if field_name == "warning_codes":
                allowed_warnings = {warning.value for warning in SearchWarningCode}
                if any(value not in allowed_warnings for value in values):
                    raise ValueError("warning_codes contains an unsupported warning code")
            object.__setattr__(
                self,
                field_name,
                values,
            )
        if self.execution_profile is not None:
            _require_enum("execution_profile", self.execution_profile, ExecutionProfile)


SearchHit: TypeAlias = CompactHit | FullHit


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Return one canonical search result projection."""

    status: SearchStatus
    query_id: str
    degraded: bool = False
    warning_codes: tuple[SearchWarningCode, ...] = ()
    hits: tuple[SearchHit, ...] = ()
    next_cursor: str | None = None
    failure_code: SearchFailureCode | None = None
    execution_summary: SearchExecutionSummary | None = None

    def __post_init__(self) -> None:
        _require_enum("status", self.status, SearchStatus)
        _require_bounded_string("query_id", self.query_id, _MAX_REF_BYTES)
        warnings = _freeze_enum_tuple(
            "warning_codes",
            self.warning_codes,
            SearchWarningCode,
        )
        if len(warnings) > _MAX_WARNING_CODES:
            raise ValueError(f"warning_codes must contain at most {_MAX_WARNING_CODES} values")
        object.__setattr__(self, "warning_codes", warnings)
        hits = tuple(self.hits)
        if any(not isinstance(hit, (CompactHit, FullHit)) for hit in hits):
            raise TypeError("hits must contain canonical search hits")
        if len(hits) > _MAX_HITS:
            raise ValueError(f"hits must contain at most {_MAX_HITS} values")
        object.__setattr__(self, "hits", hits)
        if self.degraded and not warnings:
            raise ValueError("a degraded result must contain a warning code")
        if self.next_cursor is not None:
            _require_bounded_string("next_cursor", self.next_cursor, _MAX_CURSOR_BYTES)
        if self.failure_code is not None:
            _require_enum("failure_code", self.failure_code, SearchFailureCode)
        if self.status is SearchStatus.FAILED and self.failure_code is None:
            raise ValueError("failed search result requires a failure_code")
        if self.status is SearchStatus.SUCCEEDED and self.failure_code is not None:
            raise ValueError("succeeded search result must not contain a failure_code")
        if self.execution_summary is not None and not isinstance(
            self.execution_summary,
            SearchExecutionSummary,
        ):
            raise TypeError("execution_summary must be a SearchExecutionSummary")


_EnumT = TypeVar("_EnumT", bound=StrEnum)
_IdentityT = TypeVar("_IdentityT", EmbeddingSpaceRef, HashSpaceRef)


def _require_media_source(source: object) -> None:
    if not isinstance(
        source,
        (ArtifactSource, LocalPathSource, MountSource, InlineBytesSource),
    ):
        raise TypeError("source must be a canonical MediaSource")


def _require_enum(field_name: str, value: object, enum_type: type[_EnumT]) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(f"{field_name} must be a {enum_type.__name__}")


def _require_non_blank_string(field_name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    require_non_blank(field_name, value)
    return value


def _require_bounded_string(field_name: str, value: object, maximum_bytes: int) -> str:
    normalized = _require_non_blank_string(field_name, value)
    if len(normalized.encode("utf-8")) > maximum_bytes:
        raise ValueError(f"{field_name} must be at most {maximum_bytes} UTF-8 bytes")
    return normalized


def _require_schema_version(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != 1:
        raise ValueError(f"{field_name} must be 1")


def _require_safe_opaque_ref(field_name: str, value: object) -> None:
    normalized = _require_bounded_string(field_name, value, _MAX_REF_BYTES)
    lowered = normalized.lower()
    is_drive_path = len(normalized) >= 3 and normalized[1] == ":" and normalized[2] in {"/", "\\"}
    if (
        normalized.startswith(("/", "\\"))
        or ".." in normalized
        or lowered.startswith("file:")
        or is_drive_path
        or "\x00" in normalized
    ):
        raise ValueError(f"{field_name} must be a safe opaque reference")


def _require_non_negative_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _require_bounded_milliseconds(field_name: str, value: object) -> None:
    _require_non_negative_int(field_name, value)
    if value > MAX_TEMPORAL_MILLISECONDS:  # type: ignore[operator]
        raise ValueError(f"{field_name} must fit signed 64-bit milliseconds")


def _require_positive_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


def _require_finite(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be finite")
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")


def _require_unit_score(field_name: str, value: object) -> None:
    _require_finite(field_name, value)
    if not 0 <= value <= 1:  # type: ignore[operator]
        raise ValueError(f"{field_name} must be within [0, 1]")


def _freeze_strings(field_name: str, values: Iterable[str]) -> frozenset[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} values must be an iterable of strings")
    materialized = tuple(values)
    if len(set(materialized)) != len(materialized):
        raise ValueError(f"{field_name} values must not contain duplicates")
    for value in materialized:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} values must be strings")
        require_non_blank(field_name, value)
    return frozenset(materialized)


def _freeze_bounded_strings(
    field_name: str,
    values: Iterable[str],
    maximum_bytes: int,
    maximum_values: int,
) -> frozenset[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} values must be an iterable of strings")
    materialized = tuple(islice(values, maximum_values + 1))
    if len(materialized) > maximum_values:
        raise ValueError(f"{field_name} values must contain at most {maximum_values} values")
    frozen = _freeze_strings(field_name, materialized)
    for value in frozen:
        _require_bounded_string(field_name, value, maximum_bytes)
    return frozen


def _freeze_strings_tuple(field_name: str, values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} values must be an iterable of strings")
    materialized = tuple(values)
    if len(set(materialized)) != len(materialized):
        raise ValueError(f"{field_name} values must not contain duplicates")
    for value in materialized:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} values must be strings")
        require_non_blank(field_name, value)
    return materialized


def _freeze_bounded_strings_tuple(
    field_name: str,
    values: Iterable[str],
    maximum_bytes: int,
) -> tuple[str, ...]:
    frozen = _freeze_strings_tuple(field_name, values)
    for value in frozen:
        _require_bounded_string(field_name, value, maximum_bytes)
    return frozen


def _freeze_safe_opaque_refs(field_name: str, values: Iterable[str]) -> tuple[str, ...]:
    frozen = _freeze_strings_tuple(field_name, values)
    for value in frozen:
        _require_safe_opaque_ref(field_name, value)
    return frozen


def _freeze_enum_tuple(
    field_name: str,
    values: Iterable[_EnumT],
    enum_type: type[_EnumT],
) -> tuple[_EnumT, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of {enum_type.__name__}")
    materialized = tuple(values)
    if len(set(materialized)) != len(materialized):
        raise ValueError(f"{field_name} must not contain duplicates")
    if any(not isinstance(value, enum_type) for value in materialized):
        raise TypeError(f"{field_name} must contain {enum_type.__name__} values")
    return materialized


def _freeze_identity_values(
    field_name: str,
    values: Iterable[_IdentityT],
    value_type: type[_IdentityT],
) -> tuple[_IdentityT, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of {value_type.__name__}")
    materialized = tuple(values)
    if any(not isinstance(value, value_type) for value in materialized):
        raise TypeError(f"{field_name} must contain {value_type.__name__} values")
    if len({value.space_id for value in materialized}) != len(materialized):
        raise ValueError(f"{field_name} must not repeat a space_id")
    return materialized


def _freeze_enums(
    field_name: str,
    values: Iterable[_EnumT],
    enum_type: type[_EnumT],
) -> frozenset[_EnumT]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of {enum_type.__name__}")
    materialized = tuple(values)
    if len(set(materialized)) != len(materialized):
        raise ValueError(f"{field_name} must not contain duplicates")
    if any(not isinstance(value, enum_type) for value in materialized):
        raise TypeError(f"{field_name} must contain {enum_type.__name__} values")
    return frozenset(materialized)


def _validate_temporal_locator(
    target_type: TargetType,
    timestamp_ms: int | None,
    scene_range: TemporalRange | None,
    time_range: TemporalRange | None,
) -> None:
    if timestamp_ms is not None:
        _require_bounded_milliseconds("timestamp_ms", timestamp_ms)
    if scene_range is not None and not isinstance(scene_range, TemporalRange):
        raise TypeError("scene_range must be a TemporalRange")
    if time_range is not None and not isinstance(time_range, TemporalRange):
        raise TypeError("time_range must be a TemporalRange")
    if target_type is TargetType.IMAGE:
        if timestamp_ms is not None or scene_range is not None or time_range is not None:
            raise ValueError("image targets must not contain a temporal locator")
        return
    if target_type is TargetType.KEYFRAME:
        if timestamp_ms is None:
            raise ValueError("timestamp_ms is required for a keyframe target")
        if time_range is not None:
            raise ValueError("time_range is invalid for a keyframe target")
        if scene_range is not None and not (
            scene_range.start_ms <= timestamp_ms < scene_range.end_ms
        ):
            raise ValueError("timestamp_ms must fall within scene_range")
        return
    if target_type is TargetType.TEXT_CHUNK:
        if timestamp_ms is not None or scene_range is not None or time_range is not None:
            raise ValueError("text chunk targets use only a text locator")
        return
    if target_type is TargetType.AUDIO_SEGMENT:
        if timestamp_ms is not None or scene_range is not None:
            raise ValueError("audio segment targets use only time_range")
        if time_range is None:
            raise ValueError("time_range is required for an audio segment target")
        return
    if timestamp_ms is not None or scene_range is not None:
        raise ValueError("video segment targets use only time_range")
    if time_range is None:
        raise ValueError("time_range is required for a video segment target")


def _validate_text_locator(
    target_type: TargetType,
    text_locator: TextChunkLocator | None,
) -> None:
    if text_locator is not None and not isinstance(text_locator, TextChunkLocator):
        raise TypeError("text_locator must be a TextChunkLocator")
    if target_type is TargetType.TEXT_CHUNK:
        if text_locator is None:
            raise ValueError("text_locator is required for a text chunk target")
        return
    if text_locator is not None:
        raise ValueError("text_locator is valid only for a text chunk target")


def _validate_text_snippet(target_type: TargetType, snippet: str | None) -> None:
    if target_type is TargetType.TEXT_CHUNK:
        if snippet is None:
            raise ValueError("snippet is required for a text chunk target")
        require_non_blank("snippet", snippet)
        if len(snippet) > _MAX_TEXT_SNIPPET_LENGTH:
            raise ValueError("snippet must contain at most 2048 characters")
        normalized = snippet.casefold()
        if "<em>" in normalized or "</em>" in normalized:
            raise ValueError("snippet must not contain provider highlight markup")
        return
    if snippet is not None:
        raise ValueError("snippet is valid only for a text chunk target")


def _canonical_json_value(value: object, field_name: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must contain only finite numbers")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field_name} mapping keys must be strings")
            normalized[key] = _canonical_json_value(item, f"{field_name}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _canonical_json_value(item, f"{field_name}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{field_name} contains a non-JSON value: {type(value).__name__}")


__all__ = [
    "MAX_AUDIO_STREAM_INDEX",
    "MAX_CURSOR_BYTES",
    "MAX_FILTER_VALUES",
    "MAX_INLINE_IMAGE_BYTES",
    "MAX_LANGUAGE_BYTES",
    "MAX_MEDIA_TYPE_BYTES",
    "MAX_PATH_BYTES",
    "MAX_REF_BYTES",
    "MAX_TEMPORAL_MILLISECONDS",
    "MAX_TEXT_BYTES",
    "RETRIEVAL_CONTRACT_VERSION",
    "ArtifactSource",
    "AudioClipQuery",
    "CharacterRange",
    "CompactHit",
    "DetailLevel",
    "DistanceMetric",
    "EmbeddingSpaceRef",
    "ExecutionProfile",
    "FullHit",
    "HashSpaceRef",
    "ImageQuery",
    "IndexGeneration",
    "IndexParametersRef",
    "InlineBytesSource",
    "LaneScore",
    "LocalPathSource",
    "MatchKind",
    "MediaSource",
    "MountSource",
    "ProcessingFingerprint",
    "QueryModality",
    "SearchExecutionSummary",
    "SearchFailureCode",
    "SearchFilters",
    "SearchHit",
    "SearchIntent",
    "SearchQuery",
    "SearchRequest",
    "SearchResult",
    "SearchStatus",
    "SearchWarningCode",
    "SearchableUnit",
    "SourceType",
    "TargetType",
    "TemporalRange",
    "TextChunkLocator",
    "TextOrigin",
    "TextQuery",
    "VectorNormalization",
    "VideoClipQuery",
]
