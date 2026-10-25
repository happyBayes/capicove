"""Canonical offline Tool boundary for multimodal search."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol, TypeVar

from capicove.contracts import (
    MAX_CURSOR_BYTES,
    MAX_FILTER_VALUES,
    MAX_INLINE_IMAGE_BYTES,
    MAX_LANGUAGE_BYTES,
    MAX_MEDIA_TYPE_BYTES,
    MAX_PATH_BYTES,
    MAX_REF_BYTES,
    MAX_TEMPORAL_MILLISECONDS,
    MAX_TEXT_BYTES,
    ArtifactSource,
    CompactHit,
    DetailLevel,
    DistanceMetric,
    EmbeddingSpaceRef,
    ExecutionProfile,
    FullHit,
    HashSpaceRef,
    ImageQuery,
    InlineBytesSource,
    InvocationContext,
    LaneScore,
    LocalPathSource,
    MatchKind,
    MountSource,
    PrivacyClass,
    ProcessingFingerprint,
    QueryModality,
    SearchExecutionSummary,
    SearchFailureCode,
    SearchFilters,
    SearchIntent,
    SearchRequest,
    SearchResult,
    SearchStatus,
    SearchWarningCode,
    SideEffect,
    TargetType,
    TemporalRange,
    TextQuery,
    ToolError,
    ToolErrorCode,
    ToolResources,
    ToolResult,
    ToolSpec,
    ToolStatus,
    VectorNormalization,
    VideoClipQuery,
)
from capicove.retrieval.search_models import InvocationAuthorizationContext
from capicove.tools.registry import ToolRegistry


MAX_INLINE_BASE64_CHARS = 4 * ((MAX_INLINE_IMAGE_BYTES + 2) // 3)
MAX_WARNING_CODES = 16
MAX_ERROR_MESSAGE_BYTES = 512
MAX_EVIDENCE_REFS = 100
MAX_LINEAGE_REFS = 64
MAX_LANE_SCORES = 4
MAX_SPACES = 4
MAX_HIT_ID_BYTES = 128
MAX_EVIDENCE_ID_BYTES = 256
MAX_COMPACT_OUTPUT_BYTES = 544 * 1_024
_EnumT = TypeVar("_EnumT", bound=StrEnum)
_SequenceT = TypeVar("_SequenceT")

_SHA256_SCHEMA: dict[str, object] = {
    "type": "string",
    "pattern": "^[0-9a-f]{64}$",
}
_TEMPORAL_RANGE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["start_ms", "end_ms"],
    "properties": {
        "start_ms": {
            "type": "integer",
            "minimum": 0,
            "maximum": MAX_TEMPORAL_MILLISECONDS,
        },
        "end_ms": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_TEMPORAL_MILLISECONDS,
        },
    },
}
_PROCESSING_FINGERPRINT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["sha256"],
    "properties": {"sha256": _SHA256_SCHEMA},
}
_EMBEDDING_SPACE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "space_id",
        "provider_id",
        "model_revision",
        "preprocessing_version",
        "dimension",
        "normalization",
        "distance_metric",
        "query_modalities",
        "target_modalities",
        "schema_version",
    ],
    "properties": {
        "space_id": {"type": "string", "minLength": 1, "maxLength": MAX_REF_BYTES},
        "provider_id": {"type": "string", "minLength": 1, "maxLength": MAX_REF_BYTES},
        "model_revision": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_REF_BYTES,
        },
        "preprocessing_version": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_REF_BYTES,
        },
        "dimension": {"type": "integer", "minimum": 1, "maximum": 16_384},
        "normalization": {"enum": ["none", "l2"]},
        "distance_metric": {"enum": ["cosine", "inner_product", "euclidean"]},
        "query_modalities": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "uniqueItems": True,
            "items": {"enum": ["text", "image", "video_clip"]},
        },
        "target_modalities": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "uniqueItems": True,
            "items": {"enum": ["image", "keyframe", "video_segment"]},
        },
        "schema_version": {"const": 1},
    },
}
_HASH_SPACE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "space_id",
        "provider_id",
        "provider_version",
        "preprocessing_version",
        "algorithm",
        "bit_width",
        "hash_version",
        "target_modalities",
        "schema_version",
    ],
    "properties": {
        "space_id": {"type": "string", "minLength": 1, "maxLength": MAX_REF_BYTES},
        "provider_id": {"type": "string", "minLength": 1, "maxLength": MAX_REF_BYTES},
        "provider_version": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_REF_BYTES,
        },
        "preprocessing_version": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_REF_BYTES,
        },
        "algorithm": {"const": "phash"},
        "bit_width": {"const": 64},
        "hash_version": {"const": "phash-64-v1"},
        "target_modalities": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "uniqueItems": True,
            "items": {"enum": ["image", "keyframe", "video_segment"]},
        },
        "schema_version": {"const": 1},
    },
}
_LANE_SCORE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "lane_id",
        "raw_score",
        "normalized_score",
        "score_semantics",
        "calibration_version",
        "rank",
    ],
    "properties": {
        "lane_id": {"enum": ["content", "perceptual", "instance", "concept"]},
        "raw_score": {"type": "number"},
        "normalized_score": {"type": "number", "minimum": 0, "maximum": 1},
        "score_semantics": {"type": "string", "minLength": 1, "maxLength": 128},
        "calibration_version": {
            "type": "string",
            "minLength": 1,
            "maxLength": 128,
        },
        "rank": {"type": "integer", "minimum": 1, "maximum": 100_000},
    },
}
_COMPACT_HIT_PROPERTIES: dict[str, object] = {
    "id": {"type": "string", "minLength": 1, "maxLength": MAX_HIT_ID_BYTES},
    "target_type": {"enum": ["image", "keyframe", "video_segment"]},
    "brief_score": {"type": "number", "minimum": 0, "maximum": 1},
    "evidence_id": {
        "type": "string",
        "minLength": 1,
        "maxLength": MAX_EVIDENCE_ID_BYTES,
    },
    "timestamp_ms": {
        "type": ["integer", "null"],
        "minimum": 0,
        "maximum": MAX_TEMPORAL_MILLISECONDS,
    },
    "time_range": {"anyOf": [_TEMPORAL_RANGE_SCHEMA, {"type": "null"}]},
}
COMPACT_HIT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "id",
        "target_type",
        "brief_score",
        "evidence_id",
        "timestamp_ms",
        "time_range",
    ],
    "properties": _COMPACT_HIT_PROPERTIES,
}
_FULL_HIT_PROPERTIES: dict[str, object] = {
    **_COMPACT_HIT_PROPERTIES,
    "artifact_ref": {
        "type": ["string", "null"],
        "minLength": 1,
        "maxLength": MAX_REF_BYTES,
    },
    "searchable_unit_ref": {
        "type": ["string", "null"],
        "minLength": 1,
        "maxLength": MAX_REF_BYTES,
    },
    "representation_refs": {
        "type": "array",
        "maxItems": MAX_EVIDENCE_REFS,
        "uniqueItems": True,
        "items": {"type": "string", "minLength": 1, "maxLength": MAX_REF_BYTES},
    },
    "scene_range": {"anyOf": [_TEMPORAL_RANGE_SCHEMA, {"type": "null"}]},
    "match_kinds": {
        "type": "array",
        "maxItems": 7,
        "uniqueItems": True,
        "items": {
            "enum": [
                "near_duplicate",
                "semantic",
                "visual",
                "ocr",
                "asr",
                "direct_segment",
                "assembled_segment",
            ]
        },
    },
    "lane_scores": {
        "type": "array",
        "maxItems": MAX_LANE_SCORES,
        "items": _LANE_SCORE_SCHEMA,
    },
    "embedding_spaces": {
        "type": "array",
        "maxItems": MAX_SPACES,
        "items": _EMBEDDING_SPACE_SCHEMA,
    },
    "processing_fingerprint": {"anyOf": [_PROCESSING_FINGERPRINT_SCHEMA, {"type": "null"}]},
    "index_generation_ids": {
        "type": "array",
        "maxItems": MAX_LANE_SCORES,
        "uniqueItems": True,
        "items": {"type": "string", "minLength": 1, "maxLength": MAX_REF_BYTES},
    },
    "provider_ids": {
        "type": "array",
        "maxItems": MAX_LANE_SCORES,
        "uniqueItems": True,
        "items": {"type": "string", "minLength": 1, "maxLength": MAX_REF_BYTES},
    },
    "execution_profile": {"anyOf": [{"enum": ["auto", "cpu", "gpu"]}, {"type": "null"}]},
    "warning_codes": {
        "type": "array",
        "maxItems": MAX_WARNING_CODES,
        "uniqueItems": True,
        "items": {"enum": [code.value for code in SearchWarningCode]},
    },
    "lineage_refs": {
        "type": "array",
        "maxItems": MAX_LINEAGE_REFS,
        "uniqueItems": True,
        "items": {"type": "string", "minLength": 1, "maxLength": MAX_REF_BYTES},
    },
}
FULL_HIT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "id",
        "target_type",
        "brief_score",
        "evidence_id",
        "timestamp_ms",
        "time_range",
        "artifact_ref",
        "searchable_unit_ref",
        "representation_refs",
        "scene_range",
        "match_kinds",
        "lane_scores",
        "embedding_spaces",
        "processing_fingerprint",
        "index_generation_ids",
        "provider_ids",
        "execution_profile",
        "warning_codes",
        "lineage_refs",
    ],
    "properties": _FULL_HIT_PROPERTIES,
}
EXECUTION_SUMMARY_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "search_intent",
        "selected_lanes",
        "provider_ids",
        "embedding_spaces",
        "hash_spaces",
        "index_generation_ids",
        "bundle_manifest_sha256s",
        "resolved_profile_ref",
        "execution_profile",
        "activation_tier",
        "schema_version",
    ],
    "properties": {
        "search_intent": {"enum": ["instance", "concept", "balanced"]},
        "selected_lanes": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_LANE_SCORES,
            "uniqueItems": True,
            "items": {"enum": ["content", "perceptual", "instance", "concept"]},
        },
        "provider_ids": {
            "type": "array",
            "maxItems": MAX_LANE_SCORES,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1, "maxLength": MAX_REF_BYTES},
        },
        "embedding_spaces": {
            "type": "array",
            "maxItems": MAX_SPACES,
            "items": _EMBEDDING_SPACE_SCHEMA,
        },
        "hash_spaces": {
            "type": "array",
            "maxItems": MAX_SPACES,
            "items": _HASH_SPACE_SCHEMA,
        },
        "index_generation_ids": {
            "type": "array",
            "maxItems": MAX_LANE_SCORES,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1, "maxLength": MAX_REF_BYTES},
        },
        "bundle_manifest_sha256s": {
            "type": "array",
            "maxItems": 2,
            "uniqueItems": True,
            "items": _SHA256_SCHEMA,
        },
        "resolved_profile_ref": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_REF_BYTES,
        },
        "execution_profile": {"enum": ["auto", "cpu", "gpu"]},
        "activation_tier": {"enum": ["preview", "formal", "candidate_evaluation"]},
        "schema_version": {"const": 1},
    },
}
_SOURCE_SCHEMA: dict[str, object] = {
    "oneOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "artifact_ref"],
            "properties": {
                "type": {"const": "artifact"},
                "artifact_ref": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_REF_BYTES,
                },
            },
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "path"],
            "properties": {
                "type": {"const": "local_path"},
                "path": {"type": "string", "minLength": 1, "maxLength": MAX_PATH_BYTES},
            },
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "mount_ref", "relative_path"],
            "properties": {
                "type": {"const": "mount"},
                "mount_ref": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_REF_BYTES,
                },
                "relative_path": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_PATH_BYTES,
                },
            },
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "base64", "media_type"],
            "properties": {
                "type": {"const": "inline_bytes"},
                "base64": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_INLINE_BASE64_CHARS,
                },
                "media_type": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_MEDIA_TYPE_BYTES,
                },
            },
        },
    ]
}
SEARCH_INPUT_SCHEMA: dict[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["query", "targets"],
    "properties": {
        "query": {
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["type", "text"],
                    "properties": {
                        "type": {"const": "text"},
                        "text": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_TEXT_BYTES,
                        },
                        "language": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_LANGUAGE_BYTES,
                        },
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["type", "source"],
                    "properties": {"type": {"const": "image"}, "source": _SOURCE_SCHEMA},
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["type", "source"],
                    "properties": {
                        "type": {"const": "video_clip"},
                        "source": _SOURCE_SCHEMA,
                        "time_range": _TEMPORAL_RANGE_SCHEMA,
                    },
                },
            ]
        },
        "targets": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "uniqueItems": True,
            "items": {"enum": ["image", "keyframe", "video_segment"]},
        },
        "search_intent": {
            "enum": ["instance", "concept", "balanced"],
            "default": "balanced",
        },
        "top_k": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
        "cursor": {"type": "string", "minLength": 1, "maxLength": MAX_CURSOR_BYTES},
        "detail_level": {"enum": ["compact", "full"], "default": "compact"},
        "retrieval_profile_ref": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_REF_BYTES,
            "default": "local-default",
        },
        "execution_profile": {"enum": ["auto", "cpu", "gpu"], "default": "auto"},
        "data_policy_ref": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_REF_BYTES,
            "default": "local-only",
        },
        "filters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "corpus_refs": {
                    "type": "array",
                    "maxItems": MAX_FILTER_VALUES,
                    "uniqueItems": True,
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_REF_BYTES,
                    },
                },
                "artifact_refs": {
                    "type": "array",
                    "maxItems": MAX_FILTER_VALUES,
                    "uniqueItems": True,
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_REF_BYTES,
                    },
                },
                "media_types": {
                    "type": "array",
                    "maxItems": MAX_FILTER_VALUES,
                    "uniqueItems": True,
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_MEDIA_TYPE_BYTES,
                    },
                },
                "time_range": _TEMPORAL_RANGE_SCHEMA,
                "schema_version": {"const": 1},
            },
        },
    },
}
SEARCH_OUTPUT_SCHEMA: dict[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "status",
        "query_id",
        "degraded",
        "warning_codes",
        "hits",
        "next_cursor",
        "failure_code",
        "execution_summary",
    ],
    "properties": {
        "status": {"enum": ["succeeded", "failed"]},
        "query_id": {"type": "string", "minLength": 1, "maxLength": MAX_REF_BYTES},
        "degraded": {"type": "boolean"},
        "warning_codes": {
            "type": "array",
            "maxItems": MAX_WARNING_CODES,
            "uniqueItems": True,
            "items": {"enum": [code.value for code in SearchWarningCode]},
        },
        "hits": {
            "type": "array",
            "maxItems": 100,
            "items": {"oneOf": [COMPACT_HIT_SCHEMA, FULL_HIT_SCHEMA]},
        },
        "next_cursor": {
            "type": ["string", "null"],
            "minLength": 1,
            "maxLength": MAX_CURSOR_BYTES,
        },
        "failure_code": {
            "anyOf": [
                {"enum": [code.value for code in SearchFailureCode]},
                {"type": "null"},
            ]
        },
        "execution_summary": {"anyOf": [EXECUTION_SUMMARY_SCHEMA, {"type": "null"}]},
    },
}
SEARCH_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(
        {"input": SEARCH_INPUT_SCHEMA, "output": SEARCH_OUTPUT_SCHEMA},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()
SEARCH_SPEC = ToolSpec(
    tool_id="capicove.search.image",
    contract_version=1,
    description=(
        "Search authorized local evidence. Use instance for the same person, "
        "object, scene, shot, diagram, slide, prop, or visual arrangement; use "
        "concept for theme, topic, genre, style, mood, or educational meaning; "
        "use balanced only when context cannot sufficiently support either "
        "interpretation. The server does not infer intent."
    ),
    input_schema=SEARCH_INPUT_SCHEMA,
    output_schema=SEARCH_OUTPUT_SCHEMA,
    modalities=("text", "image", "video"),
    required_permissions=frozenset({"retrieval:search"}),
    privacy_class=PrivacyClass.CONFIDENTIAL,
    side_effect=SideEffect.READ,
    network_access=False,
    deterministic=False,
    idempotent=True,
    timeout_seconds=30.0,
    redacted_fields=(
        "query.text",
        "query.source.artifact_ref",
        "query.source.base64",
        "query.source.path",
        "query.source.mount_ref",
        "query.source.relative_path",
        "query.source.media_type",
        "filters.corpus_refs.*",
        "filters.artifact_refs.*",
        "filters.media_types.*",
    ),
    resources=ToolResources(filesystem_read=True),
)


class SearchServicePort(Protocol):
    """Narrow search surface consumed by the Tool boundary."""

    def search(
        self,
        request: SearchRequest,
        *,
        authorization: InvocationAuthorizationContext,
        deadline: datetime,
    ) -> SearchResult:
        """Execute one canonical search under trusted invocation authority."""


class SearchAuthorizationFactory(Protocol):
    """Create trusted search authority from startup and invocation state only."""

    def create(self, context: InvocationContext) -> InvocationAuthorizationContext:
        """Create authority without observing request-controlled arguments."""


@dataclass(frozen=True, slots=True)
class StaticSearchAuthorizationFactory:
    """Bind startup-owned search policy to authenticated invocation identity."""

    authorization_scope_ref: str
    permissions: frozenset[str]
    allowed_data_policy_refs: frozenset[str]
    allowed_source_root_refs: frozenset[str]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.authorization_scope_ref, str)
            or not self.authorization_scope_ref.strip()
        ):
            raise ValueError("authorization_scope_ref must be a non-blank string")
        permissions = _immutable_string_set(self.permissions, "permissions", allow_empty=True)
        policies = _immutable_string_set(
            self.allowed_data_policy_refs,
            "allowed_data_policy_refs",
            allow_empty=False,
        )
        source_roots = _immutable_string_set(
            self.allowed_source_root_refs,
            "allowed_source_root_refs",
            allow_empty=True,
        )
        object.__setattr__(self, "permissions", permissions)
        object.__setattr__(self, "allowed_data_policy_refs", policies)
        object.__setattr__(self, "allowed_source_root_refs", source_roots)

    def create(self, context: InvocationContext) -> InvocationAuthorizationContext:
        """Intersect startup permission policy with authenticated permissions."""
        return InvocationAuthorizationContext(
            request_id=context.request_id,
            workspace_id=context.workspace_id,
            actor_id=context.actor_id,
            environment_id=context.environment_id,
            operation=SEARCH_SPEC.tool_id,
            authorization_scope_ref=self.authorization_scope_ref,
            permissions=self.permissions.intersection(context.permissions),
            allowed_data_policy_refs=self.allowed_data_policy_refs,
            allowed_source_root_refs=self.allowed_source_root_refs,
            privacy_class=context.privacy_class,
        )


_FAILURE_MAPPING: dict[
    SearchFailureCode,
    tuple[ToolStatus, ToolErrorCode, bool],
] = {
    SearchFailureCode.SOURCE_NOT_AUTHORIZED: (
        ToolStatus.REJECTED,
        ToolErrorCode.PERMISSION_DENIED,
        False,
    ),
    SearchFailureCode.PERMISSION_DENIED: (
        ToolStatus.REJECTED,
        ToolErrorCode.PERMISSION_DENIED,
        False,
    ),
    SearchFailureCode.CLIENT_INGRESS_DENIED: (
        ToolStatus.REJECTED,
        ToolErrorCode.POLICY_REJECTED,
        False,
    ),
    SearchFailureCode.INVALID_QUERY: (ToolStatus.FAILED, ToolErrorCode.INVALID_INPUT, False),
    SearchFailureCode.STALE_CURSOR: (ToolStatus.FAILED, ToolErrorCode.INVALID_INPUT, False),
    SearchFailureCode.SOURCE_CHANGED: (ToolStatus.FAILED, ToolErrorCode.INVALID_INPUT, False),
    SearchFailureCode.INPUT_TOO_LARGE: (ToolStatus.FAILED, ToolErrorCode.INVALID_INPUT, False),
    SearchFailureCode.INVALID_MEDIA: (ToolStatus.FAILED, ToolErrorCode.INVALID_INPUT, False),
    SearchFailureCode.PROVIDER_UNAVAILABLE: (
        ToolStatus.FAILED,
        ToolErrorCode.UNAVAILABLE,
        True,
    ),
    SearchFailureCode.INDEX_GENERATION_UNAVAILABLE: (
        ToolStatus.FAILED,
        ToolErrorCode.UNAVAILABLE,
        True,
    ),
    SearchFailureCode.PROFILE_UNAVAILABLE: (
        ToolStatus.FAILED,
        ToolErrorCode.UNAVAILABLE,
        False,
    ),
    SearchFailureCode.CAPACITY_EXCEEDED: (
        ToolStatus.FAILED,
        ToolErrorCode.UNAVAILABLE,
        False,
    ),
    SearchFailureCode.BACKPRESSURE: (
        ToolStatus.FAILED,
        ToolErrorCode.UNAVAILABLE,
        True,
    ),
    SearchFailureCode.EMBEDDING_SPACE_INCOMPATIBLE: (
        ToolStatus.FAILED,
        ToolErrorCode.UNAVAILABLE,
        False,
    ),
    SearchFailureCode.HASH_SPACE_INCOMPATIBLE: (
        ToolStatus.FAILED,
        ToolErrorCode.UNAVAILABLE,
        False,
    ),
    SearchFailureCode.INDEX_CORRUPT: (ToolStatus.FAILED, ToolErrorCode.INTERNAL, False),
    SearchFailureCode.DEADLINE_EXCEEDED: (ToolStatus.FAILED, ToolErrorCode.TIMEOUT, True),
    SearchFailureCode.INTERNAL: (ToolStatus.FAILED, ToolErrorCode.INTERNAL, False),
}


class SearchToolHandler:
    """Validate and invoke one offline search without performing media I/O."""

    def __init__(
        self,
        service: SearchServicePort,
        authorization_factory: SearchAuthorizationFactory,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not callable(getattr(service, "search", None)):
            raise TypeError("service must provide search")
        if not callable(getattr(authorization_factory, "create", None)):
            raise TypeError("authorization_factory must provide create")
        self._service = service
        self._authorization_factory = authorization_factory
        self._clock = clock or (lambda: datetime.now(UTC))

    def __call__(
        self,
        context: InvocationContext,
        arguments: Mapping[str, object],
        /,
    ) -> ToolResult:
        started_at = self._clock()
        if not SEARCH_SPEC.required_permissions.issubset(context.permissions):
            return _tool_failure(
                context,
                started_at,
                started_at,
                ToolStatus.REJECTED,
                ToolErrorCode.PERMISSION_DENIED,
                "Search permission denied",
                False,
            )
        if _deadline_reached(context.deadline, started_at):
            return _tool_failure(
                context,
                started_at,
                started_at,
                ToolStatus.FAILED,
                ToolErrorCode.TIMEOUT,
                "Search deadline exceeded",
                True,
            )
        try:
            request = _parse_search_request(arguments)
        except (TypeError, ValueError, binascii.Error, UnicodeError):
            return _tool_failure(
                context,
                started_at,
                started_at,
                ToolStatus.FAILED,
                ToolErrorCode.INVALID_INPUT,
                "Invalid search input",
                False,
            )
        except Exception:
            return _tool_failure(
                context,
                started_at,
                started_at,
                ToolStatus.FAILED,
                ToolErrorCode.INTERNAL,
                "Search failed internally",
                False,
            )
        try:
            authorization = self._authorization_factory.create(context)
            _validate_authorization(context, authorization)
        except Exception:
            return _tool_failure(
                context,
                started_at,
                started_at,
                ToolStatus.FAILED,
                ToolErrorCode.INTERNAL,
                "Search authorization failed internally",
                False,
            )
        try:
            search_result = self._service.search(
                request,
                authorization=authorization,
                deadline=context.deadline,
            )
        except Exception:
            return _tool_failure(
                context,
                started_at,
                started_at,
                ToolStatus.FAILED,
                ToolErrorCode.INTERNAL,
                "Search failed internally",
                False,
            )
        try:
            completed_at = self._clock()
        except Exception:
            return _tool_failure(
                context,
                started_at,
                started_at,
                ToolStatus.FAILED,
                ToolErrorCode.INTERNAL,
                "Search clock failed internally",
                False,
            )
        if _deadline_reached(context.deadline, completed_at):
            return _tool_failure(
                context,
                started_at,
                completed_at,
                ToolStatus.FAILED,
                ToolErrorCode.TIMEOUT,
                "Search deadline exceeded",
                True,
            )
        try:
            if not isinstance(search_result, SearchResult):
                raise TypeError("search service returned a non-canonical result")
            serialized = _serialize_search_result(search_result)
            if request.detail_level is DetailLevel.COMPACT:
                if any(isinstance(hit, FullHit) for hit in search_result.hits):
                    raise ValueError("compact request returned a full hit")
                _require_compact_output_budget(serialized)
            evidence_refs = _evidence_refs(search_result)
            if search_result.status is SearchStatus.SUCCEEDED:
                return ToolResult(
                    request_id=context.request_id,
                    trace_id=context.trace_id,
                    tool_id=SEARCH_SPEC.tool_id,
                    tool_version=SEARCH_SPEC.contract_version,
                    status=ToolStatus.SUCCEEDED,
                    result=serialized,
                    evidence_refs=evidence_refs,
                    started_at=started_at,
                    completed_at=completed_at,
                )
            failure_code = search_result.failure_code
            if failure_code is None or failure_code not in _FAILURE_MAPPING:
                raise ValueError("failed search result lacks one canonical failure code")
            status, error_code, retryable = _FAILURE_MAPPING[failure_code]
            return ToolResult(
                request_id=context.request_id,
                trace_id=context.trace_id,
                tool_id=SEARCH_SPEC.tool_id,
                tool_version=SEARCH_SPEC.contract_version,
                status=status,
                result=serialized,
                evidence_refs=evidence_refs,
                error=ToolError(
                    code=error_code,
                    message=_bounded_message(f"Search failed: {failure_code.value}"),
                    retryable=retryable,
                ),
                started_at=started_at,
                completed_at=completed_at,
            )
        except Exception:
            return _tool_failure(
                context,
                started_at,
                completed_at,
                ToolStatus.FAILED,
                ToolErrorCode.INTERNAL,
                "Search failed internally",
                False,
            )


def register_search_tool(
    registry: ToolRegistry,
    service: SearchServicePort,
    authorization_factory: SearchAuthorizationFactory,
    *,
    clock: Callable[[], datetime] | None = None,
) -> SearchToolHandler:
    """Register the sole canonical search Tool and return its handler."""
    handler = SearchToolHandler(service, authorization_factory, clock=clock)
    registry.register(SEARCH_SPEC, handler)
    return handler


def _deadline_reached(deadline: datetime, now: datetime) -> bool:
    return now >= deadline


def _validate_authorization(
    context: InvocationContext,
    authorization: InvocationAuthorizationContext,
) -> None:
    if not isinstance(authorization, InvocationAuthorizationContext):
        raise RuntimeError("authorization factory returned an invalid value")
    expected = (
        context.request_id,
        context.workspace_id,
        context.actor_id,
        context.environment_id,
        context.privacy_class,
    )
    actual = (
        authorization.request_id,
        authorization.workspace_id,
        authorization.actor_id,
        authorization.environment_id,
        authorization.privacy_class,
    )
    if actual != expected or authorization.operation != SEARCH_SPEC.tool_id:
        raise RuntimeError("authorization factory returned mismatched authority")
    if not SEARCH_SPEC.required_permissions.issubset(authorization.permissions):
        raise RuntimeError("authorization factory did not grant required startup authority")


def _parse_search_request(arguments: Mapping[str, object]) -> SearchRequest:
    root = _closed_mapping(
        arguments,
        allowed={
            "query",
            "targets",
            "search_intent",
            "top_k",
            "cursor",
            "detail_level",
            "retrieval_profile_ref",
            "execution_profile",
            "data_policy_ref",
            "filters",
        },
        required={"query", "targets"},
        name="arguments",
    )
    targets_raw = _array(root["targets"], "targets", minimum=1, maximum=3)
    target_values = tuple(_enum(value, TargetType, "target") for value in targets_raw)
    if len(set(target_values)) != len(target_values):
        raise ValueError("targets must be unique")
    top_k = _integer(root.get("top_k", 20), "top_k", minimum=1, maximum=100)
    cursor = (
        None
        if "cursor" not in root
        else _bounded_string(root["cursor"], "cursor", MAX_CURSOR_BYTES)
    )
    return SearchRequest(
        query=_parse_query(root["query"]),
        targets=frozenset(target_values),
        top_k=top_k,
        cursor=cursor,
        detail_level=_enum(root.get("detail_level", "compact"), DetailLevel, "detail_level"),
        retrieval_profile_ref=_bounded_string(
            root.get("retrieval_profile_ref", "local-default"),
            "retrieval_profile_ref",
            MAX_REF_BYTES,
        ),
        execution_profile=_enum(
            root.get("execution_profile", "auto"),
            ExecutionProfile,
            "execution_profile",
        ),
        data_policy_ref=_bounded_string(
            root.get("data_policy_ref", "local-only"),
            "data_policy_ref",
            MAX_REF_BYTES,
        ),
        filters=_parse_filters(root.get("filters", {})),
        search_intent=_enum(
            root.get("search_intent", "balanced"),
            SearchIntent,
            "search_intent",
        ),
    )


def _parse_query(value: object) -> TextQuery | ImageQuery | VideoClipQuery:
    query = _closed_mapping(
        value,
        allowed={"type", "text", "language", "source", "time_range"},
        required={"type"},
        name="query",
    )
    query_type = query["type"]
    if query_type == "text":
        exact = _closed_mapping(
            query,
            allowed={"type", "text", "language"},
            required={"type", "text"},
            name="text query",
        )
        return TextQuery(
            text=_bounded_string(exact["text"], "text", MAX_TEXT_BYTES),
            language=(
                None
                if "language" not in exact
                else _bounded_string(exact["language"], "language", MAX_LANGUAGE_BYTES)
            ),
        )
    if query_type == "image":
        exact = _closed_mapping(
            query,
            allowed={"type", "source"},
            required={"type", "source"},
            name="image query",
        )
        return ImageQuery(source=_parse_source(exact["source"]))
    if query_type == "video_clip":
        exact = _closed_mapping(
            query,
            allowed={"type", "source", "time_range"},
            required={"type", "source"},
            name="video clip query",
        )
        return VideoClipQuery(
            source=_parse_source(exact["source"]),
            time_range=(
                None if "time_range" not in exact else _parse_time_range(exact["time_range"])
            ),
        )
    raise ValueError("unsupported query type")


def _parse_source(
    value: object,
) -> ArtifactSource | LocalPathSource | MountSource | InlineBytesSource:
    source = _closed_mapping(
        value,
        allowed={
            "type",
            "artifact_ref",
            "path",
            "mount_ref",
            "relative_path",
            "base64",
            "media_type",
        },
        required={"type"},
        name="source",
    )
    source_type = source["type"]
    if source_type == "artifact":
        exact = _closed_mapping(
            source,
            allowed={"type", "artifact_ref"},
            required={"type", "artifact_ref"},
            name="artifact source",
        )
        return ArtifactSource(_bounded_string(exact["artifact_ref"], "artifact_ref", MAX_REF_BYTES))
    if source_type == "local_path":
        exact = _closed_mapping(
            source,
            allowed={"type", "path"},
            required={"type", "path"},
            name="local path source",
        )
        return LocalPathSource(_bounded_string(exact["path"], "path", MAX_PATH_BYTES))
    if source_type == "mount":
        exact = _closed_mapping(
            source,
            allowed={"type", "mount_ref", "relative_path"},
            required={"type", "mount_ref", "relative_path"},
            name="mount source",
        )
        return MountSource(
            mount_ref=_bounded_string(exact["mount_ref"], "mount_ref", MAX_REF_BYTES),
            relative_path=_bounded_string(
                exact["relative_path"],
                "relative_path",
                MAX_PATH_BYTES,
            ),
        )
    if source_type == "inline_bytes":
        exact = _closed_mapping(
            source,
            allowed={"type", "base64", "media_type"},
            required={"type", "base64", "media_type"},
            name="inline source",
        )
        encoded = _bounded_string(
            exact["base64"],
            "base64",
            MAX_INLINE_BASE64_CHARS,
            strip=False,
        )
        payload = base64.b64decode(encoded, validate=True)
        if not payload or len(payload) > MAX_INLINE_IMAGE_BYTES:
            raise ValueError("inline data must contain at most 4 MiB")
        return InlineBytesSource(
            data=payload,
            media_type=_bounded_string(exact["media_type"], "media_type", MAX_MEDIA_TYPE_BYTES),
        )
    raise ValueError("unsupported source type")


def _parse_filters(value: object) -> SearchFilters:
    filters = _closed_mapping(
        value,
        allowed={"corpus_refs", "artifact_refs", "media_types", "time_range", "schema_version"},
        required=set(),
        name="filters",
    )
    schema_version = _integer(
        filters.get("schema_version", 1),
        "schema_version",
        minimum=1,
        maximum=1,
    )
    return SearchFilters(
        corpus_refs=frozenset(
            _string_array(filters.get("corpus_refs", []), "corpus_refs", MAX_REF_BYTES)
        ),
        artifact_refs=frozenset(
            _string_array(filters.get("artifact_refs", []), "artifact_refs", MAX_REF_BYTES)
        ),
        media_types=frozenset(
            _string_array(filters.get("media_types", []), "media_types", MAX_MEDIA_TYPE_BYTES)
        ),
        time_range=(
            None if "time_range" not in filters else _parse_time_range(filters["time_range"])
        ),
        schema_version=schema_version,
    )


def _parse_time_range(value: object) -> TemporalRange:
    temporal = _closed_mapping(
        value,
        allowed={"start_ms", "end_ms"},
        required={"start_ms", "end_ms"},
        name="time_range",
    )
    return TemporalRange(
        start_ms=_integer(
            temporal["start_ms"],
            "start_ms",
            minimum=0,
            maximum=MAX_TEMPORAL_MILLISECONDS,
        ),
        end_ms=_integer(
            temporal["end_ms"],
            "end_ms",
            minimum=1,
            maximum=MAX_TEMPORAL_MILLISECONDS,
        ),
    )


def _closed_mapping(
    value: object,
    *,
    allowed: set[str],
    required: set[str],
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    if len(value) > len(allowed):
        raise ValueError(f"{name} has too many fields")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    keys = set(value)
    if not required.issubset(keys) or not keys.issubset(allowed):
        raise ValueError(f"{name} has missing or unknown fields")
    return value


def _array(value: object, name: str, *, minimum: int, maximum: int) -> list[object]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ValueError(f"{name} must be a bounded array")
    return value


def _string_array(value: object, name: str, maximum_bytes: int) -> tuple[str, ...]:
    items = _array(value, name, minimum=0, maximum=MAX_FILTER_VALUES)
    normalized = tuple(_bounded_string(item, name, maximum_bytes) for item in items)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return normalized


def _bounded_string(
    value: object,
    name: str,
    maximum_bytes: int,
    *,
    strip: bool = True,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or (strip and not value.strip()):
        raise ValueError(f"{name} must be non-blank")
    if len(value) > maximum_bytes or len(value.encode("utf-8")) > maximum_bytes:
        raise ValueError(f"{name} exceeds its byte limit")
    return value


def _immutable_string_set(
    value: object,
    name: str,
    *,
    allow_empty: bool,
) -> frozenset[str]:
    if not isinstance(value, frozenset):
        raise TypeError(f"{name} must be a frozenset")
    if (not allow_empty and not value) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{name} must contain non-blank strings")
    return value


def _integer(
    value: object,
    name: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its bounds")
    return value


def _enum(value: object, enum_type: type[_EnumT], name: str) -> _EnumT:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    try:
        return enum_type(value)
    except ValueError as error:
        raise ValueError(f"{name} is unsupported") from error


def validate_serialized_search_output(value: object) -> dict[str, object]:
    """Return a canonical bounded defensive copy of one serialized search output."""
    root = _closed_mapping(
        value,
        allowed={
            "status",
            "query_id",
            "degraded",
            "warning_codes",
            "hits",
            "next_cursor",
            "failure_code",
            "execution_summary",
        },
        required={
            "status",
            "query_id",
            "degraded",
            "warning_codes",
            "hits",
            "next_cursor",
            "failure_code",
            "execution_summary",
        },
        name="search output",
    )
    degraded = _boolean(root["degraded"], "degraded")
    warnings = _serialized_enum_array(
        root["warning_codes"],
        "warning_codes",
        SearchWarningCode,
        maximum=MAX_WARNING_CODES,
    )
    hits = tuple(
        _deserialize_search_hit(item)
        for item in _array(root["hits"], "hits", minimum=0, maximum=100)
    )
    failure_code = (
        None
        if root["failure_code"] is None
        else _enum(root["failure_code"], SearchFailureCode, "failure_code")
    )
    summary = (
        None
        if root["execution_summary"] is None
        else _deserialize_execution_summary(root["execution_summary"])
    )
    canonical = SearchResult(
        status=_enum(root["status"], SearchStatus, "status"),
        query_id=_bounded_string(root["query_id"], "query_id", MAX_REF_BYTES),
        degraded=degraded,
        warning_codes=warnings,
        hits=hits,
        next_cursor=_optional_bounded_string(
            root["next_cursor"],
            "next_cursor",
            MAX_CURSOR_BYTES,
        ),
        failure_code=failure_code,
        execution_summary=summary,
    )
    serialized = _serialize_search_result(canonical)
    if all(not isinstance(hit, FullHit) for hit in canonical.hits):
        _require_compact_output_budget(serialized)
    return serialized


def _deserialize_search_hit(value: object) -> CompactHit | FullHit:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError("hit must be an object with string keys")
    compact_fields = {
        "id",
        "target_type",
        "brief_score",
        "evidence_id",
        "timestamp_ms",
        "time_range",
    }
    full_fields = compact_fields | {
        "artifact_ref",
        "searchable_unit_ref",
        "representation_refs",
        "scene_range",
        "match_kinds",
        "lane_scores",
        "embedding_spaces",
        "processing_fingerprint",
        "index_generation_ids",
        "provider_ids",
        "execution_profile",
        "warning_codes",
        "lineage_refs",
    }
    fields = set(value)
    if fields == compact_fields:
        return CompactHit(
            id=_bounded_string(value["id"], "hit id", MAX_HIT_ID_BYTES),
            target_type=_enum(value["target_type"], TargetType, "target_type"),
            brief_score=_json_unit_number(value["brief_score"], "brief_score"),
            evidence_id=_bounded_string(
                value["evidence_id"],
                "evidence_id",
                MAX_EVIDENCE_ID_BYTES,
            ),
            timestamp_ms=_optional_milliseconds(value["timestamp_ms"], "timestamp_ms"),
            time_range=_deserialize_temporal(value["time_range"], "time_range"),
        )
    if fields != full_fields:
        raise ValueError("hit does not match the compact or full closed shape")
    return FullHit(
        id=_bounded_string(value["id"], "hit id", MAX_HIT_ID_BYTES),
        target_type=_enum(value["target_type"], TargetType, "target_type"),
        brief_score=_json_unit_number(value["brief_score"], "brief_score"),
        evidence_id=_bounded_string(
            value["evidence_id"],
            "evidence_id",
            MAX_EVIDENCE_ID_BYTES,
        ),
        timestamp_ms=_optional_milliseconds(value["timestamp_ms"], "timestamp_ms"),
        time_range=_deserialize_temporal(value["time_range"], "time_range"),
        artifact_ref=_optional_bounded_string(value["artifact_ref"], "artifact_ref", MAX_REF_BYTES),
        searchable_unit_ref=_optional_bounded_string(
            value["searchable_unit_ref"],
            "searchable_unit_ref",
            MAX_REF_BYTES,
        ),
        representation_refs=_serialized_string_array(
            value["representation_refs"],
            "representation_refs",
            maximum=MAX_EVIDENCE_REFS,
        ),
        scene_range=_deserialize_temporal(value["scene_range"], "scene_range"),
        match_kinds=_serialized_enum_array(
            value["match_kinds"],
            "match_kinds",
            MatchKind,
            maximum=len(MatchKind),
        ),
        lane_scores=tuple(
            _deserialize_lane_score(item)
            for item in _array(
                value["lane_scores"],
                "lane_scores",
                minimum=0,
                maximum=MAX_LANE_SCORES,
            )
        ),
        embedding_spaces=tuple(
            _deserialize_embedding_space(item)
            for item in _array(
                value["embedding_spaces"],
                "embedding_spaces",
                minimum=0,
                maximum=MAX_SPACES,
            )
        ),
        processing_fingerprint=(
            None
            if value["processing_fingerprint"] is None
            else _deserialize_processing_fingerprint(value["processing_fingerprint"])
        ),
        index_generation_ids=_serialized_string_array(
            value["index_generation_ids"],
            "index_generation_ids",
            maximum=MAX_LANE_SCORES,
        ),
        provider_ids=_serialized_string_array(
            value["provider_ids"],
            "provider_ids",
            maximum=MAX_LANE_SCORES,
        ),
        execution_profile=(
            None
            if value["execution_profile"] is None
            else _enum(value["execution_profile"], ExecutionProfile, "execution_profile")
        ),
        warning_codes=tuple(
            item.value
            for item in _serialized_enum_array(
                value["warning_codes"],
                "warning_codes",
                SearchWarningCode,
                maximum=MAX_WARNING_CODES,
            )
        ),
        lineage_refs=_serialized_string_array(
            value["lineage_refs"],
            "lineage_refs",
            maximum=MAX_LINEAGE_REFS,
        ),
    )


def _deserialize_execution_summary(value: object) -> SearchExecutionSummary:
    summary = _closed_mapping(
        value,
        allowed={
            "search_intent",
            "selected_lanes",
            "provider_ids",
            "embedding_spaces",
            "hash_spaces",
            "index_generation_ids",
            "bundle_manifest_sha256s",
            "resolved_profile_ref",
            "execution_profile",
            "activation_tier",
            "schema_version",
        },
        required={
            "search_intent",
            "selected_lanes",
            "provider_ids",
            "embedding_spaces",
            "hash_spaces",
            "index_generation_ids",
            "bundle_manifest_sha256s",
            "resolved_profile_ref",
            "execution_profile",
            "activation_tier",
            "schema_version",
        },
        name="execution_summary",
    )
    return SearchExecutionSummary(
        search_intent=_enum(summary["search_intent"], SearchIntent, "search_intent"),
        selected_lanes=_serialized_string_array(
            summary["selected_lanes"],
            "selected_lanes",
            minimum=1,
            maximum=MAX_LANE_SCORES,
            maximum_bytes=128,
        ),
        provider_ids=_serialized_string_array(
            summary["provider_ids"],
            "provider_ids",
            maximum=MAX_LANE_SCORES,
        ),
        embedding_spaces=tuple(
            _deserialize_embedding_space(item)
            for item in _array(
                summary["embedding_spaces"],
                "embedding_spaces",
                minimum=0,
                maximum=MAX_SPACES,
            )
        ),
        hash_spaces=tuple(
            _deserialize_hash_space(item)
            for item in _array(
                summary["hash_spaces"],
                "hash_spaces",
                minimum=0,
                maximum=MAX_SPACES,
            )
        ),
        index_generation_ids=_serialized_string_array(
            summary["index_generation_ids"],
            "index_generation_ids",
            maximum=MAX_LANE_SCORES,
        ),
        bundle_manifest_sha256s=tuple(
            _sha256(item)
            for item in _serialized_string_array(
                summary["bundle_manifest_sha256s"],
                "bundle_manifest_sha256s",
                maximum=2,
                maximum_bytes=64,
            )
        ),
        resolved_profile_ref=_bounded_string(
            summary["resolved_profile_ref"],
            "resolved_profile_ref",
            MAX_REF_BYTES,
        ),
        execution_profile=_enum(
            summary["execution_profile"],
            ExecutionProfile,
            "execution_profile",
        ),
        activation_tier=_bounded_string(summary["activation_tier"], "activation_tier", 64),
        schema_version=_integer(
            summary["schema_version"],
            "schema_version",
            minimum=1,
            maximum=1,
        ),
    )


def _deserialize_embedding_space(value: object) -> EmbeddingSpaceRef:
    space = _closed_mapping(
        value,
        allowed={
            "space_id",
            "provider_id",
            "model_revision",
            "preprocessing_version",
            "dimension",
            "normalization",
            "distance_metric",
            "query_modalities",
            "target_modalities",
            "schema_version",
        },
        required={
            "space_id",
            "provider_id",
            "model_revision",
            "preprocessing_version",
            "dimension",
            "normalization",
            "distance_metric",
            "query_modalities",
            "target_modalities",
            "schema_version",
        },
        name="embedding_space",
    )
    return EmbeddingSpaceRef(
        space_id=_bounded_string(space["space_id"], "space_id", MAX_REF_BYTES),
        provider_id=_bounded_string(space["provider_id"], "provider_id", MAX_REF_BYTES),
        model_revision=_bounded_string(
            space["model_revision"],
            "model_revision",
            MAX_REF_BYTES,
        ),
        preprocessing_version=_bounded_string(
            space["preprocessing_version"],
            "preprocessing_version",
            MAX_REF_BYTES,
        ),
        dimension=_integer(space["dimension"], "dimension", minimum=1, maximum=16_384),
        normalization=_enum(space["normalization"], VectorNormalization, "normalization"),
        distance_metric=_enum(space["distance_metric"], DistanceMetric, "distance_metric"),
        query_modalities=frozenset(
            _serialized_enum_array(
                space["query_modalities"],
                "query_modalities",
                QueryModality,
                minimum=1,
                maximum=len(QueryModality),
            )
        ),
        target_modalities=frozenset(
            _serialized_enum_array(
                space["target_modalities"],
                "target_modalities",
                TargetType,
                minimum=1,
                maximum=len(TargetType),
            )
        ),
        schema_version=_integer(
            space["schema_version"],
            "schema_version",
            minimum=1,
            maximum=1,
        ),
    )


def _deserialize_hash_space(value: object) -> HashSpaceRef:
    space = _closed_mapping(
        value,
        allowed={
            "space_id",
            "provider_id",
            "provider_version",
            "preprocessing_version",
            "algorithm",
            "bit_width",
            "hash_version",
            "target_modalities",
            "schema_version",
        },
        required={
            "space_id",
            "provider_id",
            "provider_version",
            "preprocessing_version",
            "algorithm",
            "bit_width",
            "hash_version",
            "target_modalities",
            "schema_version",
        },
        name="hash_space",
    )
    return HashSpaceRef(
        space_id=_bounded_string(space["space_id"], "space_id", MAX_REF_BYTES),
        provider_id=_bounded_string(space["provider_id"], "provider_id", MAX_REF_BYTES),
        provider_version=_bounded_string(
            space["provider_version"],
            "provider_version",
            MAX_REF_BYTES,
        ),
        preprocessing_version=_bounded_string(
            space["preprocessing_version"],
            "preprocessing_version",
            MAX_REF_BYTES,
        ),
        algorithm=_bounded_string(space["algorithm"], "algorithm", MAX_REF_BYTES),
        bit_width=_integer(space["bit_width"], "bit_width", minimum=64, maximum=64),
        hash_version=_bounded_string(space["hash_version"], "hash_version", MAX_REF_BYTES),
        target_modalities=frozenset(
            _serialized_enum_array(
                space["target_modalities"],
                "target_modalities",
                TargetType,
                minimum=1,
                maximum=len(TargetType),
            )
        ),
        schema_version=_integer(
            space["schema_version"],
            "schema_version",
            minimum=1,
            maximum=1,
        ),
    )


def _deserialize_lane_score(value: object) -> LaneScore:
    score = _closed_mapping(
        value,
        allowed={
            "lane_id",
            "raw_score",
            "normalized_score",
            "score_semantics",
            "calibration_version",
            "rank",
        },
        required={
            "lane_id",
            "raw_score",
            "normalized_score",
            "score_semantics",
            "calibration_version",
            "rank",
        },
        name="lane_score",
    )
    return LaneScore(
        lane_id=_bounded_string(score["lane_id"], "lane_id", 128),
        raw_score=_json_finite_number(score["raw_score"], "raw_score"),
        normalized_score=_json_unit_number(score["normalized_score"], "normalized_score"),
        score_semantics=_bounded_string(score["score_semantics"], "score_semantics", 128),
        calibration_version=_bounded_string(
            score["calibration_version"],
            "calibration_version",
            128,
        ),
        rank=_integer(score["rank"], "rank", minimum=1, maximum=100_000),
    )


def _deserialize_processing_fingerprint(value: object) -> ProcessingFingerprint:
    fingerprint = _closed_mapping(
        value,
        allowed={"sha256"},
        required={"sha256"},
        name="processing_fingerprint",
    )
    digest = _bounded_string(fingerprint["sha256"], "sha256", 64)
    return ProcessingFingerprint(_sha256(digest))


def _deserialize_temporal(value: object, name: str) -> TemporalRange | None:
    if value is None:
        return None
    temporal = _closed_mapping(
        value,
        allowed={"start_ms", "end_ms"},
        required={"start_ms", "end_ms"},
        name=name,
    )
    return TemporalRange(
        start_ms=_integer(
            temporal["start_ms"],
            "start_ms",
            minimum=0,
            maximum=MAX_TEMPORAL_MILLISECONDS,
        ),
        end_ms=_integer(
            temporal["end_ms"],
            "end_ms",
            minimum=1,
            maximum=MAX_TEMPORAL_MILLISECONDS,
        ),
    )


def _serialized_string_array(
    value: object,
    name: str,
    *,
    maximum: int,
    minimum: int = 0,
    maximum_bytes: int = MAX_REF_BYTES,
) -> tuple[str, ...]:
    values = tuple(
        _bounded_string(item, name, maximum_bytes)
        for item in _array(value, name, minimum=minimum, maximum=maximum)
    )
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def _serialized_enum_array(
    value: object,
    name: str,
    enum_type: type[_EnumT],
    *,
    maximum: int,
    minimum: int = 0,
) -> tuple[_EnumT, ...]:
    values = tuple(
        _enum(item, enum_type, name)
        for item in _array(value, name, minimum=minimum, maximum=maximum)
    )
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def _optional_bounded_string(value: object, name: str, maximum_bytes: int) -> str | None:
    if value is None:
        return None
    return _bounded_string(value, name, maximum_bytes)


def _optional_milliseconds(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _integer(
        value,
        name,
        minimum=0,
        maximum=MAX_TEMPORAL_MILLISECONDS,
    )


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _json_finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    return _finite_number(value, name)


def _json_unit_number(value: object, name: str) -> float:
    normalized = _json_finite_number(value, name)
    if not 0 <= normalized <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return normalized


def _require_compact_output_budget(value: Mapping[str, object]) -> None:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > MAX_COMPACT_OUTPUT_BYTES:
        raise ValueError("compact result exceeds its byte budget")


def _serialize_search_result(result: SearchResult) -> dict[str, object]:
    if len(result.warning_codes) > MAX_WARNING_CODES or len(result.hits) > 100:
        raise ValueError("search result exceeds output bounds")
    query_id = _bounded_string(result.query_id, "query_id", MAX_REF_BYTES)
    next_cursor = (
        None
        if result.next_cursor is None
        else _bounded_string(result.next_cursor, "next_cursor", MAX_CURSOR_BYTES)
    )
    return {
        "status": result.status.value,
        "query_id": query_id,
        "degraded": result.degraded,
        "warning_codes": [code.value for code in result.warning_codes],
        "hits": [_serialize_hit(hit) for hit in result.hits],
        "next_cursor": next_cursor,
        "failure_code": None if result.failure_code is None else result.failure_code.value,
        "execution_summary": (
            None
            if result.execution_summary is None
            else _serialize_execution_summary(result.execution_summary)
        ),
    }


def _serialize_hit(hit: CompactHit | FullHit) -> dict[str, object]:
    serialized: dict[str, object] = {
        "id": _bounded_string(hit.id, "hit id", MAX_HIT_ID_BYTES),
        "target_type": hit.target_type.value,
        "brief_score": _unit_number(hit.brief_score, "brief_score"),
        "evidence_id": _bounded_string(
            hit.evidence_id,
            "evidence_id",
            MAX_EVIDENCE_ID_BYTES,
        ),
        "timestamp_ms": hit.timestamp_ms,
        "time_range": _serialize_temporal(hit.time_range),
    }
    if not isinstance(hit, FullHit):
        return serialized
    serialized.update(
        {
            "artifact_ref": _optional_ref(hit.artifact_ref, "artifact_ref"),
            "searchable_unit_ref": _optional_ref(
                hit.searchable_unit_ref,
                "searchable_unit_ref",
            ),
            "representation_refs": _bounded_refs(
                hit.representation_refs,
                "representation_refs",
                MAX_EVIDENCE_REFS,
            ),
            "scene_range": _serialize_temporal(hit.scene_range),
            "match_kinds": [kind.value for kind in hit.match_kinds],
            "lane_scores": [
                _serialize_lane_score(score)
                for score in _bounded_sequence(hit.lane_scores, "lane_scores", MAX_LANE_SCORES)
            ],
            "embedding_spaces": [
                _serialize_embedding_space(space)
                for space in _bounded_sequence(hit.embedding_spaces, "embedding_spaces", MAX_SPACES)
            ],
            "processing_fingerprint": (
                None
                if hit.processing_fingerprint is None
                else {"sha256": _sha256(hit.processing_fingerprint.sha256)}
            ),
            "index_generation_ids": _bounded_refs(
                hit.index_generation_ids,
                "index_generation_ids",
                MAX_LANE_SCORES,
            ),
            "provider_ids": _bounded_refs(hit.provider_ids, "provider_ids", MAX_LANE_SCORES),
            "execution_profile": (
                None if hit.execution_profile is None else hit.execution_profile.value
            ),
            "warning_codes": [
                SearchWarningCode(code).value
                for code in _bounded_sequence(
                    hit.warning_codes,
                    "warning_codes",
                    MAX_WARNING_CODES,
                )
            ],
            "lineage_refs": _bounded_refs(
                hit.lineage_refs,
                "lineage_refs",
                MAX_LINEAGE_REFS,
            ),
        }
    )
    return serialized


def _serialize_execution_summary(summary: SearchExecutionSummary) -> dict[str, object]:
    return {
        "search_intent": summary.search_intent.value,
        "selected_lanes": list(
            _bounded_sequence(summary.selected_lanes, "selected_lanes", MAX_LANE_SCORES)
        ),
        "provider_ids": _bounded_refs(summary.provider_ids, "provider_ids", MAX_LANE_SCORES),
        "embedding_spaces": [
            _serialize_embedding_space(space)
            for space in _bounded_sequence(summary.embedding_spaces, "embedding_spaces", MAX_SPACES)
        ],
        "hash_spaces": [
            _serialize_hash_space(space)
            for space in _bounded_sequence(summary.hash_spaces, "hash_spaces", MAX_SPACES)
        ],
        "index_generation_ids": _bounded_refs(
            summary.index_generation_ids,
            "index_generation_ids",
            MAX_LANE_SCORES,
        ),
        "bundle_manifest_sha256s": [
            _sha256(value)
            for value in _bounded_sequence(
                summary.bundle_manifest_sha256s,
                "bundle_manifest_sha256s",
                2,
            )
        ],
        "resolved_profile_ref": _bounded_string(
            summary.resolved_profile_ref,
            "resolved_profile_ref",
            MAX_REF_BYTES,
        ),
        "execution_profile": summary.execution_profile.value,
        "activation_tier": _bounded_string(summary.activation_tier, "activation_tier", 64),
        "schema_version": summary.schema_version,
    }


def _serialize_embedding_space(space: EmbeddingSpaceRef) -> dict[str, object]:
    if space.dimension > 16_384:
        raise ValueError("embedding dimension exceeds output bound")
    return {
        "space_id": _bounded_string(space.space_id, "space_id", MAX_REF_BYTES),
        "provider_id": _bounded_string(space.provider_id, "provider_id", MAX_REF_BYTES),
        "model_revision": _bounded_string(
            space.model_revision,
            "model_revision",
            MAX_REF_BYTES,
        ),
        "preprocessing_version": _bounded_string(
            space.preprocessing_version,
            "preprocessing_version",
            MAX_REF_BYTES,
        ),
        "dimension": space.dimension,
        "normalization": space.normalization.value,
        "distance_metric": space.distance_metric.value,
        "query_modalities": sorted(value.value for value in space.query_modalities),
        "target_modalities": sorted(value.value for value in space.target_modalities),
        "schema_version": space.schema_version,
    }


def _serialize_hash_space(space: HashSpaceRef) -> dict[str, object]:
    return {
        "space_id": _bounded_string(space.space_id, "space_id", MAX_REF_BYTES),
        "provider_id": _bounded_string(space.provider_id, "provider_id", MAX_REF_BYTES),
        "provider_version": _bounded_string(
            space.provider_version,
            "provider_version",
            MAX_REF_BYTES,
        ),
        "preprocessing_version": _bounded_string(
            space.preprocessing_version,
            "preprocessing_version",
            MAX_REF_BYTES,
        ),
        "algorithm": space.algorithm,
        "bit_width": space.bit_width,
        "hash_version": space.hash_version,
        "target_modalities": sorted(value.value for value in space.target_modalities),
        "schema_version": space.schema_version,
    }


def _serialize_lane_score(score: LaneScore) -> dict[str, object]:
    if score.lane_id not in {"content", "perceptual", "instance", "concept"}:
        raise ValueError("lane score has unsupported lane")
    if score.rank > 100_000:
        raise ValueError("lane score rank exceeds output bound")
    return {
        "lane_id": score.lane_id,
        "raw_score": _finite_number(score.raw_score, "raw_score"),
        "normalized_score": _unit_number(score.normalized_score, "normalized_score"),
        "score_semantics": _bounded_string(score.score_semantics, "score_semantics", 128),
        "calibration_version": _bounded_string(
            score.calibration_version,
            "calibration_version",
            128,
        ),
        "rank": score.rank,
    }


def _serialize_temporal(value: TemporalRange | None) -> dict[str, int] | None:
    if value is None:
        return None
    return {"start_ms": value.start_ms, "end_ms": value.end_ms}


def _optional_ref(value: str | None, name: str) -> str | None:
    return None if value is None else _bounded_string(value, name, MAX_REF_BYTES)


def _bounded_refs(values: Sequence[str], name: str, maximum: int) -> list[str]:
    items = _bounded_sequence(values, name, maximum)
    normalized = [_bounded_string(value, name, MAX_REF_BYTES) for value in items]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} contains duplicates")
    return normalized


def _bounded_sequence(
    values: Sequence[_SequenceT],
    name: str,
    maximum: int,
) -> Sequence[_SequenceT]:
    if len(values) > maximum:
        raise ValueError(f"{name} exceeds output bounds")
    return values


def _finite_number(value: float, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _unit_number(value: float, name: str) -> float:
    normalized = _finite_number(value, name)
    if not 0 <= normalized <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return normalized


def _sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("digest must be lowercase SHA-256")
    return value


def _evidence_refs(result: SearchResult) -> tuple[str, ...]:
    refs = tuple(dict.fromkeys(hit.evidence_id for hit in result.hits))
    if len(refs) > MAX_EVIDENCE_REFS:
        raise ValueError("too many evidence refs")
    for ref in refs:
        _bounded_string(ref, "evidence_ref", MAX_EVIDENCE_ID_BYTES)
    return refs


def _bounded_message(message: str) -> str:
    encoded = message.encode("utf-8")
    if len(encoded) <= MAX_ERROR_MESSAGE_BYTES:
        return message
    return encoded[:MAX_ERROR_MESSAGE_BYTES].decode("utf-8", errors="ignore") or "Search failed"


def _tool_failure(
    context: InvocationContext,
    started_at: datetime,
    completed_at: datetime,
    status: ToolStatus,
    code: ToolErrorCode,
    message: str,
    retryable: bool,
) -> ToolResult:
    if completed_at < started_at:
        completed_at = started_at
    return ToolResult(
        request_id=context.request_id,
        trace_id=context.trace_id,
        tool_id=SEARCH_SPEC.tool_id,
        tool_version=SEARCH_SPEC.contract_version,
        status=status,
        error=ToolError(code=code, message=_bounded_message(message), retryable=retryable),
        started_at=started_at,
        completed_at=completed_at,
    )


__all__ = [
    "COMPACT_HIT_SCHEMA",
    "EXECUTION_SUMMARY_SCHEMA",
    "FULL_HIT_SCHEMA",
    "MAX_COMPACT_OUTPUT_BYTES",
    "MAX_INLINE_BASE64_CHARS",
    "MAX_INLINE_IMAGE_BYTES",
    "SEARCH_INPUT_SCHEMA",
    "SEARCH_OUTPUT_SCHEMA",
    "SEARCH_SCHEMA_SHA256",
    "SEARCH_SPEC",
    "SearchAuthorizationFactory",
    "SearchServicePort",
    "SearchToolHandler",
    "StaticSearchAuthorizationFactory",
    "register_search_tool",
    "validate_serialized_search_output",
]
