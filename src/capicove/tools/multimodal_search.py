"""Closed Tool contracts for shared-space multimodal Search and Evidence (RFC-0032).

Arguments select only the query, targets, filters and projection. Models, roots,
index paths and providers are startup authority and can never appear on the wire.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Protocol, cast

from capicove.contracts import (
    InvocationContext,
    PrivacyClass,
    SideEffect,
    ToolError,
    ToolErrorCode,
    ToolResult,
    ToolSpec,
    ToolStatus,
)
from capicove.contracts.retrieval import (
    MAX_INLINE_IMAGE_BYTES,
    AudioClipQuery,
    DetailLevel,
    FullHit,
    ImageQuery,
    InlineBytesSource,
    LocalPathSource,
    SearchFilters,
    SearchRequest,
    SearchResult,
    TargetType,
    TemporalRange,
    TextOrigin,
    TextQuery,
)
from capicove.tools.evidence import CAPICOVE_EVIDENCE_SPEC


_RANGE = {
    "type": "object",
    "additionalProperties": False,
    "required": ["start_ms", "end_ms"],
    "properties": {
        "start_ms": {"type": "integer", "minimum": 0},
        "end_ms": {"type": "integer", "minimum": 1},
    },
}
_STRINGS = {"type": "array", "maxItems": 256, "items": {"type": "string", "minLength": 1}}
CAPICOVE_MULTIMODAL_SEARCH_SPEC = ToolSpec(
    tool_id="capicove.search.multimodal",
    contract_version=1,
    description="Search the active shared-space generation with a text or audio query.",
    input_schema={
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
                            "text": {"type": "string", "minLength": 1, "maxLength": 16384},
                            "language": {"type": "string", "maxLength": 128},
                        },
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["type"],
                        "properties": {
                            "type": {"const": "image"},
                            "path": {"type": "string", "minLength": 1, "maxLength": 4096},
                            "data_base64": {"type": "string", "minLength": 1},
                            "media_type": {"type": "string", "maxLength": 255},
                        },
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["type"],
                        "properties": {
                            "type": {"const": "audio_clip"},
                            "path": {"type": "string", "minLength": 1, "maxLength": 4096},
                            "data_base64": {"type": "string", "minLength": 1},
                            "media_type": {"type": "string", "maxLength": 255},
                            "start_ms": {"type": "integer", "minimum": 0},
                            "end_ms": {"type": "integer", "minimum": 1},
                            "stream_index": {"type": "integer", "minimum": 0, "maximum": 63},
                        },
                    },
                ]
            },
            "targets": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"enum": [item.value for item in TargetType]},
            },
            "top_k": {"type": "integer", "minimum": 1, "maximum": 100},
            "detail_level": {"enum": ["compact", "full"]},
            "retrieval_profile_ref": {"type": "string", "minLength": 1},
            "corpus_refs": _STRINGS,
            "artifact_refs": _STRINGS,
            "media_types": _STRINGS,
            "text_origins": {"type": "array", "items": {"enum": [o.value for o in TextOrigin]}},
            "time_range": _RANGE,
        },
    },
    output_schema={"type": "object"},
    modalities=("text", "image", "audio"),
    required_permissions=frozenset({"evidence:search"}),
    privacy_class=PrivacyClass.CONFIDENTIAL,
    side_effect=SideEffect.READ,
    network_access=False,
    timeout_seconds=120,
    redacted_fields=("query",),
)
CAPICOVE_MULTIMODAL_EVIDENCE_SPEC = ToolSpec(
    tool_id="capicove.multimodal.get_evidence",
    contract_version=1,
    description="Read authorized metadata and lineage of shared-space retrieval evidence.",
    input_schema=CAPICOVE_EVIDENCE_SPEC.input_schema,
    output_schema={"type": "object"},
    modalities=("text", "audio"),
    required_permissions=frozenset({"evidence:read"}),
    privacy_class=PrivacyClass.CONFIDENTIAL,
    side_effect=SideEffect.READ,
    network_access=False,
    timeout_seconds=30,
    redacted_fields=("evidence_id",),
)
_TOP_LEVEL = frozenset(
    cast(Mapping[str, object], CAPICOVE_MULTIMODAL_SEARCH_SPEC.input_schema["properties"])
)


class MultimodalSearchOperation(Protocol):
    """Narrow operation consumed by the Tool boundary."""

    def search(self, request: SearchRequest) -> SearchResult: ...


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    return value


def _strings(value: object, field: str) -> frozenset[str]:
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{field} must be a list of strings")
    return frozenset(value)


def _media_source(raw: Mapping[str, object]) -> LocalPathSource | InlineBytesSource:
    if ("path" in raw) == ("data_base64" in raw):
        raise ValueError("media queries require exactly one of path or data_base64")
    if "path" in raw:
        if "media_type" in raw or not isinstance(raw["path"], str):
            raise ValueError("invalid media path query")
        return LocalPathSource(raw["path"])
    data, media_type = raw["data_base64"], raw.get("media_type")
    if not isinstance(data, str) or not isinstance(media_type, str):
        raise ValueError("inline media requires data_base64 and media_type")
    if len(data) > (MAX_INLINE_IMAGE_BYTES * 4) // 3 + 4:
        raise ValueError("inline media exceeds the ingress limit")
    try:
        payload = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("inline media is not valid base64") from None
    return InlineBytesSource(payload, media_type)


def _query(raw: object) -> TextQuery | ImageQuery | AudioClipQuery:
    if not isinstance(raw, Mapping):
        raise TypeError("query must be a tagged object")
    kind = raw.get("type")
    if kind == "text":
        if set(raw) - {"type", "text", "language"} or not isinstance(raw.get("text"), str):
            raise ValueError("invalid text query")
        language = raw.get("language")
        if language is not None and not isinstance(language, str):
            raise TypeError("language must be a string")
        return TextQuery(raw["text"], language)
    if kind == "image":
        if set(raw) - {"type", "path", "data_base64", "media_type"}:
            raise ValueError("invalid image query")
        return ImageQuery(_media_source(raw))
    if kind != "audio_clip":
        raise ValueError("unsupported query type")
    if set(raw) - {
        "type",
        "path",
        "data_base64",
        "media_type",
        "start_ms",
        "end_ms",
        "stream_index",
    }:
        raise ValueError("invalid audio query")
    if ("start_ms" in raw) != ("end_ms" in raw):
        raise ValueError("audio ranges require both endpoints")
    span = (
        TemporalRange(_integer(raw["start_ms"], "start_ms"), _integer(raw["end_ms"], "end_ms"))
        if "start_ms" in raw
        else None
    )
    stream = raw.get("stream_index")
    if stream is not None:
        stream = _integer(stream, "stream_index")
    return AudioClipQuery(_media_source(raw), span, stream)


def decode_multimodal_arguments(arguments: Mapping[str, object], *, profile: str) -> SearchRequest:
    """Decode closed arguments; the configured profile is the only selectable profile."""
    if not isinstance(arguments, Mapping) or set(arguments) - _TOP_LEVEL:
        raise ValueError("unsupported search argument")
    if "targets" not in arguments or "query" not in arguments:
        raise ValueError("query and targets are required")
    reference = arguments.get("retrieval_profile_ref", profile)
    if reference != profile:
        raise ValueError("retrieval_profile_ref must match the configured profile")
    time_range = arguments.get("time_range")
    span = None
    if time_range is not None:
        if not isinstance(time_range, Mapping) or set(time_range) != {"start_ms", "end_ms"}:
            raise ValueError("invalid time_range")
        span = TemporalRange(
            _integer(time_range["start_ms"], "start_ms"), _integer(time_range["end_ms"], "end_ms")
        )
    return SearchRequest(
        query=_query(arguments["query"]),
        targets=frozenset(TargetType(value) for value in _strings(arguments["targets"], "targets")),
        top_k=_integer(arguments.get("top_k", 20), "top_k"),
        detail_level=DetailLevel(str(arguments.get("detail_level", "compact"))),
        retrieval_profile_ref=profile,
        filters=SearchFilters(
            corpus_refs=_strings(arguments.get("corpus_refs", ()), "corpus_refs"),
            artifact_refs=_strings(arguments.get("artifact_refs", ()), "artifact_refs"),
            media_types=_strings(arguments.get("media_types", ()), "media_types"),
            text_origins=frozenset(
                TextOrigin(value)
                for value in _strings(arguments.get("text_origins", ()), "text_origins")
            ),
            time_range=span,
        ),
    )


def encode_multimodal_result(result: SearchResult) -> dict[str, object]:
    """Project canonical hits without vectors, private paths or provider payloads."""
    hits: list[dict[str, object]] = []
    for hit in result.hits:
        item: dict[str, object] = {
            "id": hit.id,
            "target_type": hit.target_type.value,
            "brief_score": hit.brief_score,
            "evidence_id": hit.evidence_id,
        }
        if hit.time_range is not None:
            item["time_range"] = {
                "start_ms": hit.time_range.start_ms,
                "end_ms": hit.time_range.end_ms,
            }
        if hit.timestamp_ms is not None:
            item["timestamp_ms"] = hit.timestamp_ms
        if hit.text_locator is not None:
            item["text_locator"] = {
                "origin": hit.text_locator.origin.value,
                "start_char": hit.text_locator.character_range.start_char,
                "end_char": hit.text_locator.character_range.end_char,
            }
        if hit.snippet is not None:
            item["snippet"] = hit.snippet
        if isinstance(hit, FullHit):
            item["match_kinds"] = [kind.value for kind in hit.match_kinds]
            item["lane_scores"] = [
                {
                    "lane_id": lane.lane_id,
                    "raw_score": lane.raw_score,
                    "score_semantics": lane.score_semantics,
                    "calibration_version": lane.calibration_version,
                    "rank": lane.rank,
                }
                for lane in hit.lane_scores
            ]
            item["index_generation_ids"] = list(hit.index_generation_ids)
            item["space_ids"] = [space.space_id for space in hit.embedding_spaces]
        hits.append(item)
    payload: dict[str, object] = {
        "status": result.status.value,
        "query_id": result.query_id,
        "degraded": result.degraded,
        "warning_codes": [code.value for code in result.warning_codes],
        "hits": hits,
    }
    if result.failure_code is not None:
        payload["failure_code"] = result.failure_code.value
    return payload


class MultimodalSearchToolHandler:
    """Check permissions and deadlines, decode, dispatch and encode one search."""

    def __init__(self, operation: MultimodalSearchOperation, *, profile: str) -> None:
        self._operation = operation
        self._profile = profile

    def __call__(
        self, context: InvocationContext, arguments: Mapping[str, object], /
    ) -> ToolResult:
        """Execute one search; denial and invalid input never reach the operation."""
        started = datetime.now(UTC)
        code: ToolErrorCode | None = None
        if CAPICOVE_MULTIMODAL_SEARCH_SPEC.required_permissions - context.permissions:
            code = ToolErrorCode.PERMISSION_DENIED
        elif started >= context.deadline:
            code = ToolErrorCode.TIMEOUT
        if code is None:
            try:
                request = decode_multimodal_arguments(arguments, profile=self._profile)
            except (TypeError, ValueError, KeyError):
                code = ToolErrorCode.INVALID_INPUT
        payload: dict[str, object] | None = None
        evidence: tuple[str, ...] = ()
        if code is None:
            try:
                result = self._operation.search(request)
                payload = encode_multimodal_result(result)
                evidence = tuple(hit.evidence_id for hit in result.hits)
            except Exception:
                code = ToolErrorCode.INTERNAL
        return ToolResult(
            request_id=context.request_id,
            trace_id=context.trace_id,
            tool_id=CAPICOVE_MULTIMODAL_SEARCH_SPEC.tool_id,
            tool_version=CAPICOVE_MULTIMODAL_SEARCH_SPEC.contract_version,
            status=ToolStatus.SUCCEEDED if code is None else ToolStatus.FAILED,
            result=payload,
            error=None
            if code is None
            else ToolError(code=code, message="Multimodal search failed.", retryable=False),
            evidence_refs=evidence,
            started_at=started,
            completed_at=datetime.now(UTC),
        )


__all__ = [
    "CAPICOVE_MULTIMODAL_EVIDENCE_SPEC",
    "CAPICOVE_MULTIMODAL_SEARCH_SPEC",
    "MultimodalSearchToolHandler",
    "decode_multimodal_arguments",
    "encode_multimodal_result",
]
