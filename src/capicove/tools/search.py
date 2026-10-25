"""Canonical Tool boundary for provider-neutral text search."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Protocol, cast, runtime_checkable

from capicove.contracts import (
    DetailLevel,
    FullHit,
    InvocationContext,
    PrivacyClass,
    SearchFilters,
    SearchRequest,
    SearchResult,
    SideEffect,
    TargetType,
    TemporalRange,
    TextOrigin,
    TextQuery,
    ToolError,
    ToolErrorCode,
    ToolResources,
    ToolResult,
    ToolSpec,
    ToolStatus,
)
from capicove.retrieval import TextSearchError
from capicove.tools.search_video import decode_video_query, encode_video_hit


CAPICOVE_SEARCH_SPEC = ToolSpec(
    tool_id="capicove.search",
    contract_version=3,
    description="Search immutable text and video evidence using text or image queries.",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "required": ["query"],
        "properties": {
            "query": {
                "oneOf": [
                    {"type": "string", "minLength": 1},
                    {
                        "type": "object",
                        "required": ["type"],
                        "properties": {
                            "type": {"enum": ["text", "image"]},
                            "text": {"type": "string"},
                            "language": {"type": "string"},
                            "source": {"type": "object"},
                        },
                        "additionalProperties": False,
                    },
                ]
            },
            "targets": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"enum": ["keyframe", "video_segment", "text_chunk"]},
            },
            "detail_level": {"enum": ["compact", "full"]},
            "cursor": {"type": "string", "minLength": 1},
            "time_range": {
                "type": "object",
                "additionalProperties": False,
                "required": ["start_ms", "end_ms"],
                "properties": {
                    "start_ms": {"type": "integer", "minimum": 0},
                    "end_ms": {"type": "integer", "minimum": 1},
                },
            },
            "language": {"type": "string", "minLength": 1},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 100},
            "retrieval_profile_ref": {"type": "string", "minLength": 1},
            "corpus_refs": {"type": "array", "items": {"type": "string"}},
            "artifact_refs": {"type": "array", "items": {"type": "string"}},
            "media_types": {"type": "array", "items": {"type": "string"}},
            "text_origins": {
                "type": "array",
                "items": {"enum": ["document", "ocr", "asr"]},
            },
        },
    },
    output_schema={"type": "object"},
    modalities=("text", "image"),
    required_permissions=frozenset({"evidence:search"}),
    privacy_class=PrivacyClass.CONFIDENTIAL,
    side_effect=SideEffect.READ,
    network_access=True,
    timeout_seconds=30.0,
    redacted_fields=("query",),
    resources=ToolResources(filesystem_read=True),
)


class TextSearchOperation(Protocol):
    """Narrow operation consumed by the Tool boundary."""

    def search(self, request: SearchRequest) -> SearchResult: ...


@runtime_checkable
class DeadlineSearchOperation(Protocol):
    """Optional operation that preserves the Tool's absolute execution budget."""

    def search_with_deadline(
        self, request: SearchRequest, *, deadline: datetime
    ) -> SearchResult: ...


class SearchToolHandler:
    """Decode canonical arguments and map stable search outcomes."""

    def __init__(self, operation: TextSearchOperation) -> None:
        self._operation = operation

    def __call__(
        self, context: InvocationContext, arguments: Mapping[str, object], /
    ) -> ToolResult:
        """Execute one canonical text search invocation."""
        started = datetime.now(UTC)
        if CAPICOVE_SEARCH_SPEC.required_permissions.difference(context.permissions):
            return _failure(
                context, started, ToolErrorCode.PERMISSION_DENIED, "Search permission denied."
            )
        try:
            if datetime.now(UTC) >= context.deadline:
                return _failure(
                    context, started, ToolErrorCode.TIMEOUT, "Search deadline exceeded."
                )
            request = decode_search_arguments(arguments)
            if isinstance(self._operation, DeadlineSearchOperation):
                result = self._operation.search_with_deadline(request, deadline=context.deadline)
            else:
                result = self._operation.search(request)
        except (TypeError, ValueError):
            return _failure(
                context, started, ToolErrorCode.INVALID_INPUT, "Search arguments are invalid."
            )
        except TextSearchError as error:
            code = _error_code(error.code)
            message = (
                "Text retrieval is unavailable."
                if code is ToolErrorCode.UNAVAILABLE
                else "Text retrieval failed."
            )
            return _failure(context, started, code, message)
        try:
            payload = encode_search_result(result)
            if datetime.now(UTC) >= context.deadline:
                return _failure(
                    context, started, ToolErrorCode.TIMEOUT, "Search deadline exceeded."
                )
        except (TypeError, ValueError):
            return _failure(context, started, ToolErrorCode.INTERNAL, "Search result is invalid.")
        return ToolResult(
            request_id=context.request_id,
            trace_id=context.trace_id,
            tool_id=CAPICOVE_SEARCH_SPEC.tool_id,
            tool_version=CAPICOVE_SEARCH_SPEC.contract_version,
            status=ToolStatus.SUCCEEDED,
            result=payload,
            evidence_refs=tuple(hit.evidence_id for hit in result.hits),
            started_at=started,
            completed_at=datetime.now(UTC),
        )


def decode_search_arguments(arguments: Mapping[str, object]) -> SearchRequest:
    """Decode only closed provider-neutral text search arguments."""
    allowed = {
        "query",
        "language",
        "top_k",
        "retrieval_profile_ref",
        "corpus_refs",
        "artifact_refs",
        "media_types",
        "text_origins",
        "targets",
        "detail_level",
        "cursor",
        "time_range",
    }
    if set(arguments) - allowed:
        raise ValueError("unsupported search argument")
    query = arguments.get("query")
    language = arguments.get("language")
    if isinstance(query, Mapping):
        if language is not None or "targets" not in arguments:
            raise ValueError("tagged queries require explicit targets and nested language")
        canonical_query = decode_video_query(query)
        targets = frozenset(TargetType(value) for value in _strings(arguments["targets"]))
        if TargetType.TEXT_CHUNK in targets and targets != {TargetType.TEXT_CHUNK}:
            raise ValueError("mixed text and video targets are unsupported")
    elif isinstance(query, str) and "targets" not in arguments:
        if language is not None and not isinstance(language, str):
            raise TypeError("language must be a string")
        canonical_query = TextQuery(query, language)
        targets = frozenset({TargetType.TEXT_CHUNK})
    else:
        raise ValueError("invalid query or legacy targets")
    time_range = arguments.get("time_range")
    span = None
    if time_range is not None:
        if not isinstance(time_range, Mapping) or set(time_range) != {"start_ms", "end_ms"}:
            raise ValueError("invalid time_range")
        span = TemporalRange(time_range["start_ms"], time_range["end_ms"])
    filters = SearchFilters(
        time_range=span,
        corpus_refs=_strings(arguments.get("corpus_refs", ())),
        artifact_refs=_strings(arguments.get("artifact_refs", ())),
        media_types=_strings(arguments.get("media_types", ())),
        text_origins=frozenset(
            TextOrigin(value) for value in _strings(arguments.get("text_origins", ()))
        ),
    )
    return SearchRequest(
        query=canonical_query,
        targets=targets,
        detail_level=DetailLevel(str(arguments.get("detail_level", "compact"))),
        cursor=arguments.get("cursor"),  # type: ignore[arg-type]
        top_k=arguments.get("top_k", 20),  # type: ignore[arg-type]
        retrieval_profile_ref=arguments.get(
            "retrieval_profile_ref",
            "local-default" if targets == {TargetType.TEXT_CHUNK} else "video-local-exact",
        ),  # type: ignore[arg-type]
        filters=filters,
    )


def encode_search_result(result: SearchResult) -> dict[str, object]:
    """Encode one canonical compact result without provider data."""
    hits: list[dict[str, object]] = []
    for hit in result.hits:
        if hit.target_type in {TargetType.KEYFRAME, TargetType.VIDEO_SEGMENT}:
            hits.append(encode_video_hit(hit))
            continue
        locator = hit.text_locator
        if locator is None:
            raise ValueError("text search hit requires a locator")
        temporal = locator.time_range
        hits.append(
            {
                "id": hit.id,
                "target_type": hit.target_type.value,
                "brief_score": hit.brief_score,
                "evidence_id": hit.evidence_id,
                "text_locator": {
                    "origin": locator.origin.value,
                    "start_char": locator.character_range.start_char,
                    "end_char": locator.character_range.end_char,
                    "page_number": locator.page_number,
                    "start_ms": None if temporal is None else temporal.start_ms,
                    "end_ms": None if temporal is None else temporal.end_ms,
                },
                "snippet": hit.snippet,
            }
        )
    payload: dict[str, object] = {
        "status": result.status.value,
        "query_id": result.query_id,
        "degraded": result.degraded,
        "warning_codes": list(result.warning_codes),
        "hits": hits,
    }
    if result.next_cursor is not None:
        payload["next_cursor"] = result.next_cursor
    if (
        len(result.hits) <= 20
        and not any(isinstance(hit, FullHit) for hit in result.hits)
        and len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) > 16_384
    ):
        raise ValueError("compact response size exceeds limit")
    return payload


def _strings(value: object) -> frozenset[str]:
    if isinstance(value, (str, bytes)):
        raise TypeError("filter values must be arrays")
    try:
        materialized: tuple[object, ...] = tuple(value)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError("filter values must be arrays") from None
    if any(not isinstance(item, str) for item in materialized):
        raise TypeError("filter values must be strings")
    return frozenset(cast(tuple[str, ...], materialized))


def _error_code(code: str) -> ToolErrorCode:
    if code in {"invalid_query", "unsupported_query_target", "stale_cursor", "unsupported_filter"}:
        return ToolErrorCode.INVALID_INPUT
    if code == "permission_denied":
        return ToolErrorCode.PERMISSION_DENIED
    if code == "deadline_exceeded":
        return ToolErrorCode.TIMEOUT
    if code in {
        "provider_unavailable",
        "index_generation_unavailable",
        "embedding_space_incompatible",
    }:
        return ToolErrorCode.UNAVAILABLE
    return ToolErrorCode.INTERNAL


def _failure(
    context: InvocationContext, started: datetime, code: ToolErrorCode, message: str
) -> ToolResult:
    return ToolResult(
        request_id=context.request_id,
        trace_id=context.trace_id,
        tool_id=CAPICOVE_SEARCH_SPEC.tool_id,
        tool_version=CAPICOVE_SEARCH_SPEC.contract_version,
        status=ToolStatus.FAILED,
        error=ToolError(code, message, retryable=False),
        started_at=started,
        completed_at=datetime.now(UTC),
    )


__all__ = [
    "CAPICOVE_SEARCH_SPEC",
    "SearchToolHandler",
    "TextSearchOperation",
    "decode_search_arguments",
    "encode_search_result",
]
