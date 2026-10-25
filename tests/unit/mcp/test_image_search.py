"""Tests for the transport-only ``capicove_search_image`` MCP mapper."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

import capicove.tools.image_search as search_tool_module
from capicove.contracts import (
    MAX_CURSOR_BYTES,
    MAX_REF_BYTES,
    InvocationContext,
    PrivacyClass,
    SearchRequest,
    SearchResult,
    ToolError,
    ToolErrorCode,
    ToolResult,
    ToolStatus,
)
from capicove.mcp.image_search import (
    MCP_SEARCH_NAME,
    MCP_SEARCH_OUTPUT_SCHEMA,
    create_search_mcp_server,
    encode_search_mcp_result,
    register_search_mcp,
)
from capicove.mcp.video_split import McpUnavailableError
from capicove.retrieval.search_models import InvocationAuthorizationContext
from capicove.tools.image_search import (
    MAX_COMPACT_OUTPUT_BYTES,
    MAX_EVIDENCE_ID_BYTES,
    MAX_EVIDENCE_REFS,
    MAX_HIT_ID_BYTES,
    SEARCH_SPEC,
    SearchToolHandler,
    StaticSearchAuthorizationFactory,
)


NOW = datetime(2026, 8, 27, 8, 0, tzinfo=UTC)


def test_mcp_package_exposes_lazy_search_public_api() -> None:
    from capicove.mcp.image_search import MCP_SEARCH_NAME as package_search_name
    from capicove.mcp.image_search import register_search_mcp as package_register_search_mcp

    assert package_search_name == MCP_SEARCH_NAME
    assert package_register_search_mcp is register_search_mcp


def _canonical_result() -> dict[str, object]:
    return {
        "status": "succeeded",
        "query_id": "query-1",
        "degraded": False,
        "warning_codes": [],
        "hits": [],
        "next_cursor": None,
        "failure_code": None,
        "execution_summary": None,
    }


def _maximum_compact_result() -> dict[str, object]:
    result = _canonical_result()
    result["query_id"] = "q" * MAX_REF_BYTES
    result["next_cursor"] = "c" * MAX_CURSOR_BYTES
    result["hits"] = [
        {
            "id": f"{index:03d}-" + "i" * (MAX_HIT_ID_BYTES - 4),
            "target_type": "image",
            "brief_score": 0.12345678901234568,
            "evidence_id": f"{index:03d}-" + "e" * (MAX_EVIDENCE_ID_BYTES - 4),
            "timestamp_ms": None,
            "time_range": None,
        }
        for index in range(100)
    ]
    return result


def _tool_result(result: Mapping[str, object]) -> ToolResult:
    return ToolResult(
        request_id="request-1",
        trace_id="trace-1",
        tool_id=SEARCH_SPEC.tool_id,
        tool_version=SEARCH_SPEC.contract_version,
        status=ToolStatus.SUCCEEDED,
        result=result,
        started_at=NOW,
        completed_at=NOW,
    )


class _Handler:
    def __init__(self, status: ToolStatus = ToolStatus.SUCCEEDED) -> None:
        self.status = status
        self.calls: list[tuple[InvocationContext, Mapping[str, object]]] = []

    def __call__(
        self,
        context: InvocationContext,
        arguments: Mapping[str, object],
        /,
    ) -> ToolResult:
        self.calls.append((context, arguments))
        error = None
        result: Mapping[str, object] | None = _canonical_result()
        if self.status is not ToolStatus.SUCCEEDED:
            error = ToolError(
                ToolErrorCode.PERMISSION_DENIED,
                "Search permission denied",
                False,
            )
            result = None
        return ToolResult(
            request_id=context.request_id,
            trace_id=context.trace_id,
            tool_id=SEARCH_SPEC.tool_id,
            tool_version=SEARCH_SPEC.contract_version,
            status=self.status,
            result=result,
            evidence_refs=("evidence-1",),
            error=error,
            started_at=NOW,
            completed_at=NOW,
        )


class _ContextFactory:
    def __init__(
        self,
        deadline: datetime | None = None,
        *,
        permissions: frozenset[str] = frozenset({"retrieval:search"}),
    ) -> None:
        self.deadline = deadline
        self.permissions = permissions
        self.calls: list[tuple[str, datetime]] = []

    def create(self, *, tool_id: str, deadline_cap: datetime) -> InvocationContext:
        self.calls.append((tool_id, deadline_cap))
        return InvocationContext(
            request_id="request-1",
            trace_id="trace-1",
            workspace_id="workspace-1",
            actor_id="actor-1",
            environment_id="environment-1",
            deadline=self.deadline or deadline_cap,
            permissions=self.permissions,
            privacy_class=PrivacyClass.CONFIDENTIAL,
        )


class _Server:
    def __init__(self) -> None:
        self.name: str | None = None
        self.description: str | None = None
        self.input_schema: Mapping[str, object] | None = None
        self.callback: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None

    def tool(
        self,
        *,
        name: str,
        description: str,
        input_schema: Mapping[str, object],
    ) -> Callable[
        [Callable[[Mapping[str, object]], Mapping[str, object]]],
        Callable[[Mapping[str, object]], Mapping[str, object]],
    ]:
        self.name = name
        self.description = description
        self.input_schema = input_schema

        def decorate(
            callback: Callable[[Mapping[str, object]], Mapping[str, object]],
        ) -> Callable[[Mapping[str, object]], Mapping[str, object]]:
            self.callback = callback
            return callback

        return decorate


class _NeverService:
    def search(
        self,
        request: SearchRequest,
        *,
        authorization: InvocationAuthorizationContext,
        deadline: datetime,
    ) -> SearchResult:
        raise AssertionError("permission rejection must precede search")


def test_registration_is_one_canonical_name_schema_and_guidance() -> None:
    server = _Server()

    callback = register_search_mcp(
        server,
        _Handler(),
        context_factory=_ContextFactory(),
        clock=lambda: NOW,
    )

    assert server.name == MCP_SEARCH_NAME == "capicove_search_image"
    assert server.input_schema == SEARCH_SPEC.input_schema
    assert callable(callback)
    assert server.description is not None
    for phrase in (
        "same person",
        "object",
        "scene",
        "shot",
        "diagram",
        "slide",
        "prop",
        "visual arrangement",
        "theme",
        "topic",
        "genre",
        "style",
        "mood",
        "educational meaning",
        "balanced only when context cannot sufficiently support either interpretation",
    ):
        assert phrase in server.description


@pytest.mark.parametrize(
    "status",
    [ToolStatus.SUCCEEDED, ToolStatus.FAILED, ToolStatus.REJECTED],
)
def test_callback_preserves_arguments_context_and_exact_envelope(status: ToolStatus) -> None:
    arguments = {"query": {"type": "image"}, "targets": ["image"]}
    handler = _Handler(status)
    factory = _ContextFactory(deadline=NOW + timedelta(seconds=5))
    callback = register_search_mcp(
        _Server(),
        handler,
        context_factory=factory,
        clock=lambda: NOW,
    )

    encoded = callback(arguments)

    assert handler.calls[0][1] is arguments
    assert handler.calls[0][0].deadline == NOW + timedelta(seconds=5)
    assert factory.calls == [
        (SEARCH_SPEC.tool_id, NOW + timedelta(seconds=SEARCH_SPEC.timeout_seconds))
    ]
    assert set(encoded) == {
        "request_id",
        "trace_id",
        "tool_id",
        "tool_version",
        "status",
        "result",
        "evidence_refs",
        "error",
    }
    Draft202012Validator(MCP_SEARCH_OUTPUT_SCHEMA).validate(dict(encoded))


def test_context_factory_deadline_is_capped_without_granting_permissions() -> None:
    late_factory = _ContextFactory(
        deadline=NOW + timedelta(hours=1),
        permissions=frozenset(),
    )
    handler = _Handler()
    callback = register_search_mcp(
        _Server(),
        handler,
        context_factory=late_factory,
        clock=lambda: NOW,
    )

    callback({})

    context = handler.calls[0][0]
    assert context.deadline == NOW + timedelta(seconds=SEARCH_SPEC.timeout_seconds)
    assert context.permissions == frozenset()


def test_mcp_permission_rejection_matches_direct_tool_without_self_grant() -> None:
    arguments = {
        "query": {
            "type": "image",
            "source": {"type": "artifact", "artifact_ref": "artifact:query"},
        },
        "targets": ["image"],
    }
    tool = SearchToolHandler(
        _NeverService(),
        StaticSearchAuthorizationFactory(
            authorization_scope_ref="scope:search",
            permissions=frozenset({"retrieval:search"}),
            allowed_data_policy_refs=frozenset({"local-only"}),
            allowed_source_root_refs=frozenset(),
        ),
        clock=lambda: NOW,
    )
    factory = _ContextFactory(
        deadline=NOW + timedelta(seconds=5),
        permissions=frozenset(),
    )
    callback = register_search_mcp(
        _Server(),
        tool,
        context_factory=factory,
        clock=lambda: NOW,
    )

    encoded = callback(arguments)
    direct_context = factory.create(
        tool_id=SEARCH_SPEC.tool_id,
        deadline_cap=NOW + timedelta(seconds=SEARCH_SPEC.timeout_seconds),
    )
    direct = encode_search_mcp_result(
        tool(direct_context, arguments),
        expected_context=direct_context,
    )

    assert encoded == direct
    assert encoded["status"] == "rejected"
    error = encoded["error"]
    assert isinstance(error, Mapping)
    assert error["code"] == "permission_denied"


@pytest.mark.parametrize(
    "deadline",
    [NOW, NOW - timedelta(seconds=1), datetime(2026, 8, 27, 8, 0)],
)
def test_invalid_factory_deadline_fails_before_handler(deadline: datetime) -> None:
    handler = _Handler()
    callback = register_search_mcp(
        _Server(),
        handler,
        context_factory=_ContextFactory(deadline=deadline),
        clock=lambda: NOW,
    )

    with pytest.raises(ValueError, match="deadline"):
        callback({})

    assert handler.calls == []


def test_encoder_rejects_invalid_identity_duplicates_and_unbounded_errors() -> None:
    valid = _Handler()(
        _ContextFactory().create(
            tool_id=SEARCH_SPEC.tool_id, deadline_cap=NOW + timedelta(seconds=1)
        ),
        {},
    )
    Draft202012Validator.check_schema(MCP_SEARCH_OUTPUT_SCHEMA)
    Draft202012Validator(MCP_SEARCH_OUTPUT_SCHEMA).validate(encode_search_mcp_result(valid))

    wrong_tool = ToolResult(
        request_id="request-1",
        trace_id="trace-1",
        tool_id="capicove.video.split",
        tool_version=1,
        status=ToolStatus.SUCCEEDED,
        result=_canonical_result(),
        started_at=NOW,
        completed_at=NOW,
    )
    duplicate_evidence = ToolResult(
        request_id="request-1",
        trace_id="trace-1",
        tool_id=SEARCH_SPEC.tool_id,
        tool_version=1,
        status=ToolStatus.SUCCEEDED,
        result=_canonical_result(),
        evidence_refs=("same", "same"),
        started_at=NOW,
        completed_at=NOW,
    )
    wrong_status = replace(
        valid,
        status=cast(ToolStatus, "failed"),
        result=None,
        error=ToolError(ToolErrorCode.INTERNAL, "Search failed", False),
    )
    too_many_evidence = replace(
        valid,
        evidence_refs=tuple(f"evidence-{index}" for index in range(MAX_EVIDENCE_REFS + 1)),
    )
    invalid_error_code = replace(
        valid,
        status=ToolStatus.FAILED,
        result=None,
        error=ToolError(cast(ToolErrorCode, "unknown"), "Search failed", False),
    )
    blank_evidence = replace(valid, evidence_refs=(" ",))
    multibyte_evidence = replace(
        valid,
        evidence_refs=("界" * (MAX_REF_BYTES // 3 + 1),),
    )
    for invalid in (
        wrong_tool,
        duplicate_evidence,
        wrong_status,
        too_many_evidence,
        invalid_error_code,
        blank_evidence,
        multibyte_evidence,
    ):
        with pytest.raises(ValueError):
            encode_search_mcp_result(invalid)


def test_encoder_rejects_non_result_and_context_identity_mismatch() -> None:
    context = _ContextFactory().create(
        tool_id=SEARCH_SPEC.tool_id,
        deadline_cap=NOW + timedelta(seconds=1),
    )
    valid = _Handler()(context, {})

    with pytest.raises(TypeError, match="ToolResult"):
        encode_search_mcp_result(cast(ToolResult, object()))
    with pytest.raises(ValueError, match="invocation identity"):
        encode_search_mcp_result(
            valid,
            expected_context=replace(context, request_id="request-2"),
        )


@pytest.mark.parametrize("case", ["long_query_id", "too_many_hits", "open_hit"])
def test_encoder_rejects_nested_search_output_that_violates_canonical_bounds(case: str) -> None:
    result = _canonical_result()
    compact_hit: dict[str, object] = {
        "id": "image-1",
        "target_type": "image",
        "brief_score": 0.9,
        "evidence_id": "evidence-1",
        "timestamp_ms": None,
        "time_range": None,
    }
    if case == "long_query_id":
        result["query_id"] = "x" * 1_025
    elif case == "too_many_hits":
        result["hits"] = [dict(compact_hit) for _ in range(101)]
    elif case == "open_hit":
        result["hits"] = [{**compact_hit, "private_path": "/private/secret"}]
    else:  # pragma: no cover - closed parametrization guard
        raise AssertionError(f"unexpected case: {case}")
    invalid = ToolResult(
        request_id="request-1",
        trace_id="trace-1",
        tool_id=SEARCH_SPEC.tool_id,
        tool_version=SEARCH_SPEC.contract_version,
        status=ToolStatus.SUCCEEDED,
        result=result,
        started_at=NOW,
        completed_at=NOW,
    )

    with pytest.raises((TypeError, ValueError)):
        encode_search_mcp_result(invalid)


def test_encoder_returns_a_defensive_copy_of_nested_search_output() -> None:
    hit: dict[str, object] = {
        "id": "image-1",
        "target_type": "image",
        "brief_score": 0.9,
        "evidence_id": "evidence-1",
        "timestamp_ms": None,
        "time_range": None,
    }
    hits = [hit]
    result = _canonical_result()
    result["hits"] = hits
    tool_result = ToolResult(
        request_id="request-1",
        trace_id="trace-1",
        tool_id=SEARCH_SPEC.tool_id,
        tool_version=SEARCH_SPEC.contract_version,
        status=ToolStatus.SUCCEEDED,
        result=result,
        started_at=NOW,
        completed_at=NOW,
    )

    encoded = encode_search_mcp_result(tool_result)
    hit["id"] = "mutated"
    hits.append(dict(hit))

    encoded_result = encoded["result"]
    assert isinstance(encoded_result, Mapping)
    encoded_hits = encoded_result["hits"]
    assert isinstance(encoded_hits, list)
    assert len(encoded_hits) == 1
    assert encoded_hits[0]["id"] == "image-1"


def test_encoder_preserves_maximum_compact_page_within_shared_output_budget() -> None:
    result = _maximum_compact_result()
    result_hits = cast(list[dict[str, object]], result["hits"])
    tool_result = replace(
        _tool_result(result),
        evidence_refs=tuple(cast(str, hit["evidence_id"]) for hit in result_hits),
    )

    encoded = encode_search_mcp_result(tool_result)

    assert MAX_COMPACT_OUTPUT_BYTES == 544 * 1_024
    encoded_result = encoded["result"]
    assert isinstance(encoded_result, Mapping)
    encoded_hits = encoded_result["hits"]
    assert isinstance(encoded_hits, list)
    assert [hit["id"] for hit in encoded_hits] == [hit["id"] for hit in result_hits]
    assert encoded["evidence_refs"] == list(tool_result.evidence_refs)
    payload = json.dumps(
        dict(encoded_result),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert 16 * 1_024 < len(payload) <= MAX_COMPACT_OUTPUT_BYTES
    Draft202012Validator(MCP_SEARCH_OUTPUT_SCHEMA).validate(encoded)


def test_encoder_uses_same_inclusive_compact_output_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _canonical_result()
    exact_size = len(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )
    monkeypatch.setattr(search_tool_module, "MAX_COMPACT_OUTPUT_BYTES", exact_size)

    assert encode_search_mcp_result(_tool_result(result))["result"] == result

    monkeypatch.setattr(search_tool_module, "MAX_COMPACT_OUTPUT_BYTES", exact_size - 1)
    with pytest.raises(ValueError, match="byte budget"):
        encode_search_mcp_result(_tool_result(result))


def test_real_server_reports_actionable_missing_optional_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    real_import = importlib.import_module

    def blocked(name: str, package: str | None = None) -> object:
        if name.split(".", maxsplit=1)[0] == "mcp":
            raise ImportError(name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", blocked)
    with pytest.raises(McpUnavailableError, match=r"capicove\[mcp\]"):
        create_search_mcp_server(
            _Handler(),
            context_factory=_ContextFactory(),
            clock=lambda: NOW,
        )
