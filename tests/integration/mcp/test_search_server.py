"""Integration tests for the optional canonical search MCP server."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from datetime import UTC, datetime, timedelta
from threading import Event, Lock, current_thread
from typing import Any

import pytest
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

from capicove.contracts import (
    CompactHit,
    InvocationContext,
    PrivacyClass,
    SearchRequest,
    SearchResult,
    SearchStatus,
    TargetType,
)
from capicove.mcp.image_search import (
    MCP_SEARCH_NAME,
    MCP_SEARCH_OUTPUT_SCHEMA,
    create_search_mcp_server,
    encode_search_mcp_result,
)
from capicove.retrieval.search_models import InvocationAuthorizationContext
from capicove.tools.image_search import (
    SEARCH_SPEC,
    SearchToolHandler,
    StaticSearchAuthorizationFactory,
)


pytestmark = pytest.mark.integration
NOW = datetime(2026, 8, 27, 8, 0, tzinfo=UTC)


def test_real_server_bounds_workers_and_retains_cancelled_admission() -> None:
    pytest.importorskip("mcp", reason="install capicove[mcp]")
    from mcp import types

    release = Event()
    started = Event()
    finished = Event()
    lock = Lock()
    counts = {"started": 0, "finished": 0}

    class BlockingService(_Service):
        def search(
            self,
            request: SearchRequest,
            *,
            authorization: InvocationAuthorizationContext,
            deadline: datetime,
        ) -> SearchResult:
            with lock:
                counts["started"] += 1
                if counts["started"] == 4:
                    started.set()
            try:
                if not release.wait(5):
                    raise TimeoutError("Test worker was not released")
                return super().search(request, authorization=authorization, deadline=deadline)
            finally:
                with lock:
                    counts["finished"] += 1
                    if counts["finished"] == 4:
                        finished.set()

    handler = SearchToolHandler(
        BlockingService(),
        StaticSearchAuthorizationFactory(
            authorization_scope_ref="scope:search",
            permissions=frozenset({"retrieval:search"}),
            allowed_data_policy_refs=frozenset({"local-only"}),
            allowed_source_root_refs=frozenset(),
        ),
        clock=lambda: NOW,
    )
    server: Any = create_search_mcp_server(
        handler, context_factory=_ContextFactory(), clock=lambda: NOW
    )

    async def invoke() -> Any:
        return await server.request_handlers[types.CallToolRequest](
            types.CallToolRequest(
                params=types.CallToolRequestParams(name=MCP_SEARCH_NAME, arguments=_arguments())
            )
        )

    async def scenario() -> None:
        requests = [asyncio.create_task(invoke()) for _ in range(4)]
        try:
            assert await asyncio.to_thread(started.wait, 2)
            rejected = await asyncio.wait_for(invoke(), timeout=1)
            assert rejected.root.isError is True
            requests[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await requests[0]
            rejected_after_cancel = await asyncio.wait_for(invoke(), timeout=1)
            assert rejected_after_cancel.root.isError is True
            assert counts["started"] == 4
        finally:
            release.set()
            await asyncio.gather(*requests, return_exceptions=True)
        assert await asyncio.to_thread(finished.wait, 2)
        recovered = await asyncio.wait_for(invoke(), timeout=1)
        assert recovered.root.isError is False

    asyncio.run(scenario())


async def _resolve(awaitable: Awaitable[Any]) -> Any:
    return await awaitable


class _Service:
    def __init__(self) -> None:
        self.thread_names: list[str] = []

    def search(
        self,
        request: SearchRequest,
        *,
        authorization: InvocationAuthorizationContext,
        deadline: datetime,
    ) -> SearchResult:
        self.thread_names.append(current_thread().name)
        assert authorization.operation == SEARCH_SPEC.tool_id
        assert deadline == NOW + timedelta(seconds=SEARCH_SPEC.timeout_seconds)
        return SearchResult(
            SearchStatus.SUCCEEDED,
            "query-1",
            hits=(CompactHit("unit-1", TargetType.IMAGE, 0.9, "evidence-1"),),
        )


class _ContextFactory:
    def create(self, *, tool_id: str, deadline_cap: datetime) -> InvocationContext:
        assert tool_id == SEARCH_SPEC.tool_id
        return InvocationContext(
            request_id="request-1",
            trace_id="trace-1",
            workspace_id="workspace-1",
            actor_id="actor-1",
            environment_id="environment-1",
            deadline=deadline_cap,
            permissions=frozenset({"retrieval:search"}),
            privacy_class=PrivacyClass.CONFIDENTIAL,
        )


def _arguments() -> dict[str, object]:
    return {
        "query": {
            "type": "image",
            "source": {"type": "artifact", "artifact_ref": "artifact:query-1"},
        },
        "targets": ["image"],
    }


def _handler() -> SearchToolHandler:
    return SearchToolHandler(
        _Service(),
        StaticSearchAuthorizationFactory(
            authorization_scope_ref="scope:search",
            permissions=frozenset({"retrieval:search"}),
            allowed_data_policy_refs=frozenset({"local-only"}),
            allowed_source_root_refs=frozenset(),
        ),
        clock=lambda: NOW,
    )


def test_real_server_lists_one_search_contract_and_matches_python() -> None:
    pytest.importorskip("mcp", reason="install capicove[mcp]")
    from mcp import types
    from mcp.server import Server

    handler = _handler()
    context_factory = _ContextFactory()
    server = create_search_mcp_server(
        handler,
        context_factory=context_factory,
        clock=lambda: NOW,
    )

    assert isinstance(server, Server)
    list_handler = server.request_handlers[types.ListToolsRequest]
    response: Any = asyncio.run(_resolve(list_handler(types.ListToolsRequest())))
    listed = response.root.tools
    assert len(listed) == 1
    assert listed[0].name == MCP_SEARCH_NAME
    assert listed[0].inputSchema == dict(SEARCH_SPEC.input_schema)
    assert listed[0].outputSchema == MCP_SEARCH_OUTPUT_SCHEMA

    call_handler = server.request_handlers[types.CallToolRequest]
    call_response: Any = asyncio.run(
        _resolve(
            call_handler(
                types.CallToolRequest(
                    params=types.CallToolRequestParams(
                        name=MCP_SEARCH_NAME,
                        arguments=_arguments(),
                    )
                )
            )
        )
    )
    assert call_response.root.isError is False
    encoded = call_response.root.structuredContent
    Draft202012Validator(MCP_SEARCH_OUTPUT_SCHEMA).validate(encoded)

    direct_context = context_factory.create(
        tool_id=SEARCH_SPEC.tool_id,
        deadline_cap=NOW + timedelta(seconds=SEARCH_SPEC.timeout_seconds),
    )
    direct = encode_search_mcp_result(
        handler(direct_context, _arguments()),
        expected_context=direct_context,
    )
    assert encoded == direct


def test_real_server_offloads_synchronous_search_from_event_loop() -> None:
    pytest.importorskip("mcp", reason="install capicove[mcp]")
    from mcp import types

    service = _Service()
    handler = SearchToolHandler(
        service,
        StaticSearchAuthorizationFactory(
            authorization_scope_ref="scope:search",
            permissions=frozenset({"retrieval:search"}),
            allowed_data_policy_refs=frozenset({"local-only"}),
            allowed_source_root_refs=frozenset(),
        ),
        clock=lambda: NOW,
    )
    server: Any = create_search_mcp_server(
        handler, context_factory=_ContextFactory(), clock=lambda: NOW
    )
    call_handler = server.request_handlers[types.CallToolRequest]
    caller_thread = current_thread().name

    asyncio.run(
        _resolve(
            call_handler(
                types.CallToolRequest(
                    params=types.CallToolRequestParams(name=MCP_SEARCH_NAME, arguments=_arguments())
                )
            )
        )
    )

    assert service.thread_names
    assert service.thread_names[0] != caller_thread


def test_real_server_rejects_unknown_alias() -> None:
    pytest.importorskip("mcp", reason="install capicove[mcp]")
    from mcp import types

    server: Any = create_search_mcp_server(
        _handler(),
        context_factory=_ContextFactory(),
        clock=lambda: NOW,
    )
    call_handler = server.request_handlers[types.CallToolRequest]
    response: Any = asyncio.run(
        _resolve(
            call_handler(
                types.CallToolRequest(
                    params=types.CallToolRequestParams(
                        name="capicove_reverse_image_search",
                        arguments=_arguments(),
                    )
                )
            )
        )
    )
    assert response.root.isError is True
    assert response.root.structuredContent is None
