"""Exercise actual MCP request handlers, concurrency, and exception privacy."""

import asyncio
import threading
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import anyio
from mcp import types
from mcp.server import Server

from capicove.contracts import InvocationContext, ToolResult
from capicove.mcp.server import create_bound_mcp_server
from tests.unit.mcp.test_bindings import SPEC, binding, handler


async def call(server: Server[Any], name: str, arguments: dict[str, Any]) -> Any:
    return await server.request_handlers[types.CallToolRequest](
        types.CallToolRequest(params=types.CallToolRequestParams(name=name, arguments=arguments))
    )


def test_real_catalog_and_call() -> None:
    server = create_bound_mcp_server((binding(),), server_name="test")

    async def run() -> None:
        listed: Any = await server.request_handlers[types.ListToolsRequest](
            types.ListToolsRequest()
        )
        assert listed.root.tools[0].inputSchema == dict(SPEC.input_schema)
        result = await call(server, "capicove_test_echo", {})
        assert result.root.structuredContent["result"] == {"ok": True}
        assert not result.root.isError
        bad = await call(server, "capicove_secret_unknown", {})
        assert bad.root.isError
        assert "secret" not in str(bad)
        bad = await call(server, "capicove_test_echo", {"secret-input": "private"})
        assert bad.root.isError
        assert "private" not in str(bad)

    asyncio.run(run())


def test_handler_exception_does_not_echo_input() -> None:
    def broken(context: InvocationContext, arguments: Mapping[str, object], /) -> ToolResult:
        raise RuntimeError("secret-provider-value")

    server = create_bound_mcp_server((replace(binding(), handler=broken),), server_name="test")
    result = asyncio.run(call(server, "capicove_test_echo", {}))
    assert result.root.isError
    assert "secret-provider-value" not in str(result)


def test_blocking_handler_does_not_block_listing() -> None:
    entered = threading.Event()
    release = threading.Event()

    def slow(context: InvocationContext, arguments: Mapping[str, object], /) -> ToolResult:
        entered.set()
        if not release.wait(3):
            raise RuntimeError("test deadlock")
        return handler(context, arguments)

    server = create_bound_mcp_server((replace(binding(), handler=slow),), server_name="test")

    async def run() -> None:
        pending = asyncio.create_task(call(server, "capicove_test_echo", {}))
        try:
            for _ in range(100):
                if entered.is_set():
                    break
                await asyncio.sleep(0.01)
            assert entered.is_set()
            listed = await asyncio.wait_for(
                server.request_handlers[types.ListToolsRequest](types.ListToolsRequest()), 0.5
            )
            assert listed is not None
        finally:
            release.set()
            await pending

    asyncio.run(run())


def test_protocol_cancellation_drains_worker_before_return() -> None:
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    returned = threading.Event()

    def slow(context: InvocationContext, arguments: Mapping[str, object], /) -> ToolResult:
        entered.set()
        if not release.wait(3):
            raise RuntimeError("test deadlock")
        finished.set()
        return handler(context, arguments)

    server = create_bound_mcp_server((replace(binding(), handler=slow),), server_name="test")

    async def run() -> None:
        scope = anyio.CancelScope()

        async def request() -> None:
            with scope:
                await call(server, "capicove_test_echo", {})
            assert finished.is_set()
            returned.set()

        async with anyio.create_task_group() as group:
            group.start_soon(request)
            try:
                with anyio.fail_after(2):
                    while not entered.is_set():
                        await anyio.sleep(0.01)
                scope.cancel()
                await anyio.sleep(0.02)
                assert not returned.is_set()
            finally:
                release.set()
        assert returned.is_set()

    anyio.run(run)


def test_decoder_output_and_identity_failures_are_sanitized() -> None:
    def decode(arguments: Mapping[str, object]) -> dict[str, object]:
        raise ValueError("secret-decoder-value")

    def wrong_identity(
        context: InvocationContext, arguments: Mapping[str, object], /
    ) -> ToolResult:
        return replace(handler(context, arguments), request_id="secret-wrong-request")

    def bad_output(context: InvocationContext, arguments: Mapping[str, object], /) -> ToolResult:
        return replace(handler(context, arguments), result={"ok": "secret-invalid-output"})

    for item in (
        replace(binding(), decode_arguments=decode),
        replace(binding(), handler=wrong_identity),
        replace(binding(), handler=bad_output),
    ):
        server = create_bound_mcp_server((item,), server_name="test")
        result = asyncio.run(call(server, "capicove_test_echo", {}))
        assert result.root.isError
        assert "secret" not in str(result)


def test_legacy_video_filter_catalog_preserves_exact_schemas() -> None:
    from capicove.mcp.video_filter import (
        MCP_VIDEO_FILTER_NAME,
        MCP_VIDEO_FILTER_OUTPUT_SCHEMA,
        create_video_filter_mcp_server,
    )
    from capicove.tools.video_filter import VIDEO_FILTER_SPEC

    server = create_video_filter_mcp_server(handler)
    assert isinstance(server, Server)

    async def run() -> None:
        response = await server.request_handlers[types.ListToolsRequest](types.ListToolsRequest())
        assert isinstance(response.root, types.ListToolsResult)
        listed = response.root.tools
        assert len(listed) == 1
        assert listed[0].name == MCP_VIDEO_FILTER_NAME
        assert listed[0].inputSchema == dict(VIDEO_FILTER_SPEC.input_schema)
        assert listed[0].outputSchema == MCP_VIDEO_FILTER_OUTPUT_SCHEMA
        assert listed[0].annotations is not None
        assert listed[0].annotations.readOnlyHint is True
        assert listed[0].annotations.destructiveHint is False

    asyncio.run(run())
