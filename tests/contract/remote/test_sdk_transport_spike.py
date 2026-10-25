"""Exercise the pinned SDK's wire and per-request context before composition."""

import asyncio
import importlib.metadata
import json
from typing import Any

import httpx
import pytest
from mcp import types
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request


@pytest.mark.parametrize("json_response", [True, False])
def test_sdk_transports_attach_each_http_request(json_response: bool) -> None:
    """An initialized session must not freeze the initial HTTP request's actor."""
    assert importlib.metadata.version("mcp") == "1.30.0"
    server: Any = Server("capicove-transport-spike")

    async def list_tools() -> list[types.Tool]:
        request = server.request_context.request
        assert isinstance(request, Request)
        actor = request.headers["x-fixture-actor"]
        return [types.Tool(name=f"fixture_{actor}", inputSchema={"type": "object"})]

    server.list_tools()(list_tools)
    manager = StreamableHTTPSessionManager(
        server,
        json_response=json_response,
        security_settings=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=["testserver"],
            allowed_origins=["https://testserver"],
        ),
    )

    async def run() -> None:
        async with manager.run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.handle_request),
                base_url="https://testserver",
                headers={"accept": "application/json, text/event-stream"},
            ) as client:
                response = await client.post(
                    "/mcp",
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2025-11-25",
                            "capabilities": {},
                            "clientInfo": {"name": "spike", "version": "1"},
                        },
                    },
                )
                assert response.status_code == 200
                client.headers["mcp-session-id"] = response.headers["mcp-session-id"]
                client.headers["mcp-protocol-version"] = "2025-11-25"
                notified = await client.post(
                    "/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}
                )
                assert notified.status_code == 202
                assert notified.content == b""
                for request_id, actor in enumerate(("alice", "bob"), start=2):
                    response = await client.post(
                        "/mcp",
                        headers={"x-fixture-actor": actor},
                        json={"jsonrpc": "2.0", "id": request_id, "method": "tools/list"},
                    )
                    assert response.status_code == 200
                    if json_response:
                        payload = response.json()
                    else:
                        assert response.headers["content-type"].startswith("text/event-stream")
                        payload = json.loads(
                            next(
                                line.removeprefix("data: ")
                                for line in response.text.splitlines()
                                if line.startswith("data: ")
                            )
                        )
                    assert payload["result"]["tools"][0]["name"] == f"fixture_{actor}"
                assert (await client.delete("/mcp")).status_code == 200

    asyncio.run(run())


@pytest.mark.parametrize("json_response", [True, False])
def test_python_sdk_initializes_lists_and_calls(json_response: bool) -> None:
    """Use the actual Python client rather than hand-authored JSON-RPC requests."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    server: Any = Server("capicove-sdk-client-spike")

    async def list_tools() -> list[types.Tool]:
        return [types.Tool(name="fixture_echo", inputSchema={"type": "object"})]

    async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
        assert name == "fixture_echo"
        assert arguments == {"message": "local-fixture"}
        request = server.request_context.request
        assert isinstance(request, Request)
        assert request.headers["x-fixture-actor"] == "alice"
        return [types.TextContent(type="text", text="observed-call")]

    server.list_tools()(list_tools)
    server.call_tool()(call_tool)
    manager = StreamableHTTPSessionManager(
        server,
        json_response=json_response,
        security_settings=TransportSecuritySettings(
            enable_dns_rebinding_protection=True, allowed_hosts=["testserver"]
        ),
    )

    async def run() -> None:
        async with manager.run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.handle_request),
                headers={"x-fixture-actor": "alice"},
            ) as client:
                async with streamable_http_client("https://testserver/mcp", http_client=client) as (
                    read,
                    write,
                    session_id,
                ):
                    async with ClientSession(read, write) as session:
                        initialized = await session.initialize()
                        assert initialized.protocolVersion == "2025-11-25"
                        assert session_id() is not None
                        assert [tool.name for tool in (await session.list_tools()).tools] == [
                            "fixture_echo"
                        ]
                        result = await session.call_tool(
                            "fixture_echo", {"message": "local-fixture"}
                        )
                        assert result.isError is False
                        assert result.content == [
                            types.TextContent(type="text", text="observed-call")
                        ]

    asyncio.run(asyncio.wait_for(run(), timeout=10))
