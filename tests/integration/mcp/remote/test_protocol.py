"""Actual SDK clients exercise authenticated, authority-bound HTTP calls."""

import importlib
from pathlib import Path
from typing import Any

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from capicove.contracts.remote import Principal
from tests.unit.runtime.remote.test_remote_dispatch import dispatcher


@pytest.mark.anyio
@pytest.mark.parametrize("json_response", [True, False])
async def test_authenticated_sdk_read_call_and_missing_credential(
    tmp_path: Path, json_response: bool
) -> None:
    assert importlib.util.find_spec("capicove.mcp.remote") is not None
    module = importlib.import_module("capicove.mcp.remote")
    dispatch, principal, calls = dispatcher(tmp_path)

    class Verifier:
        async def verify(self, credential: str, *, administrative: bool = False) -> Principal:
            if credential != "fixture-token":
                raise ValueError("rejected")
            return principal

    host: Any = module.RemoteReadHost(
        Verifier(),
        {principal.key: dispatch},
        allowed_hosts=("testserver",),
        json_response=json_response,
    )
    async with host.run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=host), base_url="https://testserver"
        ) as client:
            assert (await client.post("/mcp", json={})).status_code == 401
            assert calls == []
            client.headers["authorization"] = "Bearer fixture-token"
            async with streamable_http_client("https://testserver/mcp", http_client=client) as (
                read,
                write,
                session_id,
            ):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    assert session_id() is not None
                    tools = await session.list_tools()
                    assert [tool.name for tool in tools.tools] == ["capicove_test_echo"]
                    result = await session.call_tool("capicove_test_echo", {})
                    assert result.isError is False
                    assert len(calls) == 1


@pytest.mark.anyio
async def test_session_scope_growth_and_identity_switch_are_denied(tmp_path: Path) -> None:
    from capicove.contracts.remote import PrincipalKey

    module = importlib.import_module("capicove.mcp.remote")
    dispatch, principal, calls = dispatcher(tmp_path)
    claims = {
        "initial": principal,
        "grown": Principal(
            principal.key, frozenset({"read", "write"}), principal.expires_at, principal.audience
        ),
        "other": Principal(
            PrincipalKey("https://issuer.test", "tenant", "app", "human", "bob"),
            principal.token_permissions,
            principal.expires_at,
            principal.audience,
        ),
        "reduced": Principal(principal.key, frozenset(), principal.expires_at, principal.audience),
    }

    class Verifier:
        async def verify(self, credential: str, *, administrative: bool = False) -> Principal:
            return claims[credential]

    host = module.RemoteReadHost(
        Verifier(), {principal.key: dispatch}, allowed_hosts=("testserver",)
    )
    async with host.run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=host),
            base_url="https://testserver",
            headers={
                "authorization": "Bearer initial",
                "accept": "application/json, text/event-stream",
            },
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
                        "clientInfo": {"name": "fixture", "version": "1"},
                    },
                },
            )
            assert response.status_code == 200
            client.headers["mcp-session-id"] = response.headers["mcp-session-id"]
            client.headers["mcp-protocol-version"] = "2025-11-25"
            await client.post(
                "/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}
            )
            for credential in ("grown", "other"):
                response = await client.post(
                    "/mcp",
                    headers={"authorization": "Bearer " + credential},
                    json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                )
                assert response.status_code == 403
            response = await client.post(
                "/mcp",
                headers={"authorization": "Bearer reduced"},
                json={"jsonrpc": "2.0", "id": 3, "method": "tools/list"},
            )
            assert response.json()["result"]["tools"] == []
            assert calls == []


@pytest.mark.anyio
async def test_unqualified_initialize_version_rejected(tmp_path: Path) -> None:
    module = importlib.import_module("capicove.mcp.remote")
    dispatch, principal, calls = dispatcher(tmp_path)

    class Verifier:
        async def verify(self, credential: str, *, administrative: bool = False) -> Principal:
            return principal

    host = module.RemoteReadHost(
        Verifier(), {principal.key: dispatch}, allowed_hosts=("testserver",)
    )
    async with host.run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=host),
            base_url="https://testserver",
            headers={
                "authorization": "Bearer fixture",
                "accept": "application/json, text/event-stream",
            },
        ) as client:
            response = await client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "fixture", "version": "1"},
                    },
                },
            )
            assert response.status_code == 400
            assert calls == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [
        '{"jsonrpc":"2.0","id":1,"id":2,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"fixture","version":"1"}}}',
        '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"fixture","version":"1"},"extra":NaN}}',
        '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"fixture","version":"1"},"extra":1e999}}',
    ],
)
async def test_ambiguous_json_rejected_before_session_creation(tmp_path: Path, body: str) -> None:
    module = importlib.import_module("capicove.mcp.remote")
    dispatch, principal, calls = dispatcher(tmp_path)

    class Verifier:
        async def verify(self, credential: str, *, administrative: bool = False) -> Principal:
            return principal

    host = module.RemoteReadHost(
        Verifier(), {principal.key: dispatch}, allowed_hosts=("testserver",)
    )
    async with host.run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=host),
            base_url="https://testserver",
            headers={
                "authorization": "Bearer fixture",
                "accept": "application/json, text/event-stream",
                "content-type": "application/json",
            },
        ) as client:
            response = await client.post("/mcp", content=body)
            assert response.status_code == 400
            assert "mcp-session-id" not in response.headers
            assert calls == []


@pytest.mark.anyio
@pytest.mark.parametrize("json_response", [True, False])
async def test_request_ids_cannot_be_reused_after_scope_downgrade(
    tmp_path: Path, json_response: bool
) -> None:
    from dataclasses import replace

    module = importlib.import_module("capicove.mcp.remote")
    dispatch, principal, calls = dispatcher(tmp_path)

    class Verifier:
        async def verify(self, credential: str, *, administrative: bool = False) -> Principal:
            return (
                principal
                if credential == "broad"
                else replace(principal, token_permissions=frozenset())
            )

    host = module.RemoteReadHost(
        Verifier(),
        {principal.key: dispatch},
        allowed_hosts=("testserver",),
        json_response=json_response,
    )
    async with host.run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=host),
            base_url="https://testserver",
            headers={
                "authorization": "Bearer broad",
                "accept": "application/json, text/event-stream",
            },
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
                        "clientInfo": {"name": "fixture", "version": "1"},
                    },
                },
            )
            client.headers["mcp-session-id"] = response.headers["mcp-session-id"]
            client.headers["mcp-protocol-version"] = "2025-11-25"
            await client.post(
                "/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}
            )
            first = await client.post(
                "/mcp", json={"jsonrpc": "2.0", "id": 10, "method": "tools/list"}
            )
            assert first.status_code == 200
            second = await client.post(
                "/mcp",
                headers={"authorization": "Bearer reduced"},
                json={"jsonrpc": "2.0", "id": "10", "method": "tools/list"},
            )
            assert second.status_code == 400
            assert calls == []


@pytest.mark.anyio
@pytest.mark.parametrize("version", [None, "2024-11-05"])
async def test_session_cannot_change_or_omit_pinned_protocol(
    tmp_path: Path, version: str | None
) -> None:
    module = importlib.import_module("capicove.mcp.remote")
    dispatch, principal, calls = dispatcher(tmp_path)

    class Verifier:
        async def verify(self, credential: str, *, administrative: bool = False) -> Principal:
            return principal

    host = module.RemoteReadHost(
        Verifier(), {principal.key: dispatch}, allowed_hosts=("testserver",)
    )
    async with host.run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=host),
            base_url="https://testserver",
            headers={
                "authorization": "Bearer fixture",
                "accept": "application/json, text/event-stream",
            },
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
                        "clientInfo": {"name": "fixture", "version": "1"},
                    },
                },
            )
            client.headers["mcp-session-id"] = response.headers["mcp-session-id"]
            if version is not None:
                client.headers["mcp-protocol-version"] = version
            response = await client.post(
                "/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
            )
            assert response.status_code == 400
            assert calls == []


@pytest.mark.anyio
async def test_actual_sdk_catalog_continuations_are_bounded(tmp_path: Path) -> None:
    from dataclasses import replace

    module = importlib.import_module("capicove.mcp.remote")
    dispatch, principal, _ = dispatcher(tmp_path)
    base = dispatch._bindings["capicove_test_echo"]
    dispatch._bindings = {
        f"capicove_test_page_{index:03}": replace(
            base,
            name=f"capicove_test_page_{index:03}",
            spec=replace(base.spec, tool_id=f"capicove.test.page_{index:03}"),
        )
        for index in range(51)
    }

    class Verifier:
        async def verify(self, credential: str, *, administrative: bool = False) -> Principal:
            return principal

    host = module.RemoteReadHost(
        Verifier(), {principal.key: dispatch}, allowed_hosts=("testserver",)
    )
    async with host.run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=host),
            base_url="https://testserver",
            headers={"authorization": "Bearer fixture"},
        ) as client:
            async with streamable_http_client("https://testserver/mcp", http_client=client) as (
                read,
                write,
                _,
            ):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    first = await session.list_tools()
                    assert len(first.tools) == 50
                    assert first.nextCursor
                    second = await session.list_tools(cursor=first.nextCursor)
                    assert len(second.tools) == 1
                    assert second.nextCursor is None
