"""Concurrent and abandoned requests cannot redirect protected SDK results."""

import asyncio
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import MappingProxyType
from typing import Any

import httpx
import pytest

from capicove.contracts import InvocationContext, ToolResult
from capicove.contracts.remote import Principal
from capicove.mcp.bindings import McpToolBinding
from capicove.mcp.remote import RemoteReadHost
from tests.unit.runtime.remote.test_remote_dispatch import dispatcher


@pytest.mark.anyio
@pytest.mark.parametrize("json_response", [True, False])
async def test_active_and_cancelled_request_ids_cannot_cross_scope(
    tmp_path: Path, json_response: bool
) -> None:
    """Reserve normalized IDs even when the HTTP caller abandons its producer."""
    dispatch, broad, _ = dispatcher(tmp_path)
    binding: McpToolBinding = next(iter(dispatch._bindings.values()))
    entered, release, finished = Event(), Event(), Event()
    protected_payload = "SECRET_READ_RESULT"

    def blocked(context: InvocationContext, arguments: Mapping[str, object]) -> ToolResult:
        entered.set()
        try:
            assert release.wait(5), "Test must release the blocked read"
            return replace(binding.handler(context, arguments), result={"actor": protected_payload})
        finally:
            finished.set()

    dispatch._bindings = MappingProxyType({binding.name: replace(binding, handler=blocked)})
    reduced = replace(broad, token_permissions=frozenset())

    class Verifier:
        async def verify(self, credential: str, *, administrative: bool = False) -> Principal:
            if credential == "broad":
                return broad
            if credential == "reduced":
                return reduced
            raise ValueError("Unknown fixture credential")

    host: Any = RemoteReadHost(
        Verifier(),
        {broad.key: dispatch},
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
            initialized = await client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {},
                        "clientInfo": {"name": "isolation-fixture", "version": "1"},
                    },
                },
            )
            assert initialized.status_code == 200
            client.headers["mcp-session-id"] = initialized.headers["mcp-session-id"]
            client.headers["mcp-protocol-version"] = "2025-11-25"
            notified = await client.post(
                "/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}
            )
            assert notified.status_code == 202
            original = asyncio.create_task(
                client.post(
                    "/mcp",
                    json={
                        "jsonrpc": "2.0",
                        "id": 10,
                        "method": "tools/call",
                        "params": {"name": binding.name, "arguments": {}},
                    },
                )
            )

            async def collide(identity: int | str) -> httpx.Response:
                return await asyncio.wait_for(
                    client.post(
                        "/mcp",
                        headers={"authorization": "Bearer reduced"},
                        json={"jsonrpc": "2.0", "id": identity, "method": "tools/list"},
                    ),
                    timeout=2,
                )

            try:
                assert await asyncio.to_thread(entered.wait, 2)
                concurrent = await collide("10")
                assert concurrent.status_code == 400
                assert protected_payload not in concurrent.text
                original.cancel()
                await asyncio.gather(original, return_exceptions=True)
                assert not finished.is_set(), "Cancellation must leave a live producer to fence"
                abandoned = await collide(10)
                assert abandoned.status_code == 400
                assert protected_payload not in abandoned.text
            finally:
                release.set()
                original.cancel()
                await asyncio.gather(original, return_exceptions=True)
                assert await asyncio.to_thread(finished.wait, 2)
