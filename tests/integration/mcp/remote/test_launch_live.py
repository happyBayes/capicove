"""Live Linux loopback qualification of the composed launch with the official SDK client.

The listener is a real TCP socket served by uvicorn; the client is the pinned MCP
SDK. JWKS comes from an in-memory transport, because the pinned identity adapter
accepts only HTTPS endpoints and no external identity provider is provisioned.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.integration.mcp.remote.test_launch import _setup
from tests.unit.runtime.remote.test_jwt_verifier import signing_key, token


__all__ = ["signing_key"]
pytestmark = pytest.mark.skipif(os.name != "posix", reason="private POSIX policy loading")


@pytest.mark.anyio
async def test_sdk_client_lists_and_calls_over_real_loopback_tcp(
    tmp_path: Path, signing_key: Any
) -> None:
    import uvicorn
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    from capicove.runtime import remote_launch

    jwt = importlib.import_module("jwt")
    public = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(signing_key.public_key()))
    public.update(kid="key-1", alg="RS256", use="sig")
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"keys": [public]}))
    app = remote_launch.compose_remote_app(
        remote_launch.load_launch(_setup(tmp_path, json_response=False)),
        identity_transport=transport,
    )
    credential = token(
        signing_key,
        scope="evidence:read evidence:search remote:readiness",
        exp=int(time.time()) + 120,
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, lifespan="on", log_level="error"))
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(10):
                while not server.started:
                    if serving.done():
                        await serving
                        raise AssertionError("server stopped before listening")
                    await asyncio.sleep(0.01)
            # A reverse proxy normally supplies the public Host; emulate it exactly.
            async with httpx.AsyncClient(
                trust_env=False,
                headers={"authorization": f"Bearer {credential}", "host": "mcp.test"},
            ) as client:
                endpoint = f"http://127.0.0.1:{port}/mcp"
                async with streamable_http_client(endpoint, http_client=client) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tools = await session.list_tools()
                        assert [tool.name for tool in tools.tools] == [
                            "capicove_get_evidence",
                            "capicove_remote_readiness",
                            "capicove_search",
                        ]
                        ready = await session.call_tool("capicove_remote_readiness", {})
                        assert ready.isError is False
                        assert ready.structuredContent["result"]["status"] == "ready"  # type: ignore[index]
                        missing = await session.call_tool("capicove_search", {"query": "quokka"})
                        assert missing.isError is True
                        assert (
                            missing.structuredContent["error"]["code"]  # type: ignore[index]
                            in {"unavailable", "invalid_input", "not_found"}
                        )
                wrong_host = await client.post(endpoint, json={}, headers={"host": "evil.test"})
                assert wrong_host.status_code == 400
        finally:
            server.should_exit = True
            await serving
