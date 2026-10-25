"""Exercise the remote host over a real loopback socket on the local platform."""

import asyncio
import socket
from pathlib import Path

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from capicove.contracts.remote import Principal
from capicove.mcp.remote import RemoteReadHost
from tests.unit.runtime.remote.test_remote_dispatch import dispatcher


@pytest.mark.anyio
@pytest.mark.parametrize("json_response", [True, False])
async def test_real_loopback_streamable_http(tmp_path: Path, json_response: bool) -> None:
    import uvicorn

    dispatch, principal, calls = dispatcher(tmp_path)

    class Verifier:
        async def verify(self, credential: str, *, administrative: bool = False) -> Principal:
            if credential != "local-fixture":
                raise ValueError("Credential rejected")
            return principal

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        host = RemoteReadHost(
            Verifier(),
            {principal.key: dispatch},
            allowed_hosts=(f"127.0.0.1:{port}",),
            json_response=json_response,
        )
        server = uvicorn.Server(uvicorn.Config(host, lifespan="off", log_level="error"))
        async with host.run():
            serving = asyncio.create_task(server.serve(sockets=[listener]))
            try:
                async with asyncio.timeout(5):
                    while not server.started:
                        if serving.done():
                            await serving
                            raise AssertionError("HTTP server stopped before listening")
                        await asyncio.sleep(0.01)
                async with httpx.AsyncClient(
                    trust_env=False, headers={"authorization": "Bearer local-fixture"}
                ) as client:
                    endpoint = f"http://127.0.0.1:{port}/mcp"
                    rejected = await client.post(
                        endpoint, headers={"authorization": "Bearer invalid"}, json={}
                    )
                    assert rejected.status_code == 401
                    assert calls == []
                    async with streamable_http_client(endpoint, http_client=client) as (
                        read,
                        write,
                        session_id,
                    ):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            assert session_id()
                            listed = await session.list_tools()
                            assert [tool.name for tool in listed.tools] == ["capicove_test_echo"]
                            result = await session.call_tool("capicove_test_echo", {})
                            assert not result.isError
                            assert len(calls) == 1
            finally:
                server.should_exit = True
                async with asyncio.timeout(5):
                    await serving
