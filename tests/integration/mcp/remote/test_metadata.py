"""Public resource discovery exposes only explicitly configured OAuth endpoints."""

import importlib
from typing import Any

import httpx
import pytest

from capicove.contracts.remote import Principal
from capicove.mcp.remote import RemoteReadHost


@pytest.mark.anyio
async def test_bad_host_is_denied_before_verifier_or_body() -> None:
    calls: list[str] = []

    class Verifier:
        async def verify(self, credential: str, *, administrative: bool = False) -> Principal:
            calls.append(credential)
            raise ValueError("fixture rejection")

    host = RemoteReadHost(Verifier(), {}, allowed_hosts=("testserver",))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=host), base_url="https://testserver"
    ) as client:
        for headers in ({"host": "evil.test"}, {"origin": "https://evil.test"}):
            response = await client.post(
                "/mcp", headers={"authorization": "Bearer private", **headers}, json={}
            )
            assert response.status_code == 400
        assert calls == []


@pytest.mark.anyio
async def test_public_metadata_and_challenge_have_no_private_enrollment() -> None:
    module: Any = importlib.import_module("capicove.runtime.remote_metadata")
    metadata = module.ProtectedResourceMetadata("https://testserver/mcp", "https://issuer.test")
    calls: list[str] = []

    class Verifier:
        async def verify(self, credential: str, *, administrative: bool = False) -> Principal:
            calls.append(credential)
            raise ValueError("fixture rejection")

    host = RemoteReadHost(Verifier(), {}, allowed_hosts=("testserver",), resource_metadata=metadata)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=host), base_url="https://testserver"
    ) as client:
        for path in (
            "/.well-known/oauth-protected-resource/mcp",
            "/.well-known/oauth-protected-resource",
        ):
            response = await client.get(path)
            assert response.status_code == 200
            assert response.json() == {
                "resource": "https://testserver/mcp",
                "authorization_servers": ["https://issuer.test"],
                "bearer_methods_supported": ["header"],
            }
        response = await client.post("/mcp", json={})
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == (
            'Bearer resource_metadata="https://testserver/.well-known/oauth-protected-resource/mcp"'
        )
        assert calls == []
        assert (await client.post("/.well-known/oauth-protected-resource/mcp")).status_code == 405
        assert (
            await client.get(
                "/.well-known/oauth-protected-resource/mcp", headers={"host": "evil.test"}
            )
        ).status_code == 400


@pytest.mark.parametrize(
    "resource",
    [
        "http://testserver/mcp",
        "https://user@testserver/mcp",
        "https://testserver/mcp?token=private",
        "https://testserver/mcp\r\nInjected: true",
        "https://testserver/other",
    ],
)
def test_metadata_rejects_unsafe_resource_identifiers(resource: str) -> None:
    module: Any = importlib.import_module("capicove.runtime.remote_metadata")
    with pytest.raises(ValueError):
        module.ProtectedResourceMetadata(resource, "https://issuer.test")


def test_metadata_issuer_must_match_enrolled_principals(tmp_path: Any) -> None:
    from dataclasses import replace

    from tests.unit.runtime.remote.test_remote_dispatch import dispatcher

    module: Any = importlib.import_module("capicove.runtime.remote_metadata")
    dispatch, principal, _ = dispatcher(tmp_path)
    dispatch._grant = replace(
        dispatch._grant,
        authority=replace(dispatch.authority, audience="https://testserver/mcp"),
    )
    metadata = module.ProtectedResourceMetadata(
        "https://testserver/mcp", "https://wrong-issuer.test"
    )

    class Verifier:
        async def verify(self, credential: str, *, administrative: bool = False) -> Principal:
            raise AssertionError("Configuration must fail before verification")

    with pytest.raises(ValueError):
        RemoteReadHost(
            Verifier(),
            {principal.key: dispatch},
            allowed_hosts=("testserver",),
            resource_metadata=metadata,
        )
