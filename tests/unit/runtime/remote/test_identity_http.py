"""Identity fetches never follow redirects or accept ambiguous JSON."""

import importlib
from typing import Any

import pytest


def client(handler: Any, *, enabled: bool = True) -> Any:
    assert importlib.util.find_spec("capicove.runtime._remote.identity_http") is not None
    httpx = importlib.import_module("httpx")
    module = importlib.import_module("capicove.runtime._remote.identity_http")
    return module.IdentityHttp(
        "https://issuer.test/keys", network_enabled=enabled, transport=httpx.MockTransport(handler)
    )


@pytest.mark.anyio
async def test_network_requires_explicit_opt_in() -> None:
    calls = []
    with pytest.raises(ValueError):
        await client(lambda request: calls.append(request), enabled=False).get()
    assert calls == []


@pytest.mark.anyio
async def test_bounded_identity_document() -> None:
    httpx = importlib.import_module("httpx")
    result = await client(lambda request: httpx.Response(200, json={"keys": []})).get()
    assert result == {"keys": []}


@pytest.mark.anyio
@pytest.mark.parametrize(
    "status,body,headers",
    [
        (302, b"", {"location": "https://attacker.test/keys"}),
        (200, b'{"active":false,"active":true}', {}),
        (200, b"x" * 65537, {}),
        (200, b"[]", {}),
        (500, b'{"keys":[]}', {}),
    ],
)
async def test_bad_identity_responses_fail_closed(
    status: int, body: bytes, headers: dict[str, str]
) -> None:
    httpx = importlib.import_module("httpx")
    calls = []

    def handler(request: Any) -> Any:
        calls.append(request)
        return httpx.Response(status, content=body, headers=headers)

    with pytest.raises(ValueError):
        await client(handler).get()
    assert len(calls) == 1


@pytest.mark.anyio
async def test_compressed_or_excessively_nested_identity_json_rejected() -> None:
    import gzip

    httpx = importlib.import_module("httpx")
    with pytest.raises(ValueError):
        await client(
            lambda request: httpx.Response(
                200, content=gzip.compress(b"x" * 1048576), headers={"content-encoding": "gzip"}
            )
        ).get()
    with pytest.raises(ValueError):
        await client(
            lambda request: httpx.Response(
                200, content=b'{"x":' + b"[" * 1500 + b"0" + b"]" * 1500 + b"}"
            )
        ).get()
