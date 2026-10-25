"""An authenticated stream cannot exhaust another application's gateway quota."""

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest

from capicove.contracts.remote import Principal
from capicove.mcp.remote import RemoteReadHost, _Session
from tests.unit.runtime.remote.test_remote_dispatch import dispatcher


@pytest.mark.anyio
async def test_third_stream_is_rejected_before_sdk_and_cancel_releases_slot(tmp_path: Path) -> None:
    dispatch, principal, calls = dispatcher(tmp_path)

    class Verifier:
        async def verify(self, credential: str, *, administrative: bool = False) -> Principal:
            return principal

    host = RemoteReadHost(Verifier(), {principal.key: dispatch}, allowed_hosts=("testserver",))
    entered: asyncio.Queue[bool] = asyncio.Queue()
    release = asyncio.Event()

    async def streaming_sdk(scope: Any, receive: Any, send: Any) -> None:
        entered.put_nowait(True)
        await release.wait()

    host._manager.handle_request = streaming_sdk
    # Install an enrolled session fixture to isolate admission from transport.
    # Separate protocol and TCP tests exercise actual SDK session establishment.
    host._sessions["fixture-session"] = _Session(
        dispatch.authority, principal.token_permissions, time.monotonic()
    )
    # Host bindings are reconciled against live SDK sessions; keep one live.
    host._manager._server_instances["fixture-session"] = object()
    scope = {
        "type": "http",
        "path": "/mcp",
        "method": "GET",
        "headers": [
            (b"host", b"testserver"),
            (b"authorization", b"Bearer fixture"),
            (b"mcp-session-id", b"fixture-session"),
            (b"mcp-protocol-version", b"2025-11-25"),
        ],
    }
    messages: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    first = asyncio.create_task(host(scope, receive, send))
    second = asyncio.create_task(host(scope, receive, send))
    try:
        await asyncio.wait_for(entered.get(), 1)
        await asyncio.wait_for(entered.get(), 1)
        third = asyncio.create_task(host(scope, receive, send))
        done, _ = await asyncio.wait({third}, timeout=0.1)
        if not done:
            third.cancel()
            await asyncio.gather(third, return_exceptions=True)
        assert done, "Over-capacity stream reached the SDK and blocked"
        assert messages[0]["status"] == 429
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        replacement = asyncio.create_task(host(scope, receive, send))
        await asyncio.wait_for(entered.get(), 1)
        release.set()
        await replacement
        assert calls == []
    finally:
        release.set()
        await asyncio.gather(first, second, return_exceptions=True)


@pytest.mark.anyio
async def test_post_body_receivers_share_application_capacity(tmp_path: Path) -> None:
    from capicove.runtime.remote_sessions import RequestAdmission

    dispatch, principal, _ = dispatcher(tmp_path)

    class Verifier:
        async def verify(self, credential: str, *, administrative: bool = False) -> Principal:
            return principal

    host = RemoteReadHost(Verifier(), {principal.key: dispatch}, allowed_hosts=("testserver",))
    host._admission = RequestAdmission(global_limit=1, application_limit=1)
    entered = asyncio.Event()
    release = asyncio.Event()
    receivers: list[bool] = []
    messages: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        receivers.append(True)
        entered.set()
        await release.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "path": "/mcp",
        "method": "POST",
        "headers": [(b"host", b"testserver"), (b"authorization", b"Bearer fixture")],
    }
    first = asyncio.create_task(host(scope, receive, send))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        second = asyncio.create_task(host(scope, receive, send))
        done, _ = await asyncio.wait({second}, timeout=0.1)
        if not done:
            second.cancel()
            await asyncio.gather(second, return_exceptions=True)
        assert done, "Over-capacity body receiver was admitted"
        assert len(receivers) == 1
        assert messages[0]["status"] == 429
    finally:
        release.set()
        await asyncio.gather(first, return_exceptions=True)
    assert host._admission.active_requests == 0


@pytest.mark.anyio
async def test_global_ingress_limit_precedes_token_verification(tmp_path: Path) -> None:
    import threading

    dispatch, principal, _ = dispatcher(tmp_path)
    entered = asyncio.Event()
    release = asyncio.Event()
    verifications: list[bool] = []

    class Verifier:
        async def verify(self, credential: str, *, administrative: bool = False) -> Principal:
            verifications.append(True)
            entered.set()
            await release.wait()
            raise ValueError("fixture rejection")

    host = RemoteReadHost(Verifier(), {principal.key: dispatch}, allowed_hosts=("testserver",))
    host._request_slots = threading.BoundedSemaphore(1)
    messages: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        raise AssertionError("Unauthenticated request reached body reception")

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "path": "/mcp",
        "method": "POST",
        "headers": [(b"host", b"testserver"), (b"authorization", b"Bearer fixture")],
    }
    first = asyncio.create_task(host(scope, receive, send))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        second = asyncio.create_task(host(scope, receive, send))
        done, _ = await asyncio.wait({second}, timeout=0.1)
        if not done:
            second.cancel()
            await asyncio.gather(second, return_exceptions=True)
        assert done, "Ingress overload entered token verification"
        assert len(verifications) == 1
        assert messages[0]["status"] == 429
    finally:
        release.set()
        await asyncio.gather(first, return_exceptions=True)


@pytest.mark.anyio
async def test_application_session_cap_applies_to_actual_sdk_initialization(tmp_path: Path) -> None:
    import httpx

    from capicove.runtime.remote_sessions import SessionCapacity

    dispatch, principal, _ = dispatcher(tmp_path)

    class Verifier:
        async def verify(self, credential: str, *, administrative: bool = False) -> Principal:
            return principal

    host = RemoteReadHost(Verifier(), {principal.key: dispatch}, allowed_hosts=("testserver",))
    host._session_capacity = SessionCapacity(global_limit=2, application_limit=1)
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "quota-fixture", "version": "1"},
        },
    }
    async with host.run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=host),
            base_url="https://testserver",
            headers={
                "authorization": "Bearer fixture",
                "accept": "application/json, text/event-stream",
            },
        ) as client:
            first = await client.post("/mcp", json=request)
            assert first.status_code == 200
            denied = await client.post("/mcp", json=request)
            assert denied.status_code == 429
            assert len(host._sessions) == 1
            deleted = await client.delete(
                "/mcp",
                headers={
                    "mcp-session-id": first.headers["mcp-session-id"],
                    "mcp-protocol-version": "2025-11-25",
                },
            )
            assert deleted.status_code == 200
            replacement = await client.post("/mcp", json=request)
            assert replacement.status_code == 200


def _initialize_request() -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "quota-fixture", "version": "1"},
        },
    }


def _quota_host(tmp_path: Path, *, idle_timeout: float) -> Any:
    from capicove.runtime.remote_sessions import SessionCapacity

    dispatch, principal, _ = dispatcher(tmp_path)

    class Verifier:
        async def verify(self, credential: str, *, administrative: bool = False) -> Principal:
            return principal

    host = RemoteReadHost(Verifier(), {principal.key: dispatch}, allowed_hosts=("testserver",))
    host._session_capacity = SessionCapacity(global_limit=2, application_limit=1)
    host._manager.session_idle_timeout = idle_timeout
    return host


def _client(host: Any) -> Any:
    import httpx

    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=host),
        base_url="https://testserver",
        headers={
            "authorization": "Bearer fixture",
            "accept": "application/json, text/event-stream",
        },
    )


@pytest.mark.anyio
async def test_sdk_expired_session_delete_404_retires_host_reservation(tmp_path: Path) -> None:
    host = _quota_host(tmp_path, idle_timeout=0.05)
    async with host.run(), _client(host) as client:
        first = await client.post("/mcp", json=_initialize_request())
        assert first.status_code == 200
        identity = first.headers["mcp-session-id"]
        await asyncio.sleep(0.15)
        assert identity not in host._manager._server_instances
        deleted = await client.delete(
            "/mcp",
            headers={"mcp-session-id": identity, "mcp-protocol-version": "2025-11-25"},
        )
        assert deleted.status_code == 404
        assert identity not in host._sessions
        replacement = await client.post("/mcp", json=_initialize_request())
        assert replacement.status_code == 200
        assert list(host._sessions) == [replacement.headers["mcp-session-id"]]


@pytest.mark.anyio
async def test_abandoned_expired_session_is_reclaimed_before_rejecting_init(
    tmp_path: Path,
) -> None:
    host = _quota_host(tmp_path, idle_timeout=0.05)
    async with host.run(), _client(host) as client:
        first = await client.post("/mcp", json=_initialize_request())
        assert first.status_code == 200
        await asyncio.sleep(0.15)
        replacement = await client.post("/mcp", json=_initialize_request())
        assert replacement.status_code == 200
        assert list(host._sessions) == [replacement.headers["mcp-session-id"]]


@pytest.mark.anyio
async def test_active_session_is_not_reclaimed_by_replacement(tmp_path: Path) -> None:
    host = _quota_host(tmp_path, idle_timeout=60)
    async with host.run(), _client(host) as client:
        first = await client.post("/mcp", json=_initialize_request())
        assert first.status_code == 200
        denied = await asyncio.gather(
            client.post("/mcp", json=_initialize_request()),
            client.post("/mcp", json=_initialize_request()),
        )
        assert [item.status_code for item in denied] == [429, 429]
        assert list(host._sessions) == [first.headers["mcp-session-id"]]
        assert first.headers["mcp-session-id"] in host._manager._server_instances


@pytest.mark.anyio
async def test_failed_opening_send_releases_session_reservation(tmp_path: Path) -> None:
    host = _quota_host(tmp_path, idle_timeout=60)
    body = json.dumps(_initialize_request()).encode()
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "query_string": b"",
        "headers": [
            (b"host", b"testserver"),
            (b"authorization", b"Bearer fixture"),
            (b"accept", b"application/json, text/event-stream"),
            (b"content-type", b"application/json"),
        ],
    }

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    async def broken_send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body":
            raise OSError("client disconnected")

    async with host.run(), _client(host) as client:
        with pytest.raises(OSError):
            await host(scope, receive, broken_send)
        await asyncio.sleep(0)
        assert host._sessions == {}
        replacement = await client.post("/mcp", json=_initialize_request())
        assert replacement.status_code == 200


@pytest.mark.anyio
async def test_request_outliving_its_token_ends_with_401_not_an_error(tmp_path: Path) -> None:
    from dataclasses import replace

    dispatch, principal, calls = dispatcher(tmp_path)
    short = replace(principal, expires_at=time.time() + 0.2)

    class Verifier:
        async def verify(self, credential: str, *, administrative: bool = False) -> Principal:
            return short

    host = RemoteReadHost(Verifier(), {principal.key: dispatch}, allowed_hosts=("testserver",))

    async def slow_sdk(scope: Any, receive: Any, send: Any) -> None:
        await asyncio.sleep(5)

    host._manager.handle_request = slow_sdk
    host._sessions["fixture-session"] = _Session(
        dispatch.authority, principal.token_permissions, time.monotonic()
    )
    host._manager._server_instances["fixture-session"] = object()
    scope = {
        "type": "http",
        "path": "/mcp",
        "method": "GET",
        "headers": [
            (b"host", b"testserver"),
            (b"authorization", b"Bearer fixture"),
            (b"mcp-session-id", b"fixture-session"),
            (b"mcp-protocol-version", b"2025-11-25"),
        ],
    }
    messages: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    await asyncio.wait_for(host(scope, receive, send), 3)
    assert messages[0]["type"] == "http.response.start" and messages[0]["status"] == 401
    assert calls == []
