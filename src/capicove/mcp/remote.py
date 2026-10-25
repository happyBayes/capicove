"""Programmatic authenticated read-only Streamable HTTP composition.

This composition is not a production launcher: durable effects, isolated workers,
full configuration loading and deployment qualification remain separate gates.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import math
import secrets
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from capicove.contracts.remote import AuthorityKey, Principal, PrincipalKey
from capicove.mcp.remote_catalog import CatalogPager, catalog_entry
from capicove.runtime.remote_auth import bearer_token
from capicove.runtime.remote_dispatch import RemoteDispatcher
from capicove.runtime.remote_metadata import ProtectedResourceMetadata
from capicove.runtime.remote_ports import TokenVerifier
from capicove.runtime.remote_sessions import RequestAdmission, SessionCapacity


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous fields before forwarding a request to the SDK."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _finite_float(value: str) -> float:
    """Reject numeric overflow before forwarding the same bytes to the SDK."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite JSON number")
    return result


def _reject_constant(value: str) -> Any:
    """Forbid non-JSON numeric constants accepted by the Python decoder."""
    raise ValueError("Non-finite JSON number")


@dataclass(frozen=True, slots=True)
class _Session:
    authority: AuthorityKey
    ceiling: frozenset[str]
    created_at: float
    used_request_ids: set[str] = field(default_factory=set)


class RemoteReadHost:
    """Authenticate every SDK request and pin sessions to complete authority."""

    def __init__(
        self,
        verifier: TokenVerifier,
        dispatchers: Mapping[PrincipalKey, RemoteDispatcher],
        *,
        allowed_hosts: tuple[str, ...],
        json_response: bool = True,
        resource_metadata: ProtectedResourceMetadata | None = None,
    ) -> None:
        if not allowed_hosts or any("*" in host for host in allowed_hosts):
            raise ValueError("Exact remote hosts must be explicitly enrolled")
        self._allowed_hosts = frozenset(host.lower().encode("ascii") for host in allowed_hosts)
        if resource_metadata is not None and (
            resource_metadata.host.encode("ascii") not in self._allowed_hosts
            or any(
                item.authority.audience != resource_metadata.resource
                or item.authority.principal_key.issuer != resource_metadata.authorization_server
                for item in dispatchers.values()
            )
        ):
            raise ValueError("Public metadata must match the enrolled host and grant audiences")
        self._metadata = resource_metadata
        anyio: Any = importlib.import_module("anyio")
        self._anyio = anyio
        self._request_slots = threading.BoundedSemaphore(128)
        self._admission = RequestAdmission()
        self._session_capacity = SessionCapacity()
        self._json_response = json_response
        self._worker_slots = asyncio.Semaphore(4)
        self._read_tasks: set[asyncio.Task[dict[str, object]]] = set()
        sdk: Any = importlib.import_module("mcp.server")
        types: Any = importlib.import_module("mcp.types")
        manager: Any = importlib.import_module("mcp.server.streamable_http_manager")
        security: Any = importlib.import_module("mcp.server.transport_security")
        server: Any = sdk.Server("capicove-remote-read")
        self._verifier = verifier
        self._dispatchers = MappingProxyType(dict(dispatchers))
        self._sessions: dict[str, _Session] = {}
        self._lock = asyncio.Lock()
        self._catalog = CatalogPager(secrets.token_bytes(32))

        def current() -> tuple[Principal, RemoteDispatcher]:
            principal = server.request_context.request.scope["capicove.principal"]
            return principal, self._dispatchers[principal.key]

        async def list_tools(request: Any) -> Any:
            principal, dispatcher = current()
            request_scope = server.request_context.request.scope
            bindings = await asyncio.to_thread(dispatcher.catalog, principal)
            cursor = (
                request.params.cursor
                if request is not None and request.params is not None
                else None
            )
            page = await asyncio.to_thread(
                self._catalog.page,
                bindings,
                authority=dispatcher.authority,
                session_id=request_scope["capicove.session_id"],
                permissions=principal.token_permissions,
                cursor=cursor,
            )
            return types.ListToolsResult(
                tools=[types.Tool(**catalog_entry(item)) for item in page.bindings],
                nextCursor=page.next_cursor,
                _meta={"capicove": {"catalog_sha256": page.catalog_digest}},
            )

        async def call_tool(name: str, arguments: dict[str, Any]) -> Any:
            principal, dispatcher = current()
            try:
                if len(self._read_tasks) >= 32:
                    raise ValueError("Remote read queue capacity exceeded")

                async def invoke() -> dict[str, object]:
                    async with self._worker_slots:
                        return await asyncio.to_thread(
                            dispatcher.invoke, principal, name, arguments
                        )

                task = asyncio.create_task(invoke())
                self._read_tasks.add(task)

                def completed(done: asyncio.Task[dict[str, object]]) -> None:
                    self._read_tasks.discard(done)
                    if not done.cancelled():
                        done.exception()

                task.add_done_callback(completed)
                result = await asyncio.shield(task)
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text="Capicove Tool result")],
                    structuredContent=result,
                    isError=result.get("status") != "succeeded",
                )
            except Exception:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text="Operation unavailable")],
                    isError=True,
                )

        # The SDK selects its request-aware adapter from the concrete runtime
        # annotation; keep SDK types lazy until this host is explicitly created.
        list_tools.__annotations__["request"] = types.ListToolsRequest
        server.list_tools()(list_tools)
        server.call_tool(validate_input=False)(call_tool)
        self._manager = manager.StreamableHTTPSessionManager(
            server,
            json_response=json_response,
            max_request_body_size=2 * 1024 * 1024,
            max_sessions=1000,
            session_idle_timeout=900,
            security_settings=security.TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=list(allowed_hosts),
                allowed_origins=[],
            ),
        )
        # The pinned SDK retires idle, crashed and failed-opening sessions without a
        # callback; reconciliation reads its live session table, so fail closed
        # if a different SDK no longer exposes that table.
        if not isinstance(getattr(self._manager, "_server_instances", None), dict):
            raise RuntimeError("Unsupported MCP SDK session manager")

    def run(self) -> Any:
        """Own SDK session task lifetime explicitly."""
        return self._manager.run()

    def _retire_locked(self, session_id: str) -> None:
        """Drop one host binding and release its capacity exactly once."""
        self._sessions.pop(session_id, None)
        self._session_capacity.remove(session_id)

    def _reconcile_locked(self) -> None:
        """Retire bindings the SDK no longer serves or past the maximum lifetime."""
        live = self._manager._server_instances
        cutoff = time.monotonic() - 3600
        for key, value in list(self._sessions.items()):
            if key not in live or value.created_at <= cutoff:
                self._retire_locked(key)

    async def __call__(self, scope: Mapping[str, Any], receive: Any, send: Any) -> None:
        """Bound ingress before authentication, buffering and SDK allocation."""
        if scope.get("type") != "http":
            raise ValueError("Use run() to manage the HTTP-only host lifecycle")
        if not self._request_slots.acquire(blocking=False):
            await send(
                {
                    "type": "http.response.start",
                    "status": 429,
                    "headers": [(b"content-type", b"text/plain")],
                }
            )
            await send({"type": "http.response.body", "body": b"Remote request rejected"})
            return
        try:
            await self._handle_request(scope, receive, send)
        finally:
            self._request_slots.release()

    async def _handle_request(self, scope: Mapping[str, Any], receive: Any, send: Any) -> None:
        """Reject credentials before SDK parsing and bind every response session."""

        async def reject(status: int) -> None:
            headers = [(b"content-type", b"text/plain")]
            if status == 401:
                headers.append(
                    (
                        b"www-authenticate",
                        self._metadata.challenge() if self._metadata is not None else b"Bearer",
                    )
                )
            await send({"type": "http.response.start", "status": status, "headers": headers})
            await send({"type": "http.response.body", "body": b"Remote request rejected"})

        if scope.get("type") != "http":
            raise ValueError("Use run() to manage the HTTP-only host lifecycle")
        headers = scope.get("headers", [])
        for protected in (
            b"host",
            b"origin",
            b"mcp-protocol-version",
            b"content-type",
            b"content-length",
        ):
            if sum(name.lower() == protected for name, _ in headers) > 1:
                await reject(400)
                return
        if any(
            name.lower() in {b"forwarded", b"x-forwarded-host", b"x-forwarded-for"}
            for name, _ in headers
        ):
            await reject(400)
            return
        hosts = [value.lower() for name, value in headers if name.lower() == b"host"]
        if (
            len(hosts) != 1
            or hosts[0] not in self._allowed_hosts
            or any(name.lower() == b"origin" for name, _ in headers)
        ):
            await reject(400)
            return
        if self._metadata is not None and scope.get("path") in {
            "/.well-known/oauth-protected-resource",
            "/.well-known/oauth-protected-resource/mcp",
        }:
            if scope.get("method") != "GET":
                await reject(405)
                return
            if scope.get("query_string", b""):
                await reject(400)
                return
            body = json.dumps(self._metadata.payload(), separators=(",", ":")).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"cache-control", b"no-store"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        if scope.get("path") != "/mcp":
            await reject(404)
            return
        try:
            credential = bearer_token(headers, query=scope.get("query_string", b""))
            async with asyncio.timeout(3):
                principal = await self._verifier.verify(credential)
        except (ValueError, TimeoutError):
            await reject(401)
            return
        dispatcher = self._dispatchers.get(principal.key)
        if dispatcher is None:
            await reject(403)
            return
        try:
            await asyncio.to_thread(dispatcher.validate_principal, principal)
        except (ValueError, PermissionError):
            await reject(403)
            return
        session_headers = [value for name, value in headers if name.lower() == b"mcp-session-id"]
        if len(session_headers) > 1:
            await reject(400)
            return
        session_id = (
            session_headers[0].decode("ascii", errors="replace") if session_headers else None
        )
        protocol_headers = [
            value for name, value in headers if name.lower() == b"mcp-protocol-version"
        ]
        if (session_id is not None or protocol_headers) and protocol_headers != [b"2025-11-25"]:
            await reject(400)
            return
        if session_id is None:
            async with self._lock:
                self._reconcile_locked()
                at_capacity = len(self._sessions) >= 1000
            if at_capacity:
                await reject(503)
                return
        if session_id is not None:
            async with self._lock:
                self._reconcile_locked()
                binding = self._sessions.get(session_id)
            if binding is None:
                await reject(404)
                return
            if (
                binding.authority != dispatcher.authority
                or not principal.token_permissions <= binding.ceiling
            ):
                await reject(403)
                return
        with ExitStack() as admission:
            try:
                admission.enter_context(
                    self._admission.reserve(
                        principal.key,
                        session_id=session_id,
                        streaming=scope.get("method") == "GET"
                        or (scope.get("method") == "POST" and not self._json_response),
                    )
                )
            except (PermissionError, ValueError):
                await reject(429)
                return
            register_session: Callable[[str], None] | None = None
            if session_id is None:
                try:
                    register_session = admission.enter_context(
                        self._session_capacity.reserve(principal.key)
                    )
                except PermissionError:
                    await reject(429)
                    return
            sdk_receive = receive
            if scope.get("method") == "POST":
                payload = bytearray()
                try:
                    async with asyncio.timeout(2):
                        while True:
                            chunk = await receive()
                            if chunk.get("type") != "http.request":
                                raise ValueError("Request disconnected")
                            data = chunk.get("body", b"")
                            if len(payload) + len(data) > 2 * 1024 * 1024:
                                raise ValueError("Request body exceeds byte limit")
                            payload.extend(data)
                            if not chunk.get("more_body", False):
                                break
                    request = json.loads(
                        payload,
                        object_pairs_hook=_unique_object,
                        parse_constant=_reject_constant,
                        parse_float=_finite_float,
                    )
                    if not isinstance(request, dict):
                        raise ValueError("Batched requests are not supported")
                    if request.get("method") == "initialize":
                        params = request.get("params")
                        if (
                            not isinstance(params, dict)
                            or params.get("protocolVersion") != "2025-11-25"
                        ):
                            raise ValueError("Unqualified protocol profile")
                except (ValueError, RecursionError, TimeoutError):
                    await reject(400)
                    return
                if session_id is not None and "id" in request and "method" in request:
                    request_id = request["id"]
                    if type(request_id) not in (int, str) or len(str(request_id)) > 256:
                        await reject(400)
                        return
                    # The SDK indexes response streams by str(id). Keep reservations
                    # for the session lifetime, including abandoned producer tasks.
                    async with self._lock:
                        session = self._sessions.get(session_id)
                        conflict = (
                            session is None
                            or str(request_id) in session.used_request_ids
                            or len(session.used_request_ids) >= 10000
                        )
                        if not conflict and session is not None:
                            session.used_request_ids.add(str(request_id))
                    if conflict:
                        await reject(400)
                        return
                delivered = False

                async def replay() -> Any:
                    nonlocal delivered
                    if delivered:
                        return await receive()
                    delivered = True
                    return {"type": "http.request", "body": bytes(payload), "more_body": False}

                sdk_receive = replay
            request_scope = dict(scope)
            request_scope["capicove.principal"] = principal
            request_scope["capicove.session_id"] = session_id

            sent_bytes = 0
            response_status: int | None = None
            opened: str | None = None

            async def bounded_send(message: dict[str, Any]) -> None:
                nonlocal sent_bytes, response_status, opened
                if message["type"] == "http.response.start":
                    response_status = message["status"]
                    for name, value in message.get("headers", []):
                        if name.lower() == b"mcp-session-id":
                            identity = value.decode("ascii")
                            if register_session is not None and opened is None:
                                register_session(identity)
                                opened = identity
                            elif identity not in (session_id, opened):
                                raise PermissionError("SDK session identity changed")
                            async with self._lock:
                                self._sessions.setdefault(
                                    identity,
                                    _Session(
                                        dispatcher.authority,
                                        principal.token_permissions,
                                        time.monotonic(),
                                    ),
                                )
                if message["type"] == "http.response.body" and message.get("body"):
                    sent_bytes += len(message["body"])
                    if sent_bytes > 2 * 1024 * 1024:
                        raise ValueError("Remote response byte limit exceeded")

                    async def send_before_expiry() -> None:
                        remaining = min(2, principal.expires_at - time.time())
                        if remaining <= 0:
                            raise PermissionError("Response authorization expired")
                        async with asyncio.timeout(remaining):
                            await send(message)

                    def fenced_send() -> None:
                        with dispatcher.release_permit(principal):
                            self._anyio.from_thread.run(send_before_expiry)

                    with self._anyio.CancelScope(shield=True):
                        await self._anyio.to_thread.run_sync(fenced_send, abandon_on_cancel=False)
                else:
                    await send(message)

            try:
                async with asyncio.timeout(max(0, principal.expires_at - time.time())):
                    await self._manager.handle_request(request_scope, sdk_receive, bounded_send)
            except TimeoutError:
                # The token expired mid-request: never extend execution authority. An
                # unstarted response becomes a 401 challenge; a started one just ends.
                if response_status is None:
                    with self._anyio.CancelScope(shield=True):
                        await reject(401)
            finally:
                # Shield retirement so cancellation cannot leak a reservation for a
                # session the SDK already discarded (DELETE, failed opening send).
                with self._anyio.CancelScope(shield=True):
                    async with self._lock:
                        if (
                            scope.get("method") == "DELETE"
                            and session_id is not None
                            and response_status == 200
                        ):
                            self._retire_locked(session_id)
                        for identity in (opened, session_id):
                            if (
                                identity is not None
                                and identity not in self._manager._server_instances
                            ):
                                self._retire_locked(identity)
