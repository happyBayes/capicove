"""Loopback HTTP application over the workspace host's selected Tools (RFC-0020).

The application owns authentication, routing, body bounds, and admission only.
Every invocation is delegated to an invoker that runs the host-composed binding,
so HTTP never gains a second Tool implementation or its own authority.
"""

from __future__ import annotations

import hmac
import json
import threading
from collections.abc import Awaitable, Callable, Mapping
from typing import Protocol

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from capicove.mcp.bindings import InvalidToolArguments


MAX_BODY_BYTES = 2 * 1024 * 1024
_ALIASES = {"/v1/search": "capicove.search", "/v1/evidence": "capicove.get_evidence"}


class WorkerTimeout(Exception):
    """The worker did not answer within the Tool deadline."""


class WorkerUnavailable(Exception):
    """The worker stopped, for example during shutdown."""


class ToolInvocationFailed(Exception):
    """The binding raised outside the canonical result envelope."""


class ToolInvoker(Protocol):
    """Run one selected Tool through the host-composed binding."""

    def invoke(
        self, tool_id: str, arguments: Mapping[str, object], *, timeout: float
    ) -> dict[str, object]: ...


class _BodyTooLarge(Exception):
    pass


def create_workspace_app(
    *,
    catalog: Mapping[str, object],
    token: str,
    invoker: ToolInvoker,
    timeouts: Mapping[str, float] | None = None,
) -> Starlette:
    """Create the authenticated application for one composed catalog."""
    tools = catalog.get("tools")
    if not isinstance(tools, list):
        raise ValueError("catalog requires a tool list")
    selected = {str(item["tool_id"]) for item in tools if isinstance(item, Mapping)}
    deadlines = dict(timeouts or {})
    admission = threading.Lock()
    closing = threading.Event()
    expected = f"Bearer {token}".encode()

    def authorized(request: Request) -> bool:
        supplied = request.headers.get("authorization", "").encode()
        return hmac.compare_digest(supplied, expected)

    def guarded(
        endpoint: Callable[[Request], Awaitable[Response]],
    ) -> Callable[[Request], Awaitable[Response]]:
        async def wrapper(request: Request) -> Response:
            if not authorized(request):
                return _error(401, "unauthorized", "A valid bearer token is required.")
            return await endpoint(request)

        return wrapper

    async def health(request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    async def list_tools(request: Request) -> Response:
        return JSONResponse(dict(catalog))

    async def invoke(request: Request, tool_id: str) -> Response:
        if tool_id not in selected:
            return _error(404, "not_found", "The Tool is not selected by this host.")
        if closing.is_set():
            return _error(503, "unavailable", "The server is shutting down.")
        try:
            arguments = await _read_arguments(request)
        except _BodyTooLarge:
            return _error(413, "payload_too_large", "The request body exceeds 2 MiB.")
        except ValueError:
            return _error(400, "invalid_request", "The body must be a JSON object.")
        if not admission.acquire(blocking=False):
            return _error(429, "busy", "Another invocation is running.")
        try:
            result = await run_in_threadpool(
                invoker.invoke, tool_id, arguments, timeout=deadlines.get(tool_id, 30.0)
            )
        except InvalidToolArguments:
            return _error(400, "invalid_arguments", "The arguments do not match the Tool.")
        except WorkerTimeout:
            return _error(504, "deadline_exceeded", "The Tool exceeded its deadline.")
        except WorkerUnavailable:
            return _error(503, "unavailable", "The Tool worker is unavailable.")
        except ToolInvocationFailed:
            return _error(500, "internal", "The Tool invocation failed.")
        finally:
            admission.release()
        return JSONResponse(result)

    async def tool_route(request: Request) -> Response:
        return await invoke(request, str(request.path_params["tool_id"]))

    routes = [
        Route("/v1/health", health, methods=["GET"]),
        Route("/v1/tools", guarded(list_tools), methods=["GET"]),
        Route("/v1/tools/{tool_id}", guarded(tool_route), methods=["POST"]),
    ]
    for path, tool_id in _ALIASES.items():

        async def alias(request: Request, tool_id: str = tool_id) -> Response:
            return await invoke(request, tool_id)

        routes.append(Route(path, guarded(alias), methods=["POST"]))
    app = Starlette(routes=routes)
    app.state.closing = closing
    return app


async def _read_arguments(request: Request) -> dict[str, object]:
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        raise _BodyTooLarge
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_BODY_BYTES:
            raise _BodyTooLarge
    try:
        value = json.loads(bytes(body).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise ValueError("invalid JSON body") from None
    if not isinstance(value, dict):
        raise ValueError("body must be an object")
    return value


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


__all__ = [
    "MAX_BODY_BYTES",
    "ToolInvocationFailed",
    "ToolInvoker",
    "WorkerTimeout",
    "WorkerUnavailable",
    "create_workspace_app",
]
