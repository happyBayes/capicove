"""Shared bounded HTTP execution and authentication for model services."""

from __future__ import annotations

import asyncio
import hmac
from collections.abc import Callable
from functools import partial
from typing import Any, ParamSpec, TypeVar

import anyio
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse


_P = ParamSpec("_P")
_T = TypeVar("_T")


def error(status: int, code: str) -> JSONResponse:
    """Return server-authored diagnostics without model or upload contents."""
    return JSONResponse(
        {"error": {"code": code, "message": code.replace("_", " ")}},
        status_code=status,
        headers={"WWW-Authenticate": "Bearer"} if status == 401 else None,
    )


def authenticate(app: FastAPI, api_key: str | None) -> None:
    """Authenticate before consuming any body, allowing only public health checks."""
    if api_key is not None and (
        not api_key or len(api_key) > 8192 or any(not 33 <= ord(c) <= 126 for c in api_key)
    ):
        raise ValueError("invalid service API key")

    @app.middleware("http")
    async def check(request: Request, call_next: Any) -> Any:
        if (
            request.url.path != "/health"
            and api_key is not None
            and not hmac.compare_digest(
                request.headers.get("authorization", "").encode(),
                ("Bearer " + api_key).encode("ascii"),
            )
        ):
            return error(401, "invalid_api_key")
        return await call_next(request)


async def run_worker(function: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs) -> _T:
    """Drain native work before cancellation releases its files or admission slot."""
    with anyio.CancelScope(shield=True):
        worker = asyncio.create_task(anyio.to_thread.run_sync(partial(function, *args, **kwargs)))
        cancelled = False
        while True:
            try:
                result = await asyncio.shield(worker)
            except asyncio.CancelledError:
                if worker.done():
                    if not worker.cancelled():
                        worker.exception()
                    raise
                cancelled = True
            except Exception:
                if cancelled:
                    raise asyncio.CancelledError from None
                raise
            else:
                if cancelled:
                    raise asyncio.CancelledError
                return result
    raise asyncio.CancelledError
