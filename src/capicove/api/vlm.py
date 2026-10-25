"""Bounded Chat Completions service over registered public VLM factory ports."""

from __future__ import annotations

import hmac
from collections.abc import AsyncIterator
from concurrent.futures import Future
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any

import anyio
from fastapi import FastAPI, Request
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from capicove.api.model_http import run_worker as run_in_threadpool
from capicove.model import ModelFactory
from capicove.model.vlm import MAX_REQUEST_BYTES, VlmError, parse_chat_request, parse_json
from capicove.model.vlm_service import model_info, registered_vlms


def _error(status: int, code: str) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "code": code,
                "type": "invalid_request_error" if status < 500 else "server_error",
                "message": code.replace("_", " "),
            }
        },
        status_code=status,
        headers={"WWW-Authenticate": "Bearer"} if status == 401 else None,
    )


def create_app(factory: ModelFactory, *, api_key: str | None = None) -> FastAPI:
    """Own a configured factory and one in-flight request, including body intake."""
    if api_key is not None and (
        not api_key or len(api_key) > 8192 or any(not 33 <= ord(c) <= 126 for c in api_key)
    ):
        raise ValueError("invalid VLM service API key")
    catalog = registered_vlms(factory.configuration)
    aliases = {name: binding for name, (binding, _) in catalog.items()}
    for _, (binding, profile) in catalog.items():
        matches = [p for _, p in catalog.values() if p.model == profile.model]
        if len(matches) == 1 and profile.model not in aliases:
            aliases[profile.model] = binding
    admission = Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await run_in_threadpool(factory.close)

    app = FastAPI(title="Capicove VLM", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def authenticate(request: Request, call_next: Any) -> Any:
        if request.url.path != "/health" and api_key is not None:
            supplied = request.headers.get("authorization", "").encode("utf-8")
            if not hmac.compare_digest(supplied, ("Bearer " + api_key).encode("ascii")):
                return _error(401, "invalid_api_key")
        return await call_next(request)

    @app.exception_handler(HTTPException)
    async def route_error(request: Request, error: HTTPException) -> JSONResponse:
        return _error(error.status_code, "route_unavailable")

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"status": "ok", "upstream_checked": False, "models": list(catalog)}

    @app.get("/v1/models")
    async def models() -> dict[str, object]:
        return {"object": "list", "data": [model_info(p) for _, p in catalog.values()]}

    @app.get("/v1/models/{model:path}")
    async def model(model: str) -> JSONResponse:
        if model not in aliases:
            return _error(404, "model_not_found")
        name = factory.configuration.bindings[aliases[model]]
        return JSONResponse(model_info(catalog[name][1]))

    def execute(ownership: Future[None], payload: bytes, count: bool = False) -> JSONResponse:
        if not ownership.set_running_or_notify_cancel():
            raise RuntimeError("generation_cancelled_before_start")
        try:
            try:
                raw = parse_json(payload)
                requested_model = raw.get("model")
                if not isinstance(requested_model, str):
                    return _error(400, "model_required")
                if requested_model not in aliases:
                    return _error(404, "model_not_found")
                parsed = parse_chat_request(raw)
            except (ValueError, UnicodeError, RecursionError):
                return _error(400, "invalid_request")
            with factory.vlm(aliases[requested_model]) as provider:
                result = (
                    provider.count_tokens(parsed).to_dict()
                    if count
                    else provider.generate(parsed).to_openai()
                )
                result["model"] = requested_model
                return JSONResponse(result)
        finally:
            # A cancelled HTTP waiter cannot release a running worker's slot.
            admission.release()
            ownership.set_result(None)

    @app.post("/v1/count_tokens")
    @app.post("/v1/chat/completions")
    async def generate(request: Request) -> JSONResponse:
        if not admission.acquire(blocking=False):
            return _error(429, "server_busy")
        ownership: Future[None] = Future()
        try:
            payload = bytearray()
            with anyio.fail_after(30):
                async for chunk in request.stream():
                    if len(payload) + len(chunk) > MAX_REQUEST_BYTES:
                        return _error(413, "request_too_large")
                    payload.extend(chunk)
            with anyio.CancelScope(shield=True):
                result = await run_in_threadpool(
                    execute, ownership, bytes(payload), request.url.path == "/v1/count_tokens"
                )
            return result
        except TimeoutError:
            return _error(408, "request_timeout")
        except VlmError as error:
            return _error(error.status, error.code)
        except ValueError:
            return _error(400, "invalid_request_or_model_policy")
        except Exception:
            return _error(500, "generation_failed")
        finally:
            # Cancellation succeeds only before execute() atomically starts.
            if ownership.cancel():
                admission.release()

    return app
