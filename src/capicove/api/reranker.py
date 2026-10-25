"""Bounded HTTP reranking using a resident public text reranker port."""

from __future__ import annotations

import hmac
from collections.abc import AsyncIterator
from concurrent.futures import Future
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any, Protocol

import anyio
from fastapi import FastAPI, Request
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from capicove.api.model_http import run_worker as run_in_threadpool
from capicove.retrieval.reranking import (
    MAX_REQUEST_BYTES,
    TextReranker,
    parse_rerank_json,
    parse_rerank_request,
    render_rerank_result,
)


class ManagedReranker(TextReranker, Protocol):
    """Require explicit cleanup from the HTTP application's owned provider."""

    def close(self) -> None: ...


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": code, "message": message}},
        status_code=status,
        headers={"WWW-Authenticate": "Bearer"} if status == 401 else None,
    )


def create_app(
    ranker: ManagedReranker, *, api_key: str | None = None, model_name: str | None = None
) -> FastAPI:
    """Own one warm model; admit one inference and reject excess work with 429."""
    if api_key is not None and (
        not api_key or len(api_key) > 8192 or any(not 33 <= ord(c) <= 126 for c in api_key)
    ):
        raise ValueError("invalid reranker API key")
    admission = Lock()
    model_name = model_name or ranker.model_id
    names = {ranker.model_id, ranker.model_id.rsplit("/", 1)[-1], model_name}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await run_in_threadpool(ranker.close)

    app = FastAPI(title="Capicove Text Reranker", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def authenticate(request: Request, call_next: Any) -> Any:
        if request.url.path != "/health" and api_key is not None:
            supplied = request.headers.get("authorization", "").encode("utf-8")
            if not hmac.compare_digest(supplied, ("Bearer " + api_key).encode("ascii")):
                return _error(401, "invalid_api_key", "Invalid API key.")
        return await call_next(request)

    @app.exception_handler(HTTPException)
    async def route_error(request: Request, error: HTTPException) -> JSONResponse:
        return _error(error.status_code, "route_unavailable", "Route or method unavailable.")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model": ranker.model_id, "model_revision": ranker.model_revision}

    def model_info() -> dict[str, str | int]:
        return {
            "id": model_name,
            "object": "model",
            "created": 0,
            "owned_by": "local",
            "model_revision": ranker.model_revision,
        }

    @app.get("/v1/models")
    async def models() -> dict[str, object]:
        return {"object": "list", "data": [model_info()]}

    @app.get("/v1/models/{model:path}")
    async def model(model: str) -> JSONResponse:
        if model not in names:
            return _error(404, "model_not_found", "The requested model is not served.")
        return JSONResponse(model_info())

    def execute(
        ownership: Future[None], query: str, documents: tuple[str, ...], top_n: int, include: bool
    ) -> dict[str, object]:
        # Starting and cancelling are atomic. A late scheduled worker must not
        # run after the request has already returned its admission slot.
        if not ownership.set_running_or_notify_cancel():
            raise RuntimeError("rerank_cancelled_before_start")
        try:
            return render_rerank_result(
                ranker.rerank(query, documents, top_n=top_n), documents, return_documents=include
            )
        finally:
            # The worker owns the slot until native inference actually finishes,
            # including when an HTTP client disconnects or cancels its wait.
            admission.release()
            ownership.set_result(None)

    @app.post("/v1/rerank")
    async def rerank(request: Request) -> JSONResponse:
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_REQUEST_BYTES:
                return _error(413, "request_too_large", "Request body exceeds 2 MiB.")
            body.extend(chunk)
        try:
            value = parse_rerank_json(bytes(body))
            if not isinstance(value.get("model"), str):
                raise ValueError("invalid_rerank_request")
            if value["model"] not in names:
                return _error(404, "model_not_found", "The requested model is not served.")
            parsed = parse_rerank_request(value)
        except (ValueError, UnicodeError, RecursionError):
            return _error(
                400,
                "invalid_request",
                "Expected a query, 1-128 nonblank text documents and valid options.",
            )
        if not admission.acquire(blocking=False):
            return _error(429, "server_busy", "Another reranking request is running.")
        ownership: Future[None] = Future()
        try:
            with anyio.CancelScope(shield=True):
                result = await run_in_threadpool(execute, ownership, *parsed)
            result["model"] = model_name
            return JSONResponse(result)
        except ValueError as error:
            if str(error) == "rerank_input_too_long":
                return _error(
                    400,
                    "input_too_long",
                    "A query/document pair or total token count exceeds the configured limit.",
                )
            return _error(500, "inference_failed", "Local reranking inference failed.")
        except Exception:
            return _error(500, "inference_failed", "Local reranking inference failed.")
        finally:
            # Only an unstarted task can be cancelled. Running work keeps its
            # slot until execute() finishes, even if its HTTP waiter disappears.
            if ownership.cancel():
                admission.release()

    return app
