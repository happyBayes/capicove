"""Optional OpenAI-compatible HTTP boundary for verified local BGE-M3 (RFC-0026)."""

from __future__ import annotations

import base64
import hmac
import json
import struct
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Protocol

from fastapi import FastAPI, Request
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from capicove.api.model_http import run_worker as run_in_threadpool
from capicove.contracts import EmbeddingSpaceRef
from capicove.retrieval import TextEmbedding


if TYPE_CHECKING:
    from capicove.retrieval.bge import BgeTextBatchEncoder

MODEL_ID = "BAAI/bge-m3"
_MODEL_NAMES = frozenset({MODEL_ID, "bge-m3"})
_MAX_BODY_BYTES = 2 * 1024 * 1024
_MAX_INPUTS = 128
_MAX_TOTAL_TOKENS = 300_000


class EmbeddingEncoder(Protocol):
    """Consume the public BGE adapter's validated embeddings and exact usage."""

    @property
    def space(self) -> EmbeddingSpaceRef: ...

    def encode_with_token_counts(
        self,
        texts: tuple[str, ...],
        *,
        space: EmbeddingSpaceRef,
        role: str,
        max_total_tokens: int | None = None,
    ) -> tuple[tuple[TextEmbedding, ...], tuple[int, ...]]: ...

    def close(self) -> None: ...


def build_encoder(
    model_directory: Path,
    *,
    expected_model_manifest_sha256: str,
    device: str = "cpu",
    batch_size: int = 8,
    max_tokens: int = 512,
) -> BgeTextBatchEncoder:
    """Verify and warm an explicitly installed model before accepting requests."""
    from capicove.retrieval.bge import BgeTextBatchEncoder, load_bge_m3_profile

    profile = load_bge_m3_profile(
        model_directory,
        expected_manifest_sha256=expected_model_manifest_sha256,
        max_tokens=max_tokens,
    )
    encoder = BgeTextBatchEncoder(profile, device=device, batch_size=batch_size)
    try:
        encoder.encode(("warmup",), space=encoder.space, role="document")
    except Exception:
        encoder.close()
        raise
    return encoder


def _error(
    status: int, message: str, *, param: str | None = None, code: str = "invalid_request"
) -> JSONResponse:
    kind = "server_error" if status >= 500 else "invalid_request_error"
    if status == 401:
        kind = "authentication_error"
    elif status == 429:
        kind = "rate_limit_error"
    return JSONResponse(
        {"error": {"message": message, "type": kind, "param": param, "code": code}},
        status_code=status,
        headers={"WWW-Authenticate": "Bearer"} if status == 401 else None,
    )


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _parse(
    value: Any, names: frozenset[str] = _MODEL_NAMES
) -> tuple[tuple[str, ...], str] | JSONResponse:
    if not isinstance(value, dict) or set(value) - {
        "input",
        "model",
        "encoding_format",
        "dimensions",
        "user",
    }:
        return _error(400, "Expected an embeddings request with supported fields.")
    if not isinstance(value.get("model"), str):
        return _error(400, "Model must be a string.", param="model")
    if value["model"] not in names:
        return _error(
            404, "The requested model is not served.", param="model", code="model_not_found"
        )
    raw = value.get("input")
    texts = [raw] if isinstance(raw, str) else raw
    if (
        not isinstance(texts, list)
        or not 1 <= len(texts) <= _MAX_INPUTS
        or any(not isinstance(text, str) or not text.strip() for text in texts)
    ):
        return _error(
            400, "Input must be one nonblank string or 1-128 nonblank strings.", param="input"
        )
    try:
        for text in texts:
            text.encode("utf-8")
    except UnicodeError:
        return _error(400, "Input must contain valid Unicode.", param="input")
    encoding = value.get("encoding_format", "float")
    if not isinstance(encoding, str) or encoding not in {"float", "base64"}:
        return _error(400, "Encoding must be float or base64.", param="encoding_format")
    if "dimensions" in value and (
        type(value["dimensions"]) is not int or value["dimensions"] != 1024
    ):
        return _error(400, "BGE-M3 supports exactly 1024 dimensions.", param="dimensions")
    if "user" in value and (not isinstance(value["user"], str) or len(value["user"]) > 256):
        return _error(400, "User must be a string of at most 256 characters.", param="user")
    return tuple(texts), encoding


def create_app(
    encoder: EmbeddingEncoder, *, api_key: str | None = None, model_name: str = MODEL_ID
) -> FastAPI:
    """Serve text embeddings with bounded admission and optional bearer authentication.

    The application owns the encoder and closes it at normal lifespan shutdown.
    This single-owner model API is independent of workspace host permissions.
    """
    if encoder.space.dimension != 1024:
        raise ValueError("bge_m3_requires_1024_dimensions")
    if api_key is not None and (
        not api_key or len(api_key) > 8192 or any(not 33 <= ord(c) <= 126 for c in api_key)
    ):
        raise ValueError("invalid_embedding_api_key")
    admission = Lock()
    names = _MODEL_NAMES | {model_name}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await run_in_threadpool(encoder.close)

    app = FastAPI(title="Capicove BGE-M3 Embeddings", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def authenticate(request: Request, call_next: Any) -> Any:
        if request.url.path != "/health" and api_key is not None:
            supplied = request.headers.get("authorization", "").encode("utf-8")
            if not hmac.compare_digest(supplied, ("Bearer " + api_key).encode("ascii")):
                return _error(401, "Invalid API key.", code="invalid_api_key")
        return await call_next(request)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, error: HTTPException) -> JSONResponse:
        return _error(error.status_code, "The requested route or method is not available.")

    @app.get("/health")
    async def health() -> dict[str, str | int]:
        return {"status": "ok", "model": model_name, "dimension": 1024}

    def model_info() -> dict[str, str | int]:
        return {"id": model_name, "object": "model", "created": 0, "owned_by": "local"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [model_info()]}

    @app.get("/v1/models/{model:path}")
    async def model(model: str) -> Any:
        if model not in names:
            return _error(
                404, "The requested model is not served.", param="model", code="model_not_found"
            )
        return model_info()

    def embed(texts: tuple[str, ...], encoding: str) -> dict[str, Any]:
        vectors, counts = encoder.encode_with_token_counts(
            texts, space=encoder.space, role="document", max_total_tokens=_MAX_TOTAL_TOKENS
        )
        data = []
        for index, vector in enumerate(vectors):
            values = list(vector.vector)
            encoded: list[float] | str = values
            if encoding == "base64":
                encoded = base64.b64encode(struct.pack("<1024f", *values)).decode("ascii")
            data.append({"object": "embedding", "index": index, "embedding": encoded})
        total = sum(counts)
        return {
            "object": "list",
            "data": data,
            "model": model_name,
            "usage": {"prompt_tokens": total, "total_tokens": total},
        }

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> JSONResponse:
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > _MAX_BODY_BYTES:
                return _error(413, "Request body exceeds 2 MiB.", code="request_too_large")
            body.extend(chunk)
        try:
            value = json.loads(body, object_pairs_hook=_unique)
        except (ValueError, UnicodeError, RecursionError):
            return _error(400, "Request body must be valid JSON.")
        parsed = _parse(value, names)
        if isinstance(parsed, JSONResponse):
            return parsed
        if not admission.acquire(blocking=False):
            return _error(429, "Another embedding request is running.", code="server_busy")
        try:
            result = await run_in_threadpool(embed, *parsed)
            return JSONResponse(result)
        except ValueError as error:
            if str(error) in {"embedding_input_too_long", "embedding_batch_too_long"}:
                return _error(400, "Input exceeds the configured token limit.", param="input")
            return _error(500, "Local embedding inference failed.", code="inference_failed")
        except Exception:
            return _error(500, "Local embedding inference failed.", code="inference_failed")
        finally:
            admission.release()

    return app
