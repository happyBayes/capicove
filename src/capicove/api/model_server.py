"""One authenticated listener dispatching to existing capability applications."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any
from urllib.parse import urlencode

import anyio
from fastapi import FastAPI
from starlette.requests import Request
from starlette.types import Message, Receive, Scope, Send

from capicove.api.model_http import authenticate, error, run_worker
from capicove.api.ocr import OcrHttpSettings
from capicove.model import ModelCapability, ModelConfiguration, ModelFactory, ProviderRegistry
from capicove.model.http_profile import HttpModelProfile
from capicove.model.serving import ROUTES, catalog, document_binding, select_models
from capicove.model.vlm import parse_json


def _encode(value: dict[str, object]) -> bytes:
    """Keep bounded but potentially large JSON work outside the event loop."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode()


class _Owned:
    """Release a native instance exactly once across child lifespan and rollback."""

    def __init__(self, value: Any, dispose: Any) -> None:
        self.value, self.dispose = value, dispose
        self.closed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.value, name)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.dispose(self.value)


async def _application(
    configuration: ModelConfiguration,
    name: str,
    registry: ProviderRegistry,
    stack: AsyncExitStack,
    ocr_settings: OcrHttpSettings,
) -> FastAPI:
    profile = configuration.profiles[name]
    if isinstance(profile, HttpModelProfile):
        from capicove.api.vlm import create_app

        factory = ModelFactory(ModelConfiguration({name: profile}, {name: name}), registry=registry)
        stack.push_async_callback(run_worker, factory.close)
        with factory.vlm(name):
            pass
        return create_app(factory)
    definition = registry.resolve(profile.provider)
    definition.validate_profile(profile)
    created: list[_Owned] = []

    def construct() -> _Owned:
        instance = _Owned(definition.create_instance(profile), definition.dispose)
        created.append(instance)
        return instance

    try:
        instance = await run_worker(construct)
    except BaseException:
        # The worker is drained before cancellation propagates; retain its result
        # until ownership has transferred to the lifespan stack.
        if created:
            await run_worker(created[0].close)
        raise
    stack.push_async_callback(run_worker, instance.close)
    if profile.capability == ModelCapability.TEXT_EMBEDDING:
        from capicove.api.bge_m3 import create_app as embeddings_app

        return embeddings_app(instance, model_name=name)
    if profile.capability == ModelCapability.TEXT_RERANKING:
        from capicove.api.reranker import create_app as reranker_app

        return reranker_app(instance, model_name=name)
    if profile.capability == ModelCapability.AUDIO_TRANSCRIPTION:
        from capicove.api.audio import create_app as audio_app

        return audio_app(instance, model=name)
    if profile.capability == ModelCapability.IMAGE_TEXT_EMBEDDING:
        from capicove.api.siglip2 import create_app as siglip_app
        from capicove.retrieval import Siglip2RetrievalService

        return siglip_app(Siglip2RetrievalService(instance))
    if profile.capability == ModelCapability.IMAGE_OCR:
        from capicove.api.ocr import create_app as ocr_app

        return ocr_app(lambda: instance, settings=ocr_settings)
    if profile.capability == ModelCapability.DOCUMENT_PARSER:
        from fastapi.exceptions import RequestValidationError
        from starlette.exceptions import HTTPException

        from capicove.api.documents import document_router
        from capicove.api.ocr import ApiError, _failure
        from capicove.documents import (
            DocumentInputError,
            DocumentProcessingError,
            DocumentRejectedError,
        )
        from capicove.runtime.ocr_service import OcrRuntime

        # PDF parsing uses the runtime's admission/worker methods, not its OCR provider.
        app = FastAPI()
        for error_type in (
            Exception,
            ApiError,
            DocumentInputError,
            DocumentRejectedError,
            DocumentProcessingError,
            RequestValidationError,
            HTTPException,
        ):
            app.add_exception_handler(error_type, _failure)

        @app.middleware("http")
        async def request_identity(request: Request, call_next: Any) -> Any:
            from uuid import uuid4

            request.state.request_id = uuid4().hex
            return await call_next(request)

        app.include_router(
            document_router(
                document_binding(profile, instance),
                OcrRuntime(lambda: instance),
                ocr_settings,
            )
        )
        return app
    raise ValueError("unsupported model service capability")


def _aliases(configuration: ModelConfiguration) -> dict[str, str]:
    """Resolve unique upstream VLM names without shadowing registered names."""
    aliases = {name: name for name in configuration.profiles}
    upstream: dict[str, list[str]] = {}
    for name, profile in configuration.profiles.items():
        if isinstance(profile, HttpModelProfile):
            upstream.setdefault(profile.model, []).append(name)
    for model, names in upstream.items():
        if len(names) == 1 and model not in aliases:
            aliases[model] = names[0]
    return aliases


class ModelDispatch:
    """Bound intake and serialize each selected model independently."""

    def __init__(self, configuration: ModelConfiguration, applications: dict[str, FastAPI]) -> None:
        self.configuration, self.applications = configuration, applications
        self.aliases = _aliases(configuration)
        self.busy: set[str] = set()
        self.intake = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await send({"type": "websocket.close", "code": 1008})
            return
        path = scope["path"]
        candidates = [
            n for n, p in self.configuration.profiles.items() if path in ROUTES[p.capability]
        ]
        if not candidates or scope["method"] != "POST":
            await error(404 if not candidates else 405, "route_unavailable")(scope, receive, send)
            return
        if self.intake >= 4:
            await error(429, "upload_busy")(scope, receive, send)
            return
        self.intake += 1
        selected: str | None = None
        locked = False
        response_started = False

        async def observed_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            request = Request(scope, receive)
            payload = bytearray()
            limit = 51 * 1024 * 1024 if path == "/v1/documents/parse" else 33 * 1024 * 1024
            with anyio.fail_after(30):
                async for chunk in request.stream():
                    if len(payload) + len(chunk) > limit:
                        await error(413, "request_too_large")(scope, receive, send)
                        return
                    payload.extend(chunk)
            body = bytes(payload)
            raw: dict[str, object] | None = None
            query = list(request.query_params.multi_items())
            if len([k for k, _ in query if k == "model"]) > 1:
                raise ValueError("duplicate model selector")
            requested: object = request.query_params.get("model")
            is_json = path in {
                "/v1/embeddings",
                "/v1/rerank",
                "/v1/chat/completions",
                "/v1/count_tokens",
                "/v1/embeddings/text",
            }
            if is_json:
                raw = await run_worker(parse_json, body)
                if requested is not None and raw.get("model", requested) != requested:
                    raise ValueError("conflicting model selectors")
                requested = raw.get("model", requested)
            elif path == "/v1/audio/transcriptions":

                async def form_receive() -> Message:
                    return {"type": "http.request", "body": body, "more_body": False}

                with anyio.CancelScope(shield=True):
                    async with Request(scope, form_receive).form(
                        max_files=1, max_fields=3, max_part_size=1024
                    ) as form:
                        if len(form.multi_items()) != len(form):
                            raise ValueError("duplicate multipart fields")
                        if requested is not None and form.get("model") != requested:
                            raise ValueError("conflicting model selectors")
                        requested = form.get("model")
            if (
                requested is None
                and len(candidates) == 1
                and path
                not in {
                    "/v1/embeddings",
                    "/v1/rerank",
                    "/v1/chat/completions",
                    "/v1/count_tokens",
                    "/v1/audio/transcriptions",
                }
            ):
                requested = candidates[0]
            selected = self.aliases.get(requested) if isinstance(requested, str) else None
            if selected not in candidates:
                await error(404, "model_not_found")(scope, receive, send)
                return
            assert selected is not None
            if selected in self.busy:
                await error(429, "server_busy")(scope, receive, send)
                return
            self.busy.add(selected)
            locked = True
            profile = self.configuration.profiles[selected]
            if raw is not None:
                if profile.capability == ModelCapability.IMAGE_TEXT_EMBEDDING:
                    raw.pop("model", None)
                body = await run_worker(_encode, raw)
            child_scope = dict(scope)
            child_scope["query_string"] = urlencode(
                [(k, v) for k, v in query if k != "model"]
            ).encode()
            child_scope["headers"] = [
                (k, v) for k, v in scope["headers"] if k.lower() != b"content-length"
            ] + [(b"content-length", str(len(body)).encode())]
            delivered = False

            async def replay() -> Message:
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            with anyio.CancelScope(shield=True):
                await self.applications[selected](child_scope, replay, observed_send)
        except TimeoutError:
            if response_started:
                raise
            await error(408, "upload_timeout")(scope, receive, send)
        except (ValueError, UnicodeError, RecursionError):
            if response_started:
                raise
            await error(400, "invalid_request")(scope, receive, send)
        finally:
            if locked and selected is not None:
                self.busy.remove(selected)
            self.intake -= 1


def create_app(
    configuration: ModelConfiguration,
    *,
    models: tuple[str, ...] | None = None,
    api_key: str | None = None,
    registry: ProviderRegistry | None = None,
    ocr_settings: OcrHttpSettings | None = None,
) -> FastAPI:
    """Serve selected registered instances; roll back all resources on startup failure."""
    from capicove.model.providers import builtin_registry

    selected = select_models(configuration, models)
    ocr_settings = ocr_settings or OcrHttpSettings()
    registry = registry or builtin_registry()
    # Validate provider/capability combinations without loading native libraries.
    ModelFactory(selected, registry=registry).close()
    applications: dict[str, FastAPI] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            for name in selected.profiles:
                child = await _application(selected, name, registry, stack, ocr_settings)
                await stack.enter_async_context(child.router.lifespan_context(child))
                applications[name] = child
            schema = app.openapi()
            for name, application in applications.items():
                child_schema = application.openapi()
                for route in ROUTES[selected.profiles[name].capability]:
                    if route in child_schema["paths"]:
                        schema["paths"][route] = child_schema["paths"][route]
                for category, entries in child_schema.get("components", {}).items():
                    schema.setdefault("components", {}).setdefault(category, {}).update(entries)
            app.openapi_schema = schema
            try:
                yield
            finally:
                applications.clear()

    app = FastAPI(title="Capicove Models", lifespan=lifespan)
    authenticate(app, api_key)

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"status": "ok", "models": list(applications), "upstream_checked": False}

    @app.get("/v1/models")
    async def models_route() -> dict[str, object]:
        return {"object": "list", "data": catalog(selected)}

    @app.get("/v1/models/{name:path}")
    async def model_route(name: str) -> Any:
        return next(
            (item for item in catalog(selected) if item["id"] == _aliases(selected).get(name)),
            error(404, "model_not_found"),
        )

    app.mount("/", ModelDispatch(selected, applications))
    return app
