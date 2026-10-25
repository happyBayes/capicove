"""OCR application composition, transport policy and image routes (RFC-0027/0028)."""

from __future__ import annotations

import hmac
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict
from starlette.exceptions import HTTPException
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse

from capicove.documents import (
    DocumentBackendUnavailableError,
    DocumentInputError,
    DocumentLimitError,
    DocumentLimits,
    DocumentParserBinding,
    DocumentProcessingError,
    DocumentRejectedError,
)
from capicove.processing.image import ImageOcr, ImageOcrResult
from capicove.processing.image.ocr import (
    OCR_MEDIA_TYPES,
    OcrProviderUnavailableError,
    validate_ocr_languages,
)
from capicove.processing.image.preparation import ImagePixelLimitError, ImagePreparationError
from capicove.runtime.ocr_service import (
    OcrProvider as OcrProvider,
)
from capicove.runtime.ocr_service import (
    OcrRuntime as OcrRuntime,
)
from capicove.runtime.ocr_service import (
    OcrServiceBusyError,
    OcrServiceClosedError,
)


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OcrHttpSettings:
    """Startup-owned ingress and processing limits, shared by CLI and app factory."""

    max_body_bytes: int = 10 * 1024 * 1024
    max_pixels: int = 20_000_000
    max_document_bytes: int = 50 * 1024 * 1024
    max_document_pages: int = 10
    max_document_page_points: int = 2000
    document_temp_directory: Path | None = None
    languages: tuple[str, ...] = ("chi_sim", "eng")

    def __post_init__(self) -> None:
        object.__setattr__(self, "languages", validate_ocr_languages(tuple(self.languages)))
        for name, maximum in (
            ("max_body_bytes", 100 * 1024 * 1024),
            ("max_pixels", 100_000_000),
            ("max_document_bytes", 100 * 1024 * 1024),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be within 1-{maximum}")
        DocumentLimits(self.max_document_pages, self.max_document_page_points)
        root = self.document_temp_directory
        if root is not None and (
            not root.is_absolute() or root.resolve() != root or not root.is_dir()
        ):
            raise ValueError(
                "document_temp_directory must be an existing absolute unlinked directory"
            )

    @property
    def document_limits(self) -> DocumentLimits:
        """Construct the capability's validated PDF budget without duplicating its rules."""
        return DocumentLimits(self.max_document_pages, self.max_document_page_points)


_DEFAULT_SETTINGS = OcrHttpSettings()


class ApiError(Exception):
    """A deliberate HTTP rejection with server-authored public text."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(code)
        self.status, self.code, self.message = status, code, message


class ErrorBody(BaseModel):
    """A stable failure code and server-authored public message."""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """The shared HTTP error envelope."""

    error: ErrorBody


class OcrResponse(BaseModel):
    """The validated public image evidence envelope."""

    model_config = ConfigDict(extra="forbid")
    result: ImageOcrResult
    processing_fingerprint: str


def error_responses(*statuses: int) -> dict[int | str, dict[str, Any]]:
    """Declare the same error schema used by the application's exception handlers."""
    return {status: {"model": ErrorResponse} for status in statuses}


async def upload_chunks(request: Request, limit: int, *, empty_code: str) -> AsyncIterator[bytes]:
    """Bound streamed uploads regardless of Content-Length or transfer encoding."""
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise ApiError(413, "request_too_large", "Upload exceeds the byte limit.")
        yield chunk
    if not size:
        raise ApiError(400, empty_code, "An upload body is required.")


def _failure(request: Request, error: Exception) -> JSONResponse:
    if isinstance(error, ApiError):
        status, code, message = error.status, error.code, error.message
    elif isinstance(error, OcrServiceBusyError):
        status, code, message = 429, "server_busy", "Another image or document request is running."
    elif isinstance(error, OcrServiceClosedError):
        status, code, message = 503, "shutting_down", "The service is shutting down."
    elif isinstance(error, DocumentInputError):
        status = 413 if isinstance(error, DocumentLimitError) else 400
        messages = {
            "invalid_document": "Unable to read PDF content.",
            "encrypted_document": "Encrypted PDFs are not accepted.",
            "invalid_page_range": "Page range exceeds PDF bounds.",
            "too_many_pages": "Select fewer pages for this request.",
            "page_too_large": "PDF page dimensions exceed limits.",
        }
        code = error.code if error.code in messages else "invalid_document"
        message = messages[code]
    elif isinstance(error, DocumentRejectedError):
        status, code, message = 422, "document_rejected", "The configured parser rejected this PDF."
    elif isinstance(error, (DocumentBackendUnavailableError, OcrProviderUnavailableError)):
        status, code, message = (
            503,
            "backend_unavailable",
            "The configured model backend is unavailable.",
        )
    elif isinstance(error, DocumentProcessingError):
        status, code, message = 500, error.code, "Local document parsing failed."
    elif isinstance(error, ImagePixelLimitError):
        status, code, message = 413, "image_too_large", "Decoded image exceeds the pixel limit."
    elif isinstance(error, ImagePreparationError):
        status, code, message = 400, "invalid_image", "Unable to decode image content."
    elif isinstance(error, RequestValidationError):
        status, code, message = 400, "invalid_page_range", "Use valid integer page bounds."
    elif isinstance(error, ClientDisconnect):
        status, code, message = 400, "client_disconnected", "Upload was interrupted."
    elif isinstance(error, HTTPException):
        status, code, message = error.status_code, "http_error", "Route or method is unavailable."
    else:
        code = (
            "document_parse_failed"
            if request.url.path == "/v1/documents/parse"
            else "inference_failed"
        )
        status, message = 500, "Local inference failed."
    if status >= 500:
        cause = error.__cause__ or error
        # Log only server-generated identity and exception type, never payloads,
        # credentials, exception messages, local paths or traceback locals.
        _LOGGER.error(
            "model_request_failed request_id=%s code=%s error_type=%s",
            request.state.request_id,
            code,
            type(cause).__name__,
        )
    return JSONResponse(
        ErrorResponse(error=ErrorBody(code=code, message=message)).model_dump(),
        status_code=status,
        headers={"WWW-Authenticate": "Bearer"} if status == 401 else None,
    )


def _image_router(runtime: OcrRuntime, settings: OcrHttpSettings) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/v1/ocr",
        response_model=OcrResponse,
        responses=error_responses(400, 401, 413, 415, 429, 500, 503),
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    media: {"schema": {"type": "string", "format": "binary"}}
                    for media in OCR_MEDIA_TYPES
                },
            }
        },
    )
    async def recognize(request: Request) -> OcrResponse:
        media = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if media not in OCR_MEDIA_TYPES:
            raise ApiError(415, "unsupported_media_type", "Use image/png, jpeg, webp or bmp.")
        async with runtime.admit():
            data = bytearray()
            async for chunk in upload_chunks(
                request, settings.max_body_bytes, empty_code="invalid_image"
            ):
                data.extend(chunk)
            result = await runtime.run(
                ImageOcr(runtime.provider).recognize_upload,
                bytes(data),
                media_type=media,
                max_pixels=settings.max_pixels,
                languages=settings.languages,
            )
            return OcrResponse(result=result, processing_fingerprint=runtime.fingerprint)

    return router


def create_app(
    provider_factory: Callable[[], OcrProvider],
    *,
    api_key: str | None = None,
    languages: tuple[str, ...] = _DEFAULT_SETTINGS.languages,
    max_body_bytes: int = _DEFAULT_SETTINGS.max_body_bytes,
    max_pixels: int = _DEFAULT_SETTINGS.max_pixels,
    document_parser: DocumentParserBinding | None = None,
    max_document_bytes: int = _DEFAULT_SETTINGS.max_document_bytes,
    max_document_pages: int = _DEFAULT_SETTINGS.max_document_pages,
    max_document_page_points: int = _DEFAULT_SETTINGS.max_document_page_points,
    document_temp_directory: Path | None = None,
    settings: OcrHttpSettings | None = None,
) -> FastAPI:
    """Compose typed routes with one provider owner; retain legacy keyword arguments.

    ``settings``, when supplied, owns all limit and language configuration. An
    OcrRuntime supplied as the factory can be warmed before listener startup.
    """
    config = OcrHttpSettings(
        max_body_bytes,
        max_pixels,
        max_document_bytes,
        max_document_pages,
        max_document_page_points,
        document_temp_directory,
        languages,
    )
    if settings is not None:
        if config != _DEFAULT_SETTINGS:
            raise ValueError("settings cannot be combined with legacy configuration arguments")
        config = settings
    if api_key is not None and (
        not api_key or len(api_key) > 8192 or any(not 33 <= ord(c) <= 126 for c in api_key)
    ):
        raise ValueError("invalid OCR API key")
    runtime = (
        provider_factory
        if isinstance(provider_factory, OcrRuntime)
        else OcrRuntime(provider_factory)
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        try:
            await runtime.run(runtime.start)
            yield
        finally:
            await runtime.run(runtime.close)

    application = FastAPI(
        title="Capicove OCR and Document Parsing", version="1.0.0", lifespan=lifespan
    )
    for error_type in (
        ApiError,
        OcrServiceBusyError,
        OcrServiceClosedError,
        DocumentInputError,
        DocumentRejectedError,
        DocumentProcessingError,
        ImagePreparationError,
        OcrProviderUnavailableError,
        RequestValidationError,
        HTTPException,
        ClientDisconnect,
    ):
        application.add_exception_handler(error_type, _failure)

    @application.middleware("http")
    async def authenticate(request: Request, call_next: Any) -> Any:
        request.state.request_id = uuid4().hex
        supplied = request.headers.get("authorization", "").encode("utf-8")
        if (
            request.url.path != "/health"
            and api_key is not None
            and not hmac.compare_digest(supplied, ("Bearer " + api_key).encode("ascii"))
        ):
            response = _failure(request, ApiError(401, "invalid_api_key", "Invalid API key."))
        else:
            try:
                response = await call_next(request)
            except Exception as error:
                response = _failure(request, error)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @application.get("/health")
    async def health() -> dict[str, str | None]:
        return {
            "status": "ok",
            "document_parser": None
            if document_parser is None
            else document_parser.backend.parser_id,
            "provider_id": runtime.provider.provider_id,
            "provider_version": runtime.provider.provider_version,
            "processing_fingerprint": runtime.fingerprint,
        }

    application.include_router(_image_router(runtime, config))
    if document_parser is not None:
        from capicove.api.documents import document_router

        application.include_router(document_router(document_parser, runtime, config))
    return application
