"""Typed PDF HTTP translation; document policy lives in capicove.documents."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Annotated, BinaryIO

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from capicove.api.ocr import ApiError, OcrHttpSettings, OcrRuntime, error_responses, upload_chunks
from capicove.contracts import (
    DocumentBlockType,
    DocumentOutlineEntry,
    DocumentPageStatus,
    DocumentParserReceipt,
)
from capicove.documents import (
    DocumentParserBinding,
    DocumentParsingService,
    document_parse_result_to_json,
)


# Preserve the existing import path while the binding belongs to the capability.
__all__ = ["DocumentParserBinding", "document_router"]


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DocumentBlockResponse(_WireModel):
    """One canonical block with normalized array coordinates."""

    block_id: str
    page_index: Annotated[int, Field(ge=0)]
    block_type: DocumentBlockType
    exact_text: str
    normalized_text: str
    box: tuple[
        Annotated[float, Field(ge=0, le=1)],
        Annotated[float, Field(ge=0, le=1)],
        Annotated[float, Field(ge=0, le=1)],
        Annotated[float, Field(ge=0, le=1)],
    ]
    reading_order: Annotated[int, Field(ge=0)]
    confidence: Annotated[float, Field(ge=0, le=1)]


class DocumentPageResponse(_WireModel):
    """Original PDF page identity, dimensions, status and ordered blocks."""

    page_index: Annotated[int, Field(ge=0)]
    printed_page_label: str | None
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]
    status: DocumentPageStatus
    blocks: tuple[DocumentBlockResponse, ...]
    error: str | None


class DocumentResultResponse(_WireModel):
    """The existing document codec shape with complete evidence lineage."""

    source_sha256: str
    representation_sha256: str
    pages: tuple[DocumentPageResponse, ...]
    receipt: DocumentParserReceipt
    outline: tuple[DocumentOutlineEntry, ...]
    warnings: tuple[str, ...]


class DocumentResponse(_WireModel):
    """Validated document result envelope."""

    result: DocumentResultResponse


class PageSelection(BaseModel):
    """Transport page selection before capability-level bounds inspection."""

    start: int = 0
    end: int | None = None


def _selection(
    request: Request,
    start_page_index: Annotated[int, Query(ge=0)] = 0,
    end_page_index: Annotated[int | None, Query(ge=1)] = None,
) -> PageSelection:
    seen: set[str] = set()
    for name, raw in request.query_params.multi_items():
        if (
            name not in {"start_page_index", "end_page_index"}
            or name in seen
            or not raw.isascii()
            or not raw.isdecimal()
            or len(raw) > 9
        ):
            raise ApiError(400, "invalid_page_range", "Use unique integer page bounds.")
        seen.add(name)
    if end_page_index is not None and end_page_index <= start_page_index:
        raise ApiError(400, "invalid_page_range", "Page range must be nonempty.")
    return PageSelection(start=start_page_index, end=end_page_index)


class _Upload:
    """Own private staging until processing has drained, including cancellation."""

    def __init__(self, parent: Path | None) -> None:
        self._parent = parent
        self._directory: tempfile.TemporaryDirectory[str] | None = None
        self._stream: BinaryIO | None = None
        self._digest = hashlib.sha256()
        self.path: Path

    def open(self) -> None:
        self._directory = tempfile.TemporaryDirectory(prefix="capicove-http-pdf-", dir=self._parent)
        self.path = Path(self._directory.name) / "source.pdf"
        self._stream = self.path.open("xb")

    def write(self, data: bytes) -> None:
        if self._stream is None:
            raise RuntimeError("upload is not open")
        self._stream.write(data)
        self._digest.update(data)

    def finish(self) -> str:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        return self._digest.hexdigest()

    def close(self) -> None:
        try:
            self.finish()
        finally:
            if self._directory is not None:
                self._directory.cleanup()
                self._directory = None


def document_router(
    binding: DocumentParserBinding, runtime: OcrRuntime, settings: OcrHttpSettings
) -> APIRouter:
    """Expose one parser without giving requests filesystem or model authority."""
    router = APIRouter()
    service = DocumentParsingService(binding, settings.document_limits)

    @router.post(
        "/v1/documents/parse",
        response_model=DocumentResponse,
        responses=error_responses(400, 401, 413, 415, 422, 429, 500, 503),
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {"application/pdf": {"schema": {"type": "string", "format": "binary"}}},
            }
        },
    )
    async def parse(
        request: Request, pages: Annotated[PageSelection, Depends(_selection)]
    ) -> DocumentResponse:
        if (
            request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            != "application/pdf"
        ):
            raise ApiError(415, "unsupported_media_type", "Use application/pdf.")
        async with runtime.admit():
            upload = _Upload(settings.document_temp_directory)
            try:
                try:
                    await runtime.run(upload.open)
                    async for chunk in upload_chunks(
                        request, settings.max_document_bytes, empty_code="invalid_document"
                    ):
                        await runtime.run(upload.write, chunk)
                    digest = await runtime.run(upload.finish)
                    result = await runtime.run(
                        service.parse, upload.path, digest, start=pages.start, end=pages.end
                    )
                    return DocumentResponse.model_validate(
                        {"result": document_parse_result_to_json(result)}
                    )
                finally:
                    await runtime.run(upload.close)
            except OSError as error:
                raise ApiError(
                    500, "document_upload_failed", "Unable to stage PDF upload."
                ) from error

    return router
