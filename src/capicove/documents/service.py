"""Bounded PDF acceptance and parsing through public backend contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from capicove.contracts import (
    DocumentPageRange,
    DocumentParseRequest,
    DocumentParseResult,
    LocalPathSource,
)
from capicove.documents.ports import DocumentParserBackend


class DocumentInputError(ValueError):
    """Reject invalid PDF content or page selection with a stable reason code."""

    def __init__(self, code: str = "invalid_document") -> None:
        super().__init__(code)
        self.code = code


class DocumentLimitError(DocumentInputError):
    """Reject a document exceeding the configured processing budget."""


class DocumentRejectedError(ValueError):
    """The configured parser explicitly does not support this document."""


class DocumentProcessingError(RuntimeError):
    """Document processing failed for a reason unrelated to caller validation."""

    code = "document_parse_failed"


class DocumentBackendUnavailableError(DocumentProcessingError):
    """Installed assets or the selected parser runtime are unavailable."""

    code = "backend_unavailable"


class DocumentResultError(DocumentProcessingError):
    """The parser returned evidence inconsistent with the authorized request."""

    code = "invalid_parser_result"


@dataclass(frozen=True, slots=True)
class DocumentLimits:
    """Bound selected pages and both raw and physical PDF box side lengths."""

    max_pages: int = 10
    max_page_points: int = 2000

    def __post_init__(self) -> None:
        for name, maximum in (("max_pages", 1000), ("max_page_points", 14400)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be within 1-{maximum}")


@dataclass(frozen=True, slots=True)
class DocumentParserBinding:
    """Bind a public parser to an immutable startup-selected profile."""

    backend: DocumentParserBackend
    profile_id: str
    profile_version: int
    language_hints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.backend, DocumentParserBackend):
            raise TypeError("a public document parser backend is required")
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("a nonblank document profile is required")
        if type(self.profile_version) is not int or self.profile_version < 1:
            raise ValueError("a positive document profile version is required")
        hints = tuple(self.language_hints)
        if (
            len(hints) > 16
            or any(
                not isinstance(hint, str) or not hint.strip() or len(hint) > 64 for hint in hints
            )
            or len(set(hints)) != len(hints)
        ):
            raise ValueError("language hints must be unique bounded strings")
        object.__setattr__(self, "language_hints", hints)


class DocumentParsingService:
    """Inspect a private immutable upload before admitting it to native parsing."""

    def __init__(self, binding: DocumentParserBinding, limits: DocumentLimits) -> None:
        self.binding = binding
        self.limits = limits

    def parse(
        self, source: Path, source_sha256: str, *, start: int = 0, end: int | None = None
    ) -> DocumentParseResult:
        """Return evidence for exactly the requested source, pages and profile."""
        selected = self._inspect(source, start, end)
        request = DocumentParseRequest(
            source=LocalPathSource(str(source)),
            source_sha256=source_sha256,
            media_type="application/pdf",
            profile_id=self.binding.profile_id,
            profile_version=self.binding.profile_version,
            language_hints=self.binding.language_hints,
            page_range=selected,
        )
        try:
            result = self.binding.backend.parse(request)
        except (DocumentRejectedError, DocumentProcessingError):
            raise
        except Exception as error:
            raise DocumentProcessingError("document backend failed") from error
        if (
            not isinstance(result, DocumentParseResult)
            or result.source_sha256 != source_sha256
            or tuple(page.page_index for page in result.pages)
            != tuple(range(selected.start_page_index, selected.end_page_index))
            or result.receipt.profile_id != self.binding.profile_id
            or result.receipt.profile_version != self.binding.profile_version
        ):
            raise DocumentResultError("parser result does not match the request")
        return result

    def _inspect(self, source: Path, start: int, end: int | None) -> DocumentPageRange:
        try:
            from pypdf import PdfReader
            from pypdf.errors import PyPdfError
        except ImportError as error:
            raise DocumentBackendUnavailableError("install a compatible PDF reader") from error
        if (
            type(start) is not int
            or start < 0
            or (end is not None and (type(end) is not int or end <= start))
        ):
            raise DocumentInputError("invalid_page_range")
        try:
            with source.open("rb") as stream:
                reader = PdfReader(stream)
                if reader.is_encrypted:
                    raise DocumentInputError("encrypted_document")
                count = len(reader.pages)
                stop = count if end is None else end
                if not 0 <= start < stop <= count:
                    raise DocumentInputError("invalid_page_range")
                if stop - start > self.limits.max_pages:
                    raise DocumentLimitError("too_many_pages")
                for index in range(start, stop):
                    page = reader.pages[index]
                    unit = float(page.get("/UserUnit", 1))
                    raw = [
                        float(value)
                        for box in (page.mediabox, page.cropbox)
                        for value in (box.width, box.height)
                    ]
                    # PDFium may ignore UserUnit. Both representations must fit.
                    sizes = raw + [value * unit for value in raw]
                    if (
                        not math.isfinite(unit)
                        or unit <= 0
                        or any(not math.isfinite(value) or value <= 0 for value in sizes)
                    ):
                        raise DocumentInputError()
                    if any(value > self.limits.max_page_points for value in sizes):
                        raise DocumentLimitError("page_too_large")
                return DocumentPageRange(start, stop)
        except DocumentInputError:
            raise
        except OSError as error:
            raise DocumentProcessingError("unable to read staged PDF") from error
        except (
            PyPdfError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            OverflowError,
            RecursionError,
        ) as error:
            raise DocumentInputError() from error
