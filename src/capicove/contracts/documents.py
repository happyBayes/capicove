"""Canonical contracts for immutable document parsing results."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from capicove.contracts._validation import (
    require_non_blank,
    require_non_negative_int,
    require_positive_int,
    require_sha256,
)
from capicove.contracts.retrieval import (
    ArtifactSource,
    InlineBytesSource,
    LocalPathSource,
    MediaSource,
    MountSource,
)


DOCUMENT_PARSER_CONTRACT_VERSION = 1
_MEDIA_SOURCE_TYPES = (
    ArtifactSource,
    InlineBytesSource,
    LocalPathSource,
    MountSource,
)


def _require_strict_int(field_name: str, value: object) -> int:
    """Require an integer without accepting booleans as integer values."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    return value


class DocumentBlockType(StrEnum):
    """Identify one parsed document layout block."""

    DOCUMENT_TITLE = "document_title"
    PARAGRAPH_TITLE = "paragraph_title"
    TEXT = "text"
    LIST = "list"
    TABLE = "table"
    IMAGE = "image"
    FORMULA = "formula"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    OTHER = "other"


class DocumentPageStatus(StrEnum):
    """Describe whether one requested PDF page parsed successfully."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class NormalizedBox:
    """Locate one block using normalized page coordinates."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        values = (self.left, self.top, self.right, self.bottom)
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
            raise ValueError("box coordinates must be finite numbers")
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("box coordinates must be within [0, 1]")
        if self.left >= self.right:
            raise ValueError("left must be less than right")
        if self.top >= self.bottom:
            raise ValueError("top must be less than bottom")


@dataclass(frozen=True, slots=True)
class DocumentPageRange:
    """Select a non-empty half-open range of zero-based PDF pages."""

    start_page_index: int
    end_page_index: int

    def __post_init__(self) -> None:
        _require_strict_int("start_page_index", self.start_page_index)
        _require_strict_int("end_page_index", self.end_page_index)
        require_non_negative_int("start_page_index", self.start_page_index)
        require_non_negative_int("end_page_index", self.end_page_index)
        if self.end_page_index <= self.start_page_index:
            raise ValueError("end_page_index must exceed start_page_index")


@dataclass(frozen=True, slots=True)
class DocumentBlock:
    """Preserve one immutable page-layout block and its source evidence."""

    block_id: str
    page_index: int
    block_type: DocumentBlockType
    exact_text: str
    normalized_text: str
    box: NormalizedBox
    reading_order: int
    confidence: float

    def __post_init__(self) -> None:
        require_non_blank("block_id", self.block_id)
        _require_strict_int("page_index", self.page_index)
        require_non_negative_int("page_index", self.page_index)
        if not isinstance(self.block_type, DocumentBlockType):
            raise TypeError("block_type must be a DocumentBlockType")
        if not isinstance(self.exact_text, str) or not isinstance(self.normalized_text, str):
            raise TypeError("block text must be a string")
        if not isinstance(self.box, NormalizedBox):
            raise TypeError("box must be a NormalizedBox")
        _require_strict_int("reading_order", self.reading_order)
        require_non_negative_int("reading_order", self.reading_order)
        if not isinstance(self.confidence, (int, float)) or not math.isfinite(self.confidence):
            raise ValueError("confidence must be finite")
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("confidence must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class DocumentPage:
    """Preserve one requested PDF page and its ordered layout blocks."""

    page_index: int
    printed_page_label: str | None
    width: int
    height: int
    status: DocumentPageStatus
    blocks: tuple[DocumentBlock, ...]
    error: str | None = None

    def __post_init__(self) -> None:
        _require_strict_int("page_index", self.page_index)
        require_non_negative_int("page_index", self.page_index)
        if self.printed_page_label is not None:
            require_non_blank("printed_page_label", self.printed_page_label)
        _require_strict_int("width", self.width)
        _require_strict_int("height", self.height)
        require_positive_int("width", self.width)
        require_positive_int("height", self.height)
        if not isinstance(self.status, DocumentPageStatus):
            raise TypeError("status must be a DocumentPageStatus")
        blocks = tuple(self.blocks)
        if any(not isinstance(block, DocumentBlock) for block in blocks):
            raise TypeError("blocks must contain DocumentBlock values")
        if any(block.page_index != self.page_index for block in blocks):
            raise ValueError("block page_index must match the containing page")
        block_ids = tuple(block.block_id for block in blocks)
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("block_id values must be unique within a page")
        orders = tuple(block.reading_order for block in blocks)
        if orders != tuple(sorted(orders)) or len(set(orders)) != len(orders):
            raise ValueError("block reading_order values must be unique and increasing")
        if self.status is DocumentPageStatus.SUCCEEDED and self.error is not None:
            raise ValueError("a succeeded page forbids an error")
        if self.status is DocumentPageStatus.FAILED and (blocks or not self.error):
            raise ValueError("a failed page requires an error and forbids blocks")
        object.__setattr__(self, "blocks", blocks)


@dataclass(frozen=True, slots=True)
class DocumentParserReceipt:
    """Record the parser, profile, models, configuration, and processing identity."""

    plugin_id: str
    plugin_version: str
    profile_id: str
    profile_version: int
    model_manifest_sha256: str
    configuration_sha256: str
    processing_sha256: str

    def __post_init__(self) -> None:
        for name in ("plugin_id", "plugin_version", "profile_id"):
            require_non_blank(name, getattr(self, name))
        _require_strict_int("profile_version", self.profile_version)
        require_positive_int("profile_version", self.profile_version)
        for name in (
            "model_manifest_sha256",
            "configuration_sha256",
            "processing_sha256",
        ):
            require_sha256(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class DocumentOutlineEntry:
    """Preserve one embedded PDF outline entry and its nested children."""

    title: str
    level: int
    page_index: int
    children: tuple[DocumentOutlineEntry, ...] = ()

    def __post_init__(self) -> None:
        require_non_blank("title", self.title)
        _require_strict_int("level", self.level)
        _require_strict_int("page_index", self.page_index)
        require_positive_int("level", self.level)
        require_non_negative_int("page_index", self.page_index)
        children = tuple(self.children)
        if any(not isinstance(child, DocumentOutlineEntry) for child in children):
            raise TypeError("children must contain DocumentOutlineEntry values")
        if any(child.level <= self.level for child in children):
            raise ValueError("outline children must be deeper than their parent")
        child_pages = tuple(child.page_index for child in children)
        if child_pages != tuple(sorted(child_pages)):
            raise ValueError("outline children must be ordered by page_index")
        object.__setattr__(self, "children", children)


@dataclass(frozen=True, slots=True)
class DocumentParseRequest:
    """Request a deterministic PDF representation from an authorized source."""

    source: MediaSource
    source_sha256: str
    media_type: str
    profile_id: str
    profile_version: int
    language_hints: tuple[str, ...] = ()
    page_range: DocumentPageRange | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, _MEDIA_SOURCE_TYPES):
            raise TypeError("source must be a supported MediaSource")
        require_sha256("source_sha256", self.source_sha256)
        if self.media_type != "application/pdf":
            raise ValueError("media_type must be application/pdf")
        require_non_blank("profile_id", self.profile_id)
        _require_strict_int("profile_version", self.profile_version)
        require_positive_int("profile_version", self.profile_version)
        hints = tuple(self.language_hints)
        if len(hints) > 16 or any(not item.strip() or len(item) > 64 for item in hints):
            raise ValueError("language_hints must contain at most 16 bounded values")
        if len(set(hints)) != len(hints):
            raise ValueError("language_hints must be unique")
        if self.page_range is not None and not isinstance(self.page_range, DocumentPageRange):
            raise TypeError("page_range must be a DocumentPageRange")
        object.__setattr__(self, "language_hints", hints)


@dataclass(frozen=True, slots=True)
class DocumentParseResult:
    """Return immutable page evidence and reproducibility lineage."""

    source_sha256: str
    representation_sha256: str
    pages: tuple[DocumentPage, ...]
    receipt: DocumentParserReceipt
    outline: tuple[DocumentOutlineEntry, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_sha256("source_sha256", self.source_sha256)
        require_sha256("representation_sha256", self.representation_sha256)
        pages = tuple(self.pages)
        if not pages or any(not isinstance(page, DocumentPage) for page in pages):
            raise ValueError("pages must contain at least one DocumentPage")
        indices = tuple(page.page_index for page in pages)
        if indices != tuple(sorted(indices)) or len(set(indices)) != len(indices):
            raise ValueError("page indices must be unique and strictly increasing")
        block_ids = tuple(block.block_id for page in pages for block in page.blocks)
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("block_id values must be unique within a parse result")
        if not isinstance(self.receipt, DocumentParserReceipt):
            raise TypeError("receipt must be a DocumentParserReceipt")
        outline = tuple(self.outline)
        if any(not isinstance(item, DocumentOutlineEntry) for item in outline):
            raise TypeError("outline must contain DocumentOutlineEntry values")
        outline_pages = tuple(item.page_index for item in _walk_outline(outline))
        if any(index not in set(indices) for index in outline_pages):
            raise ValueError("outline page_index must resolve to a parsed page")
        root_pages = tuple(item.page_index for item in outline)
        if root_pages != tuple(sorted(root_pages)):
            raise ValueError("outline roots must be ordered by page_index")
        warnings = tuple(self.warnings)
        if (
            len(warnings) > 64
            or any(
                not isinstance(item, str) or not item.strip() or len(item) > 500
                for item in warnings
            )
            or len(set(warnings)) != len(warnings)
        ):
            raise ValueError("warnings must contain unique bounded strings")
        object.__setattr__(self, "pages", pages)
        object.__setattr__(self, "outline", outline)
        object.__setattr__(self, "warnings", warnings)


def _walk_outline(
    entries: tuple[DocumentOutlineEntry, ...],
) -> tuple[DocumentOutlineEntry, ...]:
    values: list[DocumentOutlineEntry] = []
    for entry in entries:
        values.append(entry)
        values.extend(_walk_outline(entry.children))
    return tuple(values)


__all__ = [
    "DOCUMENT_PARSER_CONTRACT_VERSION",
    "DocumentBlock",
    "DocumentBlockType",
    "DocumentOutlineEntry",
    "DocumentPage",
    "DocumentPageRange",
    "DocumentPageStatus",
    "DocumentParseRequest",
    "DocumentParseResult",
    "DocumentParserReceipt",
    "NormalizedBox",
]
