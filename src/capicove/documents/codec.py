"""Closed JSON codec for immutable document parse results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from capicove.contracts import (
    DocumentBlock,
    DocumentBlockType,
    DocumentOutlineEntry,
    DocumentPage,
    DocumentPageStatus,
    DocumentParseResult,
    DocumentParserReceipt,
    NormalizedBox,
)


def document_parse_result_to_json(result: DocumentParseResult) -> dict[str, object]:
    """Encode one parse result into a deterministic JSON-compatible mapping."""
    return {
        "source_sha256": result.source_sha256,
        "representation_sha256": result.representation_sha256,
        "pages": [_page_to_json(page) for page in result.pages],
        "receipt": {
            "plugin_id": result.receipt.plugin_id,
            "plugin_version": result.receipt.plugin_version,
            "profile_id": result.receipt.profile_id,
            "profile_version": result.receipt.profile_version,
            "model_manifest_sha256": result.receipt.model_manifest_sha256,
            "configuration_sha256": result.receipt.configuration_sha256,
            "processing_sha256": result.receipt.processing_sha256,
        },
        "outline": [_outline_to_json(item) for item in result.outline],
        "warnings": list(result.warnings),
    }


def document_parse_result_from_json(value: object) -> DocumentParseResult:
    """Decode one closed parse-result mapping and reject unknown fields."""
    root = _mapping(value, "document parse result")
    _fields(
        root,
        {
            "source_sha256",
            "representation_sha256",
            "pages",
            "receipt",
            "outline",
            "warnings",
        },
    )
    receipt = _mapping(root["receipt"], "receipt")
    _fields(
        receipt,
        {
            "plugin_id",
            "plugin_version",
            "profile_id",
            "profile_version",
            "model_manifest_sha256",
            "configuration_sha256",
            "processing_sha256",
        },
    )
    return DocumentParseResult(
        source_sha256=_string(root["source_sha256"], "source_sha256"),
        representation_sha256=_string(root["representation_sha256"], "representation_sha256"),
        pages=tuple(_page_from_json(item) for item in _list(root["pages"], "pages")),
        receipt=DocumentParserReceipt(
            _string(receipt["plugin_id"], "plugin_id"),
            _string(receipt["plugin_version"], "plugin_version"),
            _string(receipt["profile_id"], "profile_id"),
            _integer(receipt["profile_version"], "profile_version"),
            _string(receipt["model_manifest_sha256"], "model_manifest_sha256"),
            _string(receipt["configuration_sha256"], "configuration_sha256"),
            _string(receipt["processing_sha256"], "processing_sha256"),
        ),
        outline=tuple(_outline_from_json(item) for item in _list(root["outline"], "outline")),
        warnings=tuple(_string(item, "warning") for item in _list(root["warnings"], "warnings")),
    )


def _page_to_json(page: DocumentPage) -> dict[str, object]:
    return {
        "page_index": page.page_index,
        "printed_page_label": page.printed_page_label,
        "width": page.width,
        "height": page.height,
        "status": page.status.value,
        "blocks": [
            {
                "block_id": block.block_id,
                "page_index": block.page_index,
                "block_type": block.block_type.value,
                "exact_text": block.exact_text,
                "normalized_text": block.normalized_text,
                "box": [block.box.left, block.box.top, block.box.right, block.box.bottom],
                "reading_order": block.reading_order,
                "confidence": block.confidence,
            }
            for block in page.blocks
        ],
        "error": page.error,
    }


def _page_from_json(value: object) -> DocumentPage:
    page = _mapping(value, "page")
    _fields(
        page, {"page_index", "printed_page_label", "width", "height", "status", "blocks", "error"}
    )
    blocks: list[DocumentBlock] = []
    for raw in _list(page["blocks"], "blocks"):
        block = _mapping(raw, "block")
        _fields(
            block,
            {
                "block_id",
                "page_index",
                "block_type",
                "exact_text",
                "normalized_text",
                "box",
                "reading_order",
                "confidence",
            },
        )
        box = _list(block["box"], "box")
        if len(box) != 4:
            raise ValueError("box must contain four coordinates")
        blocks.append(
            DocumentBlock(
                _string(block["block_id"], "block_id"),
                _integer(block["page_index"], "page_index"),
                DocumentBlockType(_string(block["block_type"], "block_type")),
                _string(block["exact_text"], "exact_text", allow_empty=True),
                _string(block["normalized_text"], "normalized_text", allow_empty=True),
                NormalizedBox(*(_number(item, "box coordinate") for item in box)),
                _integer(block["reading_order"], "reading_order"),
                _number(block["confidence"], "confidence"),
            )
        )
    label = page["printed_page_label"]
    error = page["error"]
    return DocumentPage(
        _integer(page["page_index"], "page_index"),
        None if label is None else _string(label, "printed_page_label"),
        _integer(page["width"], "width"),
        _integer(page["height"], "height"),
        DocumentPageStatus(_string(page["status"], "status")),
        tuple(blocks),
        None if error is None else _string(error, "error"),
    )


def _outline_to_json(entry: DocumentOutlineEntry) -> dict[str, object]:
    return {
        "title": entry.title,
        "level": entry.level,
        "page_index": entry.page_index,
        "children": [_outline_to_json(child) for child in entry.children],
    }


def _outline_from_json(value: object) -> DocumentOutlineEntry:
    entry = _mapping(value, "outline entry")
    _fields(entry, {"title", "level", "page_index", "children"})
    return DocumentOutlineEntry(
        _string(entry["title"], "title"),
        _integer(entry["level"], "level"),
        _integer(entry["page_index"], "page_index"),
        tuple(_outline_from_json(child) for child in _list(entry["children"], "children")),
    )


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object")
    return cast(dict[str, object], dict(value))


def _fields(value: Mapping[str, object], expected: set[str]) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown:
        raise ValueError(f"unknown fields: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"missing fields: {', '.join(missing)}")


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _string(value: object, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{name} must be a string")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(value)


__all__ = ["document_parse_result_from_json", "document_parse_result_to_json"]
