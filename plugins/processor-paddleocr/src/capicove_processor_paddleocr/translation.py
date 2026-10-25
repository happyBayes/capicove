"""Translate PP-StructureV3 result dictionaries into canonical evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

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


_LABELS = {
    "doc_title": DocumentBlockType.DOCUMENT_TITLE,
    "document_title": DocumentBlockType.DOCUMENT_TITLE,
    "paragraph_title": DocumentBlockType.PARAGRAPH_TITLE,
    "text": DocumentBlockType.TEXT,
    "content": DocumentBlockType.TEXT,
    "list": DocumentBlockType.LIST,
    "table": DocumentBlockType.TABLE,
    "image": DocumentBlockType.IMAGE,
    "figure": DocumentBlockType.IMAGE,
    "formula": DocumentBlockType.FORMULA,
    "header": DocumentBlockType.HEADER,
    "footer": DocumentBlockType.FOOTER,
    "number": DocumentBlockType.PAGE_NUMBER,
    "page_number": DocumentBlockType.PAGE_NUMBER,
}


@dataclass(frozen=True, slots=True)
class _OcrLine:
    """Bind one exact OCR line to its provider geometry and score."""

    text: str
    box: tuple[float, float, float, float]
    confidence: float


def translate_pages(
    raw_pages: Sequence[Mapping[str, object]],
    *,
    source_sha256: str,
    receipt: DocumentParserReceipt,
    outline: tuple[DocumentOutlineEntry, ...] = (),
    page_labels: Mapping[int, str] | None = None,
) -> DocumentParseResult:
    """Translate ordered JSON-compatible Paddle results."""
    labels = {} if page_labels is None else page_labels
    pages = tuple(
        _page(raw, fallback_index=index, page_labels=labels) for index, raw in enumerate(raw_pages)
    )
    payload = json.dumps(
        {
            "source_sha256": source_sha256,
            "receipt": {
                "plugin_id": receipt.plugin_id,
                "plugin_version": receipt.plugin_version,
                "profile_id": receipt.profile_id,
                "profile_version": receipt.profile_version,
                "model_manifest_sha256": receipt.model_manifest_sha256,
                "configuration_sha256": receipt.configuration_sha256,
                "processing_sha256": receipt.processing_sha256,
            },
            "pages": [_page_identity(page) for page in pages],
            "outline": [_outline_identity(item) for item in outline],
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return DocumentParseResult(
        source_sha256,
        hashlib.sha256(payload).hexdigest(),
        pages,
        receipt,
        outline,
    )


def _page(
    raw: Mapping[str, object],
    *,
    fallback_index: int,
    page_labels: Mapping[int, str],
) -> DocumentPage:
    value = raw.get("res", raw)
    if not isinstance(value, Mapping):
        raise ValueError("PP-StructureV3 page result must be an object")
    page_index = _integer(value.get("page_index", fallback_index), "page_index")
    width = _positive_int(value.get("width"), "width")
    height = _positive_int(value.get("height"), "height")
    parsed = value.get("parsing_res_list")
    if not isinstance(parsed, Sequence) or isinstance(parsed, (str, bytes)):
        raise ValueError("parsing_res_list must be an array")
    ocr_lines = _ocr_lines(value, width=width, height=height)
    assigned_lines, unassigned_lines = _assign_ocr_lines(parsed, ocr_lines)
    if any(line.text.strip() for line in unassigned_lines):
        raise ValueError("OCR lines are not covered by layout blocks")
    blocks: list[DocumentBlock] = []
    for position, item in enumerate(parsed):
        if not isinstance(item, Mapping):
            raise ValueError("parsed block must be an object")
        box = item.get("block_bbox")
        if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
            raise ValueError("block_bbox must contain four coordinates")
        block_number = _integer(item.get("block_id", position), "block_id")
        # The parsing list is ordered; native ranks omit furniture and may repeat.
        reading_order = position
        label = _text(item.get("block_label"), "block_label")
        exact = _text(item.get("block_content", ""), "block_content", allow_empty=True)
        lines = assigned_lines[position]
        if label == "table" and not exact.strip():
            exact = "\n".join(line.text for line in lines if line.text.strip())
        layout_confidence = _layout_confidence(value.get("layout_det_res"), box, label)
        confidence = _block_confidence(exact, layout_confidence, lines)
        left, top, right, bottom = (_number(part, "block_bbox") for part in box)
        blocks.append(
            DocumentBlock(
                f"p{page_index:04d}:b{block_number:06d}",
                page_index,
                _LABELS.get(label, DocumentBlockType.OTHER),
                exact,
                "\n".join(" ".join(line.split()) for line in exact.splitlines()).strip(),
                NormalizedBox(
                    max(0.0, left / width),
                    max(0.0, top / height),
                    min(1.0, right / width),
                    min(1.0, bottom / height),
                ),
                reading_order,
                confidence,
            )
        )
    if not any(block.exact_text.strip() for block in blocks):
        raise ValueError("OCR page contains no usable evidence")
    return DocumentPage(
        page_index,
        page_labels.get(page_index),
        width,
        height,
        DocumentPageStatus.SUCCEEDED,
        tuple(sorted(blocks, key=lambda block: block.reading_order)),
    )


def _ocr_lines(value: Mapping[str, object], *, width: int, height: int) -> tuple[_OcrLine, ...]:
    overall = value.get("overall_ocr_res")
    if overall is None:
        return ()
    if not isinstance(overall, Mapping):
        raise ValueError("overall_ocr_res must be an object")
    texts = overall.get("rec_texts")
    boxes = overall.get("rec_boxes")
    scores = overall.get("rec_scores")
    if texts is None and boxes is None and scores is None:
        return ()
    values = (texts, boxes, scores)
    if any(not isinstance(item, Sequence) or isinstance(item, (str, bytes)) for item in values):
        raise ValueError("OCR texts, boxes, and scores must be aligned arrays")
    if len(texts) != len(boxes) or len(texts) != len(scores):  # type: ignore[arg-type]
        raise ValueError("OCR texts, boxes, and scores must be aligned arrays")
    lines: list[_OcrLine] = []
    for text, box, score in zip(texts, boxes, scores, strict=True):  # type: ignore[arg-type]
        exact = _text(text, "rec_text", allow_empty=True)
        bounds = _ocr_box(box, width=width, height=height)
        confidence = _number(score, "rec_score")
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("rec_score must be within [0, 1]")
        lines.append(_OcrLine(exact, bounds, confidence))
    return tuple(lines)


def _ocr_box(value: object, *, width: int, height: int) -> tuple[float, float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError("rec_box must contain four coordinates")
    left, top, right, bottom = (_number(part, "rec_box") for part in value)
    if not 0.0 <= left < right <= width or not 0.0 <= top < bottom <= height:
        raise ValueError("rec_box must be ordered within the page")
    return left, top, right, bottom


def _assign_ocr_lines(
    parsed: Sequence[object], lines: tuple[_OcrLine, ...]
) -> tuple[tuple[tuple[_OcrLine, ...], ...], tuple[_OcrLine, ...]]:
    assignments: list[list[_OcrLine]] = [[] for _ in parsed]
    unassigned: list[_OcrLine] = []
    parsed_boxes: list[tuple[float, float, float, float, float, int]] = []
    for position, item in enumerate(parsed):
        if not isinstance(item, Mapping):
            raise ValueError("parsed block must be an object")
        box = item.get("block_bbox")
        if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
            raise ValueError("block_bbox must contain four coordinates")
        left, top, right, bottom = (_number(part, "block_bbox") for part in box)
        if left >= right or top >= bottom:
            raise ValueError("block_bbox coordinates must be ordered")
        parsed_boxes.append((left, top, right, bottom, (right - left) * (bottom - top), position))
    for line in lines:
        center_x = (line.box[0] + line.box[2]) / 2
        center_y = (line.box[1] + line.box[3]) / 2
        candidates = [
            (area, position)
            for left, top, right, bottom, area, position in parsed_boxes
            if left <= center_x <= right and top <= center_y <= bottom
        ]
        if candidates:
            _, position = min(candidates)
            assignments[position].append(line)
        else:
            unassigned.append(line)
    return tuple(tuple(items) for items in assignments), tuple(unassigned)


def _block_confidence(
    exact_text: str, layout_confidence: float, lines: tuple[_OcrLine, ...]
) -> float:
    if not exact_text.strip():
        return layout_confidence
    if not lines:
        return 0.0
    exact_identity = _evidence_text(exact_text)
    line_identity = " ".join(identity for line in lines if (identity := _evidence_text(line.text)))
    if not exact_identity or exact_identity != line_identity:
        return 0.0
    return min(layout_confidence, *(line.confidence for line in lines))


def _evidence_text(value: str) -> str:
    """Normalize Unicode whitespace without changing evidence characters."""
    return " ".join(value.split())


def _layout_confidence(value: object, coordinates: Sequence[object], label: str) -> float:
    """Never borrow the confidence of a differently ordered layout detection."""
    if not isinstance(value, Mapping):
        return 0.0
    boxes = value.get("boxes")
    if not isinstance(boxes, Sequence) or isinstance(boxes, (str, bytes)):
        return 0.0
    scores: list[float] = []
    for box in boxes:
        if not isinstance(box, Mapping) or box.get("label") != label:
            continue
        bounds = box.get("coordinate")
        if not isinstance(bounds, Sequence) or isinstance(bounds, (str, bytes)) or len(bounds) != 4:
            continue
        if all(
            abs(_number(left, "coordinate") - _number(right, "block_bbox")) <= 1
            for left, right in zip(bounds, coordinates, strict=True)
        ):
            scores.append(_number(box.get("score", 0.0), "score"))
    return min(scores) if scores else 0.0


def _page_identity(page: DocumentPage) -> dict[str, object]:
    return {
        "page_index": page.page_index,
        "printed_page_label": page.printed_page_label,
        "width": page.width,
        "height": page.height,
        "status": page.status.value,
        "error": page.error,
        "blocks": [
            {
                "id": block.block_id,
                "type": block.block_type.value,
                "text": block.exact_text,
                "box": [block.box.left, block.box.top, block.box.right, block.box.bottom],
                "order": block.reading_order,
                "confidence": block.confidence,
            }
            for block in page.blocks
        ],
    }


def _outline_identity(entry: DocumentOutlineEntry) -> dict[str, object]:
    return {
        "title": entry.title,
        "level": entry.level,
        "page_index": entry.page_index,
        "children": [_outline_identity(child) for child in entry.children],
    }


def _text(value: object, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{name} must be a string")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _positive_int(value: object, name: str) -> int:
    result = _integer(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


__all__ = ["translate_pages"]
