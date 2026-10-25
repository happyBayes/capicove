"""Extract embedded PDF outline and printed page labels lazily."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from capicove.contracts import DocumentOutlineEntry
from capicove_processor_paddleocr.runtime import import_verified_runtime_module


def extract_pdf_metadata(
    path: Path,
) -> tuple[tuple[DocumentOutlineEntry, ...], dict[int, str]]:
    """Read deterministic structural metadata without importing pypdf at discovery."""
    pypdf = import_verified_runtime_module("pypdf")
    reader = pypdf.PdfReader(path)
    if getattr(reader, "is_encrypted", False):
        raise ValueError("encrypted PDFs are not supported")
    raw_outline = getattr(reader, "outline", ())
    outline = translate_outline(
        raw_outline,
        page_number=reader.get_destination_page_number,
    )
    # pypdf synthesizes physical page numbers when /PageLabels is absent.
    if "/PageLabels" not in reader.trailer["/Root"]:
        return outline, {}
    labels = {
        index: label
        for index, label in enumerate(getattr(reader, "page_labels", ()))
        if isinstance(label, str) and label.strip()
    }
    return outline, labels


def translate_outline(
    values: object,
    *,
    page_number: Callable[[object], int],
    level: int = 1,
) -> tuple[DocumentOutlineEntry, ...]:
    """Translate pypdf's destination/list representation into canonical entries."""
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return ()
    entries: list[DocumentOutlineEntry] = []
    index = 0
    while index < len(values):
        value = values[index]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            index += 1
            continue
        children: tuple[DocumentOutlineEntry, ...] = ()
        if index + 1 < len(values):
            nested = values[index + 1]
            if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
                children = translate_outline(nested, page_number=page_number, level=level + 1)
                index += 1
        title = _destination_title(value)
        try:
            page_index = page_number(value)
        except (AttributeError, KeyError, TypeError, ValueError):
            index += 1
            continue
        if title is not None and isinstance(page_index, int) and not isinstance(page_index, bool):
            entries.append(
                DocumentOutlineEntry(
                    title,
                    level,
                    page_index,
                    tuple(sorted(children, key=lambda child: child.page_index)),
                )
            )
        index += 1
    return tuple(sorted(entries, key=lambda entry: entry.page_index))


def _destination_title(value: object) -> str | None:
    title = getattr(value, "title", None)
    if title is None and isinstance(value, Mapping):
        title = value.get("/Title")
    if not isinstance(title, str) or not title.strip():
        return None
    return title


__all__ = ["extract_pdf_metadata", "translate_outline"]
