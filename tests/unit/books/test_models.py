from __future__ import annotations

import pytest

from capicove.books import (
    BookSection,
    BookStructure,
    StructureConfidence,
    StructureMethod,
)


def _section(**overrides: object) -> BookSection:
    values: dict[str, object] = {
        "section_id": "section:one",
        "exact_title": "Chapter 1",
        "normalized_title": "Chapter 1",
        "level": 1,
        "parent_id": None,
        "children_ids": (),
        "start_page_index": 1,
        "end_page_index": 3,
        "printed_start_label": "1",
        "printed_end_label": "2",
        "heading_block_ids": ("p0001:b000001",),
        "method": StructureMethod.LAYOUT,
        "confidence": StructureConfidence.HIGH,
        "reason_codes": ("numbered_heading",),
        "warnings": (),
    }
    values.update(overrides)
    return BookSection(**values)  # type: ignore[arg-type]


def test_book_section_requires_a_nonempty_half_open_range() -> None:
    with pytest.raises(ValueError, match="end_page_index"):
        _section(end_page_index=1)


def test_book_structure_rejects_unknown_child() -> None:
    section = _section(children_ids=("section:missing",))

    with pytest.raises(ValueError, match="children"):
        BookStructure(
            "a" * 64,
            "b" * 64,
            1,
            (section,),
            (section.section_id,),
        )
