from __future__ import annotations

from capicove.books import (
    StructureConfidence,
    StructureMethod,
    build_book_structure,
    structure_review_tasks,
)
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


def _block(
    page: int,
    number: int,
    text: str,
    block_type: DocumentBlockType,
    *,
    confidence: float = 0.95,
) -> DocumentBlock:
    return DocumentBlock(
        f"p{page:04d}:b{number:06d}",
        page,
        block_type,
        text,
        " ".join(text.split()),
        NormalizedBox(0.1, 0.1 + number * 0.05, 0.9, 0.14 + number * 0.05),
        number,
        confidence,
    )


def _result(
    page_blocks: dict[int, tuple[DocumentBlock, ...]],
    *,
    outline: tuple[DocumentOutlineEntry, ...] = (),
) -> DocumentParseResult:
    pages = tuple(
        DocumentPage(
            page,
            str(page),
            1000,
            1400,
            DocumentPageStatus.SUCCEEDED,
            blocks,
        )
        for page, blocks in sorted(page_blocks.items())
    )
    receipt = DocumentParserReceipt(
        "capicove.processor.paddleocr",
        "0.1.0.dev0",
        "book-balanced",
        1,
        "c" * 64,
        "d" * 64,
        "e" * 64,
    )
    return DocumentParseResult("a" * 64, "b" * 64, pages, receipt, outline)


def test_embedded_outline_has_priority_and_passes_without_agent_review() -> None:
    result = _result(
        {
            1: (_block(1, 0, "Chapter 1", DocumentBlockType.DOCUMENT_TITLE),),
            2: (_block(2, 0, "1.1 Scope", DocumentBlockType.PARAGRAPH_TITLE),),
            3: (_block(3, 0, "Chapter 2", DocumentBlockType.DOCUMENT_TITLE),),
            4: (_block(4, 0, "Body", DocumentBlockType.TEXT),),
        },
        outline=(
            DocumentOutlineEntry(
                "Chapter 1",
                1,
                1,
                (DocumentOutlineEntry("1.1 Scope", 2, 2),),
            ),
            DocumentOutlineEntry("Chapter 2", 1, 3),
        ),
    )

    structure = build_book_structure(result)

    assert [section.method for section in structure.sections] == [
        StructureMethod.OUTLINE,
        StructureMethod.OUTLINE,
        StructureMethod.OUTLINE,
    ]
    assert all(section.confidence is StructureConfidence.HIGH for section in structure.sections)
    assert structure.sections[0].children_ids == (structure.sections[1].section_id,)
    assert [(item.start_page_index, item.end_page_index) for item in structure.sections] == [
        (1, 3),
        (2, 3),
        (3, 5),
    ]
    assert structure_review_tasks(structure, result) == ()


def test_numbered_layout_builds_stable_hierarchy_and_ignores_running_furniture() -> None:
    result = _result(
        {
            0: (
                _block(0, 0, "Running title", DocumentBlockType.HEADER),
                _block(0, 1, "Chapter 1 Foundations", DocumentBlockType.DOCUMENT_TITLE),
            ),
            1: (
                _block(1, 0, "Running title", DocumentBlockType.HEADER),
                _block(1, 1, "1.1 Concepts", DocumentBlockType.PARAGRAPH_TITLE),
            ),
            2: (_block(2, 0, "1.2 Practice", DocumentBlockType.PARAGRAPH_TITLE),),
            3: (_block(3, 0, "Chapter 2 Systems", DocumentBlockType.DOCUMENT_TITLE),),
        }
    )

    first = build_book_structure(result)
    second = build_book_structure(result)

    assert first == second
    assert [section.level for section in first.sections] == [1, 2, 2, 1]
    assert [section.normalized_title for section in first.sections] == [
        "Chapter 1 Foundations",
        "1.1 Concepts",
        "1.2 Practice",
        "Chapter 2 Systems",
    ]
    assert [(section.start_page_index, section.end_page_index) for section in first.sections] == [
        (0, 3),
        (1, 2),
        (2, 3),
        (3, 4),
    ]
    assert all(section.confidence is StructureConfidence.HIGH for section in first.sections)


def test_printed_toc_alignment_is_preferred_over_layout_only_evidence() -> None:
    result = _result(
        {
            0: (
                _block(0, 0, "Contents", DocumentBlockType.DOCUMENT_TITLE),
                _block(0, 1, "Chapter 1 Foundations .... 1", DocumentBlockType.LIST),
                _block(0, 2, "Chapter 2 Systems .... 3", DocumentBlockType.LIST),
            ),
            1: (_block(1, 0, "Chapter 1 Foundations", DocumentBlockType.DOCUMENT_TITLE),),
            2: (_block(2, 0, "Body", DocumentBlockType.TEXT),),
            3: (_block(3, 0, "Chapter 2 Systems", DocumentBlockType.DOCUMENT_TITLE),),
        }
    )

    structure = build_book_structure(result)

    assert [item.method for item in structure.sections] == [
        StructureMethod.TOC,
        StructureMethod.TOC,
    ]
    assert "toc_title_alignment" in structure.sections[0].reason_codes


def test_only_low_confidence_local_region_emits_bounded_review_task() -> None:
    result = _result(
        {
            0: (_block(0, 0, "Preface", DocumentBlockType.DOCUMENT_TITLE, confidence=0.7),),
            1: (_block(1, 0, "Body one", DocumentBlockType.TEXT),),
            2: (_block(2, 0, "Background", DocumentBlockType.PARAGRAPH_TITLE),),
            3: (_block(3, 0, "Body two", DocumentBlockType.TEXT),),
        }
    )

    structure = build_book_structure(result)
    tasks = structure_review_tasks(structure, result)

    assert all(section.confidence is StructureConfidence.LOW for section in structure.sections)
    assert len(tasks) == len(structure.sections)
    assert all(len(task.context_block_ids) <= 32 for task in tasks)
    assert {task.section_id for task in tasks} == {
        section.section_id for section in structure.sections
    }


def test_one_ambiguous_heading_does_not_send_the_whole_book_for_review() -> None:
    result = _result(
        {
            0: (_block(0, 0, "Chapter 1 Foundations", DocumentBlockType.DOCUMENT_TITLE),),
            1: (_block(1, 0, "1.1 Concepts", DocumentBlockType.PARAGRAPH_TITLE),),
            2: (_block(2, 0, "Interlude", DocumentBlockType.PARAGRAPH_TITLE),),
            3: (_block(3, 0, "Chapter 2 Systems", DocumentBlockType.DOCUMENT_TITLE),),
        }
    )

    structure = build_book_structure(result)
    tasks = structure_review_tasks(structure, result)

    assert [section.confidence for section in structure.sections] == [
        StructureConfidence.HIGH,
        StructureConfidence.HIGH,
        StructureConfidence.LOW,
        StructureConfidence.HIGH,
    ]
    assert [task.section_id for task in tasks] == [structure.sections[2].section_id]


def test_plain_section_numbers_reset_under_each_explicit_chapter() -> None:
    titles = ["第一章 运动", "1. 位置", "2. 时间", "第二章 力", "1. 重力"]
    result = _result(
        {
            index: (_block(index, 0, title, DocumentBlockType.PARAGRAPH_TITLE),)
            for index, title in enumerate(titles)
        }
    )

    structure = build_book_structure(result)

    assert [section.level for section in structure.sections] == [1, 2, 2, 1, 2]
    assert structure.sections[1].parent_id == structure.sections[0].section_id
    assert structure.sections[4].parent_id == structure.sections[3].section_id


def test_multiline_toc_aligns_image_chapter_and_unnumbered_body_heading() -> None:
    result = _result(
        {
            0: (_block(0, 0, "目录\n第一章 运动10\n1. 位置11\n2. 时间15", DocumentBlockType.TEXT),),
            14: (_block(14, 0, "I 第一章运动\nChapter introduction.", DocumentBlockType.IMAGE),),
            15: (
                _block(15, 0, "位置", DocumentBlockType.PARAGRAPH_TITLE),
                _block(15, 1, "问题", DocumentBlockType.PARAGRAPH_TITLE),
            ),
            19: (_block(19, 0, "时间", DocumentBlockType.PARAGRAPH_TITLE),),
        }
    )

    structure = build_book_structure(result)

    assert [s.level for s in structure.sections] == [1, 2, 2]
    assert [s.start_page_index for s in structure.sections] == [14, 15, 19]
    assert all(s.method is StructureMethod.TOC for s in structure.sections)
    assert structure.sections[0].heading_block_ids == ("p0014:b000000",)


def test_contradictory_section_numbering_is_not_high_confidence() -> None:
    result = _result(
        {
            0: (_block(0, 0, "Chapter 1", DocumentBlockType.DOCUMENT_TITLE),),
            1: (_block(1, 0, "2.1 Misnumbered", DocumentBlockType.PARAGRAPH_TITLE),),
        }
    )

    structure = build_book_structure(result)

    assert structure.sections[0].confidence is StructureConfidence.HIGH
    assert structure.sections[1].confidence is StructureConfidence.LOW
    assert len(structure_review_tasks(structure, result)) == 1


def test_chapter_absent_from_printed_toc_requires_local_review() -> None:
    result = _result(
        {
            0: (_block(0, 0, "Contents\nChapter 1 Foundations .... 1", DocumentBlockType.TEXT),),
            1: (_block(1, 0, "Chapter 1 Foundations", DocumentBlockType.DOCUMENT_TITLE),),
            2: (_block(2, 0, "Chapter 2 Callout", DocumentBlockType.DOCUMENT_TITLE),),
        }
    )
    structure = build_book_structure(result)
    assert [section.confidence for section in structure.sections] == [
        StructureConfidence.HIGH,
        StructureConfidence.LOW,
    ]


def test_unmatched_chapter_cannot_hide_later_toc_matched_chapter() -> None:
    titles = [
        "Chapter 1 Foundations",
        "Chapter 2 Callout",
        "1.1 Concepts",
        "Chapter 2 Systems",
        "Chapter 2 Systems",
    ]
    result = _result(
        {
            0: (
                _block(
                    0,
                    0,
                    "Contents\nChapter 1 Foundations .... 1\n1.1 Concepts .... 2\nChapter 2 Systems .... 3",
                    DocumentBlockType.TEXT,
                ),
            ),
            **{
                index: (_block(index, 0, title, DocumentBlockType.PARAGRAPH_TITLE),)
                for index, title in enumerate(titles, 1)
            },
        }
    )
    structure = build_book_structure(result)
    assert [section.normalized_title for section in structure.sections] == titles[:4]
    assert [section.confidence for section in structure.sections] == [
        StructureConfidence.HIGH,
        StructureConfidence.LOW,
        StructureConfidence.HIGH,
        StructureConfidence.HIGH,
    ]
