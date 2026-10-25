from __future__ import annotations

import pytest

from capicove.books import (
    BookSection,
    BookStructure,
    StructureConfidence,
    StructureMethod,
    StructureReviewSubmission,
    StructureReviewTask,
    apply_structure_reviews,
)
from capicove.contracts import (
    DocumentBlock,
    DocumentBlockType,
    DocumentPage,
    DocumentPageStatus,
    DocumentParseResult,
    DocumentParserReceipt,
    NormalizedBox,
)


def test_merge_cannot_retarget_an_unrelated_book_root() -> None:
    structure, result, task = _review_fixture()
    with pytest.raises(ValueError, match="local"):
        apply_structure_reviews(
            structure,
            result,
            (task,),
            (StructureReviewSubmission(task.task_id, "merge", target_section_id="section:other"),),
        )


def _review_fixture() -> tuple[
    BookStructure,
    DocumentParseResult,
    StructureReviewTask,
]:
    pages = []
    for page_index, title in enumerate(("Root", "Ambiguous", "Split Here", "Sibling", "Other")):
        block = DocumentBlock(
            f"p{page_index:04d}:b000000",
            page_index,
            DocumentBlockType.PARAGRAPH_TITLE,
            title,
            title,
            NormalizedBox(0.1, 0.1, 0.9, 0.2),
            0,
            0.9,
        )
        pages.append(
            DocumentPage(
                page_index,
                str(page_index + 1),
                100,
                200,
                DocumentPageStatus.SUCCEEDED,
                (block,),
            )
        )
    receipt = DocumentParserReceipt("plugin", "1", "profile", 1, "c" * 64, "d" * 64, "e" * 64)
    result = DocumentParseResult("a" * 64, "b" * 64, tuple(pages), receipt)
    child = BookSection(
        "section:child",
        "Ambiguous",
        "Ambiguous",
        2,
        "section:root",
        (),
        1,
        3,
        "2",
        "3",
        ("p0001:b000000",),
        StructureMethod.LAYOUT,
        StructureConfidence.LOW,
        ("layout_heading",),
        (),
    )
    sibling = BookSection(
        "section:sibling",
        "Sibling",
        "Sibling",
        2,
        "section:root",
        (),
        3,
        4,
        "4",
        "4",
        ("p0003:b000000",),
        StructureMethod.LAYOUT,
        StructureConfidence.HIGH,
        ("numbered_heading",),
        (),
    )
    root = BookSection(
        "section:root",
        "Root",
        "Root",
        1,
        None,
        (child.section_id, sibling.section_id),
        0,
        4,
        "1",
        "4",
        ("p0000:b000000",),
        StructureMethod.LAYOUT,
        StructureConfidence.HIGH,
        ("numbered_heading",),
        (),
    )
    other = BookSection(
        "section:other",
        "Other",
        "Other",
        1,
        None,
        (),
        4,
        5,
        "5",
        "5",
        ("p0004:b000000",),
        StructureMethod.LAYOUT,
        StructureConfidence.HIGH,
        ("numbered_heading",),
        (),
    )
    structure = BookStructure(
        "a" * 64,
        "b" * 64,
        1,
        (root, child, sibling, other),
        (root.section_id, other.section_id),
    )
    task = StructureReviewTask(
        "structure-review:" + "2" * 24,
        child.section_id,
        child.heading_block_ids,
        ("p0001:b000000", "p0002:b000000"),
        (root.section_id, other.section_id),
        ("accept", "reject", "merge", "split", "reparent"),
    )
    return structure, result, task


def test_accept_review_promotes_only_the_bounded_low_confidence_section() -> None:
    block = DocumentBlock(
        "p0000:b000000",
        0,
        DocumentBlockType.PARAGRAPH_TITLE,
        "Background",
        "Background",
        NormalizedBox(0.1, 0.1, 0.9, 0.2),
        0,
        0.9,
    )
    page = DocumentPage(0, "1", 100, 200, DocumentPageStatus.SUCCEEDED, (block,))
    receipt = DocumentParserReceipt("plugin", "1", "profile", 1, "c" * 64, "d" * 64, "e" * 64)
    result = DocumentParseResult("a" * 64, "b" * 64, (page,), receipt)
    section = BookSection(
        "section:one",
        "Background",
        "Background",
        1,
        None,
        (),
        0,
        1,
        "1",
        "1",
        (block.block_id,),
        StructureMethod.LAYOUT,
        StructureConfidence.LOW,
        ("layout_heading",),
        (),
    )
    structure = BookStructure("a" * 64, "b" * 64, 1, (section,), (section.section_id,))
    task = StructureReviewTask(
        "structure-review:" + "1" * 24,
        section.section_id,
        (block.block_id,),
        (block.block_id,),
        (),
        ("accept", "reject", "merge", "split", "reparent"),
    )

    reviewed = apply_structure_reviews(
        structure,
        result,
        (task,),
        (StructureReviewSubmission(task.task_id, "accept"),),
    )

    assert reviewed.sections[0].confidence is StructureConfidence.HIGH
    assert reviewed.sections[0].method is StructureMethod.AGENT_REVIEW
    assert reviewed.sections[0].heading_block_ids == (block.block_id,)


def test_reject_review_removes_section_and_rebuilds_parent_children() -> None:
    structure, result, task = _review_fixture()

    reviewed = apply_structure_reviews(
        structure,
        result,
        (task,),
        (StructureReviewSubmission(task.task_id, "reject"),),
    )

    assert "section:child" not in {item.section_id for item in reviewed.sections}
    assert reviewed.sections[0].children_ids == ("section:sibling",)


def test_merge_review_combines_heading_evidence_and_ranges() -> None:
    structure, result, task = _review_fixture()

    reviewed = apply_structure_reviews(
        structure,
        result,
        (task,),
        (
            StructureReviewSubmission(
                task.task_id,
                "merge",
                target_section_id="section:sibling",
            ),
        ),
    )

    merged = next(item for item in reviewed.sections if item.section_id == "section:sibling")
    assert (merged.start_page_index, merged.end_page_index) == (1, 4)
    assert merged.heading_block_ids == ("p0003:b000000", "p0001:b000000")


def test_reparent_review_uses_only_allowed_noncyclic_parent() -> None:
    structure, result, task = _review_fixture()

    reviewed = apply_structure_reviews(
        structure,
        result,
        (task,),
        (
            StructureReviewSubmission(
                task.task_id,
                "reparent",
                target_section_id="section:other",
            ),
        ),
    )

    child = next(item for item in reviewed.sections if item.section_id == "section:child")
    assert child.parent_id == "section:other"
    assert child.method is StructureMethod.AGENT_REVIEW


def test_split_review_creates_stable_section_from_local_heading_block() -> None:
    structure, result, task = _review_fixture()

    reviewed = apply_structure_reviews(
        structure,
        result,
        (task,),
        (
            StructureReviewSubmission(
                task.task_id,
                "split",
                selected_block_ids=("p0002:b000000",),
            ),
        ),
    )

    split = next(item for item in reviewed.sections if item.exact_title == "Split Here")
    original = next(item for item in reviewed.sections if item.section_id == "section:child")
    assert (original.start_page_index, original.end_page_index) == (1, 2)
    assert (split.start_page_index, split.end_page_index) == (2, 3)
    assert split.parent_id == "section:root"
