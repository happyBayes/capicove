from __future__ import annotations

from dataclasses import replace

import pytest

from capicove.books import (
    AtomicClaim,
    BookSection,
    BookStructure,
    StructureConfidence,
    StructureMethod,
    SummarySubmission,
    create_summary_tasks,
    validate_summary_submission,
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


def _section(
    identifier: str,
    level: int,
    start: int,
    end: int,
    *,
    parent: str | None = None,
    children: tuple[str, ...] = (),
) -> BookSection:
    return BookSection(
        identifier,
        identifier,
        identifier,
        level,
        parent,
        children,
        start,
        end,
        str(start),
        str(end - 1),
        (f"p{start:04d}:b000000",),
        StructureMethod.LAYOUT,
        StructureConfidence.HIGH,
        ("numbered_heading",),
        (),
    )


def _evidence() -> tuple[BookStructure, DocumentParseResult]:
    child = _section("section:child", 2, 1, 3, parent="section:root")
    root = _section("section:root", 1, 0, 3, children=(child.section_id,))
    structure = BookStructure("a" * 64, "b" * 64, 1, (root, child), (root.section_id,))
    pages = []
    for page in range(3):
        block = DocumentBlock(
            f"p{page:04d}:b000000",
            page,
            DocumentBlockType.TEXT,
            f"Evidence {page}",
            f"Evidence {page}",
            NormalizedBox(0.1, 0.1, 0.9, 0.2),
            0,
            0.99,
        )
        pages.append(
            DocumentPage(
                page,
                str(page),
                100,
                200,
                DocumentPageStatus.SUCCEEDED,
                (block,),
            )
        )
    receipt = DocumentParserReceipt("plugin", "1", "profile", 1, "c" * 64, "d" * 64, "e" * 64)
    return structure, DocumentParseResult("a" * 64, "b" * 64, tuple(pages), receipt)


def test_summary_tasks_are_bottom_up_bounded_and_deterministic() -> None:
    structure, evidence = _evidence()

    first = create_summary_tasks(structure, evidence)
    second = create_summary_tasks(structure, evidence)

    assert first == second
    assert [task.section_id for task in first] == ["section:child", "section:root"]
    assert first[1].dependency_task_ids == (first[0].task_id,)
    assert first[0].allowed_citation_ids == (
        "p0001:b000000",
        "p0002:b000000",
    )
    assert first[1].allowed_citation_ids == ("p0000:b000000",)
    assert all(len(task.evidence) <= 128 for task in first)


def test_submission_rejects_stale_dependency_incomplete_and_uncited_claims() -> None:
    structure, evidence = _evidence()
    child, root = create_summary_tasks(structure, evidence)
    valid = SummarySubmission(
        child.task_id,
        child.idempotency_key,
        "Child abstract",
        ("Point",),
        (AtomicClaim("claim:one", "A factual claim", ("p0001:b000000",)),),
    )

    validate_summary_submission(child, valid, completed_task_ids=frozenset())

    with pytest.raises(ValueError, match="idempotency"):
        validate_summary_submission(
            child,
            SummarySubmission(
                child.task_id,
                "f" * 64,
                valid.abstract,
                valid.key_points,
                valid.claims,
            ),
            completed_task_ids=frozenset(),
        )
    with pytest.raises(ValueError, match="dependencies"):
        validate_summary_submission(root, valid, completed_task_ids=frozenset())
    with pytest.raises(ValueError, match="citation"):
        validate_summary_submission(
            child,
            SummarySubmission(
                child.task_id,
                child.idempotency_key,
                "Abstract",
                ("Point",),
                (AtomicClaim("claim:bad", "Unsupported", ("p0000:b000000",)),),
            ),
            completed_task_ids=frozenset(),
        )


def test_factual_claim_requires_at_least_one_citation() -> None:
    with pytest.raises(ValueError, match="citation"):
        AtomicClaim("claim:uncited", "Fact", ())


def test_evidence_overflow_is_not_silently_truncated() -> None:
    structure, evidence = _evidence()
    page = evidence.pages[1]
    blocks = tuple(
        replace(page.blocks[0], block_id=f"p0001:b{index:06d}", reading_order=index)
        for index in range(141)
    )
    evidence = replace(
        evidence, pages=(evidence.pages[0], replace(page, blocks=blocks), evidence.pages[2])
    )

    with pytest.raises(ValueError, match=r"128.*split"):
        create_summary_tasks(structure, evidence)


def test_parent_heading_evidence_survives_child_start_on_same_page() -> None:
    structure, evidence = _evidence()
    root, child = structure.sections
    child = replace(child, start_page_index=0, heading_block_ids=("p0000:b000001",))
    structure = replace(structure, sections=(root, child))
    page = evidence.pages[0]
    child_heading = replace(page.blocks[0], block_id="p0000:b000001", reading_order=1)
    evidence = replace(
        evidence, pages=(replace(page, blocks=(*page.blocks, child_heading)), *evidence.pages[1:])
    )

    child_task, root_task = create_summary_tasks(structure, evidence)

    assert root_task.allowed_citation_ids == ("p0000:b000000",)
    assert child_task.allowed_citation_ids == ("p0000:b000001", "p0001:b000000", "p0002:b000000")


def test_parent_without_direct_evidence_can_submit_cited_child_synthesis() -> None:
    structure, evidence = _evidence()
    root, child = structure.sections
    root = replace(root, heading_block_ids=(), method=StructureMethod.OUTLINE)
    child = replace(child, start_page_index=0, heading_block_ids=("p0000:b000000",))
    structure = replace(structure, sections=(root, child))

    child_task, root_task = create_summary_tasks(structure, evidence)
    submission = SummarySubmission(
        root_task.task_id,
        root_task.idempotency_key,
        "Synthesis of the completed child",
        ("Point",),
        (AtomicClaim("claim:synthesis", "Fact", ("p0001:b000000",)),),
    )

    validate_summary_submission(
        root_task, submission, completed_task_ids=frozenset({child_task.task_id})
    )
    assert root_task.allowed_citation_ids == ("p0000:b000000", "p0001:b000000", "p0002:b000000")
    assert root_task.dependency_task_ids == (child_task.task_id,)
