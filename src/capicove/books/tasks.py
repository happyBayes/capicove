"""Provider-neutral summary task graph and submission validation."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from capicove.books.models import BookSection, BookStructure
from capicove.contracts import DocumentBlockType, DocumentParseResult


SUMMARY_TASK_CONTRACT_VERSION = 1
_EXCLUDED_EVIDENCE_TYPES = frozenset(
    {
        DocumentBlockType.HEADER,
        DocumentBlockType.FOOTER,
        DocumentBlockType.PAGE_NUMBER,
    }
)


@dataclass(frozen=True, slots=True)
class SummaryEvidence:
    """Expose one bounded immutable source block to a caller-owned Agent."""

    citation_id: str
    page_index: int
    printed_page_label: str | None
    block_type: str
    exact_text: str
    exact_text_sha256: str

    def __post_init__(self) -> None:
        if not self.citation_id.strip():
            raise ValueError("citation_id must not be blank")
        if isinstance(self.page_index, bool) or self.page_index < 0:
            raise ValueError("page_index must be non-negative")
        if self.printed_page_label is not None and not self.printed_page_label.strip():
            raise ValueError("printed_page_label must not be blank")
        if not self.block_type.strip() or not isinstance(self.exact_text, str):
            raise ValueError("evidence type and text must be valid")
        if re.fullmatch(r"[0-9a-f]{64}", self.exact_text_sha256) is None:
            raise ValueError("exact_text_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class SummaryTask:
    """Request one section summary over bounded evidence and completed children."""

    task_id: str
    section_id: str
    objective: str
    evidence: tuple[SummaryEvidence, ...]
    allowed_citation_ids: tuple[str, ...]
    dependency_task_ids: tuple[str, ...]
    output_contract_version: int
    idempotency_key: str

    def __post_init__(self) -> None:
        if re.fullmatch(r"summary:[0-9a-f]{24}", self.task_id) is None:
            raise ValueError("task_id must be canonical")
        if not self.section_id.strip() or not self.objective.strip():
            raise ValueError("section_id and objective must not be blank")
        evidence = tuple(self.evidence)
        if len(evidence) > 128:
            raise ValueError("summary evidence must contain at most 128 blocks")
        citations = tuple(self.allowed_citation_ids)
        if citations != tuple(item.citation_id for item in evidence):
            raise ValueError("allowed citations must exactly match evidence order")
        dependencies = tuple(self.dependency_task_ids)
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("dependency task IDs must be unique")
        if self.output_contract_version != SUMMARY_TASK_CONTRACT_VERSION:
            raise ValueError("unsupported summary output contract version")
        if re.fullmatch(r"[0-9a-f]{64}", self.idempotency_key) is None:
            raise ValueError("idempotency_key must be a lowercase SHA-256 digest")
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "allowed_citation_ids", citations)
        object.__setattr__(self, "dependency_task_ids", dependencies)


@dataclass(frozen=True, slots=True)
class AtomicClaim:
    """Bind one factual statement to task-authorized source blocks."""

    claim_id: str
    text: str
    citation_ids: tuple[str, ...]
    quotation: str | None = None

    def __post_init__(self) -> None:
        if re.fullmatch(r"claim:[A-Za-z0-9._:-]+", self.claim_id) is None:
            raise ValueError("claim_id must be canonical")
        if not self.text.strip():
            raise ValueError("claim text must not be blank")
        citations = tuple(self.citation_ids)
        if not citations or len(set(citations)) != len(citations):
            raise ValueError("each factual claim requires unique citations")
        if any(not item.strip() for item in citations):
            raise ValueError("citation IDs must not be blank")
        if self.quotation is not None and not self.quotation.strip():
            raise ValueError("quotation must not be blank")
        object.__setattr__(self, "citation_ids", citations)


@dataclass(frozen=True, slots=True)
class SummarySubmission:
    """Carry one caller-generated structured summary for validation."""

    task_id: str
    idempotency_key: str
    abstract: str
    key_points: tuple[str, ...]
    claims: tuple[AtomicClaim, ...]
    provider: str | None = None
    model: str | None = None

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must not be blank")
        if re.fullmatch(r"[0-9a-f]{64}", self.idempotency_key) is None:
            raise ValueError("idempotency_key must be a lowercase SHA-256 digest")
        if not self.abstract.strip():
            raise ValueError("abstract must not be blank")
        key_points = tuple(self.key_points)
        claims = tuple(self.claims)
        if not key_points or any(not item.strip() for item in key_points):
            raise ValueError("key_points must contain nonblank values")
        if not claims or any(not isinstance(item, AtomicClaim) for item in claims):
            raise ValueError("claims must contain AtomicClaim values")
        claim_ids = tuple(item.claim_id for item in claims)
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("claim IDs must be unique within a submission")
        for value in (self.provider, self.model):
            if value is not None and not value.strip():
                raise ValueError("provider and model must not be blank")
        object.__setattr__(self, "key_points", key_points)
        object.__setattr__(self, "claims", claims)


def create_summary_tasks(
    structure: BookStructure,
    result: DocumentParseResult,
) -> tuple[SummaryTask, ...]:
    """Create a stable bottom-up task graph with direct section evidence."""
    sections = {section.section_id: section for section in structure.sections}
    task_ids = {
        section.section_id: _task_id(structure.representation_sha256, section.section_id)
        for section in structure.sections
    }
    pages = {page.page_index: page for page in result.pages}
    ranges = _section_block_ranges(structure, result)
    tasks_by_section: dict[str, SummaryTask] = {}

    def build(section: BookSection) -> None:
        for child_id in section.children_ids:
            build(sections[child_id])
        child_ranges = tuple(ranges[child_id] for child_id in section.children_ids)
        section_start, section_end = ranges[section.section_id]
        evidence: list[SummaryEvidence] = []
        for page_index in range(section.start_page_index, section.end_page_index):
            page = pages.get(page_index)
            if page is None:
                continue
            for block in page.blocks:
                position = (page_index, block.reading_order)
                if not section_start <= position < section_end:
                    continue
                if any(start <= position < end for start, end in child_ranges):
                    continue
                if block.block_type in _EXCLUDED_EVIDENCE_TYPES or not block.exact_text.strip():
                    continue
                evidence.append(
                    SummaryEvidence(
                        block.block_id,
                        page.page_index,
                        page.printed_page_label,
                        block.block_type.value,
                        block.exact_text,
                        hashlib.sha256(block.exact_text.encode("utf-8")).hexdigest(),
                    )
                )
        if not evidence and section.children_ids:
            # Outline-only parents still need source citations for child synthesis.
            evidence.extend(
                item
                for child_id in section.children_ids
                for item in tasks_by_section[child_id].evidence
            )
        if len(evidence) > 128:
            raise ValueError(
                f"section {section.section_id} exceeds 128 evidence blocks; "
                "split the section before creating summary tasks"
            )
        bounded = tuple(evidence)
        dependencies = tuple(task_ids[child_id] for child_id in section.children_ids)
        idempotency_payload = "|".join(
            (
                structure.representation_sha256,
                section.section_id,
                *dependencies,
                *(item.exact_text_sha256 for item in bounded),
            )
        )
        tasks_by_section[section.section_id] = SummaryTask(
            task_ids[section.section_id],
            section.section_id,
            f"Summarize section: {section.exact_title}",
            bounded,
            tuple(item.citation_id for item in bounded),
            dependencies,
            SUMMARY_TASK_CONTRACT_VERSION,
            hashlib.sha256(idempotency_payload.encode("utf-8")).hexdigest(),
        )

    for root_id in structure.root_ids:
        build(sections[root_id])
    ordered: list[SummaryTask] = []

    def append_postorder(section: BookSection) -> None:
        for child_id in section.children_ids:
            append_postorder(sections[child_id])
        ordered.append(tasks_by_section[section.section_id])

    for root_id in structure.root_ids:
        append_postorder(sections[root_id])
    return tuple(ordered)


def _section_block_ranges(
    structure: BookStructure,
    result: DocumentParseResult,
) -> dict[str, tuple[tuple[int, int], tuple[int, int]]]:
    """Locate section boundaries inside shared pages without dropping parent text."""
    blocks = {block.block_id: block for page in result.pages for block in page.blocks}
    starts = {
        section.section_id: min(
            (
                (blocks[key].page_index, blocks[key].reading_order)
                for key in section.heading_block_ids
                if key in blocks
            ),
            default=(section.start_page_index, 0),
        )
        for section in structure.sections
    }
    ranges = {}
    for index, section in enumerate(structure.sections):
        end = (section.end_page_index, 0)
        for following in structure.sections[index + 1 :]:
            if following.level <= section.level:
                end = min(end, starts[following.section_id])
                break
        ranges[section.section_id] = (starts[section.section_id], end)
    return ranges


def validate_summary_submission(
    task: SummaryTask,
    submission: SummarySubmission,
    *,
    completed_task_ids: frozenset[str],
) -> None:
    """Reject dependency-incomplete, stale, or citation-invalid submissions."""
    missing = tuple(
        dependency
        for dependency in task.dependency_task_ids
        if dependency not in completed_task_ids
    )
    if missing:
        raise ValueError("summary task dependencies are incomplete")
    if submission.task_id != task.task_id:
        raise ValueError("submission task_id does not match the task")
    if submission.idempotency_key != task.idempotency_key:
        raise ValueError("submission idempotency key is stale")
    allowed = frozenset(task.allowed_citation_ids)
    if any(
        citation not in allowed for claim in submission.claims for citation in claim.citation_ids
    ):
        raise ValueError("submission contains a citation outside the task allowance")


def _task_id(representation_sha256: str, section_id: str) -> str:
    value = hashlib.sha256(f"{representation_sha256}|{section_id}".encode()).hexdigest()
    return f"summary:{value[:24]}"


__all__ = [
    "SUMMARY_TASK_CONTRACT_VERSION",
    "AtomicClaim",
    "SummaryEvidence",
    "SummarySubmission",
    "SummaryTask",
    "create_summary_tasks",
    "validate_summary_submission",
]
