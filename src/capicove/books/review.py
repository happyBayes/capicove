"""Validate and apply bounded caller-owned structure review decisions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

from capicove.books.models import (
    BookSection,
    BookStructure,
    StructureConfidence,
    StructureMethod,
    StructureReviewTask,
)
from capicove.contracts import DocumentBlock, DocumentParseResult


_OPERATIONS = frozenset({"accept", "reject", "merge", "split", "reparent"})


@dataclass(frozen=True, slots=True)
class StructureReviewSubmission:
    """Carry one closed correction over a bounded structure review task."""

    task_id: str
    operation: str
    selected_block_ids: tuple[str, ...] = ()
    target_section_id: str | None = None

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must not be blank")
        if self.operation not in _OPERATIONS:
            raise ValueError("unknown structure review operation")
        selected = tuple(self.selected_block_ids)
        if len(set(selected)) != len(selected) or any(not item.strip() for item in selected):
            raise ValueError("selected_block_ids must contain unique nonblank IDs")
        if self.target_section_id is not None and not self.target_section_id.strip():
            raise ValueError("target_section_id must not be blank")
        if self.operation == "split" and not selected:
            raise ValueError("split requires selected source blocks")
        if self.operation in {"merge", "reparent"} and self.target_section_id is None:
            raise ValueError(f"{self.operation} requires target_section_id")
        if self.operation in {"accept", "reject"} and (
            selected or self.target_section_id is not None
        ):
            raise ValueError(f"{self.operation} forbids correction parameters")
        object.__setattr__(self, "selected_block_ids", selected)


def apply_structure_reviews(
    structure: BookStructure,
    result: DocumentParseResult,
    tasks: tuple[StructureReviewTask, ...],
    submissions: tuple[StructureReviewSubmission, ...],
    *,
    require_complete: bool = True,
) -> BookStructure:
    """Apply exactly one validated decision per low-confidence local task."""
    tasks_by_id = {task.task_id: task for task in tasks}
    decisions = {submission.task_id: submission for submission in submissions}
    if len(tasks_by_id) != len(tasks) or len(decisions) != len(submissions):
        raise ValueError("structure review task and submission IDs must be unique")
    if not set(decisions) <= set(tasks_by_id) or (
        require_complete and set(decisions) != set(tasks_by_id)
    ):
        raise ValueError("all and only current structure review tasks require decisions")
    sections = {section.section_id: section for section in structure.sections}
    order = [section.section_id for section in structure.sections]
    blocks = {block.block_id: block for page in result.pages for block in page.blocks}
    for task in tasks:
        if task.task_id not in decisions:
            continue
        decision = decisions[task.task_id]
        if decision.operation not in task.permitted_operations:
            raise ValueError("structure review operation is outside the task allowance")
        section = sections.get(task.section_id)
        if section is None:
            raise ValueError("structure review section no longer exists")
        if any(block_id not in task.context_block_ids for block_id in decision.selected_block_ids):
            raise ValueError("structure review selected a block outside local context")
        if decision.operation == "accept":
            sections[section.section_id] = _reviewed(section, "agent_accept")
        elif decision.operation == "reject":
            _remove_section(section.section_id, sections, order, replacement=section.parent_id)
        elif decision.operation == "merge":
            target_id = _target(decision)
            target = sections.get(target_id)
            if target is None or target_id == section.section_id:
                raise ValueError("merge target must be another current section")
            if not _local_merge_target(section, target, sections, order):
                raise ValueError("merge target is outside local parent or adjacent siblings")
            sections[target_id] = _reviewed(
                replace(
                    target,
                    start_page_index=min(target.start_page_index, section.start_page_index),
                    end_page_index=max(target.end_page_index, section.end_page_index),
                    heading_block_ids=tuple(
                        dict.fromkeys((*target.heading_block_ids, *section.heading_block_ids))
                    ),
                ),
                "agent_merge",
            )
            _remove_section(section.section_id, sections, order, replacement=target_id)
        elif decision.operation == "reparent":
            target_id = _target(decision)
            if target_id not in task.allowed_parent_ids or target_id not in sections:
                raise ValueError("reparent target is outside allowed parent positions")
            if _is_descendant(target_id, section.section_id, sections):
                raise ValueError("reparent would create a structure cycle")
            target = sections[target_id]
            delta = target.level + 1 - section.level
            for descendant_id in _subtree_ids(section.section_id, sections):
                current = sections[descendant_id]
                sections[descendant_id] = replace(current, level=current.level + delta)
            sections[section.section_id] = _reviewed(
                replace(sections[section.section_id], parent_id=target_id),
                "agent_reparent",
            )
        elif decision.operation == "split":
            selected = sorted(
                (blocks[block_id] for block_id in decision.selected_block_ids),
                key=lambda block: (block.page_index, block.reading_order, block.block_id),
            )
            _split_section(section, selected, sections, order, structure)
    rebuilt = _rebuild(
        tuple(sections[identifier] for identifier in order if identifier in sections)
    )
    if require_complete and any(
        section.confidence is not StructureConfidence.HIGH for section in rebuilt
    ):
        raise ValueError("structure review left unresolved sections")
    return BookStructure(
        structure.source_sha256,
        structure.representation_sha256,
        structure.profile_version,
        rebuilt,
        tuple(section.section_id for section in rebuilt if section.parent_id is None),
        structure.warnings,
    )


def _local_merge_target(
    section: BookSection,
    target: BookSection,
    sections: dict[str, BookSection],
    order: list[str],
) -> bool:
    """Permit the direct parent or an adjacent, touching sibling only."""
    if target.section_id == section.parent_id:
        return True
    if target.parent_id != section.parent_id:
        return False
    siblings = [
        identifier for identifier in order if sections[identifier].parent_id == section.parent_id
    ]
    if abs(siblings.index(target.section_id) - siblings.index(section.section_id)) != 1:
        return False
    return (
        target.start_page_index <= section.end_page_index
        and section.start_page_index <= target.end_page_index
    )


def _split_section(
    section: BookSection,
    selected: list[DocumentBlock],
    sections: dict[str, BookSection],
    order: list[str],
    structure: BookStructure,
) -> None:
    split_blocks = [
        block
        for block in selected
        if section.start_page_index < block.page_index < section.end_page_index
    ]
    if not split_blocks:
        raise ValueError("split blocks must occur inside the section after its heading")
    starts: list[DocumentBlock] = []
    seen_pages: set[int] = set()
    for block in split_blocks:
        if block.page_index not in seen_pages:
            starts.append(block)
            seen_pages.add(block.page_index)
    original_end = section.end_page_index
    sections[section.section_id] = _reviewed(
        replace(section, end_page_index=starts[0].page_index),
        "agent_split",
    )
    insertion = order.index(section.section_id) + 1
    for position, block in enumerate(starts):
        end = starts[position + 1].page_index if position + 1 < len(starts) else original_end
        identity = f"{structure.representation_sha256}|{section.section_id}|{block.block_id}"
        identifier = f"section:{hashlib.sha256(identity.encode()).hexdigest()[:24]}"
        sections[identifier] = BookSection(
            identifier,
            block.exact_text,
            block.normalized_text,
            section.level,
            section.parent_id,
            (),
            block.page_index,
            end,
            None,
            None,
            (block.block_id,),
            StructureMethod.AGENT_REVIEW,
            StructureConfidence.HIGH,
            ("agent_split",),
            (),
        )
        order.insert(insertion + position, identifier)


def _remove_section(
    identifier: str,
    sections: dict[str, BookSection],
    order: list[str],
    *,
    replacement: str | None,
) -> None:
    for child_id in sections[identifier].children_ids:
        child = sections[child_id]
        sections[child_id] = replace(child, parent_id=replacement)
    del sections[identifier]
    order.remove(identifier)


def _rebuild(values: tuple[BookSection, ...]) -> tuple[BookSection, ...]:
    known = frozenset(section.section_id for section in values)
    children: dict[str, list[str]] = {section.section_id: [] for section in values}
    normalized: list[BookSection] = []
    for section in values:
        parent = section.parent_id if section.parent_id in known else None
        normalized.append(replace(section, parent_id=parent, children_ids=()))
        if parent is not None:
            children[parent].append(section.section_id)
    return tuple(
        replace(section, children_ids=tuple(children[section.section_id])) for section in normalized
    )


def _reviewed(section: BookSection, reason: str) -> BookSection:
    return replace(
        section,
        method=StructureMethod.AGENT_REVIEW,
        confidence=StructureConfidence.HIGH,
        reason_codes=tuple(dict.fromkeys((*section.reason_codes, reason))),
    )


def _target(decision: StructureReviewSubmission) -> str:
    if decision.target_section_id is None:
        raise ValueError("structure review target is required")
    return decision.target_section_id


def _subtree_ids(identifier: str, sections: dict[str, BookSection]) -> tuple[str, ...]:
    values = [identifier]
    for child_id in sections[identifier].children_ids:
        values.extend(_subtree_ids(child_id, sections))
    return tuple(values)


def _is_descendant(
    candidate_id: str,
    ancestor_id: str,
    sections: dict[str, BookSection],
) -> bool:
    return candidate_id in _subtree_ids(ancestor_id, sections)


__all__ = ["StructureReviewSubmission", "apply_structure_reviews"]
