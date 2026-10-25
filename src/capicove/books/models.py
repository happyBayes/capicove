"""Canonical dependency-light book structure and review task values."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class StructureConfidence(StrEnum):
    """Classify deterministic chapter-structure confidence."""

    HIGH = "high"
    LOW = "low"
    UNRESOLVED = "unresolved"


class StructureMethod(StrEnum):
    """Identify the strongest evidence used to construct a section."""

    OUTLINE = "outline"
    TOC = "toc"
    LAYOUT = "layout"
    AGENT_REVIEW = "agent_review"


@dataclass(frozen=True, slots=True)
class BookSection:
    """Describe one ordered section and its immutable source evidence."""

    section_id: str
    exact_title: str
    normalized_title: str
    level: int
    parent_id: str | None
    children_ids: tuple[str, ...]
    start_page_index: int
    end_page_index: int
    printed_start_label: str | None
    printed_end_label: str | None
    heading_block_ids: tuple[str, ...]
    method: StructureMethod
    confidence: StructureConfidence
    reason_codes: tuple[str, ...]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if re.fullmatch(r"section:[0-9a-f]{24}|section:[a-z0-9._-]+", self.section_id) is None:
            raise ValueError("section_id must be canonical")
        if not self.exact_title.strip() or not self.normalized_title.strip():
            raise ValueError("section titles must not be blank")
        if isinstance(self.level, bool) or not isinstance(self.level, int) or self.level <= 0:
            raise ValueError("level must be a positive integer")
        if self.parent_id is not None and not self.parent_id.strip():
            raise ValueError("parent_id must not be blank")
        if (
            isinstance(self.start_page_index, bool)
            or not isinstance(self.start_page_index, int)
            or self.start_page_index < 0
        ):
            raise ValueError("start_page_index must be non-negative")
        if (
            isinstance(self.end_page_index, bool)
            or not isinstance(self.end_page_index, int)
            or self.end_page_index <= self.start_page_index
        ):
            raise ValueError("end_page_index must exceed start_page_index")
        for label in (self.printed_start_label, self.printed_end_label):
            if label is not None and not label.strip():
                raise ValueError("printed page labels must not be blank")
        for name, values in (
            ("children_ids", self.children_ids),
            ("heading_block_ids", self.heading_block_ids),
            ("reason_codes", self.reason_codes),
            ("warnings", self.warnings),
        ):
            items = tuple(values)
            if len(set(items)) != len(items) or any(not item.strip() for item in items):
                raise ValueError(f"{name} must contain unique nonblank strings")
            object.__setattr__(self, name, items)
        if not isinstance(self.method, StructureMethod):
            raise TypeError("method must be a StructureMethod")
        if not isinstance(self.confidence, StructureConfidence):
            raise TypeError("confidence must be a StructureConfidence")


@dataclass(frozen=True, slots=True)
class BookStructure:
    """Preserve one deterministic, versioned flat projection of a section tree."""

    source_sha256: str
    representation_sha256: str
    profile_version: int
    sections: tuple[BookSection, ...]
    root_ids: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("source_sha256", "representation_sha256"):
            value = getattr(self, name)
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if (
            isinstance(self.profile_version, bool)
            or not isinstance(self.profile_version, int)
            or self.profile_version <= 0
        ):
            raise ValueError("profile_version must be positive")
        sections = tuple(self.sections)
        identifiers = tuple(section.section_id for section in sections)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("section IDs must be unique")
        known = frozenset(identifiers)
        roots = tuple(self.root_ids)
        if len(set(roots)) != len(roots) or any(item not in known for item in roots):
            raise ValueError("root_ids must identify unique known sections")
        if roots != tuple(section.section_id for section in sections if section.parent_id is None):
            raise ValueError("root_ids must match parentless section order")
        by_id = {section.section_id: section for section in sections}
        for section in sections:
            if section.parent_id is not None:
                if section.parent_id not in known:
                    raise ValueError("section parent must exist")
                parent = by_id[section.parent_id]
                if section.section_id not in parent.children_ids:
                    raise ValueError("parent children must include the section")
                if parent.level >= section.level:
                    raise ValueError("child level must be deeper than parent level")
            if any(child not in known for child in section.children_ids):
                raise ValueError("children must identify known sections")
            if any(by_id[child].parent_id != section.section_id for child in section.children_ids):
                raise ValueError("children must point back to their parent")
        warnings = tuple(self.warnings)
        if len(set(warnings)) != len(warnings) or any(not item.strip() for item in warnings):
            raise ValueError("warnings must contain unique nonblank strings")
        object.__setattr__(self, "sections", sections)
        object.__setattr__(self, "root_ids", roots)
        object.__setattr__(self, "warnings", warnings)


@dataclass(frozen=True, slots=True)
class StructureReviewTask:
    """Bound one low-confidence section review to local resolvable evidence."""

    task_id: str
    section_id: str
    candidate_heading_block_ids: tuple[str, ...]
    context_block_ids: tuple[str, ...]
    allowed_parent_ids: tuple[str, ...]
    permitted_operations: tuple[str, ...]

    def __post_init__(self) -> None:
        if re.fullmatch(r"structure-review:[0-9a-f]{24}", self.task_id) is None:
            raise ValueError("task_id must be canonical")
        if not self.section_id.strip():
            raise ValueError("section_id must not be blank")
        for name in (
            "candidate_heading_block_ids",
            "context_block_ids",
            "allowed_parent_ids",
            "permitted_operations",
        ):
            values = tuple(getattr(self, name))
            if len(set(values)) != len(values) or any(not item.strip() for item in values):
                raise ValueError(f"{name} must contain unique nonblank strings")
            object.__setattr__(self, name, values)
        if len(self.context_block_ids) > 32:
            raise ValueError("context_block_ids must be bounded")


__all__ = [
    "BookSection",
    "BookStructure",
    "StructureConfidence",
    "StructureMethod",
    "StructureReviewTask",
]
