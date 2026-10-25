"""Deterministic hybrid hierarchy construction for structured books."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from capicove.books.models import (
    BookSection,
    BookStructure,
    StructureConfidence,
    StructureMethod,
    StructureReviewTask,
)
from capicove.books.toc import chapter_number, match_toc, read_printed_toc
from capicove.contracts import (
    DocumentBlock,
    DocumentBlockType,
    DocumentOutlineEntry,
    DocumentParseResult,
)


_HEADING_TYPES = frozenset({DocumentBlockType.DOCUMENT_TITLE, DocumentBlockType.PARAGRAPH_TITLE})
_TOC_HEADINGS = frozenset({"contents", "table of contents", "目录", "目 录"})
_OPERATIONS = ("accept", "reject", "merge", "split", "reparent")


@dataclass(frozen=True, slots=True)
class _Candidate:
    exact_title: str
    normalized_title: str
    level: int
    page_index: int
    block_ids: tuple[str, ...]
    method: StructureMethod
    confidence: float
    strong_level: bool
    reasons: tuple[str, ...]


def build_book_structure(
    result: DocumentParseResult,
    *,
    profile_version: int = 1,
) -> BookStructure:
    """Build a stable tree using outline, printed TOC, then layout evidence."""
    if isinstance(profile_version, bool) or profile_version <= 0:
        raise ValueError("profile_version must be positive")
    if result.outline:
        candidates = _outline_candidates(result.outline, result)
        confidences = tuple(StructureConfidence.HIGH for _ in candidates)
    else:
        candidates = _layout_candidates(result)
        if _globally_consistent(candidates):
            confidences = tuple(StructureConfidence.HIGH for _ in candidates)
        else:
            confidences = tuple(
                StructureConfidence.HIGH
                if candidate.strong_level and candidate.confidence >= 0.8
                else StructureConfidence.LOW
                for candidate in candidates
            )
    if not candidates:
        return BookStructure(
            result.source_sha256,
            result.representation_sha256,
            profile_version,
            (),
            (),
            ("no_heading_candidates",),
        )
    maximum_page = max(page.page_index for page in result.pages) + 1
    sections = _sections(
        candidates,
        result,
        profile_version=profile_version,
        confidences=confidences,
        maximum_page=maximum_page,
    )
    return BookStructure(
        result.source_sha256,
        result.representation_sha256,
        profile_version,
        sections,
        tuple(section.section_id for section in sections if section.parent_id is None),
    )


def structure_review_tasks(
    structure: BookStructure,
    result: DocumentParseResult,
) -> tuple[StructureReviewTask, ...]:
    """Emit bounded tasks only for low-confidence or unresolved sections."""
    blocks = tuple(block for page in result.pages for block in page.blocks)
    tasks: list[StructureReviewTask] = []
    for position, section in enumerate(structure.sections):
        if section.confidence is StructureConfidence.HIGH:
            continue
        context = tuple(
            block.block_id
            for block in blocks
            if section.start_page_index - 1
            <= block.page_index
            <= min(section.end_page_index, section.start_page_index + 1)
        )[:32]
        allowed_parents = tuple(
            candidate.section_id
            for candidate in structure.sections[:position]
            if candidate.level < section.level
        )
        identity = "|".join(
            (
                structure.representation_sha256,
                str(structure.profile_version),
                section.section_id,
            )
        )
        tasks.append(
            StructureReviewTask(
                f"structure-review:{hashlib.sha256(identity.encode()).hexdigest()[:24]}",
                section.section_id,
                section.heading_block_ids,
                context,
                allowed_parents,
                _OPERATIONS,
            )
        )
    return tuple(tasks)


def _outline_candidates(
    entries: tuple[DocumentOutlineEntry, ...],
    result: DocumentParseResult,
) -> tuple[_Candidate, ...]:
    by_page = {page.page_index: page.blocks for page in result.pages}
    candidates: list[_Candidate] = []

    def append(values: tuple[DocumentOutlineEntry, ...]) -> None:
        for entry in values:
            normalized = _normalize(entry.title)
            matches = tuple(
                block.block_id
                for block in by_page.get(entry.page_index, ())
                if block.block_type in _HEADING_TYPES
                and _normalize(block.normalized_text) == normalized
            )
            candidates.append(
                _Candidate(
                    entry.title,
                    " ".join(entry.title.split()),
                    entry.level,
                    entry.page_index,
                    matches,
                    StructureMethod.OUTLINE,
                    1.0,
                    True,
                    ("embedded_outline",),
                )
            )
            append(entry.children)

    append(entries)
    return tuple(candidates)


def _layout_candidates(result: DocumentParseResult) -> tuple[_Candidate, ...]:
    toc_pages, toc_entries = read_printed_toc(result)
    candidates: list[_Candidate] = []
    current_chapter: int | None = None
    current_chapter_title: str | None = None
    for page in result.pages:
        if page.page_index in toc_pages:
            continue
        for block in page.blocks:
            if (
                block.block_type
                in {
                    DocumentBlockType.HEADER,
                    DocumentBlockType.FOOTER,
                    DocumentBlockType.PAGE_NUMBER,
                }
                or not block.exact_text.strip()
            ):
                continue
            exact_title = block.exact_text
            if block.block_type not in _HEADING_TYPES:
                lines = [line.strip() for line in exact_title.splitlines() if line.strip()]
                first = lines[0]
                if chapter_number(first) is None and match_toc(first, toc_entries) is None:
                    continue
                exact_title = first
            normalized_title = " ".join(exact_title.split())
            normalized_key = _normalize(normalized_title)
            if normalized_key in _TOC_HEADINGS:
                continue
            chapter = chapter_number(normalized_title)
            aligned_entry = match_toc(normalized_title, toc_entries)
            unmatched_chapter = bool(toc_entries) and chapter is not None and aligned_entry is None
            # With a TOC, unlisted local callouts stay inside their section.
            if toc_entries and aligned_entry is None and chapter is None:
                continue
            level, strong = _heading_level(block)
            conflict = False
            if chapter is not None:
                if chapter == current_chapter and normalized_key == current_chapter_title:
                    continue
                conflict = current_chapter is not None and chapter <= current_chapter
                if not unmatched_chapter:
                    current_chapter = chapter
                    current_chapter_title = normalized_key
                level, strong = 1, True
            else:
                numbered = re.match(r"^(\d+(?:\.\d+)*)\b", normalized_title)
                if numbered and current_chapter is not None:
                    parts = numbered.group(1).split(".")
                    level = max(2, len(parts))
                    conflict = len(parts) > 1 and int(parts[0]) != current_chapter
            aligned = aligned_entry is not None
            if aligned_entry is not None:
                level = aligned_entry.level
            method = StructureMethod.TOC if aligned else StructureMethod.LAYOUT
            reasons = ["layout_heading"]
            if aligned:
                reasons.append("toc_title_alignment")
            if strong:
                reasons.append("numbered_heading")
            if conflict:
                reasons.append("numbering_conflict")
            if unmatched_chapter:
                reasons.append("toc_heading_unmatched")
            candidates.append(
                _Candidate(
                    exact_title,
                    normalized_title,
                    level,
                    block.page_index,
                    (block.block_id,),
                    method,
                    block.confidence,
                    (strong or aligned) and not conflict and not unmatched_chapter,
                    tuple(reasons),
                )
            )
    return tuple(candidates)


def _heading_level(block: DocumentBlock) -> tuple[int, bool]:
    text = block.normalized_text.strip()
    if re.match(r"^chapter\s+\d+\b", text, flags=re.IGNORECASE) or re.match(
        r"^第[零〇一二三四五六七八九十百千万\d]+章", text
    ):
        return 1, True
    if re.match(r"^第[零〇一二三四五六七八九十百千万\d]+节", text):
        return 2, True
    numbered = re.match(r"^(\d+(?:\.\d+)*)\b", text)
    if numbered is not None:
        return len(numbered.group(1).split(".")), True
    if block.block_type is DocumentBlockType.DOCUMENT_TITLE:
        return 1, False
    return 2, False


def _globally_consistent(candidates: tuple[_Candidate, ...]) -> bool:
    if len(candidates) < 2 or not any(item.level == 1 for item in candidates):
        return False
    if any(not item.strong_level or item.confidence < 0.8 for item in candidates):
        return False
    previous_level = 1
    for candidate in candidates:
        if candidate.level > previous_level + 1:
            return False
        previous_level = candidate.level
    return True


def _sections(
    candidates: tuple[_Candidate, ...],
    result: DocumentParseResult,
    *,
    profile_version: int,
    confidences: tuple[StructureConfidence, ...],
    maximum_page: int,
) -> tuple[BookSection, ...]:
    identifiers = tuple(
        _section_id(candidate, result.representation_sha256, profile_version, position)
        for position, candidate in enumerate(candidates)
    )
    parents: list[str | None] = []
    stack: list[int] = []
    for position, candidate in enumerate(candidates):
        while stack and candidates[stack[-1]].level >= candidate.level:
            stack.pop()
        parents.append(identifiers[stack[-1]] if stack else None)
        stack.append(position)
    children: dict[str, list[str]] = {identifier: [] for identifier in identifiers}
    for identifier, parent in zip(identifiers, parents, strict=True):
        if parent is not None:
            children[parent].append(identifier)
    labels = {page.page_index: page.printed_page_label for page in result.pages}
    sections: list[BookSection] = []
    for position, candidate in enumerate(candidates):
        end = maximum_page
        for following in candidates[position + 1 :]:
            if following.level <= candidate.level:
                end = following.page_index
                break
        if end <= candidate.page_index:
            end = candidate.page_index + 1
        sections.append(
            BookSection(
                identifiers[position],
                candidate.exact_title,
                candidate.normalized_title,
                candidate.level,
                parents[position],
                tuple(children[identifiers[position]]),
                candidate.page_index,
                end,
                labels.get(candidate.page_index),
                labels.get(end - 1),
                candidate.block_ids,
                candidate.method,
                confidences[position],
                candidate.reasons,
                (),
            )
        )
    return tuple(sections)


def _section_id(
    candidate: _Candidate,
    representation_sha256: str,
    profile_version: int,
    position: int,
) -> str:
    identity = "|".join(
        (
            representation_sha256,
            str(profile_version),
            str(position),
            candidate.normalized_title,
            str(candidate.page_index),
            *candidate.block_ids,
        )
    )
    return f"section:{hashlib.sha256(identity.encode()).hexdigest()[:24]}"


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


__all__ = ["build_book_structure", "structure_review_tasks"]
