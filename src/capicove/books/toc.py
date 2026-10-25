"""Conservative printed-TOC matching over immutable parser text."""

from __future__ import annotations

import re
from dataclasses import dataclass

from capicove.contracts import DocumentParseResult


_CHAPTER = re.compile(
    r"^(?:[IVX]+\s+)?(?:chapter\s+(\d+)\b|第([零〇一二三四五六七八九十百\d]+)章)",
    re.IGNORECASE,
)
_MARKERS = frozenset({"contents", "tableofcontents", "目录"})


@dataclass(frozen=True, slots=True)
class TocEntry:
    """Retain a printed title, hierarchy hint and printed numeric target."""

    title: str
    level: int
    page_label: str


def chapter_number(text: str) -> int | None:
    """Read explicit English or Chinese chapter numbering only."""
    match = _CHAPTER.match(text.strip())
    if match is None:
        return None
    value = match.group(1) or match.group(2)
    if value.isdigit():
        return int(value)
    digits = dict(zip("零〇一二三四五六七八九", (0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9), strict=True))
    total, current = 0, 0
    for character in value:
        if character in {"十", "百"}:
            total += (current or 1) * (10 if character == "十" else 100)
            current = 0
        else:
            current = digits[character]
    return total + current


def heading_key(text: str) -> str:
    """Match title content even when OCR separates numbering or whitespace."""
    without_chapter = _CHAPTER.sub("", text.strip())
    without_number = re.sub(r"^\d+(?:\.\d+)*[.\uff0e、]?\s*", "", without_chapter)
    value = without_number.strip() or text
    return re.sub(r"\s+", "", value).casefold()


def read_printed_toc(result: DocumentParseResult) -> tuple[frozenset[int], tuple[TocEntry, ...]]:
    """Recognize TOCs inside title, content or image-classified text blocks."""
    pages: set[int] = set()
    entries: list[TocEntry] = []
    in_chapter = False
    for page in result.pages:
        lines = [line.strip() for block in page.blocks for line in block.exact_text.splitlines()]
        if not any(re.sub(r"\s+", "", line).casefold() in _MARKERS for line in lines):
            continue
        pages.add(page.page_index)
        for line in lines:
            match = re.fullmatch(r"(.*?[^\d\s])\s*(\d+)\s*", line)
            if match is None:
                match = re.fullmatch(r"(.+?)(?:\.{2,}|…+|\s+)([ivxlcdm]+)", line, re.IGNORECASE)
            if match is None:
                continue
            title = re.sub(r"[.\s…]+$", "", match.group(1))
            if not title or not any(character.isalpha() for character in title):
                continue
            if chapter_number(title) is not None:
                level, in_chapter = 1, True
            else:
                numbered = re.match(r"^(\d+(?:\.\d+)*)", title)
                level = len(numbered.group(1).split(".")) if numbered else 1
                if numbered and in_chapter:
                    level = max(2, level)
            entries.append(TocEntry(title, level, match.group(2)))
    return frozenset(pages), tuple(entries)


def match_toc(title: str, entries: tuple[TocEntry, ...]) -> TocEntry | None:
    """Accept only unambiguous normalized title matches."""
    matches = [entry for entry in entries if heading_key(entry.title) == heading_key(title)]
    return matches[0] if len(matches) == 1 else None
