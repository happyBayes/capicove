"""Conservative, versioned front-matter projections over immutable source blocks."""

from __future__ import annotations

import re
from collections import defaultdict

from capicove.contracts import DocumentBlock, DocumentBlockType


FURNITURE = {DocumentBlockType.HEADER, DocumentBlockType.FOOTER, DocumentBlockType.PAGE_NUMBER}
TOC_PARENT_PREFIX = "front-matter:toc:"
_TOC = re.compile(r"^(?:目\s*录|table of contents|contents)(?:\s*\(continued\))?$", re.I)
_CHAPTER = re.compile(r"^(?:chapter\s+\w+|第[一二三四五六七八九十百\d]+[章节])", re.I)
_CONTENT_START = re.compile(
    r"^(?:致同学们|序言|前言|导读|绪论|引言|preface\b|foreword\b|introduction\b|"
    r"读者指南|阅读指南|使用说明|学习指导|how to use\b|using this (?:book|textbook)\b|"
    r"to (?:the )?(?:student|reader)s?\b|chapter\s+\w+|第[一二三四五六七八九十百\d]+[章节])",
    re.I,
)
_COPYRIGHT = re.compile(r"版权所有|copyright\s*(?:©|\d)|©\s*\d", re.I)
_RIGHTS = re.compile(r"未经许可|不得.{0,12}复制|all rights reserved", re.I)
_IMPRINT = re.compile(r"isbn\b|责任编辑|出版|published by|printed in", re.I)
_PUBLISHER = re.compile(r"出版社|published by|university press", re.I)
_EDITION = re.compile(r"教科书|教材|必修|第.{1,6}[册冊版]|\bedition\b|\btextbook\b", re.I)


def source_roles(
    blocks: dict[str, DocumentBlock], policy: str, *, page_indices: frozenset[int] | None = None
) -> tuple[dict[str, str], dict[str, str]]:
    """Return exact exclusion reasons and TOC parent IDs derived only from source.

    Unknown content is retained. Filtering needs the actual start of the book;
    a partial range starting later is never mistaken for a cover. The policy
    identifier freezes these heuristics for replay of immutable chunk runs.
    """
    if policy not in {"preserve", "navigation-v1"}:
        raise ValueError("unknown front-matter policy")
    excluded = {
        key: block.block_type.value if block.exact_text.strip() else "empty"
        for key, block in blocks.items()
        if block.block_type in FURNITURE or not block.exact_text.strip()
    }
    navigation: dict[str, str] = {}
    pages: dict[int, list[DocumentBlock]] = defaultdict(list)
    for page_index in page_indices or ():
        pages[page_index] = []
    for block in blocks.values():
        pages[block.page_index].append(block)
    if policy == "preserve" or not pages or min(pages) != 0:
        return excluded, navigation
    may_exclude = True
    toc_parent: str | None = None
    for page_index in range(16):
        if page_index not in pages:
            break
        page = sorted(pages[page_index], key=lambda block: block.reading_order)
        eligible = [block for block in page if block.block_id not in excluded]
        text = "\n".join(block.exact_text for block in eligible)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        entry_count = sum(bool(re.search(r"\S.*\s*[0-9]+\s*$", line)) for line in lines)
        toc_heading = any(_TOC.fullmatch(line) for line in lines[:2]) or any(
            block.block_type is DocumentBlockType.HEADER
            and _TOC.fullmatch(block.exact_text.strip())
            for block in page
        )
        content_heading = any(
            _CONTENT_START.match(line) and not re.search(r"[0-9]+\s*$", line) for line in lines[:2]
        ) or any(
            block.block_type is DocumentBlockType.DOCUMENT_TITLE
            and _CONTENT_START.match(block.exact_text.strip())
            for block in eligible
        )
        is_toc = (
            bool(lines)
            and (
                toc_heading
                or (
                    not content_heading
                    and toc_parent is not None
                    and entry_count >= 2
                    and entry_count >= len(lines) * 0.6
                )
            )
            and entry_count >= 2
        )
        if is_toc:
            toc_parent = toc_parent or f"{TOC_PARENT_PREFIX}{page_index}"
            navigation.update({block.block_id: toc_parent for block in eligible})
            may_exclude = False
            continue
        toc_parent = None
        if any(_CHAPTER.match(line) for line in lines[:2]):
            break
        if content_heading:
            may_exclude = False
        if not may_exclude:
            continue
        reason = None
        ambiguous_title = any(
            block.block_type is DocumentBlockType.DOCUMENT_TITLE
            and not re.fullmatch(
                r"copyright(?:\s+\d{4})?|版权(?:页|信息)?|版权所有",
                block.exact_text.strip(),
                re.I,
            )
            for block in eligible
        )
        if (
            not ambiguous_title
            and _COPYRIGHT.search(text)
            and _RIGHTS.search(text)
            and _IMPRINT.search(text)
        ):
            reason = "front_matter:copyright"
        elif (
            len(text) <= 500
            and _PUBLISHER.search(text)
            and _EDITION.search(text)
            and any(block.block_type is DocumentBlockType.DOCUMENT_TITLE for block in eligible)
            and not re.search(r"[。\uff01\uff1f!?]|\.(?:\s|$)", text)
        ):
            reason = "front_matter:cover"
        if reason:
            excluded.update({block.block_id: reason for block in eligible})
        elif len(text) > 500 or any(
            block.block_type
            in {DocumentBlockType.DOCUMENT_TITLE, DocumentBlockType.PARAGRAPH_TITLE}
            for block in eligible
        ):
            may_exclude = False
    return excluded, navigation
