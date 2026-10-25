"""Deterministic content cues and safe fallback cut boundaries."""

from __future__ import annotations

import re
import unicodedata

from capicove.contracts import DocumentBlock, DocumentBlockType


def content_type(block: DocumentBlock) -> str:
    """Use only visible source cues, not inferred semantic certainty."""
    if block.block_type in (DocumentBlockType.TABLE, DocumentBlockType.FORMULA):
        return block.block_type.value
    text = block.exact_text.lstrip()
    if text.startswith("```") or re.match(
        r"(?:def |class |function |import |public |#include)", text
    ):
        return "code"
    for kind, pattern in (
        ("definition", r"(?:definition\b|定义|概念)"),
        ("principle", r"(?:theorem\b|principle\b|proposition\b|定理|定律|原理|命题)"),
        ("proof", r"(?:proof\b|证明)"),
        (
            "example",
            r"(?:example\b|solution\b|例题|例\s*\d|解[\uff1a:]|解析[\uff1a:]|答案[\uff1a:])",
        ),
        ("procedure", r"(?:step\s*\d|步骤|方法[\uff1a:])"),
        ("comparison", r"(?:comparison\b|比较|对比)"),
    ):
        if re.match(pattern, text, re.IGNORECASE):
            return kind
    return "text"


def continues(previous: str, following: str) -> bool:
    """Require a visible continuation cue before crossing a PDF page."""
    left, right = previous.rstrip(), following.lstrip()
    return bool(left and right) and (
        left[-1] not in ".。!?\uff01\uff1f;\uff1b}"
        or bool(re.match(r"(?:continued\b|where\b|续|其中|因此|所以)", right, re.IGNORECASE))
    )


def joins_protected_group(previous: str, following: str, kind: str, next_kind: str) -> bool:
    if kind in {"definition", "principle", "formula", "code"} and next_kind == "text":
        return bool(
            re.match(
                r"\s*(?:this means\b|where\b|here\b|the code\b|其中|式中|这意味着|上述代码)",
                following,
                re.IGNORECASE,
            )
        )
    if kind == "principle" and next_kind == "proof":
        return True
    if kind == "example" and next_kind == "example":
        return bool(
            re.match(
                r"\s*(?:solution\b|解[\uff1a:]|解析[\uff1a:]|答案[\uff1a:])",
                following,
                re.IGNORECASE,
            )
        )
    return (
        kind == next_kind
        and kind in {"code", "proof", "procedure"}
        and continues(previous, following)
    )


def cut_positions(text: str, kind: str) -> tuple[int, ...]:
    """Prefer line/paragraph/sentence boundaries; no tokenizer decoding is used."""
    if kind in {"table", "code"}:
        return tuple(match.end() for match in re.finditer(r"\n", text))
    paragraphs = tuple(match.end() for match in re.finditer(r"\n\s*\n", text))
    if paragraphs:
        return paragraphs
    return tuple(match.end() for match in re.finditer(r"[。\uff01\uff1f]|[.!?](?:\s+|$)|\n", text))


def safe_boundary(text: str, index: int) -> bool:
    """Keep combining marks and common joined emoji with their base when possible."""
    if index == len(text):
        return True
    return not (
        unicodedata.combining(text[index])
        or text[index] in "\u200d\ufe0f"
        or text[index - 1] == "\u200d"
        or "\U0001f3fb" <= text[index] <= "\U0001f3ff"
    )
