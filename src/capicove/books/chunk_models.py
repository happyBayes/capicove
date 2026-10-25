"""Immutable chunk projections over exact parser evidence."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from capicove.books.chunk_config import (
    CONTENT_TYPES,
    ChunkingConfig,
    integer_limit,
    persisted_chunk_config,
)
from capicove.books.chunk_front_matter import TOC_PARENT_PREFIX
from capicove.contracts import DocumentBlock, DocumentParseResult


def digest(value: object) -> str:
    """Hash canonical JSON rather than ambiguous delimiter concatenation."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def chunk_run_id(
    source_sha256: str,
    representation_sha256: str,
    structure_sha256: str,
    config: ChunkingConfig,
    tokenizer_fingerprint: str,
) -> str:
    """Bind versioned policy to evidence identity for creation and replay checks."""
    return (
        "chunk-run:"
        + digest(
            {
                "algorithm": "book-hybrid-v1",
                "preset_version": 1,
                "source": source_sha256,
                "representation": representation_sha256,
                "structure": structure_sha256,
                "config": config.chunk_values(),
                "tokenizer": tokenizer_fingerprint,
            }
        )[:24]
    )


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """Locate exact Unicode code points in an immutable page block."""

    block_id: str
    page_index: int
    start_char: int
    end_char: int
    text_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, str) or not self.block_id.strip():
            raise ValueError("span block ID must be nonblank")
        integer_limit("page_index", self.page_index, minimum=0, maximum=10**7)
        integer_limit("start_char", self.start_char, minimum=0, maximum=10**9)
        integer_limit("end_char", self.end_char, minimum=self.start_char + 1, maximum=10**9)
        if (
            not isinstance(self.text_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.text_sha256) is None
        ):
            raise ValueError("span text hash must be a SHA-256 digest")

    @classmethod
    def from_block(cls, block: DocumentBlock, start: int = 0, end: int | None = None) -> SourceSpan:
        stop = len(block.exact_text) if end is None else end
        if not 0 <= start < stop <= len(block.exact_text):
            raise ValueError("span exceeds source block bounds")
        return cls(
            block.block_id,
            block.page_index,
            start,
            stop,
            hashlib.sha256(block.exact_text[start:stop].encode()).hexdigest(),
        )

    def extract(self, block: DocumentBlock) -> str:
        """Validate page, bounds and exact slice hash before exposing text."""
        if (
            block.block_id != self.block_id
            or block.page_index != self.page_index
            or self.end_char > len(block.exact_text)
        ):
            raise ValueError("span does not resolve to its source block")
        value = block.exact_text[self.start_char : self.end_char]
        if hashlib.sha256(value.encode()).hexdigest() != self.text_sha256:
            raise ValueError("span source hash mismatch")
        return value

    def text(self, result: DocumentParseResult) -> str:
        for page in result.pages:
            if page.page_index == self.page_index:
                for block in page.blocks:
                    if block.block_id == self.block_id:
                        return self.extract(block)
        raise ValueError("span source block is unavailable")


def spans_text(spans: tuple[SourceSpan, ...], blocks: dict[str, DocumentBlock]) -> str:
    """Join adjacent same-block spans without introducing source characters."""
    parts: list[str] = []
    previous: SourceSpan | None = None
    for span in spans:
        if previous is not None and not (
            previous.block_id == span.block_id and previous.end_char == span.start_char
        ):
            parts.append("\n")
        parts.append(span.extract(blocks[span.block_id]))
        previous = span
    return "".join(parts)


def render_chunk(headings: tuple[str, ...], core: str, before: str = "", after: str = "") -> str:
    """Define the exact rendered payload measured by max_chunk_tokens."""
    return "\n\n".join(part for part in (" > ".join(headings), before, core, after) if part)


@dataclass(frozen=True, slots=True)
class BookChunk:
    """Own disjoint core evidence and separate, source-mapped context."""

    chunk_id: str
    parent_id: str
    heading_path: tuple[str, ...]
    core_spans: tuple[SourceSpan, ...]
    before_spans: tuple[SourceSpan, ...]
    after_spans: tuple[SourceSpan, ...]
    content_types: tuple[str, ...]
    core_tokens: int
    rendered_tokens: int
    warnings: tuple[str, ...] = ()

    @property
    def content_role(self) -> str:
        """Distinguish retained navigation from knowledge-extraction content."""
        return "navigation" if self.parent_id.startswith(TOC_PARENT_PREFIX) else "content"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.chunk_id, str)
            or re.fullmatch(r"chunk:[0-9a-f]{24}", self.chunk_id) is None
        ):
            raise ValueError("invalid chunk ID")
        if not isinstance(self.parent_id, str) or not self.parent_id.strip():
            raise ValueError("chunk parent ID must be nonblank")
        for values in (self.heading_path, self.content_types, self.warnings):
            if not isinstance(values, tuple) or any(
                not isinstance(item, str) or not item.strip() for item in values
            ):
                raise ValueError("chunk labels must be tuples of nonblank strings")
        if not self.content_types or any(
            kind not in (*CONTENT_TYPES, "text") for kind in self.content_types
        ):
            raise ValueError("invalid chunk content type")
        spans = (*self.core_spans, *self.before_spans, *self.after_spans)
        if (
            not self.core_spans
            or len(spans) > 128
            or any(not isinstance(span, SourceSpan) for span in spans)
        ):
            raise ValueError("chunk requires core evidence and at most 128 spans")
        integer_limit("core_tokens", self.core_tokens, minimum=0, maximum=131072)
        integer_limit("rendered_tokens", self.rendered_tokens, minimum=0, maximum=32768)


@dataclass(frozen=True, slots=True)
class ChunkRun:
    """Identify a complete deterministic chunk projection, independent of summaries."""

    run_id: str
    source_sha256: str
    representation_sha256: str
    structure_sha256: str
    tokenizer_fingerprint: str
    token_count_mode: str
    config_json: str
    chunks: tuple[BookChunk, ...]
    exclusions: tuple[tuple[str, str], ...]
    coverage_valid: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.run_id, str)
            or re.fullmatch(r"chunk-run:[0-9a-f]{24}", self.run_id) is None
        ):
            raise ValueError("invalid chunk run ID")
        for value in (self.source_sha256, self.representation_sha256, self.structure_sha256):
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("chunk run requires SHA-256 identities")
        if (
            self.token_count_mode not in ("estimated", "exact")
            or not isinstance(self.tokenizer_fingerprint, str)
            or not self.tokenizer_fingerprint.strip()
        ):
            raise ValueError("invalid tokenizer identity or count mode")
        if self.coverage_valid is not True:
            raise ValueError("only complete coverage may be persisted")
        if any(
            len(item) != 2 or not all(isinstance(value, str) and value for value in item)
            for item in self.exclusions
        ) or len({item[0] for item in self.exclusions}) != len(self.exclusions):
            raise ValueError("invalid chunk exclusions")
        config = persisted_chunk_config(json.loads(self.config_json))
        if len({chunk.chunk_id for chunk in self.chunks}) != len(self.chunks) or any(
            chunk.rendered_tokens > config.max_chunk_tokens for chunk in self.chunks
        ):
            raise ValueError("duplicate chunks or rendered token budget exceeded")
