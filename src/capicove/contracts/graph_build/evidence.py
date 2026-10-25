"""Immutable, source-mapped graph evidence values."""

from __future__ import annotations

import re
from dataclasses import dataclass

from capicove.contracts.retrieval import CharacterRange, TextChunkLocator


def _text(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be blank")


def _integer(name: str, value: int, minimum: int = 0) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} is below its minimum")


@dataclass(frozen=True, slots=True)
class GraphScope:
    """Identify an analysis scope; possession does not grant authorization."""

    workspace_ref: str
    scope_ref: str

    def __post_init__(self) -> None:
        _text("workspace_ref", self.workspace_ref)
        _text("scope_ref", self.scope_ref)


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """Locate exact representation text using half-open code-point offsets.

    Bounds against representation length and quote equality require an authorized
    reader and are checked when evidence is bound, not by this value alone.
    """

    evidence_ref: str
    representation_ref: str
    start_char: int
    end_char: int
    block_id: str | None = None
    page_index: int | None = None

    def __post_init__(self) -> None:
        _text("evidence_ref", self.evidence_ref)
        _text("representation_ref", self.representation_ref)
        CharacterRange(self.start_char, self.end_char)
        if (self.block_id is None) != (self.page_index is None):
            raise ValueError("block_id and page_index must be supplied together")
        if self.block_id is not None:
            _text("block_id", self.block_id)
        if self.page_index is not None:
            _integer("page_index", self.page_index)


@dataclass(frozen=True, slots=True)
class InputSegment:
    """Represent contiguous model input mapped to ordered source spans.

    Text is the concatenation of the source slices, without unmapped separators.
    Source readers must verify slice contents before model invocation.
    """

    segment_id: str
    role: str
    text: str
    source_spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        _text("segment_id", self.segment_id)
        if self.role not in ("core", "context", "title", "table_header"):
            raise ValueError("unsupported input segment role")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if not self.text:
            raise ValueError("segment text must not be empty")
        spans = tuple(self.source_spans)
        if not spans or any(not isinstance(span, SourceSpan) for span in spans):
            raise ValueError("source_spans must contain SourceSpan values")
        if sum(span.end_char - span.start_char for span in spans) != len(self.text):
            raise ValueError("source spans must cover the segment text")
        object.__setattr__(self, "source_spans", spans)


@dataclass(frozen=True, slots=True)
class TextUnit:
    """Own one core range and separately source-mapped contextual segments."""

    text_unit_id: str
    document_ref: str
    scope: GraphScope
    source_sha256: str
    representation_ref: str
    core_locator: TextChunkLocator
    segments: tuple[InputSegment, ...]
    section_path: tuple[str, ...]
    ordinal: int
    token_count: int
    chunking_fingerprint: str

    def __post_init__(self) -> None:
        for name in ("text_unit_id", "document_ref", "representation_ref"):
            _text(name, getattr(self, name))
        for name in ("source_sha256", "chunking_fingerprint"):
            value = getattr(self, name)
            _text(name, value)
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 hash")
        if not isinstance(self.scope, GraphScope):
            raise TypeError("scope must be a GraphScope")
        if not isinstance(self.core_locator, TextChunkLocator):
            raise TypeError("core_locator must be a TextChunkLocator")
        _integer("ordinal", self.ordinal)
        _integer("token_count", self.token_count, 1)
        segments = tuple(self.segments)
        if not segments or any(not isinstance(item, InputSegment) for item in segments):
            raise ValueError("segments must contain InputSegment values")
        if len({item.segment_id for item in segments}) != len(segments):
            raise ValueError("segment IDs must be unique")
        core = [span for item in segments if item.role == "core" for span in item.source_spans]
        cursor = self.core_locator.character_range.start_char
        for span in core:
            if span.representation_ref != self.representation_ref or span.start_char != cursor:
                raise ValueError("core spans must cover one contiguous representation range")
            cursor = span.end_char
        if not core or cursor != self.core_locator.character_range.end_char:
            raise ValueError("core spans must exactly cover the core locator")
        section_input: object = self.section_path
        if isinstance(section_input, str):
            raise TypeError("section_path must be a sequence of headings")
        sections = tuple(self.section_path)
        for section in sections:
            _text("section", section)
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "section_path", sections)
