"""Deterministic core/context composition over authorized exact representations."""

import re
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass

from capicove.contracts.graph_build import InputSegment, SourceSpan, TextUnit
from capicove.contracts.retrieval import CharacterRange, TextChunkLocator, TextOrigin
from capicove.graph_build.codec import fingerprint as _fingerprint
from capicove.graph_build.ports import TextTokenizer
from capicove.graph_build.profiles import ChunkingProfile
from capicove.graph_build.representations import BlockSpan, TextRepresentation


_MAX_INPUT_CODEPOINTS = 1_000_000
_MAX_COUNT_CALLS = 50_000
_MAX_COUNT_CODEPOINTS = 8_000_000


@dataclass
class _BudgetedTokenizer:
    """Bound exhaustive counting before invoking the configured tokenizer."""

    tokenizer: TextTokenizer
    calls: int = 0
    codepoints: int = 0

    @property
    def fingerprint(self) -> str:
        return self.tokenizer.fingerprint

    def count(self, text: str) -> int:
        if self.calls >= _MAX_COUNT_CALLS or self.codepoints + len(text) > _MAX_COUNT_CODEPOINTS:
            raise ValueError("Chunking tokenizer work budget exceeded")
        self.calls += 1
        self.codepoints += len(text)
        return self.tokenizer.count(text)


def _digest(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected a lowercase SHA-256 fingerprint")


def occurrence_id(
    scope_ref: str,
    document_ref: str,
    representation_ref: str,
    start: int,
    end: int,
    fingerprint: str,
) -> str:
    """Bind an occurrence to source location and interpretation, never text alone."""
    for value in (scope_ref, document_ref, representation_ref):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Occurrence references must be nonblank text")
    CharacterRange(start, end)
    _digest(fingerprint)
    digest = _fingerprint(
        {
            "version": 1,
            "scope": scope_ref,
            "document": document_ref,
            "representation": representation_ref,
            "start": start,
            "end": end,
            "chunking": fingerprint,
        }
    )
    return f"text-unit:{digest}"


def _count(tokenizer: TextTokenizer, text: str) -> int:
    count = tokenizer.count(text)
    if type(count) is not int:
        raise TypeError("Tokenizer count must be an integer")
    if count < 0 or (text.strip() and count == 0):
        raise ValueError("Tokenizer returned an invalid count")
    return count


@dataclass(frozen=True)
class _Run:
    blocks: tuple[BlockSpan, ...]
    context: tuple[BlockSpan, ...]


def _same_section(left: BlockSpan, right: BlockSpan) -> bool:
    return left.page_index == right.page_index and left.section_path == right.section_path


def _runs(rep: TextRepresentation) -> Iterator[_Run]:
    blocks: list[BlockSpan] = []
    headings: list[BlockSpan] = []
    header: BlockSpan | None = None
    context: tuple[BlockSpan, ...] = ()
    for block in rep.blocks:
        if block.exclusion_reason is not None or block.block_type in {"heading", "table_header"}:
            if blocks:
                yield _Run(tuple(blocks), context)
                blocks = []
            if block.exclusion_reason is not None:
                headings, header = [], None
            elif block.block_type == "heading":
                headings = [
                    item
                    for item in headings
                    if item.page_index == block.page_index
                    and len(item.section_path) < len(block.section_path)
                    and block.section_path[: len(item.section_path)] == item.section_path
                ] + [block]
                header = None
            else:
                header = block
            continue
        if block.start_char == block.end_char:
            continue
        if block.block_type not in {"table", "table_row", "separator"}:
            header = None
        selected = tuple(
            item
            for item in headings
            if item.page_index == block.page_index
            and block.section_path[: len(item.section_path)] == item.section_path
        )
        if header is not None and _same_section(header, block):
            selected += (header,)
        if blocks and (
            not _same_section(blocks[-1], block)
            or blocks[-1].end_char != block.start_char
            or selected != context
        ):
            yield _Run(tuple(blocks), context)
            blocks = []
        context = selected
        blocks.append(block)
    if blocks:
        yield _Run(tuple(blocks), context)


def _largest_fitting(
    text: str,
    start: int,
    ends: Sequence[int],
    tokenizer: TextTokenizer,
    limit: int,
) -> int | None:
    # Descending exhaustive checks do not assume monotonic token counts.
    for end in reversed(ends):
        if end > start and 0 < _count(tokenizer, text[start:end]) <= limit:
            return end
    return None


def _ranges(
    rep: TextRepresentation, run: _Run, tokenizer: TextTokenizer, profile: ChunkingProfile
) -> Iterator[tuple[int, int]]:
    start = run.blocks[0].start_char
    end = run.blocks[-1].end_char
    if not rep.text[start:end].strip() and all(
        block.block_type == "separator" for block in run.blocks
    ):
        return
    boundaries = [block.end_char for block in run.blocks]
    while start < end:
        chosen = _largest_fitting(rep.text, start, boundaries, tokenizer, profile.target_tokens)
        if chosen is None:
            block_end = next(
                (
                    value
                    for value in boundaries
                    if value > start and _count(tokenizer, rep.text[start:value]) > 0
                ),
                None,
            )
            if block_end is None:
                raise ValueError("No positive-token core fits the remaining source")
            if _count(tokenizer, rep.text[start:block_end]) <= profile.max_tokens:
                chosen = block_end
            else:
                sentences = [
                    start + match.end()
                    for match in re.finditer(
                        r"[.!?\u3002\uff01\uff1f](?:[ \t]+)?|\r?\n",
                        rep.text[start:block_end],
                    )
                ]
                chosen = _largest_fitting(
                    rep.text, start, sentences, tokenizer, profile.target_tokens
                )
                if chosen is None:
                    chosen = _largest_fitting(
                        rep.text, start, sentences, tokenizer, profile.max_tokens
                    )
                if chosen is None:
                    chosen = _largest_fitting(
                        rep.text,
                        start,
                        range(start + 1, block_end),
                        tokenizer,
                        profile.max_tokens,
                    )
        if chosen is None:
            raise ValueError("No nonempty source prefix fits the token budget")
        yield start, chosen
        start = chosen


def _segment(
    rep: TextRepresentation,
    representation_ref: str,
    blocks: tuple[BlockSpan, ...],
    start: int,
    end: int,
    role: str,
    segment_id: str,
) -> InputSegment:
    spans = tuple(
        SourceSpan(
            representation_ref,
            representation_ref,
            max(start, block.start_char),
            min(end, block.end_char),
            block.block_id if block.page_index is not None else None,
            block.page_index,
        )
        for block in blocks
        if max(start, block.start_char) < min(end, block.end_char)
    )
    return InputSegment(segment_id, role, rep.text[start:end], spans)


def _context_segments(
    rep: TextRepresentation,
    representation_ref: str,
    run: _Run,
    start: int,
    tokenizer: TextTokenizer,
    budget: int,
) -> tuple[InputSegment, ...]:
    segments: list[InputSegment] = []
    for block in run.context:
        if block.start_char == block.end_char:
            continue
        cost = _count(tokenizer, rep.text[block.start_char : block.end_char])
        if cost > budget:
            continue
        role = "title" if block.block_type == "heading" else "table_header"
        segments.append(
            _segment(
                rep,
                representation_ref,
                (block,),
                block.start_char,
                block.end_char,
                role,
                f"{role}:{len(segments)}",
            )
        )
        budget -= cost
    if budget > 0:
        for context_start in range(run.blocks[0].start_char, start):
            value = rep.text[context_start:start]
            if value.strip() and _count(tokenizer, value) <= budget:
                segments.append(
                    _segment(
                        rep,
                        representation_ref,
                        run.blocks,
                        context_start,
                        start,
                        "context",
                        "context",
                    )
                )
                break
    return tuple(segments)


def compose_text_units(
    rep: TextRepresentation, tokenizer: TextTokenizer, profile: ChunkingProfile
) -> tuple[TextUnit, ...]:
    """Compose disjoint source cores with bounded, separately mapped context.

    Exact representation refs also identify span evidence; authorized readers must
    resolve them to their source dependencies before model invocation. Document
    locators never imply OCR, even when block page provenance is present.
    """
    if not isinstance(rep, TextRepresentation) or not isinstance(profile, ChunkingProfile):
        raise TypeError("Expected TextRepresentation and ChunkingProfile")
    if len(rep.text) > _MAX_INPUT_CODEPOINTS:
        raise ValueError("Chunking representation input limit exceeded")
    tokenizer_fingerprint = tokenizer.fingerprint
    _digest(tokenizer_fingerprint)
    tokenizer = _BudgetedTokenizer(tokenizer)
    policy = _fingerprint(
        {
            "algorithm": "structure-core-v1",
            "profile": asdict(profile),
            "tokenizer": tokenizer_fingerprint,
            "work_limits": {
                "input_codepoints": _MAX_INPUT_CODEPOINTS,
                "count_calls": _MAX_COUNT_CALLS,
                "count_codepoints": _MAX_COUNT_CODEPOINTS,
            },
        }
    )
    representation_ref = rep.representation_ref
    units: list[TextUnit] = []
    for run in _runs(rep):
        for start, end in _ranges(rep, run, tokenizer, profile):
            core = _segment(rep, representation_ref, run.blocks, start, end, "core", "core")
            segments = (
                *_context_segments(
                    rep, representation_ref, run, start, tokenizer, profile.context_tokens
                ),
                core,
            )
            token_count = _count(tokenizer, core.text)
            if token_count > profile.max_tokens:
                raise ValueError("Emitted core exceeds the hard token limit")
            units.append(
                TextUnit(
                    occurrence_id(
                        rep.scope.scope_ref,
                        rep.document_ref,
                        representation_ref,
                        start,
                        end,
                        policy,
                    ),
                    rep.document_ref,
                    rep.scope,
                    rep.source_sha256,
                    representation_ref,
                    TextChunkLocator(TextOrigin.DOCUMENT, CharacterRange(start, end)),
                    segments,
                    run.blocks[0].section_path,
                    len(units),
                    token_count,
                    policy,
                )
            )
    if tokenizer.fingerprint != tokenizer_fingerprint:
        raise ValueError("Tokenizer identity changed during composition")
    return tuple(units)
