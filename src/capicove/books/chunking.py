"""Source-preserving, structure/type-aware bounded book chunking."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from itertools import chain

from capicove.books.chunk_config import ChunkingConfig
from capicove.books.chunk_content import (
    content_type,
    continues,
    cut_positions,
    joins_protected_group,
    safe_boundary,
)
from capicove.books.chunk_front_matter import source_roles
from capicove.books.chunk_models import (
    BookChunk,
    ChunkRun,
    SourceSpan,
    chunk_run_id,
    digest,
    render_chunk,
    spans_text,
)
from capicove.books.chunk_tokenizer import TokenCounter, Utf8ByteCounter, load_token_counter
from capicove.books.codec import book_structure_to_json
from capicove.books.models import BookStructure
from capicove.books.tasks import _section_block_ranges
from capicove.contracts import (
    DocumentBlock,
    DocumentBlockType,
    DocumentPageStatus,
    DocumentParseResult,
)


MAX_SPANS = 128
MAX_RENDERED_BYTES = 131072


@dataclass(frozen=True)
class _Piece:
    span: SourceSpan
    parent: str
    headings: tuple[str, ...]
    kind: str
    warnings: tuple[str, ...] = ()


def build_chunk_run(
    result: DocumentParseResult,
    structure: BookStructure,
    config: ChunkingConfig,
    *,
    counter: TokenCounter | None = None,
) -> ChunkRun:
    """Partition eligible source text exactly once, without model invocation."""
    if (
        result.source_sha256 != structure.source_sha256
        or result.representation_sha256 != structure.representation_sha256
    ):
        raise ValueError("structure and source representation identities disagree")
    if any(page.status is not DocumentPageStatus.SUCCEEDED for page in result.pages):
        raise ValueError("failed pages prevent complete chunk coverage")
    counter = counter or load_token_counter(config)
    blocks = {block.block_id: block for page in result.pages for block in page.blocks}
    if len(blocks) != sum(len(page.blocks) for page in result.pages):
        raise ValueError("source block IDs must be unique across the document")
    structure_hash = digest(book_structure_to_json(structure))
    run_id = chunk_run_id(
        result.source_sha256,
        result.representation_sha256,
        structure_hash,
        config,
        counter.fingerprint,
    )
    sections = {section.section_id: section for section in structure.sections}
    ranges = _section_block_ranges(structure, result)
    pieces: list[_Piece] = []
    page_indices = frozenset(page.page_index for page in result.pages)
    excluded, navigation = source_roles(
        blocks, config.front_matter_policy, page_indices=page_indices
    )
    exclusions = list(excluded.items())
    for page in result.pages:
        for block in page.blocks:
            if block.block_id in excluded:
                continue
            if len(block.exact_text.encode()) > 4 * 1024 * 1024:
                raise ValueError("source block exceeds the 4 MiB processing limit")
            position = (block.page_index, block.reading_order)
            candidates = [
                section
                for section in structure.sections
                if ranges[section.section_id][0] <= position < ranges[section.section_id][1]
            ]
            owner = max(candidates, key=lambda section: section.level, default=None)
            ceiling = {"section": 2, "chapter": 1}.get(config.boundary_level)
            while (
                owner is not None
                and ceiling is not None
                and owner.level > ceiling
                and owner.parent_id
            ):
                owner = sections[owner.parent_id]
            parent = owner.section_id if owner else "front-matter"
            headings: list[str] = []
            ancestor = owner
            while ancestor is not None:
                headings.insert(0, ancestor.exact_title)
                ancestor = sections.get(ancestor.parent_id or "")
            if block.block_id in navigation:
                parent = navigation[block.block_id]
                headings = []
            kind = content_type(block)
            warnings = (
                ("type_text_fallback",)
                if kind == "text"
                and block.block_type
                in {DocumentBlockType.TABLE, DocumentBlockType.IMAGE, DocumentBlockType.OTHER}
                else ()
            )
            piece = _Piece(SourceSpan.from_block(block), parent, tuple(headings), kind, warnings)
            pieces.extend(_split(piece, blocks, config, counter))
    batches: list[list[_Piece]] = []
    for piece in pieces:
        previous = batches[-1][-1] if batches else None
        append = previous is not None and previous.parent == piece.parent
        protected = False
        if previous is not None and append:
            prior_text = previous.span.extract(blocks[previous.span.block_id])
            next_text = piece.span.extract(blocks[piece.span.block_id])
            if previous.span.page_index != piece.span.page_index:
                append = (
                    config.page_boundary == "allow_continuations"
                    and previous.kind == piece.kind
                    and continues(prior_text, next_text)
                )
            protected = previous.kind in config.protected_types and joins_protected_group(
                prior_text, next_text, previous.kind, piece.kind
            )
            current = tuple(part.span for part in batches[-1])
            append = append and (
                (
                    config.merge_small_chunks
                    and counter.count(spans_text(current, blocks)) < config.target_core_tokens
                )
                or protected
            )
            append = (
                append
                and len(current) < MAX_SPANS
                and _fits(piece.headings, (*current, piece.span), (), (), blocks, config, counter)
            )
        if append:
            batches[-1].append(piece)
        else:
            if protected:
                piece = replace(
                    piece, warnings=(*piece.warnings, "protected_group_split:packing_limit")
                )
            batches.append([piece])
    parent_batches: dict[str, list[int]] = defaultdict(list)
    parent_positions: dict[int, int] = {}
    for index, batch in enumerate(batches):
        siblings = parent_batches[batch[0].parent]
        parent_positions[index] = len(siblings)
        siblings.append(index)
    chunks: list[BookChunk] = []
    for index, batch in enumerate(batches):
        core = tuple(part.span for part in batch)
        siblings = parent_batches[batch[0].parent]
        sibling_position = parent_positions[index]
        before_candidates = tuple(
            part.span
            for earlier in reversed(
                siblings[max(0, sibling_position - MAX_SPANS) : sibling_position]
            )
            for part in reversed(batches[earlier])
        )
        after_candidates = tuple(
            part.span
            for later in siblings[sibling_position + 1 : sibling_position + 1 + MAX_SPANS]
            for part in batches[later]
        )
        before = bounded_context(
            before_candidates, config.context_before_tokens, blocks, counter, from_end=True
        )
        after = bounded_context(after_candidates, config.context_after_tokens, blocks, counter)
        # Context is optional. Preserve complete core before any overlap text.
        while (
            len(core) + len(before) + len(after) > MAX_SPANS
            or not _fits(batch[0].headings, core, before, after, blocks, config, counter)
        ) and (before or after):
            if after:
                after = _shrink(after, blocks, from_end=True)
            else:
                before = _shrink(before, blocks, from_end=False)
        core_text = spans_text(core, blocks)
        rendered = render_chunk(
            batch[0].headings, core_text, spans_text(before, blocks), spans_text(after, blocks)
        )
        chunk_id = "chunk:" + digest({"run": run_id, "core": [asdict(span) for span in core]})[:24]
        chunk_warnings = {warning for part in batch for warning in part.warnings}
        if len({span.page_index for span in core}) > 1:
            chunk_warnings.add("cross_page_continuation")
        chunks.append(
            BookChunk(
                chunk_id,
                batch[0].parent,
                batch[0].headings,
                core,
                before,
                after,
                tuple(sorted({part.kind for part in batch})),
                counter.count(core_text),
                counter.count(rendered),
                tuple(sorted(chunk_warnings)),
            )
        )
    validate_core_coverage(
        tuple(chunks),
        blocks,
        frozenset(excluded),
        policy=config.front_matter_policy,
        page_indices=page_indices,
    )
    return ChunkRun(
        run_id,
        result.source_sha256,
        result.representation_sha256,
        structure_hash,
        counter.fingerprint,
        counter.mode,
        json.dumps(config.chunk_values(), sort_keys=True),
        tuple(chunks),
        tuple(exclusions),
        True,
    )


def _fits(
    headings: tuple[str, ...],
    core: tuple[SourceSpan, ...],
    before: tuple[SourceSpan, ...],
    after: tuple[SourceSpan, ...],
    blocks: dict[str, DocumentBlock],
    config: ChunkingConfig,
    counter: TokenCounter,
) -> bool:
    rendered = render_chunk(
        headings, spans_text(core, blocks), spans_text(before, blocks), spans_text(after, blocks)
    )
    return (
        len(rendered.encode()) <= MAX_RENDERED_BYTES
        and counter.count(rendered) <= config.max_chunk_tokens
    )


def _split(
    piece: _Piece, blocks: dict[str, DocumentBlock], config: ChunkingConfig, counter: TokenCounter
) -> list[_Piece]:
    if _fits(piece.headings, (piece.span,), (), (), blocks, config, counter):
        return [piece]
    if config.oversize_policy == "error" or piece.kind == "formula":
        raise ValueError(
            f"oversize source block {piece.span.block_id}; enlarge budget or choose a safe split policy"
        )
    block = blocks[piece.span.block_id]
    start, end = piece.span.start_char, piece.span.end_char
    pieces: list[_Piece] = []
    while start < end:
        remaining = block.exact_text[start:end]
        prefix = _fitting_prefix(remaining, piece.headings, config, counter)
        if not prefix:
            raise ValueError("oversize heading or source character cannot fit the chunk budget")
        positions = [
            position
            for position in cut_positions(remaining[:prefix], piece.kind)
            if position <= prefix
            and counter.count(render_chunk(piece.headings, remaining[:position]))
            <= config.max_chunk_tokens
        ]
        size = positions[-1] if positions and prefix < len(remaining) else prefix
        warning = "split_at_boundary" if size in positions else "forced_character_split"
        if piece.kind in config.protected_types:
            warning = "protected_group_split:" + warning
        pieces.append(
            _Piece(
                SourceSpan.from_block(block, start, start + size),
                piece.parent,
                piece.headings,
                piece.kind,
                (*piece.warnings, warning),
            )
        )
        start += size
    return pieces


def _fitting_prefix(
    text: str, headings: tuple[str, ...], config: ChunkingConfig, counter: TokenCounter
) -> int:
    limit = min(len(text), MAX_RENDERED_BYTES)
    if isinstance(counter, Utf8ByteCounter):
        budget = (
            min(config.max_chunk_tokens, MAX_RENDERED_BYTES)
            - counter.count(render_chunk(headings, ""))
            - (2 if headings else 0)
        )
        used, limit = 0, 0
        for character in text:
            used += len(character.encode())
            if used > budget:
                break
            limit += 1
    # Probe smaller windows for speed, but never infer that untested lengths fail.
    # If all probes fail, exhaustive fallback includes every bounded prefix.
    probes: list[int] = []
    probe = limit
    while probe > 0:
        probes.append(probe)
        probe //= 2
    for end in chain(probes, range(limit, 0, -1)):
        if safe_boundary(text, end):
            rendered = render_chunk(headings, text[:end])
            if (
                len(rendered.encode()) <= MAX_RENDERED_BYTES
                and counter.count(rendered) <= config.max_chunk_tokens
            ):
                return end
    return 0


def bounded_context(
    candidates: tuple[SourceSpan, ...],
    budget: int,
    blocks: dict[str, DocumentBlock],
    counter: TokenCounter,
    *,
    from_end: bool = False,
) -> tuple[SourceSpan, ...]:
    """Select nearest source context, preserving natural order and bounded size."""
    chosen: tuple[SourceSpan, ...] = ()
    if budget <= 0:
        return chosen
    for span in candidates:
        if len(chosen) >= MAX_SPANS:
            break
        trial = (span, *chosen) if from_end else (*chosen, span)
        if (
            counter.count(spans_text(trial, blocks)) <= budget
            and len(spans_text(trial, blocks).encode()) <= MAX_RENDERED_BYTES
        ):
            chosen = trial
            continue
        block = blocks[span.block_id]
        length = min(span.end_char - span.start_char, budget)
        for size in range(length, 0, -1):
            start, end = (
                (span.end_char - size, span.end_char)
                if from_end
                else (span.start_char, span.start_char + size)
            )
            if (start > 0 and not safe_boundary(block.exact_text, start)) or not safe_boundary(
                block.exact_text, end
            ):
                continue
            shortened = SourceSpan.from_block(block, start, end)
            trial = (shortened, *chosen) if from_end else (*chosen, shortened)
            if (
                len(spans_text(trial, blocks).encode()) <= MAX_RENDERED_BYTES
                and counter.count(spans_text(trial, blocks)) <= budget
            ):
                chosen = trial
                break
        break
    return chosen


def _shrink(
    spans: tuple[SourceSpan, ...], blocks: dict[str, DocumentBlock], *, from_end: bool
) -> tuple[SourceSpan, ...]:
    index = -1 if from_end else 0
    span = spans[index]
    start, end = span.start_char, span.end_char
    if from_end:
        end -= 1
        while end > start and not safe_boundary(blocks[span.block_id].exact_text, end):
            end -= 1
    else:
        start += 1
        while start < end and not safe_boundary(blocks[span.block_id].exact_text, start):
            start += 1
    rest = spans[:-1] if from_end else spans[1:]
    if start >= end:
        return rest
    replacement = SourceSpan.from_block(blocks[span.block_id], start, end)
    return (*rest, replacement) if from_end else (replacement, *rest)


def validate_core_coverage(
    chunks: tuple[BookChunk, ...],
    blocks: dict[str, DocumentBlock],
    excluded: frozenset[str],
    *,
    policy: str = "preserve",
    page_indices: frozenset[int] | None = None,
) -> None:
    """Require exact, non-overlapping coverage of every eligible source block."""
    allowed, navigation = source_roles(blocks, policy, page_indices=page_indices)
    if excluded != frozenset(allowed):
        raise ValueError("core exclusions disagree with source policy")
    positions = {key: 0 for key in blocks if key not in excluded}
    for chunk in chunks:
        for span in (*chunk.core_spans, *chunk.before_spans, *chunk.after_spans):
            if span.block_id in excluded:
                raise ValueError("chunk references excluded source")
            parent = navigation.get(span.block_id)
            if parent != (chunk.parent_id if chunk.content_role == "navigation" else None):
                raise ValueError("chunk crosses navigation boundary")
        for span in chunk.core_spans:
            if span.block_id not in positions or positions[span.block_id] != span.start_char:
                raise ValueError("core coverage contains a gap, overlap or excluded source")
            span.extract(blocks[span.block_id])
            positions[span.block_id] = span.end_char
    if any(position != len(blocks[key].exact_text) for key, position in positions.items()):
        raise ValueError("core coverage is incomplete")
