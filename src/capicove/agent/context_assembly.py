"""Agent-owned cited context budgets, deduplication and diversity."""

import json
from dataclasses import asdict, replace

from capicove.contracts.evidence import EvidenceSpan
from capicove.contracts.memory import MemoryContext, MemoryHit, MemoryRecall


def _render(hit: MemoryHit) -> str:
    body = asdict(hit)
    for citation, span in zip(body["citations"], hit.citations, strict=True):
        citation["reference"]["evidence_id"] = span.reference.evidence_id
    return json.dumps(
        body,
        default=lambda value: value.isoformat(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _prefix(hit: MemoryHit, length: int) -> MemoryHit:
    citations = []
    consumed = 0
    for span in hit.citations:
        remaining = length - consumed
        if remaining <= 0:
            break
        size = min(span.end - span.start, remaining)
        citations.append(EvidenceSpan(span.reference, span.start, span.start + size))
        consumed += size + 1  # Newline separator between quoted spans.
    return replace(hit, text=hit.text[: min(length, consumed - 1)], citations=tuple(citations))


def _fit(hit: MemoryHit, characters: int, tokens: int) -> tuple[MemoryHit, str] | None:
    rendered = _render(hit)
    if len(rendered) <= characters and len(rendered.encode()) <= tokens:
        return hit, rendered
    lower, upper = 1, len(hit.text)
    best = None
    while lower <= upper:
        middle = (lower + upper) // 2
        excerpt = _prefix(hit, middle)
        encoded = _render(excerpt)
        if len(encoded) <= characters and len(encoded.encode()) <= tokens:
            best = excerpt, encoded
            lower = middle + 1
        else:
            upper = middle - 1
    return best


def assemble_context(
    result: MemoryRecall, *, max_characters: int, max_tokens: int
) -> MemoryContext:
    selected: list[MemoryHit] = []
    chars, tokens = 0, 0
    assertions: set[str] = set()
    sources: dict[str, int] = {}
    entities: dict[str, int] = {}
    predicates: dict[str, int] = {}
    for hit in result.hits:
        if hit.assertion_hash is not None and hit.assertion_hash in assertions:
            continue
        # UTF-8 bytes are a conservative model-independent token upper estimate.
        fitted = _fit(hit, max_characters - chars, max_tokens - tokens)
        if fitted is None:
            continue
        hit, rendered = fitted
        cost = len(rendered.encode())
        refs = {span.reference.evidence_id for span in hit.citations}
        if (
            any(sources.get(ref, 0) >= 3 for ref in refs)
            or (hit.subject_ref and entities.get(hit.subject_ref, 0) >= 5)
            or (hit.predicate and predicates.get(hit.predicate, 0) >= 5)
        ):
            continue
        selected.append(hit)
        chars += len(rendered)
        tokens += cost
        if hit.assertion_hash:
            assertions.add(hit.assertion_hash)
        for ref in refs:
            sources[ref] = sources.get(ref, 0) + 1
        if hit.subject_ref:
            entities[hit.subject_ref] = entities.get(hit.subject_ref, 0) + 1
        if hit.predicate:
            predicates[hit.predicate] = predicates.get(hit.predicate, 0) + 1
    return MemoryContext(tuple(selected), chars, tokens, result.warnings)
