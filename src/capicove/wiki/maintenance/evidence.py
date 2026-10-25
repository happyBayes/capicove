"""Exact source-backed codepoint intervals; hashes do not prove entailment."""

import hashlib

from capicove.contracts import DocumentPageStatus, DocumentParseResult
from capicove.contracts.wiki_maintenance import EvidenceSpan
from capicove.wiki.maintenance.representations import (
    MarkdownRepresentation,
    markdown_representation,
)


def quote_interval(text: str, quote: str) -> tuple[int, int]:
    """Resolve a unique quote, including overlapping-match ambiguity."""
    start = text.find(quote)
    if not quote or start < 0 or text.find(quote, start + 1) >= 0:
        raise ValueError("evidence_unavailable: missing or ambiguous quotation")
    return start, start + len(quote)


def markdown_span(value: MarkdownRepresentation, start: int, end: int) -> EvidenceSpan:
    """Issue a locator only for a valid bounded nonempty interval."""
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(value.text):
        raise ValueError("evidence_unavailable: invalid interval")
    return EvidenceSpan(
        kind="markdown_span",
        source_id="source:sha256:" + value.source_sha256,
        source_sha256=value.source_sha256,
        representation_sha256=value.representation_sha256,
        start_char=start,
        end_char=end,
        exact_sha256=hashlib.sha256(value.text[start:end].encode()).hexdigest(),
        page_index=None,
        block_id=None,
    )


def resolve_markdown(span: EvidenceSpan, raw: bytes) -> str:
    """Recompute the representation from original bytes before resolving."""
    if span.kind != "markdown_span":
        raise ValueError("evidence_unavailable: wrong representation kind")
    value = markdown_representation(raw)
    expected = markdown_span(value, span.start_char, span.end_char)
    if span != expected:
        raise ValueError("evidence_digest_mismatch")
    return value.text[span.start_char : span.end_char]


def resolve_pdf(
    span: EvidenceSpan,
    result: DocumentParseResult,
    raw: bytes,
    *,
    snapshot_sha256: str,
) -> str:
    """Resolve an exact block against a host-verified snapshot commitment.

    The commitment must come from accepted Book publication history, never from
    an Agent submission. A provider's representation ID alone is not an
    integrity hash of this serialized snapshot.
    """
    import json

    from capicove.documents import document_parse_result_to_json

    encoded = json.dumps(
        document_parse_result_to_json(result),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    if (
        span.kind != "pdf_block_span"
        or hashlib.sha256(raw).hexdigest() != result.source_sha256
        or span.source_sha256 != result.source_sha256
        or span.representation_sha256 != result.representation_sha256
        or hashlib.sha256(encoded).hexdigest() != snapshot_sha256
    ):
        raise ValueError("evidence_digest_mismatch")
    for page in result.pages:
        if page.page_index != span.page_index or page.status != DocumentPageStatus.SUCCEEDED:
            continue
        for block in page.blocks:
            if block.block_id != span.block_id:
                continue
            if span.end_char > len(block.exact_text):
                raise ValueError("evidence_unavailable: invalid PDF interval")
            text = block.exact_text[span.start_char : span.end_char]
            if hashlib.sha256(text.encode()).hexdigest() != span.exact_sha256:
                raise ValueError("evidence_digest_mismatch")
            return text
    raise ValueError("evidence_unavailable: PDF page or block unavailable")
