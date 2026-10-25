"""Deterministic L0-L2 validation for page-aware book citations."""

from __future__ import annotations

import hashlib
import re
import unicodedata

from capicove.books import AtomicClaim
from capicove.contracts import DocumentParseResult
from capicove.wiki.models import (
    CitationValidationFinding,
    CitationValidationLevel,
    CitationValidationReport,
    ClaimCitation,
)


def validate_book_citations(
    citations: tuple[ClaimCitation, ...],
    claims: tuple[AtomicClaim, ...],
    result: DocumentParseResult,
) -> CitationValidationReport:
    """Resolve identities, verify evidence integrity, then check quotations."""
    findings: list[CitationValidationFinding] = []
    claims_by_id = {claim.claim_id: claim for claim in claims}
    pages = {page.page_index: page for page in result.pages}
    cited_claim_ids = frozenset(citation.claim_id for citation in citations)
    for claim in claims:
        if claim.claim_id not in cited_claim_ids:
            findings.append(
                _finding(
                    CitationValidationLevel.L0,
                    "claim_citation_missing",
                    claim.claim_id,
                    "atomic claim has no citation record",
                )
            )
    seen_ids: set[str] = set()
    for citation in citations:
        resolved_claim = claims_by_id.get(citation.claim_id)
        if citation.citation_id in seen_ids:
            findings.append(
                _finding(
                    CitationValidationLevel.L0,
                    "citation_id_duplicate",
                    citation.claim_id,
                    "citation ID is duplicated",
                )
            )
        seen_ids.add(citation.citation_id)
        if resolved_claim is None:
            findings.append(
                _finding(
                    CitationValidationLevel.L0,
                    "citation_claim_unknown",
                    citation.claim_id,
                    "citation claim does not exist",
                )
            )
        if citation.source_sha256 != result.source_sha256:
            findings.append(
                _finding(
                    CitationValidationLevel.L0,
                    "citation_source_unknown",
                    citation.claim_id,
                    "citation source does not match the representation",
                )
            )
        if citation.representation_sha256 != result.representation_sha256:
            findings.append(
                _finding(
                    CitationValidationLevel.L0,
                    "citation_representation_unknown",
                    citation.claim_id,
                    "citation representation does not match",
                )
            )
        page = pages.get(citation.page_index)
        if page is None:
            findings.append(
                _finding(
                    CitationValidationLevel.L0,
                    "citation_page_unknown",
                    citation.claim_id,
                    "citation page does not exist",
                )
            )
            continue
        blocks = {block.block_id: block for block in page.blocks}
        missing = tuple(block_id for block_id in citation.block_ids if block_id not in blocks)
        if missing:
            findings.append(
                _finding(
                    CitationValidationLevel.L0,
                    "citation_block_unknown",
                    citation.claim_id,
                    "citation block does not resolve on the cited page",
                )
            )
            continue
        if resolved_claim is not None and any(
            block_id not in resolved_claim.citation_ids for block_id in citation.block_ids
        ):
            findings.append(
                _finding(
                    CitationValidationLevel.L0,
                    "citation_block_not_claimed",
                    citation.claim_id,
                    "citation block is outside the atomic claim allowance",
                )
            )
        exact = "\n".join(blocks[block_id].exact_text for block_id in citation.block_ids)
        digest = hashlib.sha256(exact.encode("utf-8")).hexdigest()
        if digest != citation.evidence_text_sha256:
            findings.append(
                _finding(
                    CitationValidationLevel.L1,
                    "citation_digest_mismatch",
                    citation.claim_id,
                    "citation evidence digest does not match exact source text",
                )
            )
        if citation.printed_page_label != page.printed_page_label:
            findings.append(
                _finding(
                    CitationValidationLevel.L1,
                    "citation_page_label_mismatch",
                    citation.claim_id,
                    "printed page label does not match parsed evidence",
                )
            )
        if citation.quotation is not None and _normalize(citation.quotation) not in _normalize(
            exact
        ):
            findings.append(
                _finding(
                    CitationValidationLevel.L2,
                    "citation_quotation_mismatch",
                    citation.claim_id,
                    "quotation does not occur in exact cited evidence",
                )
            )
    ordered = tuple(sorted(findings, key=lambda item: (item.level.value, item.claim_id, item.code)))
    return CitationValidationReport(CitationValidationLevel.L2, ordered)


def _normalize(value: str) -> str:
    return re.sub(r"[\t\n\r\f\v ]+", " ", unicodedata.normalize("NFC", value)).strip()


def _finding(
    level: CitationValidationLevel,
    code: str,
    claim_id: str,
    message: str,
) -> CitationValidationFinding:
    return CitationValidationFinding(level, code, claim_id, message)


__all__ = ["validate_book_citations"]
