from __future__ import annotations

import hashlib

from capicove.books import AtomicClaim
from capicove.contracts import (
    DocumentBlock,
    DocumentBlockType,
    DocumentPage,
    DocumentPageStatus,
    DocumentParseResult,
    DocumentParserReceipt,
    NormalizedBox,
)
from capicove.wiki import (
    CitationValidationLevel,
    ClaimCitation,
    validate_book_citations,
)


def _result() -> DocumentParseResult:
    block = DocumentBlock(
        "p0002:b000001",
        2,
        DocumentBlockType.TEXT,
        "Capicove preserves exact evidence.",
        "Capicove preserves exact evidence.",
        NormalizedBox(0.1, 0.1, 0.9, 0.2),
        0,
        0.99,
    )
    page = DocumentPage(2, "17", 100, 200, DocumentPageStatus.SUCCEEDED, (block,))
    receipt = DocumentParserReceipt("plugin", "1", "profile", 1, "c" * 64, "d" * 64, "e" * 64)
    return DocumentParseResult("a" * 64, "b" * 64, (page,), receipt)


def _citation(**overrides: object) -> ClaimCitation:
    exact = "Capicove preserves exact evidence."
    values: dict[str, object] = {
        "citation_id": "citation:one",
        "claim_id": "claim:one",
        "page_id": "page:section-one",
        "book_id": "book:" + "1" * 24,
        "source_sha256": "a" * 64,
        "representation_sha256": "b" * 64,
        "page_index": 2,
        "printed_page_label": "17",
        "block_ids": ("p0002:b000001",),
        "evidence_text_sha256": hashlib.sha256(exact.encode()).hexdigest(),
        "quotation": "preserves exact evidence",
        "submission_version": 1,
    }
    values.update(overrides)
    return ClaimCitation(**values)  # type: ignore[arg-type]


def test_l0_l1_l2_validate_resolution_integrity_and_quotation() -> None:
    claim = AtomicClaim("claim:one", "Capicove preserves evidence.", ("p0002:b000001",))

    report = validate_book_citations((_citation(),), (claim,), _result())

    assert report.valid is True
    assert report.validated_level is CitationValidationLevel.L2


def test_citation_validation_reports_unknown_digest_and_quotation_failures() -> None:
    claim = AtomicClaim("claim:one", "A claim", ("p0002:b000001",))
    citations = (
        _citation(block_ids=("p0002:b999999",)),
        _citation(citation_id="citation:bad-digest", evidence_text_sha256="f" * 64),
        _citation(citation_id="citation:bad-quote", quotation="not in evidence"),
    )

    report = validate_book_citations(citations, (claim,), _result())

    assert report.valid is False
    assert {item.code for item in report.findings} >= {
        "citation_block_unknown",
        "citation_digest_mismatch",
        "citation_quotation_mismatch",
    }


def test_every_atomic_claim_requires_a_resolved_citation_record() -> None:
    claim = AtomicClaim("claim:one", "A claim", ("p0002:b000001",))

    report = validate_book_citations((), (claim,), _result())

    assert {item.code for item in report.findings} == {"claim_citation_missing"}
