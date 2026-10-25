# RFC-0013: Page-Aware Wiki Version 2

- Status: Accepted
- Date: 2026-09-02
- Owners: Capicove maintainers

## Summary

Wiki schema version 2 adds immutable PDF sources, parsed representations, book
pages, and a claim citation ledger. Migration from version 1 is explicit and
previewable; opening or building a version-1 Wiki never rewrites it silently.

## Durable layout

    raw/<source-sha256>.pdf
    representations/<representation-sha256>.json
    pages/books/<book-id>/<section-id>.md
    graph/citations.jsonl

Derived indexes remain disposable. Citation records bind a stable claim ID to a
Wiki page, source and representation hashes, zero-based PDF page index, optional
printed label, one or more block IDs, the exact evidence-text digest, optional
quotation, submission version, and optional semantic-verification receipt.

## Validation

L0 resolves every claim, page, source, representation, and block identifier. L1
verifies page bounds, source lineage, and evidence digests. L2 verifies quotations
against exact evidence text using NFC normalization and collapsed ASCII
whitespace. Any L0-L2 error blocks strict publication. L3 is an optional
caller-owned judgment with supported, unsupported, or uncertain; it never replaces
deterministic validation.

Every factual claim requires at least one citation. Navigation, headings, and
explicit reader prompts are not claims. Rendered links show a printed page label
when present and always preserve the zero-based PDF page target.

## Migration and mutation

Preview returns the exact files and manifest changes without writing. Applying a
migration creates missing version-2 roots and atomically updates the manifest and
schema description. Source PDFs and representations are content addressed and
never overwritten with different bytes. Citation and publication outputs are
deterministic rebuildable projections of accepted submissions.
