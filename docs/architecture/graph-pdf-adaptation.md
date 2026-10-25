# Deferred PDF Adaptation Design for the Document Graph

- Status: Full source-aware locator adaptation remains deferred
- Date: 2026-09-13
- Scope: Task 4D design only; the delivered graph representation implementation is text-only

The original baseline lacked the public parser DTO. RFC-0011 has since landed its
public result and backend/request contracts. This design does not define, import,
or emulate them. Its full `DocumentParseResult` -> `TextRepresentation` mapping
with source-aware locator fields remains future work.

Full PDF/OCR/ASR locator support is not implemented. Task 4 remains partial.
The operator-selected local composition amendment on 2026-10-13 adds a bounded
preparation entry point using the landed public parser and image ports. It
produces derived graph text, preserves source hashes and page/block coordinates,
binds parser/model identity through the existing representation fingerprint, and
returns the original PDF parse result to the caller. It does not change graph
DTOs, text-origin retrieval locators, schemas, or publication formats. The
additional source-aware fields below remain deferred. See
[RFC-0011](../rfcs/0011-document-parser-contracts.md#local-paddle-composition-amendment-2026-10-13).

## Preconditions and identity

The future adapter accepts only a result produced for an authorized immutable
document occurrence. Before construction, it must establish all of the following
from the landed public parser result and its public receipt/lineage values, without
opening a source path or importing a parser-private module:

| Required semantic value | Future mapping requirement |
| --- | --- |
| Document and scope | Match the authorized `document_ref` and `GraphScope` requested by the graph build. A result for another occurrence or scope is rejected. |
| Immutable source digest | Verify the parser result's declared source identity against the authorized immutable source. Assign that verified lowercase SHA-256 to `TextRepresentation.source_sha256`; a missing or mismatched digest is a document failure, not a text representation. |
| Parser-result identity | Retain the public result/receipt identity, result schema/version, and explicitly selected parser configuration as immutable lineage. The future representation identity must bind the adapter mapping version and this parser-result identity in addition to the existing scoped text, block, exclusion, and source identity. |
| Exact source text | Use only each ordered block's exact parser text for `TextRepresentation.text` and coordinate calculations. Normalized text is auxiliary parser metadata and never overwrites exact text or supplies offsets. |

### Required future immutable representation fields and identity

The future change extends the capability-local `TextRepresentation` itself; it does
not use an optional companion record. The added immutable fields are:

```python
origin: TextOrigin
adapter_mapping_version: str = "pdf-adaptation-v1"
parser_result_ref: str | None
parser_receipt_ref: str | None
parser_result_schema_version: str | None
parser_configuration_ref: str | None
declared_page_indices: tuple[int, ...]
partial_page_indices: tuple[int, ...]
failed_page_indices: tuple[int, ...]
omitted_page_indices: tuple[int, ...]
```

For a PDF representation, `origin` is the caller-selected `TextOrigin.DOCUMENT` or
`TextOrigin.OCR`; `adapter_mapping_version` is exactly `"pdf-adaptation-v1"`; and
each parser-result, receipt, schema-version, and configuration field is nonblank.
The future public adapter signature is
`pdf_representation(document_ref, result, scope, origin: TextOrigin)`; it rejects
an origin outside `DOCUMENT`/`OCR` before it reads any parser block.
For the existing plain-text/Markdown adapter, the future constructor uses
`TextOrigin.DOCUMENT`, its own non-PDF mapping version, and `None` for each parser
lineage field. No other `TextOrigin` is accepted by the future PDF adapter.

Every page-index tuple is sorted, unique, and zero-based. `partial_page_indices`,
`failed_page_indices`, and `omitted_page_indices` are pairwise-disjoint subsets of
`declared_page_indices`. Parser issue codes are retained in `issue_codes`; any
partial, failed, or omitted page therefore remains visible both as coverage and as
the corresponding parser issue. A successful representation cannot have failed or
omitted coverage.

The future `representation_sha256` replaces the current schema-1 calculation with
this exact canonical payload, passed to the existing `fingerprint` helper:

```python
representation_sha256 = fingerprint(
    {
        "schema_version": 2,
        "document_ref": self.document_ref,
        "scope": asdict(self.scope),
        "source_sha256": self.source_sha256,
        "text": self.text,
        "blocks": tuple(asdict(block) for block in self.blocks),
        "issue_codes": self.issue_codes,
        "origin": self.origin.value,
        "adapter_mapping_version": self.adapter_mapping_version,
        "parser_result_ref": self.parser_result_ref,
        "parser_receipt_ref": self.parser_receipt_ref,
        "parser_result_schema_version": self.parser_result_schema_version,
        "parser_configuration_ref": self.parser_configuration_ref,
        "declared_page_indices": self.declared_page_indices,
        "partial_page_indices": self.partial_page_indices,
        "failed_page_indices": self.failed_page_indices,
        "omitted_page_indices": self.omitted_page_indices,
    }
)
representation_ref = f"representation:{representation_sha256}"
```

This gives the parser result and receipt explicit identity-bearing fields rather
than treating lineage as out-of-band metadata. It must not create a substitute
`DocumentParseResult` or copy the parser's DTO schema into `capicove.graph_build`.

## Deterministic block mapping

Use the public parser result's declared reading order. Reject malformed ordering,
duplicate block identities, missing page/block provenance, or an ambiguous order;
do not repair a malformed result by guessing. For every ordered parser block:

1. Append its exact text, retaining empty text blocks as zero-width provenance rows.
2. Insert exactly `"\n\n"` between consecutive input blocks. Inserted separators
   receive their own contiguous coverage rows so `TextRepresentation.blocks` covers
   every representation code point without gaps. The separator rows have `page_index=None`,
   `block_type="separator"`, and `exclusion_reason="adapter_separator"`; they are
   coverage-only and never create a page-bearing OCR core.
3. Create a block-span row with the parser block identity, the half-open Unicode
   code-point coordinates of its exact text, the parser's zero-based page index,
   public block kind, and public section path where supplied. The page/block pair
   is retained together; block text coordinates are never UTF-8 byte offsets.
4. Preserve the parser's declared table kind and exact table text as a block. Carry
   any public row/header provenance only when the landed DTO makes it reliable; do
   not invent table rows, bounding boxes, headings, or geometry from text.

Normalization, layout interpretation, and exclusions are separate from exact-text
composition. Header/footer, decorative, or otherwise ineligible blocks remain in
the representation and block coverage with an explicit exclusion reason. They are
not silently deleted. Per-page or per-block parser failures, omitted pages, and
parser issue codes are retained in representation/coverage lineage. A failed or
partial page is never reclassified as an empty successful page, and a partial
parser result must produce a visibly partial document outcome rather than a
successful empty representation.

## Origin and locator rules

PDF provenance is not a retrieval locator. The future PDF adapter stores its
caller-selected origin in `TextRepresentation.origin`; `compose_text_units` reads
that field rather than unconditionally constructing `TextOrigin.DOCUMENT`.

For every eligible core, composition already keeps parser-page runs separate. The
future implementation must additionally reject an OCR core unless its non-excluded
source blocks have one shared, non-`None` zero-based `page_index`. It constructs
locators exactly as follows:

```python
if rep.origin is TextOrigin.DOCUMENT:
    locator = TextChunkLocator(TextOrigin.DOCUMENT, CharacterRange(start, end))
elif rep.origin is TextOrigin.OCR:
    locator = TextChunkLocator(
        TextOrigin.OCR, CharacterRange(start, end), page_number=page_index + 1
    )
else:
    raise ValueError("unsupported representation origin")
```

Thus a document-origin locator has no page field even when its block-span
provenance records a zero-based PDF page index. Only an explicitly declared OCR
origin converts page index `0` to OCR page `1`. A PDF is not evidence of OCR, and
the adapter must not relabel all PDF text as OCR. ASR is outside this mapping.

## Authorized call sequence

The integration path is ordered and effectful only at declared capability seams:

1. Authorize the requested document and every evidence/source reference for the
   requested scope, then resolve the immutable source occurrence and its expected
   digest.
2. Select an explicitly configured public parser capability and submit its public,
   authorized request. Source acquisition, network access, model acquisition, and
   implicit parser selection are outside this adapter.
3. Persist the public parse result and its receipt/lineage through the owning
   capability's approved storage boundary, then verify that result identity,
   request/source identity, parser configuration, and digest match the invocation.
4. Before a persisted result is reused, re-check authorization before reuse and
   verify the same immutable identities again. A hash or cache key never grants
   access by itself.
5. Apply the deterministic mapping above, retain document-to-result-to-
   representation lineage, and expose parser issues and declared/processed/failed/
   omitted coverage without fabricating success.

If the public parser capability is unavailable, the source is unauthorized, the
result cannot be verified, or the media type is unsupported, return the applicable
document-level failure. Do not return empty text as a substitute and do not add an
implicit fallback parser.

## Implementation gate

When the owning DTO lands, the implementer must first add synthetic fixtures that
use that public DTO and tests for source-digest mismatch, malformed ordering, empty
blocks and separators, Unicode code-point offsets, tables, exclusions, partial and
failed pages, OCR page `0` to `1` conversion, document-origin locators, unauthorized
reuse, and unsupported formats. The implementation must depend only on the landed
public parser boundary; `capicove.graph_build` must not import `capicove.books` or
parser-private modules. Re-run the complete Task 4 contract and integration suite
only after that public contract is available.
