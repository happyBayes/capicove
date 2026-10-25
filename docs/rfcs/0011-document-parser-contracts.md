# RFC-0011: Document Parser Contracts

- Status: Accepted
- Date: 2026-09-02
- Owners: Capicove maintainers

## Local Paddle composition amendment (2026-10-13)

The operator selected the existing locked Paddle three-model profile for Book,
PDF and OCR. The optional parser distribution also exports an image provider
implementing the public image OCR port. It uses only the verified detector and
recognizer from the same inventory; orientation and unwarping models are disabled.
Recognized lines retain exact text, line confidence and axis-aligned bounds.
Language aliases select the supported multilingual profile, never another model
download. Processing identity binds model inventory, runtime versions and policy.
CPU composition defaults `enable_mkldnn=False`, avoiding the observed Paddle 3.3.1
oneDNN attribute-conversion failure on this device. Parser plugin configuration
can explicitly opt into MKL-DNN; the effective choice enters the configuration
and processing receipts. Direct legacy backend construction retains its existing
unspecified SDK default. The locked model profile is unchanged.

Graph document preparation chooses `paddleocr` by default and accepts an already
authorized `DocumentParseRequest`. It retains the complete public parse result
and projects its exact text into a graph `TextRepresentation`, binding source
hash, page indices, representation digest and parser processing identity. Failed
pages are rejected. Images use the public image service with the same default.
Graph spans address derived text; PDF boxes remain in retained parser evidence.
The graph core, canonical DTOs and generation contracts are unchanged. Existing
book-balanced text-layer and raster-image restrictions are preserved.

Standalone video filtering and video retrieval bootstrap select PaddleOCR by
default, with an explicit local model root; `none` disables OCR. OCR-area filtering
reports recognized line-box union area, distinct from the legacy EasyOCR detector
metric. Video retrieval continues to use the shared image service over verified
keyframes. Shared-host video configuration retains RFC-0021's model-filter limits;
these standalone composition settings do not grant MCP/HTTP host authority.

## Summary

Contract version 1 represents one immutable PDF parse as ordered pages and ordered
layout blocks. It preserves exact parser text, normalized search text, zero-based
PDF page indices, optional printed page labels, normalized coordinates, parser and
model receipts, source hashes, and representation lineage.

Implementations remain plugins. Core defines their input, output, compatibility,
privacy, and failure boundaries without importing a PDF parser, OCR SDK, model
runtime, GPU framework, web framework, or MCP SDK.

## Scope

Version 1 accepts `application/pdf`, bounds parsing to an optional half-open page
range, and defines no source acquisition, model download, chapter segmentation,
summary generation, Wiki projection, or remote provider behavior.

The contract supports text-bearing and scanned PDFs through the same provider-
neutral result shape. It does not require every block type to contain text. It
does require every emitted page, block, coordinate, confidence value, receipt, and
hash to satisfy the closed validation rules below.

## Public contract

Version 1 publishes:

- `DOCUMENT_PARSER_CONTRACT_VERSION`;
- `DocumentBlockType`;
- `DocumentPageStatus`;
- `NormalizedBox`;
- `DocumentPageRange`;
- `DocumentBlock`;
- `DocumentPage`;
- `DocumentParserReceipt`;
- `DocumentParseRequest`;
- `DocumentParseResult`; and
- `DocumentParserBackend`.

The plugin boundary publishes `DocumentParserPlugin` through the
`capicove.document_parsers` entry-point group.

`DocumentBlockType` has the closed values `document_title`, `paragraph_title`,
`text`, `list`, `table`, `image`, `formula`, `header`, `footer`, `page_number`, and
`other`. `DocumentPageStatus` has the closed values `succeeded` and `failed`.

## Coordinate and ordering semantics

PDF page indices and block reading-order positions are zero-based. Page ranges are
non-empty and half-open. `NormalizedBox` coordinates are finite values in `[0, 1]`
and require `left < right` and `top < bottom`.

Pages are unique and strictly increasing within a result. Blocks are unique by
stable block ID across one parse result. Each block's page index equals its
containing page. Reading-order values are unique and increasing within a page.

PDF page indices and printed page labels are separate fields. A provider may omit
a printed page label rather than infer one. A failed page contains no blocks and
requires a bounded diagnostic; a successful page forbids an error diagnostic.

## Evidence and receipts

Requests carry an expected source SHA-256 and one authorized `MediaSource`. Results
carry the same source hash, a distinct representation SHA-256, and a parser receipt
containing plugin, profile, model-manifest, configuration, and processing
fingerprints.

Exact parser text is never overwritten by normalization. Normalized text is a
separate field used by later search and structure stages. Parser results do not
claim that normalized text is source-exact evidence.

The parser receipt makes the output reproducible and reviewable without embedding
provider configuration or model SDK values in downstream capabilities. Changing
the plugin version, profile version, model manifest, effective configuration, or
processing behavior changes its associated fingerprint.

## Plugin compatibility

Document parser plugins register callable factories through
`capicove.document_parsers`, declare a compatible contract range, and expose a
side-effect-free availability check. Listing entry points does not import plugin
targets. Loading is explicit and validates the factory, runtime protocol,
capability, and contract range.

Listing, loading, and availability checks do not parse documents, inspect source
paths, download models, open network connections, initialize accelerators, or
start provider processes. Backend construction remains a separate explicit
operation. The version-1 `parse` call does not acquire models or source data.

## Privacy and security

PDF contents, exact parser text, normalized text, coordinates, and derived
representations are confidential by default. Normal diagnostics and traces exclude
raw document text, full local paths, credentials, model tokens, and unrestricted
provider responses.

Providers receive only explicitly authorized sources and output locations. A
provider must verify the expected source hash before accepting a parse result.
Network-enabled parsing is outside version 1.

## Stable failures

Later Tool and transport adapters map document parsing failures to stable classes
for invalid input, unsupported or encrypted PDFs, source hash mismatch, unavailable
provider, unavailable model, incompatible profile, corrupt output, timeout, and
internal provider failure. Provider exception classes do not enter the canonical
contract.

## Acceptance gates

Implementation begins with failing tests for enum values, coordinate bounds,
strict integer validation, source shapes, hashes, page and block order, block
identity, page status, receipt fields, public exports, runtime ports, plugin
discovery, compatibility ranges, and optional-import isolation.

Canonical imports must not load PaddleOCR, PaddlePaddle, PaddleX, MCP, FastAPI, or a
model SDK. Completion requires fresh successful runs of `make quality`,
`make test`, and `make build`.

## References

- [ADR-0017](../adr/0017-document-parser-plugin-boundary.md)
- [RFC-0002](0002-multimodal-retrieval-contracts.md)
- [RFC-0008](0008-unified-text-retrieval-contracts.md)
- Approved design (archived development reference)
