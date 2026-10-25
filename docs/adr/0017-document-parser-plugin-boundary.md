# 0017: Isolate document parsers behind versioned plugins

- Status: Accepted
- Date: 2026-09-02
- Owners: Capicove maintainers

## Local composition amendment (2026-10-13)

The operator's Paddle-only selection is implemented through
[RFC-0011's local composition amendment](../rfcs/0011-document-parser-contracts.md#local-paddle-composition-amendment-2026-10-13).
The optional distribution exports an image provider for the existing public OCR
port. Graph preparation consumes public parser results and image operations;
video consumes the same image provider. No capability imports another
capability's private OCR implementation. The default selection is Paddle, while
model roots remain explicit and all acquisition remains operator-only.

## Context

PDF parsing implementations can require large Python distributions, native code,
model weights, inference engines, accelerators, and network-enabled acquisition.
Those requirements cannot enter dependency-light core imports or make unrelated
Capicove capabilities unavailable.

The first book fast-reading parser will use PaddleOCR PP-StructureV3, but canonical
document, page, block, coordinate, receipt, and failure meanings must remain usable
by other local parsers. Model installation also needs a separate explicit trust
decision because PaddleOCR can otherwise acquire models during provider use.

## Decision

Core publishes standard-library-only document contracts and a runtime parser port.
Implementations register callable factories in `capicove.document_parsers`.
Discovery is side-effect free; explicit loading validates metadata and contract
compatibility; availability checking and backend construction are separate
operations.

The version-1 plugin lifecycle is:

```text
list entry points
  -> explicitly load descriptor
  -> check local availability
  -> explicitly create backend
  -> parse an authorized immutable source
```

No step before backend creation may import a parser runtime, inspect user content,
open a network connection, initialize an accelerator, or start a provider process.
Model acquisition is outside the parser call and always requires explicit user
authorization. Parser backends receive resolved local model paths rather than
relying on provider-default implicit downloads.

## Consequences

- PaddleOCR remains in a separate `capicove-processor-paddleocr` distribution.
- Core can validate plugin compatibility without installing an OCR stack.
- Other parser implementations can reuse the same public contracts.
- Missing optional packages or models produce actionable availability results
  without breaking `import capicove`.
- Plugins execute with user privileges and remain an explicit trust decision.
- Contract ranges, plugin versions, model manifests, and configuration fingerprints
  become part of parser lineage.
- Adding the entry-point group and changing its public meanings require reviewed
  governance and compatibility tests.

## Alternatives considered

### Add PaddleOCR to the core package

Rejected because it would make a large optional runtime and its transitive native
dependencies part of every Capicove installation and import boundary.

### Invoke the PaddleOCR CLI directly from core

Rejected because provider command syntax, environment behavior, implicit model
downloads, and error output would leak into canonical behavior.

### Let every source adapter return provider-specific Markdown

Rejected because Markdown loses stable page/block coordinates and makes parser
receipts, chapter reconstruction, and deterministic citation validation depend on
provider formatting.

### Download models automatically during the first parse

Rejected because network access, licenses, storage costs, cache locations, and
model revisions must be visible and explicitly authorized.

## References

- [RFC-0011](../rfcs/0011-document-parser-contracts.md)
- [ADR-0002](0002-plugin-entry-points.md)
- Approved design (archived development reference)
