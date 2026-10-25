# ADR-0007: Establish Image Processing Boundaries

- Status: Accepted
- Date: 2026-08-12
- Last amended: 2026-08-25
- Owners: Capicove maintainers

## Local Paddle composition amendment (2026-10-13)

[RFC-0011](../rfcs/0011-document-parser-contracts.md#local-paddle-composition-amendment-2026-10-13)
adds an optional Paddle provider using the existing capability-owned OCR port.
The operator selected this provider as the default for graph document preparation
and standalone video OCR composition. Existing Tesseract and EasyOCR adapters
remain explicitly injectable legacy implementations; no fallback selects them.
Missing Paddle assets fail at invocation. Canonical Tool argument schemas and
shared-host configuration are unchanged.

## Context

Image mapping, filtering, difference analysis, and deduplication share decoding,
metadata, cropping, score aggregation, and provenance needs. Some algorithms are
small deterministic computer-vision operations, while model inference and
distributed execution bring large optional dependencies and accelerator concerns.

## Decision

Public image operations live under `capicove.processing.image`. Immutable contracts
and pure policies remain independent from storage, model, GPU, and execution SDKs.
Pillow, NumPy, and headless OpenCV are direct runtime dependencies used by private
image implementation modules, but never by canonical `capicove.contracts`.

rembg/ONNX Runtime, Transformers/PyTorch, imagededup, and Ray remain lazy optional
adapters. Missing dependencies fail at the invoked capability boundary with an
actionable error. No adapter downloads models or enables remote code by default.

Structured OCR uses a capability-owned provider port and a private Tesseract CLI
adapter. Tesseract and its language data are explicit external resources, not core
or optional Python dependencies. The adapter sends normalized PNG bytes over
standard input, reads TSV over standard output, declares the executable in Tool
resources, and uses a fixed argument template without a shell. Arbitrary OCR flags,
configuration paths, and output paths are excluded from the public contract.

Every mapper creates a new content-addressed artifact and lineage record. Scorers
produce reusable metrics; pure filter policies produce keep decisions. Ray reuses
the local pHash and grouping semantics and changes only execution coordination.

The image deduplication capability is explicitly perceptual. Its public names are
`ImagePerceptualDeduplicator` and `RayImagePerceptualDeduplicator`; documentation
states that they filter visually similar training images and do not perform
byte-exact comparison.

Public `ImagePhash` and `ImageLshSearch` capabilities separate versioned hash
production from bounded candidate retrieval. Image processing owns deterministic,
invocation-local binary LSH candidate generation and exact Hamming verification.
LSH is never represented as an image-hash provider because it does not compute or
change pHash values. Persistent index generations and cross-request search state
remain owned by retrieval.

## Consequences

- Common image operations have predictable local behavior and shared evidence.
- Core installation includes Pillow, NumPy, and headless OpenCV.
- Heavy model and distributed dependencies do not affect unrelated imports.
- Importing Capicove does not probe or start Tesseract.
- Thresholds and model scores remain reproducible through provider lineage.
- pHash collisions and threshold transitivity are visible in duplicate groups.
- Optional LSH acceleration preserves exhaustive grouping semantics and introduces
  no persistent state or runtime dependency.

## Alternatives considered

- Port Data-Juicer operators directly: rejected because their mutable samples,
  caches, and global lifecycle do not match Capicove contracts.
- Keep all image packages optional: rejected because Pillow/OpenCV are foundational
  to the requested local image-processing surface.
- Call pHash equality exact deduplication: rejected because it misstates the visual
  semantics and intended training-dataset use case.
- Let Ray keep the first arriving record: rejected because scheduling would change
  results.
- Use random multi-table LSH: rejected because probabilistic recall could silently
  omit a near-duplicate within the requested Hamming threshold.
- Add a persistent image-processing LSH index: rejected because generation,
  activation, and query leases belong to retrieval rather than processing.
- Wrap Tesseract with `pytesseract` or `tesserocr`: rejected because the direct CLI
  adapter has a smaller dependency surface and clearer deadline/security control.

## References

- Image processing design (archived development reference)
- Image processing implementation plan (archived development reference)
- Public pHash and LSH tools design (archived development reference)
- [Module boundaries](../architecture/module-boundaries.md)
