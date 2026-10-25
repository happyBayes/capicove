# RFC-0001: Versioned Image Processing Tool Contracts

- Status: Accepted
- Date: 2026-08-12
- Last amended: 2026-08-25
- Owners: Capicove maintainers

## Summary

Capicove exposes eighteen version-1, transport-neutral image tools for background
removal, quality filtering, image-pair differences, captions, structured local OCR,
public pHash and bounded LSH search, and local or Ray perceptual deduplication. The
canonical IDs and schemas are defined in `capicove.tools.image`; Python capability
results live in `capicove.processing.image.models` and are serialized by the Tool
handlers.

The Data-Juicer names `image_deduplicator` and `ray_image_deduplicator` are reference
names only. Capicove uses `image_perceptual_deduplicator` and
`ray_image_perceptual_deduplicator` because pHash indicates visual similarity, not
byte identity.

The additive `image_phash` and `image_lsh_search` names expose pHash production and
bounded, invocation-local candidate search independently. LSH is a candidate
acceleration policy; exact Hamming verification remains authoritative.

## Contract boundary

Each Tool has its own closed input schema. Image pairs require exactly two sources,
caption generation consumes the complete output of difference-area generation, and
deduplication consumes records with stable IDs and ordered image collections.
Handlers perform permission, deadline, decoded-inline-size, and explicit path-root
checks before invoking a capability. Results use canonical `ToolResult` envelopes and
include source hashes, operation version, normalized parameters, provider/model
identity, preprocessing revision, execution device, and evidence references.

Paths and inline bytes are the version-1 source representations. Artifact references
may be added by a backward-compatible schema extension after an artifact resolver is
available at the image Tool composition root.

`capicove.image.phash` accepts bounded image records, a Hamming threshold, and an
optional `enable_lsh` boolean. It returns versioned 64-bit hashes, immutable source
digests, provider lineage, and batch-local duplicate groups. Enabling LSH changes
candidate generation only and must produce the same verified edges and connected
components as exhaustive comparison.

`capicove.image.lsh-search` accepts one query hash collection and a bounded,
caller-supplied collection of compatible hash records. It uses deterministic
`radius + 1` contiguous hash bands to generate candidates, verifies candidates by
exact Hamming distance, and returns stable distance/record-ID ordering. It accepts
no image path, writes no index, and retains no state between invocations.

`capicove.image.ocr` accepts ordered image sources and optional ordered Tesseract
language identifiers. It defaults to Simplified Chinese and English
(`chi_sim+eng`) and returns full text plus word-level pixel boxes, confidence, and
page/block/paragraph/line/word hierarchy. OCR results preserve source hashes,
effective languages, operation version, and Tesseract provider/version lineage.

## Local model and network policy

Every image Tool declares `network_access=false`. Hugging Face adapters accept only
existing local directories, set `local_files_only=true`, disable remote code, and may
receive an explicit device. Learned adapters require an immutable model revision so
a mutable local path cannot silently reuse lineage for different weights. The rembg
adapter requires a caller-provisioned local
session and never creates rembg's downloading default session. Missing dependencies
or local assets fail at invocation rather than falling back to the network.

Tesseract OCR is an external local CLI resource rather than a Python package or
downloaded model. The OCR Tool declares the `tesseract` subprocess, accepts no
caller-provided flags or output paths, and never installs the executable or trained
data. Missing `tesseract`, `chi_sim`, or `eng` resources fail as unavailable.

## Persistence and compatibility

Mappers create content-addressed artifacts without modifying sources. An existing
artifact is re-hashed before reuse. Version-1 operation results are immutable Python
dataclasses; adding optional fields is compatible, while changing field meaning,
hash algorithms, source interpretation, authorization, or required fields requires a
new contract or operation version and an RFC amendment.

Filter ranges are inclusive unless the capability explicitly defines a classifier
decision boundary. The watermark filter keeps probabilities strictly below its
configured threshold, matching its reference semantics. Probability outputs must be
within `[0, 1]`; cosine similarities must be within `[-1, 1]`.

## Ray semantics

Ray changes pHash preparation coordination only. The driver authorizes and
materializes local paths into immutable bytes before scheduling work, passes the same
configured hash provider to workers, applies bounded Ray retries and an explicit
timeout bounded by the active Tool deadline, cancels already-submitted tasks after
submission or wait failures, and performs the same
connected-component reduction and stable-winner policy as the local path.
Networked stores are outside this contract and require a separate RFC.

Local LSH tables are ephemeral invocation state, not retrieval index generations.
Persistent or cross-request LSH indexes remain outside this RFC and require the
retrieval generation, validation, activation, and lease contracts.

Local decoding, model inference, loops, and artifact commits use cooperative
deadline checks. Hard interruption is delegated to an explicitly configured
killable provider worker; timeout threads are prohibited because they cannot stop
model work safely.

## Security and privacy

Image content, associated text, and recognized OCR text are confidential by default.
Inline media and text are redacted from traces. Read and write paths are confined to
explicitly configured roots; model execution and artifact writes require separate
permissions. Sources, model weights, generated indexes, trained data, and user
content are never committed to the repository.

## References

- [ADR-0007](../adr/0007-image-processing-boundaries.md)
- [Image processing architecture](../architecture/image-processing.md)
- Accepted image processing design (archived development reference)
- Public pHash and LSH tools design (archived development reference)
