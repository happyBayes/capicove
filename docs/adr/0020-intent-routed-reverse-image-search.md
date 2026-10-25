# 0020: Route Personal Reverse-Image Search by Agent Intent

- Status: Proposed
- Date: 2026-08-27
- Owners: Capicove maintainers

## Context

The internal Stage 1 research vertical demonstrates deterministic content-hash,
radius-two perceptual-hash, and exact semantic fusion over an in-memory personal
corpus. It deliberately has no public contract, authorized source resolver,
persistent index generation, real provisioned model bundle, Tool/MCP surface, or
personal quality and latency qualification.

The broader reverse-image plan combines that personal path with enterprise
30-million-image adapters. Treating both as one implementation stage makes a small
local capability depend on separate storage, ANN, hardware, and evaluation risks.
The public intent, persistence, and source-authorization boundaries also require
code-owner review before production behavior can land.

No production implementation or activation is authorized by this proposed ADR.

## Decision

If accepted, Stage 2 will deliver only the offline CPU `personal-lite` profile for
at most 10,000 independent images:

- content SHA-256 equality is the exact result class and always precedes non-exact
  results;
- a completely identified 64-bit `phash-64-v1` lane searches an exact Hamming
  radius of two;
- separate instance and concept embedding spaces use exact read-only NumPy mmap
  search;
- the calling Agent declares `instance`, `concept`, or `balanced`; retrieval does
  not run a server-side intent classifier or silently change the declared intent;
- rank-only weighted reciprocal-rank fusion combines compatible lane ranks with a
  stable unit-ID tie break; and
- Python, the image `capicove.search.image` Tool, and the `capicove_search_image` MCP
  mapping expose equivalent search and failure behavior.

The existing text Tool `capicove.search` and MCP `capicove_search` retain their
independent contracts. The proposed image names require maintainer acceptance;
they do not overwrite the text handlers.

Target representations, the read-only unit catalog, every lane index, validation
report, and generation manifest bind the same ordered unit-manifest digest. A build
creates a new complete generation set through separate build, integrity-validation,
inert-publication, qualification, and atomic-activation phases. Active payloads are
never mutated. A query lease pins one complete alias snapshot through projection,
and lifecycle changes are append-only events rather than manifest rewrites.

Every query is authorized by a host-created `InvocationAuthorizationContext` that
binds workspace, actor, operation, permissions, authorization scope, privacy class,
and allowed source policy. Request fields can only narrow that context and cannot
grant or widen authority. Local and mount paths are opened descriptor-relative
beneath approved roots with no-follow semantics. Authorization, hashing, and decode
operate on the same opened object; the runtime never authorizes a path and reopens
it later.

The reverse-image Stage 2 path remains offline after explicit model and corpus
provisioning. Model bundles, source images, generated representations, catalogs,
indexes, evaluation datasets, and credentials are not committed.

The optional `enterprise-disk` profile, FAISS/MIH adapters, and any formal
30-million-image claim are a separate Stage 3 decision and delivery. Stage 2 does
not add placeholder enterprise dependencies.

## Consequences

- A personal library gets an exact, inspectable, restart-safe local implementation
  without ANN recall loss or a new enterprise native dependency.
- Intent remains observable in requests, traces, evaluation, and failures; model
  selection remains startup-owned behind private adapters.
- Corpus updates require building and activating another complete generation.
- Startup must refuse corrupt payloads and unsupported newer persistent schemas;
  supported older schemas migrate transactionally with rollback on failure.
- The authorization boundary must retain an opened handle through hash and decode,
  which requires platform-specific descriptor-safe adapters and negative race tests.
- This proposal does not supersede ADR-0008's broader one-node CPU/GPU architecture.

## Non-goals

- Keyframe extraction/search, text or video queries, and temporal assembly.
- Crop-, rotation-, homography-, or local-region-robust duplicate detection.
- A server-side intent classifier or request-selected provider/index parameters.
- Live mutation of an active generation.
- Federation, cross-node synchronization, or remote retrieval workers.
- A separate reverse-image MCP tool or an MCP/CLI index-administration surface.
- Automatic model downloads or committed media, models, indexes, or datasets.
- Enterprise FAISS/MIH implementation or a 30-million-image support claim.

## Alternatives considered

- **Implement the personal and enterprise plans together.** Rejected for Stage 2
  because enterprise ANN, storage, hardware, and full-corpus qualification are
  independent risks and are unnecessary for a 10,000-image local library.
- **Use one semantic space for every intent.** Rejected because instance and
  concept relevance have different failure modes and must remain independently
  observable and replaceable.
- **Infer intent inside retrieval.** Rejected because it adds another model,
  latency, calibration, and a less reproducible policy boundary.
- **Mutate one live index.** Rejected because partial writes, rollback, in-flight
  consistency, and audit lineage are safer with complete immutable generations.
- **Authorize canonical paths and open them later.** Rejected because replacement
  between authorization and use can redirect the read to a different object.

## References

- [RFC-0002: Versioned multimodal retrieval contracts](../rfcs/0002-multimodal-retrieval-contracts.md)
- [ADR-0008: Establish multimodal retrieval boundaries](0008-multimodal-retrieval-boundaries.md)
- Reverse-image search design (archived development reference)
- Stage 1 research vertical (archived development reference)
- Implementation plan (archived development reference)
