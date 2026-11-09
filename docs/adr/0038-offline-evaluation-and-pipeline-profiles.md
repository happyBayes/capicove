# ADR-0038: Evaluate offline through one harness and keep query policy in fingerprinted profiles

- Status: Proposed (2026-10-29; not accepted)
- Date: 2026-10-29
- Owners: Capicove maintainers
- RFC: [RFC-0034](../rfcs/0034-offline-retrieval-evaluation-and-pipeline-profiles.md)

## Context

Retrieval stacks had per-stack scripts with inline metrics, and the KB, offline
text and unified paths embedded RRF constants and depths in code. No path used
the reranker port. There are no real models on the development host, so every
model-dependent behavior must be exercised with labelled deterministic fakes.

## Proposed decision

- One package, `capicove.evaluation`, owns frozen dataset loading, stack
  adapters, run records, scoring (reusing `capicove.retrieval.evaluation` and
  `summarize_paired`) and paired comparison. Adapters use public runtime entry
  points and import their stack lazily.
- Run records separate deterministic results (with `results_sha256`) from timing.
  Byte-for-byte equality of results is the regression gate for default profiles.
- Query-time policy lives in closed, versioned `capicove.retrieval-pipeline-profile.v1`
  documents with SHA-256 fingerprints; defaults equal the old constants.
- Reranking is a post-fusion stage (`capicove.retrieval.rerank_stage`) behind the
  existing `TextReranker` port, disabled by default, failing or degrading
  explicitly, never silently skipped.
- Fakes live in `capicove.evaluation.fakes` and `graph_fakes`, carry
  `capicove/deterministic-fake` identities, and every record using them states
  `real_model_validation: not_run`.

## Consequences

- Changing a default profile changes a pinned fingerprint and breaks baseline
  tests, so policy changes become reviewed edits.
- Fake results can be compared and regression-tested but are not quality evidence.
- The image adapter evaluates the internal research index and imports the
  retrieval-owned OpenCV pHash adapter directly (a private module); a production
  image-host adapter needs provisioned bundles.
- Code-set runs re-verify the generation on every Evidence read, so the code pool
  is bounded to 30 and a full frozen run takes minutes.
