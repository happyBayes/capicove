# RFC-0025: Native knowledge workspace

- Status: Accepted for operator-requested implementation
- Date: 2026-10-19
- Authorization: The operator requested native configuration, import/update and
  cited question-answering closure, with tests and adversarial review. This is
  implementation authority, not independent review or release approval.
- Decision: [ADR-0029](../adr/0029-native-knowledge-workspace.md).

## Decision and scope

Use a single-owner local host composition, not an external knowledge plugin.
Reuse the model factory, document preparation, Whisper runtime, compatible
structured generator, graph builder and existing HTTP/MCP host. Canonical graph
contracts and the existing ontology remain unchanged. Generic sources receive
`source` provenance; no personal subject or self-report attribution is inferred.

`capicove kb init/update/rebuild/forget/status/doctor/search/ask/evidence` provides
one lifecycle. Configuration version 1 has exactly `schema_version`,
`workspace_id`, `storage_root`, `source_roots`, `models`, `generation`, `audio`,
`graph`, and `network_enabled`. Optional configuration references are null when
not selected. Local model bindings are `knowledge.embedding`, `knowledge.ocr`,
and `knowledge.parser`; existing verified model inventories remain authoritative.
Remote generation requires both this opt-in and the generation profile opt-in.
No discovery, download, model fallback, credential persistence or telemetry occurs.

SQLite format version 1 stores workspace/source-root identity, monotonic policy
revision, active generation, immutable prepared source versions, current source
pointers, graph publication metadata and append-only attempt outcomes. Original
bytes are retained by SHA-256. Source update prepares a candidate corpus, computes
vectors, optionally builds the complete typed graph, verifies current source
hashes, and commits source pointers plus graph metadata atomically using a policy
revision compare-and-swap. Failure or partial graph results do not activate.
Rebuild reprocesses all active sources. Forget immediately removes authority and
invalidates the whole graph; immutable local history is retained, not erased.

Reads reject missing/modified sources and recheck policy around model egress and
answer acceptance. In-flight requests whose revision changes cannot return an
answer. Source aliases, relative paths, no-follow descriptor traversal, hard-link
rejection and private storage prevent callers from expanding filesystem authority.
The deployment is single owner, not a multi-tenant ACL server. Already delivered
information and in-flight network bytes cannot be recalled after revocation.

Queries fuse lexical and configured dense ranks. Optional graph entities and
community reports supply derived context. Only selected original-text chunks can
be cited. Answers contain `claims`, each with `text` and nonempty `citations`
(`evidence_id`, exact `quote`). Unknown IDs, missing citations, invented quotes,
extra fields and truncated generation fail closed. Empty claims explicitly mean
insufficient evidence. Exact-quote validation establishes traceability, not a
mathematical guarantee that a model's paraphrase is entailed by the quote.

HTTP/MCP expose query-only `capicove.knowledge.status/search/ask/evidence` via a
version-2 `knowledge` capability (`schema_version`, `config`). Configuration and
storage must be inside the host workspace; workspace identity, actor, permissions
and configured generation endpoint authority are host-owned. Mutation stays in
the operator CLI. Existing `evidence:read` and `evidence:search` permissions apply.
Question and evidence content are restricted and excluded from ordinary traces.

## Bounds and explicit behavior

At most 32 disjoint source roots, 1,000 active sources, 128 MiB per input,
2 million prepared codepoints per source and 10,000 chunks per corpus. Text
chunks preserve exact text and block/page/time coordinates. These are bounded
local-workspace defaults, not a distributed large-corpus service. Unsupported
formats and missing selected models fail explicitly. With no models, text
retrieval remains useful and responses say `evidence_only`; this mode does not
claim generated answers or semantic/graph retrieval.

Full activation is distinct from semantic extraction coverage. The graph's
coverage and issue codes remain visible; the existing fixed ontology and report
schema impose the same limitations as standalone graph building. No promise of
LightRAG feature parity is made.

## Native wire interpretations

`extraction-native-v1` leaves numeric coordinates and identifiers to the host.
Entities provide name/type/description/exact quote; relationships provide exact
endpoint names, description, polarity and quote. A unique exact match is required.
A literal uniquely quoted endpoint may declare a missing CONCEPT entity. This is
a declared interpretation of the native wire schema, not repair of rejected
legacy offsets. Neutral confidence is 0.5; no calibration claim is made. Raw
provider bytes remain the result fingerprint authority. Existing `extraction-v1`
validation remains unchanged; unknown schemas are rejected.

`report-native-v1` accepts only title and cited statements using host-issued short
aliases. The host supplies immutable task identity, long citation IDs, statement
kinds and mandatory limitations, then invokes the existing strict report validator.
This avoids asking a small model to copy operational hashes or invent offsets.
Both versions are pinned in build profiles and request fingerprints.

The `knowledge` package extra groups existing optional runtimes with PDF rendering.
Video preparation uses fixed 30-second frame OCR plus the sole audio track, with
explicit sampled coverage. Inputs with unsupported/ambiguous tracks fail. Original
video timestamps remain distinct from derived audio and frame-file identities.
