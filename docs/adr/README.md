# Architecture Decision Records

Architecture Decision Records capture long-lived decisions and their trade-offs.
They complement RFC discussion: an RFC proposes a material change; an ADR records
the accepted outcome.

Use sequential filenames:

```text
NNNN-short-kebab-case-title.md
```

Each ADR contains:

```markdown
# NNNN: Decision title

- Status: Proposed | Accepted | Superseded | Rejected
- Date: YYYY-MM-DD
- Owners: GitHub users or teams

## Context
## Decision
## Consequences
## Alternatives considered
## References
```

Do not rewrite an accepted decision to change its meaning. Add a new ADR and mark
the earlier record as superseded.

## Accepted decisions

- [0001: Start as a Python modular monolith](0001-python-modular-monolith.md)
- [0002: Discover plugins through Python entry points](0002-plugin-entry-points.md)
- [0003: Unify Tool Runtime, Trace, and controlled RSI boundaries](0003-tool-runtime-trace-rsi-boundaries.md)
- [0004: Declare Tool resource requirements](0004-tool-resource-requirements.md)
- [0005: Make video split outputs accurate by default](0005-video-split-output-accuracy.md)
- [0006: Provide one unified local video filter](0006-unified-local-video-filter.md)
- [0007: Establish image processing boundaries](0007-image-processing-boundaries.md)
- [0008: Establish multimodal retrieval boundaries](0008-multimodal-retrieval-boundaries.md)
- [0009: Validate and preserve immutable audio evidence](0009-immutable-audio-evidence-intake.md)
- [0010: Provide local offline audio transcription](0010-local-offline-audio-transcription.md)
- [0011: Provide sample-accurate local audio clipping](0011-sample-accurate-local-audio-clipping.md)
- [0012: Establish Agent trace recording and export boundaries](0012-agent-trace-recording-export-boundaries.md)
- [0013: Isolate Spark execution behind committed dataset snapshots](0013-spark-execution-and-snapshot-boundaries.md)
- [0014: Isolate Elasticsearch behind unified text retrieval contracts](0014-elasticsearch-text-retrieval-boundaries.md)
- [0015: Keep temporal memory projections with their authoritative SQLite ledger](0015-temporal-memory-sqlite-boundaries.md)
- [0016: Keep the local Wiki a replaceable knowledge projection](0016-local-wiki-projection-boundaries.md)
- [0017: Isolate document parsers behind versioned plugins](0017-document-parser-plugin-boundary.md)
- [0018: Use an immutable local workspace for book workflows](0018-local-book-workspace.md)
- [0019: Keep configurable book chunks independent from source evidence](0019-independent-configurable-book-chunks.md)
- [0021: Keep Book chunk results append-only and publish verified projections](0021-append-only-book-chunk-results.md)
- [0022: Separate durable operator jobs from MCP host lifetime](0022-durable-operator-jobs-and-admin-isolation.md)
- [0023: Compose capabilities from one versioned workspace authority](0023-versioned-workspace-host-authority.md)
- [0024: Serve offline workspace Search and Evidence from verified FTS5 generations](0024-offline-workspace-retrieval.md)
- [0025: Serve workspace Tools over loopback HTTP through an isolated worker](0025-workspace-loopback-http.md)
- [0026: Stage host media through verified private copies](0026-host-media-handoff.md)
- [0027: Manage local models through typed factory leases](0027-unified-model-factory.md)

- [0028: Explicit deployment and compatible generation](0028-explicit-deployment-and-compatible-generation.md)
- [0029: Native knowledge workspace](0029-native-knowledge-workspace.md)

## Proposed decisions

- [0020: Route personal reverse-image search by Agent intent](0020-intent-routed-reverse-image-search.md)
