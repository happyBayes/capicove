# ADR-0037: Keep embedded hybrid and code search behind existing text ports

- Status: Accepted for staged implementation with RFC-0033 on 2026-10-23 (requester direction)
- Date: 2026-10-22
- Updated: 2026-10-23
- Owners: Capicove maintainers
- RFC: [RFC-0033](../rfcs/0033-local-hybrid-code-search.md)

## Context

The base workspace profile uses SQLite FTS5 and intentionally has no dense lane.
The project has public encoder, candidate and generation ports, but its FAISS
plugin is visual and its service-backed text retrievers require separate services.
Code needs explicit source selection, meaningful chunks and cited line coordinates
in addition to generic text similarity.

## Proposed decision

Add a separate embedded text backend implementing the existing text ports with
FTS5 and, in S3, bounded exact float32 vector scoring. Keep the legacy format unchanged.
Runtime owns configuration, authorization, safe source intake, factory leases and
generation composition. Retrieval owns code transformations, typed candidates,
versioned representations, compatible-space validation and fusion. Private SQLite
and safe Git enumeration adapters remain capability-owned.

Reuse existing canonical text Search and Evidence transports. Isolate code through
its own profile, corpus and index root, rather than changing authority when a
query happens to start with `code:`. Add an explicit CLI alias for discoverability.
No new canonical origin/target or raw-provider score appears in the public schema.

Code import is an explicit bounded operator action. It cannot run project code,
follow links or enable network access. Git supplies only an ignored-aware candidate
list; Capicove still authorizes and verifies every read. Keep exact source hashes
and character-to-line mapping. Notebook, LSP and editor automation are deferred.

Deliver S1 symbol/path indexes and line chunks without a model, S2 optional
build-only AST chunks, S3 optional local hybrid ranking, and S4 bounded relation
expansion. The closed startup-owned code policy selects the stage. Retain raw
identifier keys separately from split terms; independent definition lookup runs
before final top-k. AST failure retains exact-source line coverage and recorded
fallback status. Integrity/authority failures never masquerade as parser fallback.

S4 persists conservative containment/import/direct-call facts in SQLite, with
source provenance and explicit unresolved targets. A public generation-scoped
port supplies one-hop expansion under fixed filter, fanout and snippet budgets.
No graph service, arbitrary graph query language or full dispatch guarantee is
introduced. Rebuild dependent facts on source changes/deletions before atomic
activation. Later stages must demonstrate benefit on frozen evaluation inputs.

Keep weighted RRF and unit-level evidence. Treat DigUp's file aggregation,
modality calibration and relevance gates as comparison baselines, not a replacement
for accepted Capicove score semantics or calibrated abstention.

## Consequences

- Existing FTS5 installations stay model-free and migration-free.
- Only hybrid profiles need explicitly provisioned, verified local encoder assets.
- Optional parsers load at build time; missing grammars retain the S1 fallback.
- S4 requires AST dependencies and separate edge/context quality qualification.
- Exact scoring costs grow linearly; the preview cap is a bound, not a benchmark.
- Code and notes require separate configuration and generations in this release.
- Model, chunker and identifier-policy changes produce new generation identities.
- The new persistent format and source-intake rules require maintainer acceptance
  before implementation; test results cannot substitute for that decision.

## Alternatives considered

Requiring Elasticsearch/OpenSearch does not deliver embedded operation. Adding
FAISS immediately introduces a native dependency without measured need. Replacing
the existing SQLite format would increase migration and offline compatibility
risk. Automatic folder watching would introduce an unrelated write authority and
resource lifecycle. Reusing the proposed audio/unified format would unnecessarily
couple this work to another unaccepted program.

## References

- [DigUp](https://github.com/ARahim3/DigUp/tree/d6f1f480427d151640196e4d6588906e1f3ee782)
- [Code-Graph-RAG](https://github.com/vitali87/code-graph-rag/tree/main)

These are design references, not dependencies or proof of local performance.
Pin the Code-Graph-RAG revision before comparative execution. This staged revision
remains proposed; the document update does not accept RFC-0033.
