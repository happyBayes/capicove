# 0021: Keep Book chunk results append-only and publish verified projections

- Status: Accepted
- Date: 2026-09-28
- Owners: Capicove maintainers

## Context

RFC-0014 saves immutable, reference-only chunk task graphs. Legacy Book summary
submissions neither pin chunk-run dependencies nor express a complete result DAG.
Publishing directly from a model response would bypass citation and recovery
checks.

## Decision

Book will own a new versioned, append-only chunk-result
ledger inside each authorized run. A result pins its exact task, source evidence,
and current dependency result IDs. Replacing a result appends a version and makes
dependent results stale; historical records are never edited. A publication pins
one complete non-stale DAG and emits citation-backed claims through the public
version-2 Wiki service. An incomplete publication is not query-visible.

This decision does not reinterpret legacy `SummarySubmission` or automatically
execute chunk tasks. The Book workspace remains single-writer under ADR-0018.

## Consequences

- Replay and audit can distinguish old result versions from the current
  publication view without altering source evidence.
- Reduction callers must resubmit results after a dependency changes.
- Book needs bounded task projection, idempotent append receipts, and crash
  recovery; Wiki needs an atomic visibility boundary for multi-file output.
- A model or client cannot make unsupported claims publishable by supplying
  prose without valid source citations.

## Alternatives considered

- Extend legacy summary submissions: rejected because their task and dependency
  identities do not describe a chunk run.
- Mutate a current-result row and update Wiki in place: rejected because it
  loses lineage and makes interrupted publication ambiguous.

## References

- [RFC-0015: Book chunk results and Wiki publication](../rfcs/0015-book-chunk-results-and-publication.md)
- [ADR-0018: Use an immutable local workspace for book workflows](0018-local-book-workspace.md)
