# 0016: Keep the local Wiki a replaceable knowledge projection

- Status: Accepted
- Date: 2026-09-01
- Owners: Capicove maintainers
- Approval: The user explicitly approved the previously identified Wiki projection
  boundary on 2026-09-28. Code review, merge, and release remain separate gates.

This is the accepted boundary for the local Wiki projection. Acceptance does not
assert implementation correctness or authorize merge or release.

## Context

Users need an immediately usable, offline Wiki while Temporal Memory is developed
as the persistent, automatically evolving graph authority. The reference LLM Wiki
offers useful file layout, stable identity, provenance, and linting ideas, but its
canonical-ledger assumptions conflict with Capicove's established ownership of
evidence and knowledge contracts.

## Decision

`capicove.wiki` owns a local, user-editable projection and interchange format. Its
`standalone` mode has no service dependency, but does not own canonical truth or
Capicove source evidence. Raw files are local input snapshots; Fact and Identity
Event JSONL records are reviewed interchange records until accepted by a future
knowledge or Temporal Memory public operation.

Mutating Wiki operations serialize on one operational per-Wiki lock. Durable raw
captures and pages are published create-only and never silently replace a
concurrent destination. This coordination applies to cooperating local Capicove
processes rather than arbitrary external filesystem writers.

Query and index-health checks also take that lock while validating and reading
the projection. Lock acquisition requires read/write access to its sentinel and
may clean stale operational publication files; preserving knowledge records does
not imply read-only filesystem access. A build-pending marker keeps incomplete
multi-file publication unavailable to query until a successful rebuild. This
does not promise physical atomicity or restoration of the prior projection.

Wiki owns safe filesystem access, parsing, linting, deterministic Markdown output,
and local query orchestration. PyYAML remains lazy and optional. SQLite FTS5 is a
private, disposable adapter with a versioned metadata table and content
fingerprint. Generated indexes may be deleted and rebuilt without changing Wiki
records.

Temporal Memory integration is one-way per operation and contract driven. A
projection adapter may consume a public revisioned snapshot. A submission adapter
may send explicitly selected Wiki interchange records through public memory
commands. Neither adapter accesses another capability's private implementation,
and automatic bidirectional synchronization is excluded.

## Consequences

- Capicove can support an offline Wiki without waiting for Temporal Memory,
  subject to independent implementation review and release qualification.
- A Wiki remains portable and inspectable but is not a second canonical graph.
- Evidence acceptance, automatic evolution, conflict history, and canonical graph
  identity remain outside this capability.
- Hashes and the documented schema checks support later import review. The
  version 1 validation limits and permissive extension-field behavior are
  explicit in RFC-0010; lint does not establish canonical evidence acceptance.
- Interrupted ingest operations converge on retry without replacing evidence or
  duplicating their source-identified log entry.
- A future bridge requires its own accepted contract for revision, conflict,
  authorization, and replay semantics.

## Alternatives considered

### Make Wiki ledgers canonical knowledge

Rejected because it duplicates `knowledge` and Temporal Memory ownership and
creates conflicting write authorities.

### Depend directly on Temporal Memory now

Rejected because the local Wiki must remain immediately usable and because the
memory implementation and contracts are still evolving independently.

### Use SQLite as durable Wiki truth

Rejected because it weakens portability and auditability. SQLite remains a
disposable search projection only.

## References

- [RFC-0010](../rfcs/0010-local-wiki-projection-format.md)
- [Module boundaries](../architecture/module-boundaries.md)
- Proposed design (archived development reference)
