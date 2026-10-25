# 0015: Keep temporal memory projections with their authoritative SQLite ledger

- Status: Accepted
- Date: 2026-09-17
- Owners: Capicove maintainers
- Related proposal: [RFC-0009](../rfcs/0009-temporal-memory-contracts.md)
- Approval: The user explicitly approved the previously identified Memory storage
  decision on 2026-09-28. Code review, merge, and release remain separate gates.

## Context

PR #46 placed encrypted source bodies and live projections in a second memory
database, rebuilt them during reads, and capped the lifetime at 1,000 operations.
That implementation violated source ownership and the reviewer-referenced development
plan's transactional and temporal invariants. The reviewer requested a new branch
from current `dev` and reimplementation against the original design.

## Decision

`memory.sqlite3` is the sole memory authority and live projection database. It
contains workspace heads, operations and hash-chained events, entity/episode/fact/
policy projections, evidence links, conflict state, leases/outbox, projection
revision, and persistent FTS. `BEGIN IMMEDIATE`, WAL, full synchronization, foreign
keys, compare-and-swap revisions, and scoped idempotency receipts protect atomic
commits. One writer connection and ten bounded read-only WAL connections prevent
long searches from holding the writer mutex. SQLite 3.42 or later is required for
secure deletion from FTS segments.

PR #49 repair schema v2 adds indexed integer `valid_from`/`valid_to` columns and
backfills them from v1 projection bodies in a transaction. Both world and knowledge
axes are hard SQL filters before lane limits, including typed graph neighbors.
The current-assertion scan used by entity merges is explicitly named separately.
Existing events, operation receipts and their hashes are not rewritten. A v1
binary rejects v2; rollback requires a consistent pre-upgrade backup. Unknown
formats and the rejected PR #46 store remain unsupported.

`evidence.sqlite3` is owned by Source and contains immutable canonical text.
It is not a second live memory projection. Hashes are computed from UTF-8 content;
spans refer to Unicode character offsets and are verified through Source's public
port. Memory payloads never contain a second copy of the source excerpt. FTS is
explicitly derived text and participates in erasure.

The Source privacy authority (`privacy-authority/erased-scopes.sqlite3`) is a
small, separate deny register, not another ledger or projection. Its identity is
bound to the Source database. Reads and derived writes hold authorization guards;
erasure durably denies the scope before payload/index scrubbing. Source backup
restoration must retain the current deny register. Missing/replaced authority
fails closed. Restoring both the content and an old privacy authority can defeat
this model and is prohibited operationally; protection against a malicious host
administrator is outside the local-owner threat boundary.

Normal corrections are append-only. Explicit privacy erasure redacts scoped
payloads/receipts and clears scoped projections, FTS, and Source bodies while
retaining cryptographic audit commitments and the erasure tombstone. A failed
checkpoint reports a retryable incomplete erasure; retrying the same key repeats
the checkpoint. Filesystem snapshots and storage-device remanence require the
host's separate retention/erasure policy.

Startup verifies the complete ledger, payload/receipt hashes, operation ranges,
and workspace heads. External commits trigger verification before reuse. A broken
chain enters `read_only_safe`; `doctor` reports it and new writes are denied.
No automatic rebuild or extraction occurs on reads. An explicit rebuild replays
verified events and reads source text through the Evidence port; unavailable
Evidence rolls back the rebuild. A signed/mirrored external root is not provided,
so a fully privileged attacker rewriting an entire database and all hashes is
outside the integrity guarantee.

## Consequences and boundaries

- Source acceptance can leave an unreferenced immutable blob if a later memory
  transaction fails. No accepted fact may reference missing Evidence. Orphan
  collection is a future Source retention operation, not memory rollback.
- Existing media/search contracts and commands remain intact. The common RRF
  rule is standard-library-only so memory does not import image/model providers.
- This is local SQLite composition; runtime selection of `neo4j` returns
  `backend_unavailable`, with no silent SQLite fallback. Neo4j remains design
  delivery stage 6 and needs its own implementation and contract suite.
- Local files are private (0700 directories/0600 databases), not application-level
  encrypted. Use the host's disk encryption and restrict local-user access.
- Corrupt or unknown formats are not repaired or migrated automatically. The
  rejected MVP format requires separate reviewed migration tooling if ever needed.
- This acceptance does not replace independent implementation review, merge
  authorization, or release qualification.
