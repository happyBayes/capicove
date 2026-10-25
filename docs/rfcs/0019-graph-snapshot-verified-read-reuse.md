# RFC-0019: Verified Read Reuse for Graph Snapshot Validation

- Status: Accepted 2026-10-03 (Phases 0, 1 and 2, plus Amendment 1).
- Date: 2026-10-03
- Owners: Capicove maintainers
- Review scope: Capability-private read paths of `GraphSnapshotCatalog`,
  `GraphTextReader`, `BuildStageStore`, and the graph Parquet table codec. This
  RFC amends one accepted clause of
  [Document Graph Pipeline Contracts](document-graph-pipeline-contracts.md)
  ("Task 20 accepted validation and authorization refinements") for the
  per-batch reauthorization call site, as accepted in Phase 2. It does not change canonical graph DTOs, processing snapshot
  formats, the private snapshot SQLite schema, or any CLI, API, or MCP surface.
- Related: [ADR: Document Graph Snapshot Boundary](../adr/document-graph-snapshot-boundary.md).

## Decision requested

Approve, in order:

1. **Phase 0.** Remove repeated type introspection and per-read SQLite
   connections from graph read paths. No observable behavior changes.
2. **Phase 1.** Let a graph table or stage artifact that was decoded once be
   reused in the same process. Reuse requires that the bytes read now have the
   same SHA-256 identity as the bytes decoded before. Bytes are still read and
   hashed on every call, and every authorization and semantic check still runs.
3. **Phase 2 (separate decision).** Let `GraphTextReader.require_access`, which
   runs before and after every embedding batch, reauthorize graph eligibility
   without re-verifying all graph artifacts. Full artifact verification stays
   at the start of the projection read and at binding time. This amends the
   accepted clause "Reads and reuse revalidate current source access and
   artifacts" for this one call site.

Phases 0 and 1 keep every accepted guarantee. Phase 2 changes when a mid-projection
file mutation is detected, not whether it is detected. It may be accepted,
deferred, or rejected independently after Phase 0 and Phase 1 are measured.

## Problem

Every `GraphSnapshotCatalog.get` re-runs full validation: the input snapshot,
all ten stage validators, all 17 graph tables, the cross-table checks, and the
baseline chain. A single graph build with retrieval projection calls `get` many
times. Each call re-decodes the same immutable Parquet parts and stage
artifacts into typed rows.

### Measurements

The reference run is `tests/integration/graph_build/test_service_retrieval.py::
test_service_success_records_inactive_retrieval_binding` under `cProfile`, on a
4-vCPU Linux VM, with one tiny document set (83 s under the profiler, 35–47 s
without it).

| Path | Calls | Cumulative time |
| --- | --- | --- |
| `GraphSnapshotCatalog.get` | 55 | 69.0 s |
| … from `GraphTextReader.require_access` (per embedding batch) | 40 | 50.8 s |
| … from `GraphTextReader.read` | 14 | 17.1 s |
| … from `GraphBuildService._publish` | 1 | 1.1 s |
| `GraphSnapshotCatalog._validate` | 56 | 60.1 s |
| `BuildStageStore`-backed `validate` (stage validator) | 560 | 40.9 s |
| `BuildStageStore.read` (recursive) | 17,385 | 21.6 s |
| `ParquetTableCodec.read` | 2,167 | 18.8 s |
| `datasets._convert` (typed row conversion) | 450,227 | 19.0 s |
| `typing.get_type_hints` (from uncached `_record_fields`) | 62,057 | 11.8 s |
| `dataclasses.asdict` | 28,893 | 12.8 s |
| `sqlite3.connect` | 20,160 | 3.2 s |
| SHA-256 of every Parquet part (`_file_sha256`) | 2,223 | **0.25 s** |
| Manifest hash check (`_verify_manifest`) | 2,223 | **0.36 s** |
| Unsafe-path scan (`rglob`) | 11,058 | **0.7 s** |

Byte-level integrity verification costs about 1.3 s in total. Almost all of the
time is spent turning the same verified bytes into the same typed rows again and
again.

The same pattern dominates the test suite. On 2026-10-03 the full sequential
suite took 1,977 s, and `tests/integration/graph_build/` accounted for about
1,390 s (70%) of it. The cost matters beyond tests: per-batch reauthorization
makes a real projection perform O(embedding batches × graph tables) full
decodes.

A reverted experiment that only memoized `_record_fields` cut the reference
test from 47.5 s to 38.1 s, without any other change.

### Root causes

- **R1.** `GraphTextReader.require_access` exists to recheck authorization
  before each embedding effect. It does so through `catalog.get`, which also
  re-decodes every artifact.
- **R2.** One `_validate` call decodes each output table twice: once in the
  stage validator's row comparison and once in `_table`.
- **R3.** `_record_fields` calls `typing.get_type_hints` for every converted
  record, although dataclass annotations do not change after class creation.
- **R4.** `BuildStageStore.read` opens a new SQLite connection and re-parses and
  re-canonicalizes JSON for every node of a recursive dependency read.

## Goals and non-goals

Goals:

- Remove repeated decoding of byte-identical, already-verified content.
- Keep authorization, revocation, scope, policy revision, and tamper detection at
  least as strict as today on every read.
- Avoid any persistent format, public contract, or dependency change.

Non-goals:

- Caching across processes or persisting validation results.
- Treating file metadata (size, mtime, inode) as proof of content identity.
- Changing host-owned `InputSnapshotValidator` or `StageArtifactValidator`
  Protocols, or the processing `validate_snapshot` signature.

## Proposed changes

### Phase 0: no-semantic-change fixes

1. Memoize `_record_fields(cls)` per record class. Dataclass fields and
   resolved type hints are fixed after class creation. Only graph contract
   classes, all of them `frozen=True`, reach this function.
2. Let a recursive `BuildStageStore.read` reuse one read transaction for the
   whole dependency closure, instead of one connection per node. Every node is
   still fetched, hashed, decoded, scope-checked, and authorized.

### Phase 1: content-addressed decode reuse

1. `ParquetTableCodec.read(path, role)` keeps its path safety checks (`_safe_path`,
   `O_NOFOLLOW`, regular-file check) on every call. It then reads the part fully
   into memory, computes its SHA-256, and decodes from that same buffer. This also
   closes today's window between hashing a file and decoding it again from disk.
2. A process-local, bounded, thread-safe LRU maps
   `(sha256, role, publication_identity, serializer_identity)` to the decoded
   immutable row tuple. A hit returns the stored rows only after the bytes just read hash
   to the key's digest. A miss decodes, runs all existing schema, metadata,
   row-count, and canonical-order checks, and stores the rows only if every check
   passed. Failures are never cached.
3. `BuildStageStore.read` gains the same reuse for parsed closed artifacts, keyed
   by the content-addressed `build-artifact:` ref. Every call still selects the
   stored payload, rehashes it against the ref, checks scope, and calls
   `gate.require_access` for the artifact's evidence refs at every depth. Only
   the JSON parse, canonical re-encode, and closed-shape checks, which are pure
   functions of the verified bytes, are reused.
4. The memo is owned by the capability's private adapters, is never persisted,
   and is bounded by the total source bytes of its entries (proposed default:
   64 MiB per process). An explicit `CAPICOVE_GRAPH_DECODE_REUSE=0` disables it
   for diagnosis.

Phase 1 relies on these facts. Graph rows are frozen dataclasses built only from
tuples and scalars, so a shared value cannot be mutated. Decoding is
deterministic for fixed bytes, role, publication identity, and
`serializer_identity`, which already pins `SERIALIZER_VERSION` and the PyArrow
version. SHA-256 collision resistance makes "same digest" equivalent to "same
bytes".

### Phase 2: reauthorization without artifact re-decoding

Add a capability-private `GraphSnapshotCatalog.require_eligible(snapshot_id,
scope)`. It performs the current-state part of `get`:

- scope and current policy revision;
- the stored row and its manifest digest;
- the state digest;
- the evidence inventory, reverse dependencies, and ineligibility tombstones;
- gate authorization of all recorded evidence;
- a revision fence.

It does not decode tables or rerun stage validators.

`GraphTextReader.require_access` calls `require_eligible`, then keeps the current
source checks (`gate.require_access` and `sources.require_access`). Projection
content remains the `GraphTextContent` produced by a fully verified
`GraphTextReader.read`. `_check_projection` still performs a full verified read
before reservation and before binding commit, and compares the graph text digest.
`get`, `active`, `read`, activation, and registration keep full verification.

### Amendment 1: manifest decode reuse

`GraphSnapshotCatalog._decode` recomputes the SHA-256 of the stored manifest bytes
on every call and checks it against the stored digest and the snapshot ID, as
today. When a manifest with that digest was already decoded and found canonical
in this process, the decoded frozen `GraphSnapshotManifest` is reused. The JSON
decode and the `manifest_bytes(manifest) == payload` re-encoding are skipped.
The scope check, state digest, evidence inventory, reverse dependencies and
ineligibility tombstones still run on every call.

Digests computed later from that same object (`_validate`'s fingerprint and
`active`'s `GraphSnapshotRef`) reuse the verified canonical bytes. The lookup is
keyed by object identity and requires the cached entry to hold that exact
object. Manifests are memoized in `GRAPH_MANIFESTS`, with the same per-process
byte bound and opt-out as the other memos.

## Integrity guarantees

| Guarantee | Today | Phases 0–1 | Phase 2 |
| --- | --- | --- | --- |
| Scope, policy revision, and evidence authorization on every read and per embedding batch | Yes | Unchanged; never cached | Unchanged; never cached |
| Ineligibility tombstones and catalog state digests re-read | Every `get` | Every `get` | Every `get` and every `require_eligible` |
| Parquet part bytes re-read and SHA-256 checked against the manifest | Every `get` | Every `get` | Every `get`, `read`, binding check |
| Stage artifact bytes re-fetched and checked against the content address | Every read | Every read | Every read |
| Cross-table, document-set, stage-commitment, and baseline validation | Every `get` | Every `get`, on reused typed rows | Same as Phases 0–1 |
| Late filesystem mutation after activation rejected by later `get`/`active` | Yes | Yes | Yes |
| File mutation during projection detected | At the next batch | At the next batch | At the next full read; at the latest before binding commit, which then fails |
| Decoded bytes are the hashed bytes | No (file reopened after hashing) | Yes (same buffer) | Yes |
| Cached state crosses process restarts | No | No | No |

Out of scope, as today: SHA-256 collisions, an attacker who can modify process
memory, and global atomicity across the filesystem, the policy service, and
SQLite.

## Test plan

Each production change starts with a failing test.

### Phase 0

- `_record_fields` returns identical results across calls, and the decoded rows of
  every role round-trip unchanged.
- A recursive stage-artifact read opens one SQLite connection, counted through a
  connection-factory spy. Tampering with any dependency payload still raises
  `stage_artifact_integrity_failure`, and revoking any evidence ref at any depth
  still raises `PermissionError`.

### Phase 1

- **Reuse:** Two reads of identical bytes decode once; a decode spy shows one
  call. Both reads return equal row tuples.
- **No metadata shortcut:** A same-size mutation of a part, with its original
  mtime restored through `os.utime`, is rejected by the next `catalog.get`.
- **Key separation:** The same bytes under a different role or publication
  identity miss the memo and fail as they do today.
- **Failures not cached:** A corrupt part fails, the restored part succeeds, and
  the corrupt part fails again.
- **Bounds:** Eviction at the byte bound, and correct results from concurrent
  readers across threads.
- **Opt-out:** `CAPICOVE_GRAPH_DECODE_REUSE=0` disables reuse and keeps behavior.
- **Authorization:** A revoked evidence ref still raises on a memo hit, through
  the snapshot catalog and through the stage store.
- **Deterministic work bound:** A projection build decodes each distinct table
  part at most once per process. Assert decode counts, not wall-clock time.

### Phase 2

- Revoking evidence between two embedding batches stops the next encode, as
  today.
- Mutating a graph table part after `read` and before binding leaves no bound
  generation. The attempt fails at `_check_projection`.
- `require_eligible` rejects tombstoned evidence, a corrupt state digest, a foreign
  scope, and policy revision drift.

### Existing guards that must pass unchanged

- `test_filesystem_mutation_after_verified_read_is_rejected_on_reuse`
- `test_cross_table_corruption_is_rejected`
- `test_catalog_tampering_fails_closed`
- `test_dataset_integrity_and_identity_are_rechecked`
- `test_external_attestations_are_exact_and_revalidated`
- `test_policy_revision_drift_aborts_publication`
- `test_stage_artifact_tampering_cannot_be_reused`
- `test_reuse_rejects_mutated_committed_payload`
- the `test_retrieval_binding.py` fencing and restart matrices
- `test_security_recovery.py` checkpoint recovery

### Acceptance

- `make quality`, `make test`, and `make build` pass. `make test-regression`
  stays under one minute.
- Record sequential wall time of `tests/integration/graph_build/` before and
  after each phase on the same host. The expected outcome is at least 3× faster
  after Phase 1. Results are recorded in the implementing pull request;
  decode-count assertions, not timings, gate CI.

## Alternatives considered

- **File-metadata validation cache (size, mtime, ctime, inode).** Rejected.
  Timestamp granularity makes same-tick rewrites invisible, and it would weaken
  the byte-level guarantee to save only about 1.3 s of hashing.
- **Persisted validation results in the snapshot SQLite catalog.** Rejected. It
  changes a persistent format and creates stale state across processes and
  policy changes.
- **Passing decoded rows from the stage validator to the catalog.** Rejected. It
  changes host-owned Protocols and couples the catalog to one validator
  implementation. Phase 1 removes the duplicate decode without crossing that
  boundary.
- **Only shrinking or skipping slow tests.** Done separately for the regression
  tier. It does not fix the O(batches × tables) cost in real builds.

## Rollout and compatibility

The phases land as separate pull requests, each with its own tests and
measurement. No migration is needed, because nothing persistent changes.
Disabling Phase 1 restores today's decode behavior. If Phase 2 is accepted, an
ADR amendment to the snapshot boundary records the changed detection point for
mid-projection mutation.

## Open questions

1. Is a 64 MiB default byte bound right for expected local graph sizes, or should
   the bound scale with the number of tables in the active manifest?
2. Should `CAPICOVE_GRAPH_DECODE_REUSE` exist at all, or is reuse unconditional
   once accepted?
3. Is Phase 2's change in detection timing acceptable, or must every embedding
   batch keep full artifact re-verification?

## Decision record

On 2026-10-03 the user approved Phases 0 and 1. They were implemented with the
proposed defaults: a 64 MiB per-process byte bound for each memo, and the
`CAPICOVE_GRAPH_DECODE_REUSE=0` diagnostic opt-out. Later the same day the user
approved Phase 2 and Amendment 1, which are now implemented. This answers open
question 3: the per-batch detection change is accepted. The
[document graph ADR](../adr/document-graph-snapshot-boundary.md) and the Task 20
section of the [document graph RFC](document-graph-pipeline-contracts.md) record
the amended guarantee.

Implementation notes:

- The stage store fetches a read's dependency closure in one short `BEGIN
  IMMEDIATE` transaction and verifies it after releasing the lock. Holding that
  write lock while host `gate.require_access` callbacks run could invert lock
  order with `_put`, which enters the policy guard before the store transaction.
- `_record_fields` memoizes per class in an explicit dict. mypy cannot type a
  `type[Any]` argument to an `lru_cache` wrapper.
- `GraphSnapshotCatalog.require_eligible` repeats `get` without `_validate`:
  it decodes the stored row, authorizes, checks the stored authorization
  revision, and decodes again inside the policy guard. `GraphTextReader.
  require_access` now calls it.
- Guards live in `tests/integration/graph_build/test_verified_read_reuse.py` (23
  tests) and `tests/unit/graph_build/test_dataset_schemas.py`. Each new behavior
  test was confirmed to fail on the previous code. The integrity guards pass on
  both versions by design.

Measured on the same 4-vCPU host, without the profiler, for the reference test
`test_service_success_records_inactive_retrieval_binding`:

| Code | Wall time |
| --- | --- |
| Before this RFC | 35–47 s |
| Phases 0 and 1 | 18.3 s |
| Phases 0–2 and Amendment 1 | 7.7 s |

That is about 4.6–6× faster, which meets the acceptance target. The graph build,
unit, contract and document pipeline suites (1,800 tests, run with
`pytest -n auto` on tmpfs) went from 390 s after Phases 0 and 1 to 188 s.
