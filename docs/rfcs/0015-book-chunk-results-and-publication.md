# RFC-0015: Book Chunk Results and Wiki Publication

- Status: Accepted
- Date: 2026-09-28
- Owners: Capicove maintainers
- Review scope: New public chunk-result contract, append-only workspace format,
  and publication boundary. Acceptance does not qualify an implementation or release.
- Related contracts: [RFC-0012](0012-book-agent-task-contracts.md),
  [RFC-0013](0013-page-aware-wiki-v2.md), and
  [RFC-0014](0014-configurable-book-chunk-runs.md).

## Summary

Add an independent, model-neutral result workflow for immutable Book chunk runs.
The caller obtains bounded tasks, submits cited results, reduces accepted results
through the existing chunk-to-section-to-book dependency graph, and publishes a
complete, verified result set to a version-2 Wiki. This does not reinterpret
legacy `SummaryTask`, `SummarySubmission`, `capicove.book.submit`, or
`capicove.book.publish`. The existing `chunks(tasks=true)` output remains
reference-only; it is not silently promoted to a submission API.

## Authority and scope

`capicove.books` owns task/result validation, append-only result versions,
readiness, and publication receipts in the authorized Book workspace.
`capicove.wiki` owns version-2 Wiki validation and publication through its public
service. Source PDF bytes, parser representations, BookStructure, chunk spans,
chunk-run IDs, and citation identities remain immutable. The runtime selects the
Book workspace, Wiki root, parser assets, and permissions at startup. Tool
arguments cannot select a new root, provider, model, or publication destination.
No Tool in this RFC invokes a model or downloads an asset.

## Version-1 task and result contracts

The new `ChunkResultTask` is a versioned projection of a saved chunk run. It
retains the existing deterministic `chunk-task:<24 lowercase hex>` ID and the
four task types `chunk_summary`, `reduce_group`, `section_summary`, and
`book_summary`. It includes `schema_version=1`, `book_id`, `run_id`, `task_id`,
`task_type`, `parent_id`, ordered `dependency_task_ids`, ordered `chunk_ids`,
`citation_policy`, `citation_scope_sha256`, `max_result_bytes`, and
`submission_contract="chunk-result-v1"`. Only a `chunk_summary` task has a
nonempty `chunk_ids` list; it references exactly one content-role chunk.
Navigation-role chunks never become result tasks. Reduction fan-in remains at
most 32. Task IDs and ordering cannot change merely because a result is
submitted. `citation_policy` is `chunk_core` for leaf tasks and
`dependency_claims` for reduction tasks. The scope hash identifies the ordered
core spans or the ordered accepted dependency claim citations; the task read
endpoint returns at most 50 tasks and 1 MiB per page, rather than embedding an
unbounded citation list.

`ChunkResultSubmission` is a closed JSON object with these required fields:

| Field | Version-1 rule |
| --- | --- |
| `schema_version` | Integer `1`, not a boolean. |
| `book_id`, `run_id`, `task_id` | Exact identifiers from one authorized saved run and its task graph. |
| `idempotency_key` | Caller-owned nonblank string, at most 128 UTF-8 bytes, scoped to workspace, run, and task. |
| `dependency_result_ids` | Ordered result IDs matching every dependency task's currently accepted result; empty for `chunk_summary`. |
| `abstract` | Nonblank UTF-8 text, at most 16,384 bytes. |
| `key_points` | Zero to 32 nonblank UTF-8 strings, each at most 2,048 bytes. |
| `claims` | Zero to 64 atomic factual claims. Each contains nonblank `text` (at most 2,048 bytes) and 1 to 16 citations. |
| `provenance` | Optional closed object with installed provider ID, model revision, and caller run ID; informational only, never authority. |

Each citation contains `block_id`, zero-based `page_index`, half-open Unicode
`start_char` and `end_char`, and lowercase SHA-256 of the exact source slice.
The cited slice must resolve through the saved parser representation and match
the source and representation hashes pinned by the chunk run. A
`chunk_summary` citation must lie within that task's chunk core spans; context
spans provide reading context but not additional claim authority. A reduction
task may cite only concrete source spans carried by accepted dependency results.
It may not cite another summary as if it were source evidence. Every factual
claim requires at least one valid citation. `abstract` and `key_points` are
private working text, not independently published factual Wiki content. Any
assertion published from them must also appear as a cited `claim`.

Submission validates schema, task readiness, source hashes, exact dependency
result IDs, citation bounds and hashes, and total serialized result size of at
most 1 MiB before any durable write. The result ID is
`chunk-result:<24 lowercase hex>` derived from canonical JSON of the run ID,
task ID, version number, and accepted submission content. The first accepted
submission has version 1; a later different accepted submission appends a new
version with `supersedes_result_id`. Repeating an idempotency key with identical
canonical content returns the prior result; repeating it with different content
fails with `idempotency_conflict`. A failed validation writes no result.

## Readiness, reduction, and staleness

A task is `blocked` until every dependency has an accepted result, `ready`
when its exact dependencies are available, `complete` when it has a current
accepted result, or `stale` when a pinned dependency has been superseded. A
new dependency result never edits an old parent result; it makes that parent
and its descendants stale for the current publication view. A new submission
must pin the currently accepted dependency result IDs. Read APIs expose
versions and staleness without changing historical records. The current result
set is the unique, fully connected newest non-stale DAG; ambiguous or incomplete
sets are not publishable. Reduction is performed by caller-supplied submissions,
not by Capicove inference. Capicove validates and assembles the result graph.

## Persistent format and interruption recovery

Each run has an append-only `chunk-results/v1/` area beneath its existing
workspace directory. Accepted records are closed canonical UTF-8 JSON with a
SHA-256 sidecar and immutable sequence number. An atomic receipt index maps
task/version and scoped idempotency keys to record hashes. The workspace's
single-writer policy from ADR-0018 applies; this RFC does not introduce a
distributed queue or concurrent writers. A crash before a complete record and
receipt are durable leaves a private staging file, never a visible result. A
retry verifies and completes an orphaned accepted record or discards an
uncommitted staging file; it never overwrites a different accepted version.
Unknown versions, hash mismatches, symlinks, and missing source evidence fail
closed. No automatic migration of legacy Book submissions is provided.

Publication pins `book_id`, `run_id`, the ordered current result IDs, source and
representation hashes, Wiki schema version 2, and a deterministic rendering
policy version in a `chunk-publication:<24 lowercase hex>` identity. It renders
only citation-backed claims and their section navigation into Wiki pages and
`graph/citations.jsonl`, using the public Wiki publisher. All L0-L2 citation
checks from RFC-0013 must pass before the publication becomes visible to query.
An incomplete multi-file publication carries a pending marker and is unavailable
to Wiki query; retry with the same identity verifies existing files and completes
the receipt. A conflicting destination or changed source fails without replacing
unrelated pages. A later result set produces a new publication identity; it does
not rewrite or erase an earlier accepted publication. The publisher returns
publication ID, Wiki version, page paths, claim IDs, and source/run hashes.

## CLI and Tool surface

All new Tools have contract version 1 and use the canonical result envelope.
The CLI uses the same handlers. The startup host binds Book and Wiki roots;
`book_id`, `run_id`, task ID, and result ID are selectors, not authority.

| Tool ID | CLI operation | Side effect | Permission |
| --- | --- | --- | --- |
| `capicove.book.chunk-tasks` | `book chunk-tasks` | Read, paginated at 50 tasks and 1 MiB per response | `book:read` |
| `capicove.book.chunk-submit` | `book chunk-submit` | Append one accepted result version | `book:write` |
| `capicove.book.chunk-status` | `book chunk-status` | Read bounded progress, versions and publishability | `book:read` |
| `capicove.book.chunk-publish` | `book chunk-publish` | Publish verified Wiki projection | `book:write`, `wiki:write` |

The default read-only MCP catalog may expose `chunk-tasks` and `chunk-status`.
Submission and publication require explicit enabled Tool IDs, private writable
roots, and startup permissions. A client-supplied path cannot widen those roots.
Stable failure codes include `invalid_submission`, `citation_invalid`,
`source_changed`, `dependency_incomplete`, `dependency_stale`,
`idempotency_conflict`, `publication_incomplete`, `publication_conflict`, and
`permission_denied`. Error envelopes never echo source text or submitted prose.

## Compatibility, exclusions, and acceptance

Existing Book chunk-run bytes, `chunks(tasks=true)` responses, legacy summary
submissions, and published Wiki records retain their current meaning. The new
result area and Tools are opt-in. No automatic task execution, model selection,
network transfer, Wiki-to-Memory bridge, or mutation of source evidence is in
scope. This RFC does not declare the current PDF parser/model quality sufficient
for release.

Acceptance requires a supported local PDF completing prepare, chunk, task,
submission, reduction, and searchable page-aware Wiki publication with exact
page/block citations. Tests must independently exercise stale dependencies,
resubmission, invalid and foreign citations, changed source bytes, interrupted
record and publication writes, restart recovery, unknown versions, symlinks,
authorization, CLI/MCP parity, and installed-wheel operation on each claimed OS.
`make quality`, complete `make test`, `make build`, and `make docs-check` remain
release gates rather than consequences of accepting this RFC.

## Alternatives considered

- Reuse `SummarySubmission` v1: rejected because its section-task and citation
  allowance semantics do not encode a chunk run or dependency result versions.
- Treat `chunks(tasks=true)` as executable immediately: rejected because
  RFC-0014 explicitly labels those tasks reference-only.
- Let the host model directly update Wiki pages: rejected because it bypasses
  immutable submissions, deterministic validation, and interruption recovery.
