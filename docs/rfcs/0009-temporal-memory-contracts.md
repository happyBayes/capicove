# RFC-0009: Evidence-backed temporal memory contracts

- Status: Accepted
- Date: 2026-09-17
- Owners: Capicove maintainers
- Review context: PR #46 (archived development review)
- Approval: The user explicitly approved the previously identified Memory contract
  decision on 2026-09-28. Code review, merge, and release remain separate gates.

## Scope and authority

This contract covers the SQLite portions of the
temporal graph memory design (archived development reference).
It replaces the rejected Episode-spine implementation. Acceptance of this public
contract and persistent format does not waive implementation review or release
gates. The separate Neo4j plugin and real-model benchmark campaigns are outside
this correction's delivery claim.

## Ownership and contracts

- `source` owns immutable UTF-8 text identified by `evidence:<sha256>`, scoped
  authorization, integrity checking, and half-open Unicode character spans.
- `knowledge` owns Episodes, entity identity and aliases, ontology validation,
  typed facts, conflicts, the event ledger, and temporal queries.
- `agent` owns admission, sensitivity, retention, pinning, diversity, and cited
  context budgets. A rejected or disputed fact cannot be admitted.
- `contracts` contains standard-library-only immutable values and the normalized
  `rrf-v1` scoring rule shared with existing text retrieval. No provider is
  imported through these contracts.
- CLI and MCP call canonical `tools`; host configuration owns paths, workspace,
  authorized scopes, ontology, and optional providers.

`MemoryEpisode` has an independent ID, type (`message`, `document`, `json`, or
`fact_assertion`), Evidence references/hash, event time, recorded time, source
reference, and schema version. It contains no source body. `TemporalFact` has
independent fact/version IDs, typed subject/object, assertion/slot hashes, ontology
version, Evidence spans, authority, separate extraction confidence, state, and
parent/superseding version references. Objects are entity references or string,
boolean, int64, canonical decimal, ISO date, or UTC datetime literals.

Entity names use NFKC plus casefold for scoped alias resolution. Explicit entity
merges append redirects and reconcile affected slots; identities are not inferred
from substring equality. Predicates declare subject/object types, `one`/`many`
cardinality, and `static`/`interval` temporal behavior. Authority has five closed
levels: heuristic, model, processor, signed source, and explicit user.

## Temporal and conflict rules

Intervals are half-open. World validity is independent of observation time and
knowledge recording time. Same-assertion writes reuse a fact ID and merge support.
Many-valued predicates and non-overlapping intervals coexist. A later explicit
world start clips the older interval irrespective of ingestion order; historical
backfill cannot displace a later value. Unordered overlapping equal-authority
contradictions become disputed and leave default answers. Higher authority wins;
rejected versions remain auditable. Explicit corrections append retraction and
replacement events and close knowledge intervals; they do not edit old payloads.

The initial conflict implementation classifies each assertion interval as a unit;
it does not split a same-start, partially overlapping dispute into independent
accepted/disputed segments. Such queries conservatively omit the disputed
assertion. This is a reviewable limitation, not an accuracy claim.

## Operations and query policy

Public Tools are `remember_episode`, `assert_fact`, `recall`, `get_episode`,
`get_fact_history`, `retry_extraction`, `archive_memory`, `erase_scope`, and `doctor`.
PR #49 repairs add bounded `work_extractions` and explicit `resolve_entity` through
the same Agent/Tool boundary, with the same scoped write permission.
There is no public extracted-fact insertion or individual-fact erasure Tool.

`MemoryQuery` carries workspace/scope, both time axes, types/states, center entities,
top-k, profile, policy, and minimum committed revision. Scope, authority, retention,
state, and time filters run before candidate budgets. The five lanes use weights
1.0 / 0.8 / 0.7 / 0.3 / 0.3 and candidate limits 100 / 100 / 100 / 50 / 50.
Graph expansion uses typed entity edges, at most two hops and 1,000 visited nodes.
Plain English questions use up to 64 unique content terms joined with OR in
persistent word-token FTS/BM25, excluding grammatical question/function words.
Recognized English questions may include accented Latin names. The planner
normalizes a query copy to NFC and uses the existing word index's tokenization;
it does not rewrite source evidence or migrate indexes. Unsupported scripts,
symbols and combining marks remaining after NFC normalization retain the literal
query path; decomposed Latin sequences that NFC can compose use the word index.
An all-function-word question returns no lexical matches, never a browse result.
Terse keyword queries retain conjunctive matching. CJK and symbol queries use
literal substring matching on persistent trigram indexes. Stable final ordering is
normalized RRF descending, valid start descending (unknown last), then target ID.

The optional dense callback supplies at most 100 target IDs; knowledge rechecks
all filters. No model or network access is enabled by default. Missing extractor
warnings apply to deferred extraction, not every successful recall. Context
budgets include serialized citations/times and use UTF-8 byte counts as a
conservative model-independent token estimate.

Empty queries explicitly browse rather than pretend to run lexical matching.
For text queries, utility/recency only rerank relevant candidates. The configured
dense bridge reuses `TextSearchService` with a dense-only profile and scoped corpus;
host-owned index publication remains separate. Semantic context contains the
actual cited source spans, shortened with matching offsets to fit budgets.
Today's archive/expiry policy applies even to historical knowledge queries.

Every recall lane requires a persisted admission for the same workspace, scope,
target and kind before its candidate limit. Current policy means the most recently
committed admission (ledger-assigned `recorded`, with memory ID as a deterministic
tie-breaker). Check that record's archive state and retention deadline; do not
fall back to an older active record when the latest is archived or expired.
A new admission can supersede a historical archive/expiry decision for recall;
archiving an older memory ID does not archive the new admission. Utility ranking
uses the latest admission's importance, not the historical maximum. Source erasure
still denies the whole scope and cannot be overridden by re-admission.

Base content and admission remain separate commits. Content with no completed
admission is unavailable to recall, including provider-supplied dense IDs. Retrying
the same Agent request reuses its base receipt and completes/reuses admission
without duplicating base events. Direct authorized Episode/history audit reads
remain available independently of recall eligibility.

Absent validity starts are negative infinity and participate in clipping.
Static predicates reject all finite bounds before either cardinality is resolved;
distinct `many/static` objects coexist, while `one/static` conflicts use authority
or dispute because no temporal ordering exists. Similar names require a persisted
pending resolution decision; explicit aliases/distinct entities resolve it.

## Extraction and admission

Evidence acceptance precedes an atomic base Episode/index/outbox commit. Injected
`EntityFactExtractor` work runs afterward. Fingerprints include Episode, extractor
revision, ontology, and normalization revision. Leases, attempt counts, retry
scheduling, stale-token rejection, partial completion, and dead letters are durable.
Changing provider revision permits a new extraction. Failed extraction preserves
searchable Episodes. Provider messages are not copied into error records.

Only accepted, accessible, evidence-backed facts reach Agent admission. Extracted
facts inherit the Episode's sensitivity, retention deadline, and pin policy.
Restricted admission is denied even when pinned. Archive affects recall, while
history remains available for audit. Scope erasure is irreversible for that scope;
new ingestion needs a new authorized scope.

## Compatibility and validation

See [ADR-0015](../adr/0015-temporal-memory-sqlite-boundaries.md) for storage,
restoration, and erasure boundaries, and the
review evidence (internal record; public support status is documented in the capability guide) for
regression and performance results. The rejected PR #46 database format is not
migrated or opened as this format. No existing CLI command or Tool schema changes.
