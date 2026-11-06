# RFC-0036: Experimental GraphRAG local and global queries over verified snapshots

- Status: Proposed (drafted 2026-10-29 under requester direction; not accepted)
- Date: 2026-10-29
- Owners: Capicove maintainers
- Decision: [ADR-0040](../adr/0040-graphrag-queries-over-verified-snapshots.md) (Proposed)
- Ledger: `docs/development/goal-status-2026-10-29.md`
- Feature status: **experimental preview**

## Authorization and status

The 2026-10-29 requester goal directed GraphRAG local and global queries with an
experimental Python API, CLI and evaluation adapters, tested with fake generators.
This draft records the contracts that implementation introduces. It is not
acceptance. It does not accept WP-6 ("Graph run lifecycle", still pending) and it
supersedes nothing in the accepted document-graph pipeline RFC. The 2026-09-02
design listed a graph query service as a non-goal; this RFC proposes lifting that
non-goal for an experimental, read-only query layer only.

## Proposal

### Snapshot pinning and verification

`GraphQueryIndex.open(catalog, workspace, gate, sources, graph_ref, scope)` reads a
published snapshot through `GraphSnapshotCatalog.get` (full revalidation),
recomputes the manifest SHA-256 and compares it with `graph_ref`, validates every
table commitment, re-reads each authorized source representation and checks every
committed text-unit segment against the source text. Any mismatch fails closed.
Every result carries:

```json
"pin": {"snapshot_id", "manifest_sha256", "graph_profile_fingerprint",
        "scope": {"workspace_ref", "scope_ref"}, "query_profile": {"id", "version", "fingerprint"}}
```

plus `result_sha256` over the canonical result.

### Local mode

1. Match query terms (casefolded, stopwords removed, light suffix stripping) to
   resolved entity names and aliases; a seed needs at least `min_entity_match` of
   its name terms. Seeds are capped by distinct name (`max_seed_entities`).
2. Expand breadth-first over `resolved_relationships` up to `max_hops` and
   `max_neighborhood_entities`.
3. Rank relationships whose endpoints are both in the neighborhood by path
   relevance: own query overlap plus the best overlap of a relationship that
   continues it through a non-seed endpoint; ties by hop distance, own overlap
   and assertion ID. Cap at `max_relationships`.
4. Return the original source quotes behind each relationship's mention evidence
   as citations, in relationship order, capped at `max_evidence`.

Native resolution merges entities only on authoritative identity keys or explicit
alias evidence, so the same name in two documents is two resolved entities. With
`bridge_same_name` (default `true`) the query layer links same-name, same-type
entities at query time and marks each bridged entity with `bridged_from`. This is a
query heuristic, not a resolution decision; homonyms can be falsely bridged.

### Global mode

Rank community reports by query-term overlap with title and statements (community
reports before overview reports on ties). Return the top `max_reports` reports
with every statement, each listing citation IDs for the source quotes of the
assertions it cites, following child-report statements up to `max_child_depth`.
Evidence is ordered by statement overlap with the question, then report rank.

### Citations

`GraphCitation`: `citation_id` (`graphcite:` + SHA-256 of snapshot, evidence,
representation and offsets), `document_ref`, `evidence_ref`,
`representation_ref`, `text_unit_id`, half-open `start_char`/`end_char`, the exact
`quote` and `page_index`. Derived entity and report text is navigation context and
is never citable.

### Generation (opt-in)

`answer_graph_result(result, answerer)` calls an explicit answerer only. Every
cited quote must be an exact substring of a returned citation; otherwise
`invalid_answer`. `KnowledgeBase.graph_query(..., answer=True)` uses the injected
answerer or the existing `generate_answer` path with the configured generation
profile (which already requires the network opt-in). Without opt-in the result is
evidence only.

### Formats and surfaces (new)

- `capicove.graph-query-profile.v1`: `profile_id`, `version` (1),
  `max_seed_entities` 5, `min_entity_match` 0.5, `max_hops` 2,
  `max_neighborhood_entities` 48, `max_relationships` 96, `max_evidence` 20,
  `max_reports` 5, `max_child_depth` 3, `bridge_same_name` true; strict closed
  document with SHA-256 fingerprint.
- `capicove.graphrag-result.v1`: the local/global result documents above.
- Python: `capicove.graph_build.graphrag` (`GraphQueryIndex`, `GraphQueryProfile`,
  `answer_graph_result`, `validate_cited_answer`, `load_graph_query_profile`),
  `capicove.graph_build.native.open_native_graph`, and
  `KnowledgeBase.graph_query(question, mode=, profile=, answer=)`.
- `build_native_graph(..., generator=, tokenizer=)` accepts an injected transport
  (used by offline tests); results now also name `graph_root`, the attempt
  directory, which the KB stores with its graph record so queries can reopen the
  snapshot. Graphs built before this change lack it and must be rebuilt
  (`graph_root_unrecorded_rebuild_required`).
- CLI: `capicove kb graph-query --config C [--mode local|global] [--profile P]
  [--answer] QUESTION`.
- Evaluation stacks `graphrag-local` and `graphrag-global` (RFC-0034 harness).

## Evidence so far

Offline, with the scripted fake generator replaying authored annotations through
the real native pipeline (extraction, resolution, Louvain communities, reports,
validation, activation), on the authored 20-document multi-hop set: see the
ledger. These numbers are deterministic engineering results on a set that was
also used to tune the local ranking; they are not model or extraction quality.

## Not proposed

- Embedding-based entity or report matching (lexical only today).
- LLM map-reduce over reports; activation or lifecycle changes (WP-6).
- Any production or real-model qualification: **not run**.

## Acceptance needed

1. The query profile and result formats and the pinning fields.
2. The same-name bridge default and its risk.
3. The `kb graph-query` CLI and `KnowledgeBase.graph_query` API.
4. The `graph_root` addition to the KB graph record.
5. A held-out multi-hop set and real-model extraction before any quality claim.
