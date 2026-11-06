# ADR-0040: Answer GraphRAG queries from verified snapshots and cite only source quotes

- Status: Proposed (2026-10-29; not accepted)
- Date: 2026-10-29
- Owners: Capicove maintainers
- RFC: [RFC-0036](../rfcs/0036-experimental-graphrag-queries.md)

## Context

The graph build publishes immutable, validated 17-table snapshots, but nothing
queried them except the KB's term-overlap ranking of a JSON dump. The accepted
pipeline RFC says derived entity and report text is not source evidence, and the
native resolution stage deliberately avoids merging same-name entities without
identity evidence.

## Proposed decision

- Queries read one snapshot through catalog revalidation, a recomputed manifest
  digest and a segment-by-segment source check, then work in memory. No graph
  database is added and no activation state changes.
- Results are pinned to snapshot ID, manifest SHA-256, graph-build profile
  fingerprint and query profile fingerprint.
- Only exact source quotes (offsets into the authorized representation) are
  citable; entities, relationships and reports are navigation context.
- Same-name bridging is an explicit, profile-controlled query heuristic, reported
  per entity, rather than a change to resolution.
- Generation is opt-in through an explicit answerer and validated against the
  returned quotes.

## Consequences

- Queries cost one full snapshot verification per open; long-lived indexes must be
  reopened after source or policy changes (the KB opens one per query).
- Lexical matching misses paraphrases; embedding matching is future work.
- Bridging can join homonyms; it can be disabled per profile.
- The KB now records `graph_root`; older KB graphs need a rebuild to be queried.
