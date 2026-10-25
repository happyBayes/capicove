# ADR: Document Graph Snapshot Boundary

- Status: Accepted
- Number: Awaiting maintainer assignment
- Owners: Capicove maintainers
- Related RFC: [Document graph contracts](../rfcs/document-graph-pipeline-contracts.md)

## Context

Document extraction, entity identity, community detection, and report generation
have different failure and reproducibility properties. Coordinated mutable writes
to graph, vector, and memory stores would couple this capability to independently
developed Temporal Memory and Wiki features. The existing processing and retrieval
contracts should remain the owners of dataset and index lifecycle operations.

## Proposed decision

1. Graph tables are immutable homogeneous `DatasetSnapshotRef` artifacts, published
   through public processing operations. A graph manifest references all 17 required
   roles and represents the complete declared current document set, not a delta.
2. A private SQLite catalog holds operational receipts, immutable manifest refs,
   eligibility, and activation history. Activation compares the expected revision
   transactionally. Partial or ineligible graphs cannot replace an active graph.
3. Model stages persist effects before deterministic consumers run. Temperature
   zero does not establish determinism. Retries, structural repair, response size,
   concurrency, and build time are bounded. No new general workflow engine is added.
4. Retrieval uses existing text targets over explicitly derived representations.
   An immutable post-publication binding links graph and index generations; there
   is no global atomic transaction or implicit retrieval activation.
5. The graph builder computes conservative alignment and a current-only analysis
   view. It does not own canonical memory identity or write Temporal Memory/Wiki.
6. Community projection excludes hypotheses and generic owner-hub edges, while
   original assertions remain available as evidence. Reports preserve citations,
   uncertainty, attribution, and counter-evidence.
7. Local, explicitly selected assets are the default. No network, model acquisition,
   remote code, telemetry, or document-triggered tool execution occurs implicitly.
   Public contracts never import optional model, graph, or storage SDKs.
8. Task 0R replaces the original mandatory community algorithm with a
   capability-local injectable `CommunityPartitioner` Protocol and an explicit
   immutable `CommunityPartitionerRegistry`. Runtime composition binds reviewed
   profile tags to adapters, with no entry-point discovery in this MVP. Unknown
   tags or missing adapters fail explicitly; workflows depend only on this port.
9. `PartitionProfile` is a closed discriminated union whose MVP tag is `louvain`:
   `LouvainPartitionProfile(resolution, threshold, max_level, seed, version)`.
   `CommunityProfile` schema/version 2 retains `max_size`, `max_depth`, and
   `support_cap` and contains `partition_profile`; it replaces the old flat seed,
   iterations, and resolution fields. Unsupported version-1 shapes are rejected,
   with new profile artifacts/fingerprints rather than in-place reinterpretation.
   Canonical graph DTOs and their schema version remain unchanged.
10. NetworkX Louvain is the MVP adapter, using optional `networkx>=3.6,<4` under
    `graph-community`. It is lazily imported only when selected. Record the actual
    NetworkX version, adapter/profile versions, parameters, stable node/edge order,
    and projection fingerprint. Identical seed/profile replay is only promised
    within the recorded NetworkX version and execution configuration, not across
    versions. Future Leiden support is a separately reviewed tagged profile and
    registry adapter, requiring no workflow change.
11. Input source eligibility is an explicit trusted-host decision through the
    capability-local `SourceProvenanceResolver`. Every current document is closedly
    classified as original source or generated derivation. The canonical result
    digest is pinned alongside its resolver fingerprint in all evidence-consuming
    stage identities. Generated input remains lineage-bearing graph data, but is
    excluded from diagnostic, community, and report evidence eligibility.

## Alternatives considered

- A mutable graph database would add operational state and cross-store consistency
  without being required by the MVP. Defer it behind future reviewed adapters.
- Direct memory/Wiki writes would entangle ownership and retention. Defer explicit
  imports to the owning capabilities and retain revisioned output contracts.
- Name-only or unconstrained vector merging is simpler but collapses homonyms and
  distinct occurrences. Keep bounded candidates, hard constraints, and abstention.
- A second general DAG framework duplicates infrastructure. Use a fixed workflow
  with explicit effectful stages and the existing public publication boundary.
- Reimplementing missing processing/retrieval/parser types locally would let an
  older branch run sooner but break canonical ownership. Wait for their integration.
- A mandatory algorithm-specific port couples hierarchy orchestration to one
  runtime. Use the closed profile and explicit registry to isolate the algorithm;
  separately review future adapters and avoid discovery machinery in this MVP.

## Consequences

Stages can replay without repeating every model call. Source evidence and rejected
or ambiguous decisions remain inspectable. Failed publication cannot silently
activate incomplete data, and removal of a document cannot resurrect old facts.

More intermediate artifacts and references require bounded retention, current
authorization checks, and explicit recovery. Storage is not free: models are
reused from authorized local installations; no duplicate worktree or model copy
is required. The user has explicitly selected in-place development on `dev`.

The accepted NetworkX choice uses its BSD-3-Clause license; preserve its copyright,
license, and disclaimer notices. The RFC cites the
[versioned license](https://github.com/networkx/networkx/blob/networkx-3.6/LICENSE.txt)
and records the optional dependency decision. Task 0R performs no dependency
installation or partitioner implementation; the later adapter task records the
resolved version/notices and adds packaging metadata. Capicove remains MIT.
The selected `dev` baseline must supply the RFC-listed public interfaces.

## Validation and acceptance

Test immutable publication, source-coordinate fidelity, scope isolation, bounded
retries, crash points, concurrent activation, partial builds, and citation closure.
Test that graph builds make no memory/Wiki writes and base imports load no optional
SDKs. Keep real model evaluation separate from structural correctness tests.
Exercise the real NetworkX Louvain adapter in the optional `graph-community`
integration matrix, including replay within the recorded version and bounded
unsplittable leaves; scripted partitioners alone do not qualify the adapter.

The user explicitly accepted this ADR and its companion RFC in this task.
Final numbering and dependency readiness remain pending. Task 0R records the
user-approved strategy and dependency revision while retaining Accepted status.
No public or persistent graph contract is implemented by this document; future
adapters remain subject to separate package-specific license review.

## Amendment: local document preparation (2026-10-13)

The operator selected Paddle document/image OCR and BGE-M3 dense embeddings.
Preparation consumes public parser/image results and retains their provenance;
the graph builder continues to consume derived exact text. The existing retrieval
bridge can consume BGE-M3 through `TextEmbeddingService` without a graph-owned
model adapter. Local BGE-M3 acquisition uses the pinned, safetensors-only runtime
policy in [RFC-0008](../rfcs/0008-unified-text-retrieval-contracts.md#local-adapter-amendment-2026-10-13).
This adds no automatic source reads, model downloads, LLM selection, or shared-host
graph Tool. Native model quality and platform qualification remain separate gates.

## Amendment: verified read reuse (RFC-0019)

Accepted 2026-10-03 by
[RFC-0019](../rfcs/0019-graph-snapshot-verified-read-reuse.md):

- Graph table rows, stage-artifact shape checks and decoded manifests may be
  reused within one process. Reuse applies only after the bytes read now hash to
  the identity under which they were first decoded. Authorization, scope,
  tombstone and cross-table checks still run on every read.
- `GraphTextReader.require_access`, the reauthorization before and after each
  embedding batch, checks eligibility and current access without re-verifying
  graph artifacts. A graph file changed during a projection is therefore
  detected by the full verified read in `_check_projection`, before reservation
  and before binding commit, not at the next batch. No generation can be
  reserved or bound for such a graph. `get`, `active` and verified reads keep
  full artifact verification.
