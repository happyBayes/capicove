# RFC: Document Graph Pipeline Contracts

- Status: Accepted
- Number: Awaiting maintainer assignment
- Owners: Capicove maintainers
- Scope: Contract and architecture approval; dependency readiness remains separate
- Approval: The user explicitly accepted both drafts and requested continued development.

## Summary

Introduce `capicove.graph_build` for authorized document normalization, chunking,
mention extraction, conservative entity resolution, cross-snapshot alignment,
personal-growth diagnostics, hierarchical communities, and cited reports.
Reuse public processing and retrieval surfaces. Do not introduce a graph database,
write Temporal Memory or Wiki, or acquire documents or models implicitly.

This RFC proposes the contract ledger in the
[public contract ledger](../architecture/document-graph-contract-ledger.md)
and the invariants below together. Acceptance must cover both; the ledger is not
permission for downstream tasks to add unreviewed public fields.
The [public contract ledger](../architecture/document-graph-contract-ledger.md)
supplies the algorithm and qualification requirements.

## Dependency readiness on the selected dev baseline

The initial inspected local `dev` baseline was commit `619761e`. It did not contain the
following prerequisites that existed in the earlier planning workspace:

| Prerequisite | Required integration | Current disposition |
| --- | --- | --- |
| RFC-0007, `DatasetSnapshotRef`, `DatasetSchema`, `LocalProcessingWorkspace` | Graph manifest contracts and homogeneous dataset publication | Available after authorized fast-forward to `42175bd` |
| RFC-0008, `TextChunkLocator`, text generation lifecycle | Source coordinates and derived text retrieval | Available after authorized fast-forward to `42175bd` |
| RFC-0011, `DocumentParseResult`, `DocumentParserBackend` | PDF representation input | Public contract landed; local preparation added 2026-10-13; full source-aware locator adaptation deferred |

RFC-0008 and RFC-0011 were inspected read-only on local `codex/book-parse`; their
Accepted status there does not make their implementations available on `dev`.
No dependency branch is merged or cherry-picked by this RFC. Do not duplicate
these types, import code from another checkout, or create temporary incompatible
substitutes. Recheck the actual landed public interfaces before Task 1. Text-only
delivery can defer the PDF adapter, but cannot remove mandatory publication or
text-coordinate dependencies or claim full MVP completion.

## Canonical surface and compatibility

Publish frozen, slotted standard-library values under
`capicove.contracts.graph_build`, organized as follows:

| Module | Public values |
| --- | --- |
| `evidence` | `GraphScope`, `SourceSpan`, `InputSegment`, `TextUnit` |
| `extraction` | `RelativeSpan`, `EntityMention`, `RelationshipMention`, `ValidatedExtraction` |
| `graph` | `IdentityKey`, `Entity`, `PairDecision`, `EntityAlignment`, `RelationshipAssertion`, `RelationshipAggregate`, `EntityMembership`, `ResolutionMembership` |
| `analysis` | `EvidenceBundle`, `DiagnosticFinding`, `Community`, `CommunityMembership`, `ReportStatement`, `CommunityReport` |
| `snapshots` | `TableRef`, `StageResult`, `GraphSnapshotManifest`, `GraphSnapshotRef`, `GraphRetrievalBinding` |

Fields and their ownership are the linked contract ledger. Sequences are immutable.
Opaque evidence references are nonempty strings, not a replacement source-store
contract. Reuse `DatasetSnapshotRef`, `TextChunkLocator`, and `EmbeddingSpaceRef`.
No provider SDK object or unrestricted configuration dictionary enters these values.

The initial graph schema version is 1. JSON decoders reject unknown keys, invalid
enums, nonfinite numbers, booleans used as integers, malformed hashes, excessive
nesting, and invalid references. Optional absence is explicit `None`/JSON null.
Canonical JSON sorts string keys, uses compact UTF-8 encoding, and forbids NaN.
Hashing includes scope, semantic versions, and actual dependencies; operational
timestamps and retry counters do not enter immutable content identity.

Changing field meaning, coordinate units, identity rules, or persistent schemas
requires versioned review. Readers reject unsupported versions rather than guess.
Immutable records are not upgraded in place. A new build or explicit migration
creates new artifacts and lineage; existing retrieval target meanings are unchanged.

## Coordinates, provenance, and cache binding

All text ranges are nonempty, half-open Unicode code-point ranges into an immutable
representation. For the synthetic text `A\u4e2d🙂B`, `[1, 3)` selects `\u4e2d🙂`; these are
not UTF-8 byte offsets. Negative, reversed, boolean, or out-of-bounds ranges fail.

`SourceSpan` has evidence/representation refs and absolute start/end coordinates.
PDF block ID and zero-based page index must be supplied together when present.
A PDF block-span index maps representation coordinates back to exact parser text.
For parser page index 0, an explicitly OCR-origin retrieval locator uses page 1;
a document-origin locator has no page field. Do not infer OCR origin from PDF type.

TextUnit cores do not overlap. Preceding context, title paths, and repeated headers
are separate source-mapped segments. A context anchor may resolve a relationship
endpoint, but every accepted assertion needs core evidence. Occurrence IDs include
scope, document, representation, locator, and chunking policy, not text alone.

Cached extraction uses segment-relative spans and local IDs. Every reuse checks
current authorization, exact quotes, bounds, and endpoint references, then binds
to current occurrence IDs. Repeated text in a different document never inherits an
old document's absolute coordinates or evidence references.

## Ontology, assertions, and identity authority

Use the design's closed `personal-growth-v1` three-layer ontology: base objects,
learning/work events, and diagnostic objects. Preserve epistemic status
(`explicit`, `inferred`, `hypothesized`, `disputed`), polarity, attribution,
extraction confidence, original evidence, and derivation separately. Explicit
means source-stated, not independently verified. Confidence is not truth.

Quoted first-person material is not automatically about the workspace owner.
Generated findings cannot become independent evidence through repetition.
Diagnostic synthesis retains supporting and opposing assertions, limitations,
and a proposed verification step; suggested interventions are never executed.

Input source status is supplied only by a trusted capability-local
`SourceProvenanceResolver`. Its closed `DocumentSourceProvenance` result classifies
every current document as `source` with no derivation, or `generated` with the
exact producing reference. Non-empty builds without a complete canonical
classification fail closed. The resolver fingerprint and digest of the exact
classification rows are pinned into the run, aggregation, diagnostics,
communities, reports, and their model-request identities. Reuse reauthorizes the
current document inventory and rejects resolver, policy, or classification drift.
Assertions from generated input preserve `derivation_ref` in their stable identity;
they remain inspectable but cannot qualify as diagnostic, community, report, or
independent source evidence.

Strong compatible identity keys may establish identity; conflicting authoritative
keys prohibit merging. Names, embeddings, and shared context generate candidates,
not identity proof. Events retain occurrence identity. Fuzzy auto-merge requires
a qualified policy; otherwise abstain. Cluster-level constraints prevent unsafe
transitive merges. Support counts deduplicate source and event occurrences.

Cross-snapshot alignment pins one complete compatible baseline in the same scope.
Decisions are `matched`, `distinct`, or `ambiguous`. Low similarity alone does not
justify `distinct`. A first build yields successful `no_baseline` and an empty
alignment table. The current analysis view does not import removed-source facts.
All graph identities are local or derived: Temporal Memory alone owns canonical
memory identity, historical merge/split governance, and temporal fact policy.

## Persistence, reports, and activation

The manifest references exactly these 17 homogeneous table roles:

```text
documents, text_units, entity_mentions, relationship_mentions,
entities, entity_memberships, relationship_assertions, relationship_aggregates,
entity_alignments, resolved_entities, resolved_relationships,
resolution_memberships, diagnostic_findings, communities,
community_memberships, community_reports, operation_diagnostics
```

Empty schema-valid tables are allowed; missing mandatory tables are not. Publish
real typed dataset artifacts through the public processing workspace, then validate
the complete graph manifest. A private SQLite catalog records receipts, manifest
references, eligibility, and transactional compare-and-set activation.

Task 19 adds `LocalProcessingWorkspace.validate_snapshot(snapshot_id, *, validator)`
for verified reuse by exact committed identity. It returns the catalog snapshot
or `None` for an unknown ID; unsafe paths, manifest/part drift, or rejected bounded
local validation raise `ProcessingRecoveryError`. Processing owns URI, manifest,
and part-integrity verification before invoking the graph validator on the verified
directory. Graph code never reads processing-private manifest layouts. Callers hold
the public execution lock when coordinating validation with publication; storage
integrity does not replace current source authorization. No persistent format changes.

Run status, completeness, and activation are independent. Partial graphs remain
inspectable but cannot replace the active graph. Required extraction, resolution,
alignment, diagnostic, community, report, and graph embedding stages must finish
successfully for completeness. Ordinary identity abstention is not a stage failure.
No processing publication lock is held during model calls. Crash recovery reuses
committed outputs without claiming exactly-once model execution.

Community detection consumes explicit positive non-disputed associations, excluding
generated hypotheses and generic owner-participation hub edges. The evidence graph
retains those excluded associations. Hierarchical community discovery is bounded,
versioned, and records unsplittable oversized leaves. Reports preserve statement-level
citation paths through child reports to original assertions. Counter-evidence and
omitted coverage are retained. Structural citation checks do not prove semantic entailment.

### Community strategy revision (Task 0R)

The user-approved revision replaces the original mandatory community algorithm
with a capability-local injectable `CommunityPartitioner` Protocol and an explicit
immutable `CommunityPartitionerRegistry`. The registry binds reviewed profile tags
to injected adapters; unknown tags and missing adapters fail explicitly. There is
no entry-point discovery in this MVP. Provider-native graph objects never cross
the port: it consumes the capability-owned `WeightedGraph` and `PartitionProfile`.

`PartitionProfile` is a closed discriminated union. The MVP tag is `louvain`, with
`LouvainPartitionProfile(resolution, threshold, max_level, seed, version)`.
The tag is fixed by the variant, not caller-selected arbitrary provider kwargs.
`CommunityProfile` advances from schema/version 1 to 2, retains `max_size`,
`max_depth`, and `support_cap`, and contains `partition_profile`. Seed and
algorithm parameters move into that nested profile; the old `iterations` field
is removed. Reject unsupported versions and unknown fields rather than silently
reinterpret version-1 profiles. New profile artifacts and fingerprints preserve
lineage; canonical graph DTOs and graph schema version 1 do not change.

The MVP adapter is NetworkX Louvain, lazily imported only when explicitly selected.
Record the actual NetworkX version, adapter/profile versions, all partition
parameters, stable node/edge ordering, and projection fingerprint. Identical
seed/profile replay is only promised within the recorded NetworkX version and
execution configuration, not across versions. Hierarchy depth and leaf-size bounds
are independent of Louvain's internal `max_level`. Future Leiden support requires
a separately reviewed tagged profile and registry adapter; it requires no workflow
change and adds no dynamic discovery.

## Providers, retrieval, and security

Capability-owned ports isolate authorized representation reads, extraction, pair
adjudication, diagnostic synthesis, community detection, reporting, and receipts.
Model stages are persisted effects, not deterministic planner nodes. Requests have
finite response, retry, concurrency, and time budgets. Documents have no tool access
and cannot select a model, change a profile, or expand authorization.

A public retrieval batch-text embedding service owns compatible local BGE encoding;
graph code consumes an injected narrow port. Reuse verified local assets, not
another capability's private adapter. Pin tokenizer, model revision, input role,
dimension, normalization, and metric. No silent truncation or remote fallback.

Descriptions and reports are explicitly derived text representations with lineage,
not original source evidence or new native graph retrieval target types. A separate
immutable `GraphRetrievalBinding` links an already-published graph with validated
retrieval generations, avoiding circular hashes. Binding does not activate the
global retrieval index or promise a cross-store transaction.

Every read, including a cache hit, enforces current scope and source availability.
Hashes are integrity identifiers, not access tokens. Revocation invalidates affected
eligibility; authorized erasure remains applicable. Protected artifacts include
vectors and raw responses. Logs exclude source text, full paths, credentials, and
identity keys. No network, telemetry, model acquisition, or memory/Wiki writes are
enabled implicitly. Optional runtimes are lazily imported.

## Dependency license decision

The repository license remains MIT. Task 0R records the accepted MVP optional
dependency `networkx>=3.6,<4` under the `graph-community` extra, with NetworkX's
BSD-3-Clause license. See the versioned
[NetworkX license](https://github.com/networkx/networkx/blob/networkx-3.6/LICENSE.txt)
and [Louvain API](https://networkx.org/documentation/3.6/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html).
Retain upstream copyright, license, and disclaimer notices when distributing it.
The later adapter task records the resolved package version and distribution
notices and adds the optional extra/lock entry; Task 0R installs no dependency and
implements no partitioner. NetworkX remains a lazy optional import, never a
canonical-contract or base-import dependency. Future adapters require their own
package and license review; this is not blanket redistribution clearance.

Task 17 resolves the explicit `graph-community` dependency to NetworkX 3.6.1 in
`uv.lock` and qualifies the installed 3.6.1 runtime without downloads. Its complete
BSD-3-Clause copyright, conditions, and disclaimer are retained in the packaged
`src/capicove/graph_build/NETWORKX_LICENSE.txt`, with the versioned upstream source
and license URLs. This notice ships in both Capicove wheel and source archives;
NetworkX itself remains a separately installed optional dependency.

The capability-local `hierarchical_community_run` returns an immutable
`CommunityHierarchyResult` containing a `CommunityExecutionRecord`. It binds the
adapter algorithm/version, declared actual runtime/version and configuration,
exact community/partition profile, source projection, ordered induced graphs and
partitions for each root/recursive visit, and the resulting community structure.
Generic injected strategies must declare an immutable `CommunityRuntimeIdentity`;
there is no mutable last-run metadata. `hierarchical_communities` retains the
planned tuple-returning compatibility interface. Internal communities retain all
subtree members; consumers derive leaf-only membership by excluding parent IDs.
These records are capability-local execution evidence, not new canonical DTOs or
a persistent schema change, and communities do not establish source truth.
Record construction iteratively validates complete connected-component roots,
preorder visits, induced graphs, partition eligibility and child coverage,
community IDs, ancestry, memberships, and exact stop issues before permitting an
execution fingerprint. The NetworkX adapter requires every automatic backend
priority list to be empty, including per-function overrides. It records all
priority categories and checks exact runtime identity at construction, before
graph construction and partitioning, and after partition output is materialized;
it never mutates global NetworkX configuration.

## Acceptance and verification

- Maintainer assigns the final RFC/ADR numbers and records acceptance explicitly.
- Required processing and text retrieval contracts land on the selected baseline.
- Apply the Task 0R NetworkX dependency/license decision and record resolved-version
  distribution notices with the optional adapter.
- Task 1 begins with failing source-span, canonical JSON, scope, and import tests.
- Later tasks cover Unicode, cache rebinding, false merges, conflicting assertions,
  citation failures, crash recovery, concurrent activation, and revocation.
- Real Parquet/NetworkX Louvain integration and authorized local model qualification
  remain distinct from scripted CI providers; no synthetic test proves model quality.
- Test fixtures are project-authored and non-sensitive. Apply the repository's
  detect-secrets checks; never weaken scanning to accept fixtures.
- Run `make quality`, `make test`, and `make build` before completion claims.

## Decision record

The user explicitly accepted this RFC and its companion ADR in this task.
Final numbering and dependency integration remain pending. Task 0R is an explicit
user-approved revision of these Accepted documents: NetworkX Louvain and the
capability-local strategy boundary supersede the original algorithm choice.
The resolved package and notices remain recorded at adapter delivery. See the
[accepted ADR](../adr/document-graph-snapshot-boundary.md).

After explicit user authorization, `origin/dev` was refreshed and local `dev` was
fast-forwarded to `42175bd`. Processing and text retrieval prerequisites are now
present. The user explicitly authorized Task 1 using these accepted named documents;
numeric registry assignment remains administrative follow-up. The parser remains a separate
integration prerequisite; it must not be pulled in through an unrelated books/Wiki
change without explicit authorization.

### Task 1 codec details

The evidence codec uses a closed envelope with `schema_version`, `kind`, and
`value`. Version 1 kinds are `scope`, `source_span`, `segment`, and `text_unit`.
Nested fields exactly match their dataclass fields, including explicit nulls.
Duplicate JSON keys are rejected. `InputSegment.role` is one of `core`, `context`,
`title`, and `table_header`; its text concatenates the mapped source slices without
unmapped separators. Later representation composition must map separators explicitly.
An authorized reader checks actual source lengths and quote equality; constructors
can only validate range shape, coverage lengths, and core contiguity.

### Accepted Task 2 field refinement

The user explicitly accepted the following closed-field refinement before its
implementation. Qualifiers are sorted unique pairs with string values and keys
`raw_predicate`, `occurrence_ref`, `time_text`, and `condition_text`. Feature
pairs use `name_similarity`, `embedding_similarity`, and `shared_context_score`
with finite values in [0, 1]; absent features mean unassessed. Scores are not truth
probabilities and cannot override authoritative identity conflicts.

Graph record codecs use closed version-1 envelopes with a fixed record-kind
registry. Model extraction responses instead use local IDs and `RelativeSpan`
quotes; no document IDs or absolute evidence references occur in that schema.
Model-proposed identity keys cannot declare `authoritative=true`; host-verified
identity keys remain a separate trust decision.

Schema validation is structural only. Task 6 still implements quote binding,
source authorization, unique local IDs, endpoint/type compatibility, core evidence
ownership, and partial-result handling. Source-stated hypotheses do not fabricate
derivation receipts. Actual generated assertions require derivation tracking in
their producing stage.

### Task 3 ledger representations and validation boundaries

`ReportSections` materializes the ten named content sections as immutable tuples
of `ReportStatement`. All ten keys are mandatory on the wire, including empty
sections. `ChildStatementRef(report_id, statement_id)` is the typed pair for a
child-local citation. Statement kinds are `fact`, `hypothesis`, `recommendation`,
and `limitation`. Facts need assertion or child citations; a finding alone cannot
establish a fact. Hypotheses need a citation. Report-local statement IDs are unique
across sections, and a report cannot cite itself as a child. Report generation
must still validate citation existence, child statement kinds, source closure,
and semantic entailment; the constructor does not load other reports.

Generated `DiagnosticFinding` values permit only `inferred` and `hypothesized`,
require supporting assertion references and a derivation receipt, and retain
counter references. Limitations are immutable strings; an absent proposed
verification step is explicit null. Community roots have depth zero and no parent;
other communities have a parent and positive depth. Publication must still verify
the full hierarchy and membership against the actual tables.

`StageCoverage(denominator, declared, processed, accepted, rejected, failed,
omitted)` counts one named population, with nonnegative integer counts:
`processed = accepted + rejected` and
`declared = processed + failed + omitted`. Stage policies specify the unit name;
do not combine counts of documents, mentions, or reports in one record. Failed
items have no processing result, while omitted items were not attempted. Ordinary
identity abstentions are accepted processing outcomes, not failed items. More
detailed input/output counts remain in stage artifacts and operational diagnostics.

The immutable stage statuses are `succeeded`, `partial`, `failed`, `skipped`, and
`cancelled`. Success requires an artifact reference, including successful empty
outputs; other statuses require issue codes. Completeness independently requires
all ten pre-publication stages to succeed with `accepted = declared`. Partial
manifests may omit stage results but must still include all 17 table roles.
First-build alignment records `succeeded` with `no_baseline` and zero declared
alignment comparisons. That issue code is invalid with a pinned baseline.

Manifest schema version 1 normalizes tables by role and stages by stage ID.
`manifest_bytes` encodes the semantic manifest, without the graph codec envelope;
its bytes can be hashed for `GraphSnapshotRef.manifest_sha256`. Snapshot IDs remain
opaque catalog references. Dataset schema decoding reuses the public recursive
schema codec, subject to the graph codec's existing JSON nesting bound. The
separate retrieval binding preserves complete version-1 embedding-space metadata
and unique generation/space references. Binding construction does not verify
generation contents, authorize reads, or activate an index.

### Task 20 accepted validation and authorization refinements

The Task 20 controller explicitly approved the following capability-local
boundaries after implementation exposed missing external input/stage schema
semantics. Canonical graph DTO fields and processing persistent formats remain
unchanged. `GraphSnapshotCatalog` requires an absolute private SQLite path, the
public `LocalProcessingWorkspace`, and three explicit host dependencies:

- `SnapshotAccessGate` extends `AccessGate` with a nonnegative monotonic policy
  `revision` and `guard(expected_revision)`. The guard rejects stale policy and
  serializes policy changes through the enclosed SQLite commit. Scope authority
  includes graph lifecycle invalidation; invalidating deleted evidence does not
  require reading that evidence. The host must provide these semantics; no
  fabricated revision, optional duck typing, or default authority is permitted.
- `InputSnapshotValidator.validate(ref, scope)` returns immutable
  `InputSnapshotInventory(snapshot_ref, scope, document_refs, evidence_refs,
  validation_sha256)`. It verifies the external document-set schema, exact
  immutable ref/content, complete active document inventory, source availability
  and current authorization. Its digest commits to content and validator version.
  Catalog validation independently compares exact document/evidence sets with
  the verified graph documents table and checks the processing snapshot.
- `StageArtifactValidator.validate(stage, scope, profile_fingerprint)` returns
  immutable `ValidatedStageArtifacts(stage, scope, profile_fingerprint,
  evidence_refs, validation_sha256, table_refs)`. It verifies every artifact ref,
  exact profile/version and coverage population commitment. Exact output
  `TableRef` commitments prevent mixing another build's datasets under valid
  stage metadata. The catalog retains all graph cross-table validation.

The catalog owns this closed producer map: representations → documents;
text_units → text_units; extraction → entity_mentions, relationship_mentions;
identity_embeddings → no graph table; resolution → entities,
entity_memberships, resolution_memberships; alignment → entity_alignments,
resolved_entities; aggregation → relationship_assertions,
relationship_aggregates, resolved_relationships; diagnostics → diagnostic_findings;
communities → communities, community_memberships; reports → community_reports.
`operation_diagnostics` is the catalog-validated union of typed stage diagnostics,
with no unique producer. Duplicate, foreign, wrong-stage or changed-ref
commitments are rejected. Complete manifests commit all 16 owned roles; partial
manifests commit every nonempty table through its succeeded/partial producing
stage. Failed/skipped/cancelled stages cannot claim produced tables. Typed empty
tables may remain unclaimed in partial manifests.

The closed coverage denominators, in workflow order, are documents, codepoints,
text_units, descriptors, mentions, entities, assertions, bundles, entities and
reports. Catalog checks exact table populations where their units match; accepted
report count matches published reports. Other counts require the stage validator's
exact content/coverage commitment, rather than guessed conversions. Partial
upstream outputs remain inspectable while missing downstream stages cannot
establish complete graph coverage or activation eligibility.

Task 19 publication fingerprints bind complete materialization identity, not
just the profile hash. The catalog verifies them using public
`validate_snapshot` followed by the typed Parquet codec, with exact scoped dataset
identity, schema, serializer, content and ref checks. It never imports processing
private manifest layouts or compares that fingerprint directly to a profile hash.

The private snapshot SQLite schema stores canonical manifest bytes, bounded
validation digests, observed authorization revision, normalized reverse source
dependencies, durable source-ineligibility tombstones, active pointers and
activation history. It contains no graph row copies or Memory/Wiki tables.
Registration is immutable, idempotent and inactive. Reads and reuse revalidate
current source access and artifacts. [RFC-0019](0019-graph-snapshot-verified-read-reuse.md)
amends the per-embedding-batch reauthorization: it rechecks eligibility and
current access, and artifact revalidation runs at the projection read and before
binding. Activation uses `BEGIN IMMEDIATE` and a
revision compare-and-set with history in one transaction. Source invalidation
retains artifacts and metadata, prevents later resurrection by registration,
and clears any dependent active pointer with a revision/history update.
These guarantees are local SQLite transactions plus the explicit policy guard;
they do not claim filesystem/policy-service/SQLite global atomicity.
Activation repeats verified reads of the fixed 17 table references after entering
the policy guard and immediately before the short SQLite transaction; Parquet
decoding does not hold the SQLite write lock. Work is linear in those local table
payloads, with no model calls or artifact acquisition. A filesystem mutation
after the final read can occur outside SQLite's transaction boundary; subsequent
`get`/`active`/reuse validation rejects that damaged snapshot rather than exposing
it as readable. The policy guard must keep authorization stable through commit.

Real Task 19 report schemas exposed a separate depth-bound mismatch in the
manifest codec. The accepted correction retains the generic `canonical_bytes`,
`fingerprint`, evidence and ordinary graph value JSON bound at 16, while using
a fixed private depth bound of 32 only for exact `GraphSnapshotManifest` payloads
and their closed graph-codec envelope. Decoding checks the shallow exact envelope
before choosing this bound, then reconstructs every canonical type. The public
recursive DatasetSchema depth contract is unchanged. Excessive manifest and
ordinary-value nesting still fail; there is no caller-selectable unbounded mode.
