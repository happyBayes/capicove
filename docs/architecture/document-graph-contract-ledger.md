# Document graph contract ledger

This is the public contract reference extracted from the original development
plan. It preserves the contract and policy definitions; it does not add a new
approval or qualify a model. See [the accepted RFC](../rfcs/document-graph-pipeline-contracts.md).

## Contract ledger used by all tasks

This ledger fixes names so tasks do not invent incompatible DTOs. Dataclasses are
frozen and slotted; collection inputs become immutable tuples/frozensets. Optional
values below use `None`, not omitted unknown semantics. All IDs are scope-bound;
references and hashes have validation separate from human-readable names.

| Module under `contracts/graph_build/` | Types and required fields |
| --- | --- |
| `evidence.py` | `GraphScope(workspace_ref, scope_ref)`; `SourceSpan(evidence_ref, representation_ref, start_char, end_char, block_id=None, page_index=None)`; `InputSegment(segment_id, role, text, source_spans)`; `TextUnit(text_unit_id, document_ref, scope, source_sha256, representation_ref, core_locator, segments, section_path, ordinal, token_count, chunking_fingerprint)` |
| `extraction.py` | `RelativeSpan(segment_id, start_char, end_char, quote)`; `EntityMention(mention_id, text_unit_id, name, entity_type, description, source_spans, anchor_role, extraction_confidence, identity_keys, attribution_ref)`; `RelationshipMention(mention_id, text_unit_id, source_mention_id, target_mention_id, predicate, description, source_spans, epistemic_status, polarity, extraction_confidence, attribution_ref, qualifiers)`; `ValidatedExtraction(entities, relationships, rejected_count, issue_codes, result_fingerprint)` |
| `graph.py` | `IdentityKey(namespace, value, authoritative)`; `Entity(entity_id, scope, entity_type, display_name, aliases, mention_ids, identity_keys, occurrence_ref, subject_ref)`; `PairDecision(left_id, right_id, decision, reason_codes, features, evidence_refs)`; `EntityAlignment(current_entity_id, previous_entity_id, decision, feature_scores, evidence_refs, resolver_fingerprint)`; `RelationshipAssertion(assertion_id, source_entity_id, predicate, target_entity_id, epistemic_status, polarity, attribution_ref, qualifiers, evidence_refs, support_group_id, derivation_ref, counter_assertion_ids)`; `RelationshipAggregate(aggregate_id, assertion_ids, support_group_ids, counter_assertion_ids)` |
| `analysis.py` | `EvidenceBundle(bundle_id, subject_ref, target_ref, assertion_ids, counter_assertion_ids, independent_event_ids, omitted_count)`; `DiagnosticFinding(finding_id, diagnostic_type, subject_ref, target_ref, epistemic_status, support_refs, counter_refs, explanation, limitations, verification_step, derivation_ref)`; `Community(community_id, parent_id, depth, member_ids, issue_codes)`; `ReportStatement(statement_id, text, kind, assertion_refs, finding_refs, child_statement_refs)`; `CommunityReport(report_id, report_kind, community_id, title, sections, selected_bundle_id, omitted_count, receipt_ref)` |
| `snapshots.py` | `TableRef(role, dataset_ref)`; `StageResult(stage_id, status, artifact_refs, issue_codes, coverage)`; `GraphSnapshotManifest(schema_version, scope, input_snapshot_ref, baseline_snapshot_id, profile_fingerprint, tables, stages, completeness)`; `GraphSnapshotRef(snapshot_id, manifest_sha256)`; `GraphRetrievalBinding(binding_id, graph_snapshot_id, generation_refs, embedding_spaces, unit_manifest_sha256)` |

`EvidenceRef` in this ledger is a nonempty opaque string, not a new source-store
contract. `dataset_ref` and `input_snapshot_ref` use existing `DatasetSnapshotRef`.
`embedding_spaces` use `EmbeddingSpaceRef`. `qualifiers`, `features`, `coverage`,
and report `sections` have closed typed keys in their owning schema, not unrestricted
provider JSON. `SourceSpan` uses Unicode code points and optional PDF provenance;
`core_locator` remains the existing `TextChunkLocator` with its origin rules.

Report kinds are `community` and `overview`; only overview permits
`community_id=None`. Statement IDs are report-local content IDs, namespaced by
report ID when referenced from a parent. This avoids a circular hash between a
statement and its containing report. Membership schemas are explicit typed pairs:
`EntityMembership(entity_id, mention_id)` and
`ResolutionMembership(analysis_entity_id, current_entity_id)` in `graph.py`, plus
`CommunityMembership(community_id, entity_id)` in `analysis.py`. Diagnostic and
report derivation references never resolve to raw provider objects.

New operational values in `profiles.py` are `ChunkingProfile`,
`InferencePolicy`, `ResolutionProfile`, `CommunityProfile`, `PartitionProfile`,
`LouvainPartitionProfile`, and `GraphBuildProfile`.
Their codec rejects unknown keys, bool-as-integer values, nonfinite floats,
nonpositive budgets, and incompatible tokenizer/model windows.

`GraphBuildProfile` has profile ID/version, ontology version, chunking policy,
inference policy, resolution policy, community policy, generation profile ID,
identity embedding space, retrieval embedding space, report prompt/schema
versions, and diagnostic prompt/schema versions. `ResolutionProfile` has version,
top-k, pair batch size, identity pool limit, and an optional qualification-ref
for fuzzy merge thresholds. `CommunityProfile` schema/version 2 has `max_size`,
`max_depth`, `support_cap`, and `partition_profile`. `PartitionProfile` is a closed
discriminated union whose MVP `louvain` tag is fixed by
`LouvainPartitionProfile(resolution, threshold, max_level, seed, version)`.
The nested variant replaces the old flat seed, iterations, and resolution fields.
Reject unknown tags/keys and unsupported versions rather than silently coercing
version-1 profiles. Update nested profile serialization and fingerprints with new
artifacts and lineage; canonical graph DTOs and graph schema version 1 do not change.
Store typed coverage as declared,
processed, accepted, rejected, failed and omitted counts, with stage-specific
denominator names. Validate count relationships rather than deriving completeness
from a single success percentage.

### Initial local-development policies

- Inference: 2 concurrent units, 3 total provider calls per unit including at most
  1 structural repair, 120-second call deadline, and 3600-second build deadline.
  Inject clock/sleeper for tests; production values are explicit profiles.
- Extraction: at most 64 entities, 128 relations, 256 KiB response bytes, and 4096
  new output tokens; preflight the complete prompt against the selected model.
- Candidate retrieval: top-k 10 per source, maximum 2000 pairs per batch; process
  batches with stable ordering. No qualified fuzzy merge threshold is assumed.
- Communities: depth 0-3 inclusive, 50-node target leaf size, independent-support
  cap 20. The Louvain profile starts with resolution 1.0, threshold 1e-7,
  max_level 10, seed 42, and version 1. `max_level` bounds Louvain's internal
  optimization levels independently from hierarchy `max_depth`.
- Model paths, model manifest/revision, device, tokenizer revision, input role,
  context limit, and output schema are required configuration, never guessed.
- Above limits are development policy, not benchmark findings or production
  capacity claims. Task 24 qualifies model-dependent settings on held-out data.


## Community partitioning boundary

`CommunityPartitioner` and `CommunityPartitionerRegistry` select the explicit
`PartitionProfile`. The MVP adapter uses `networkx>=3.6,<4` under BSD-3-Clause.
The hierarchy is owned by `CommunityProfile`; the Louvain profile bounds the
partitioner's optimization independently of the hierarchy depth.
