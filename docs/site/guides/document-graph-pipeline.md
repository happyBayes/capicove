# Document graph pipeline

Capicove builds an evidence-backed, immutable knowledge-graph snapshot from an
authorized text document snapshot. The fixed workflow covers representations,
deterministic chunks, extraction, identity resolution, cross-snapshot alignment,
assertions, personal-growth diagnostics, Louvain communities, cited community and
overview reports, publication of 17 typed Parquet tables, and an optional retrieval
binding.

The API is intentionally composition-first. There is no document-graph CLI or MCP
command, no implicit model discovery, and no network fallback. An explicitly
configured compatible chat server can implement the existing generation ports;
see [deployment and remote generation](deployment.md). Existing local model
loading remains offline.

## Install

!!! note "Host integration required"

    Audience: developers building a trusted host. Input is an authorized immutable
    text snapshot; output is a graph snapshot and optional retrieval binding.
    The Python blocks are composition sketches, not copy-and-run programs.
    `gate`, `inputs`, accepted profiles, readers, generators and identity services
    must be supplied by the host using the public protocols. Never replace
    authorization with an allow-all stub for real data.

Read the complete composition and publication boundaries before installing extras.
Local model assets must be provisioned explicitly; installation alone does not
create a usable graph service.

Select the adapters you actually run:

```bash
python -m pip install \
  "capicove[graph-models,text-embedding,graph-community,graph-data]"
```

- `graph-models`: verified local causal LLM/tokenizer adapter.
- `text-embedding`: verified local BGE batch encoder.
- `graph-community`: NetworkX 3.6.x Louvain.
- `graph-data`: PyArrow 25.x Parquet publication.

Assets must already exist on local storage. These extras never download a model.

## Compose the Python service

The host owns authorization, the immutable input inventory, source representation
reads, accepted profiles, local storage roots, identity attestations, and optional
retrieval implementation. Implement the public protocols and create one shared
`GraphScope` and one shared policy gate for every component.

```python
from pathlib import Path
import time

from capicove.contracts.graph_build import GraphScope
from capicove.graph_build import GraphBuildRequest, GraphBuildService
from capicove.graph_build._adapters.artifacts import FilesystemProtectedArtifacts
from capicove.graph_build._adapters.louvain import NetworkXLouvainPartitioner
from capicove.graph_build._adapters.sqlite import SqliteReceiptStore
from capicove.graph_build._stage_artifacts import BuildStageStore
from capicove.graph_build.inference import InferenceRunner
from capicove.graph_build.service import BuildPolicy
from capicove.graph_build.snapshots import GraphSnapshotCatalog
from capicove.processing import LocalProcessingWorkspace

root = Path("/absolute/private/capicove-graph")
scope = GraphScope("workspace:example", "scope:private")
workspace = LocalProcessingWorkspace(root / "processing")
stages = BuildStageStore(root / "stages" / "stages.sqlite", gate, workspace)
catalog = GraphSnapshotCatalog(root / "catalog" / "catalog.sqlite", workspace, gate, inputs, stages)


def artifact_factory(source_refs: tuple[str, ...]):
    return FilesystemProtectedArtifacts(
        root / "protected-artifacts", gate, evidence_refs=source_refs
    )


receipts = SqliteReceiptStore(
    root / "receipts" / "receipts.sqlite",
    gate,
    artifact_factory(all_input_evidence_refs),
)
inference = InferenceRunner(
    generator=local_structured_generator,
    tokenizer=local_request_tokenizer,
    gate=gate,
    artifact_factory=artifact_factory,
    receipts=receipts,
    clock=time.monotonic,
    wall_clock=time.time,
    sleeper=time.sleep,
    cancelled=lambda: False,
)

service = GraphBuildService(
    root / "runs" / "runs.sqlite",
    reader=authorized_representation_reader,
    gate=gate,
    profiles=(accepted_graph_profile,),
    inputs=inputs,
    tokenizer=local_text_tokenizer,
    inference_runner=inference,
    embedder=identity_embedding_provider,
    identity_space=identity_embedding_space,
    partitioner=NetworkXLouvainPartitioner(),
    workspace=workspace,
    catalog=catalog,
    stages=stages,
    policy=BuildPolicy(
        subject_ref="subject:explicit-host-identity",
        support_groups=source_occurrence_support_groups,
    ),
    subject_resolver=subject_resolver,
    counter_evidence=counter_evidence_resolver,
    identity_evidence=identity_evidence_resolver,
    source_provenance=source_provenance_resolver,
    retrieval=retrieval_runtime,  # or None
)
```

The lower-case names are required host objects, not hidden globals:

- `gate` implements current scope/evidence authorization plus a monotonic revision
  fence; `inputs` validates and returns the exact immutable `DocumentRow` inventory.
- `authorized_representation_reader` returns an authorized `TextRepresentation` for
  each current document and rechecks source access.
- `accepted_graph_profile` is a closed `GraphBuildProfile`. Its fingerprint binds
  ontology, prompts/schemas, tokenizer/model identities, budgets, embedding-space
  IDs, and community profile.
- local generator/tokenizer and BGE objects must be constructed from verified
  `LocalModelAssets`; see the qualification procedure linked below.
- subject, alias, occurrence, support-group, and counter-evidence resolvers consume
  trusted host metadata. Document prose is never identity authority or an
  instruction source.
- `source_provenance_resolver` returns exactly one sorted
  `DocumentSourceProvenance` for every current document. Original source documents
  have `source_kind="source"` and no derivation; generated inputs have
  `source_kind="generated"` and a non-empty trusted `derivation_ref`. Missing,
  foreign, duplicate, or changing classifications fail closed. Generated graph or
  report text cannot be recycled as independent diagnostic, community, or report
  evidence.
- `retrieval_runtime` is an optional explicit `BuildRetrievalRuntime`. It binds a
  `GraphTextReader`, `GraphEmbeddingRuntime`, text-index lifecycle/writer/catalog,
  `GraphRetrievalCatalog`, index format, and batch size. It does not activate an
  index generation.

Private `_adapters` above are package-provided local implementations, not extension
contracts. External capabilities should depend on the public protocols and canonical
contracts, not another capability's private implementation.

## Build, replay, and regenerate

An input snapshot is a typed `DatasetSnapshotRef` containing the authorized
`documents` role. Build with a pinned profile and optional complete compatible
baseline:

```python
request = GraphBuildRequest(
    input_snapshot_ref=input_snapshot_ref,
    scope=scope,
    profile_id=accepted_graph_profile.profile_id,
    baseline_snapshot_id=prior_graph_snapshot_id,  # None on first build
    mode="reuse",
)
result = service.build(request)
```

Modes are exact:

- `reuse` reuses currently authorized successful receipts and computes missing
  work.
- `replay` performs no model-generation provider call and fails if the exact
  authorized generation receipt is unavailable. Identity embeddings or optional
  retrieval/index work can still run when their exact stage or retrieval artifacts
  are absent; zero calls across every adapter requires all of those artifacts.
- `regenerate` creates distinct generation identities; it never overwrites a prior
  accepted response.

Every reuse rechecks current scope and evidence authorization. A cache key is not
an authorization token. Runtime, profile, input, baseline, source-lineage, or
policy-revision drift fails closed.

After an interruption, preserve the stores and resume the exact durable request:

```python
resumed = service.resume(result.run_id, scope)
```

`resume` does not accept a changed profile, input snapshot, baseline, or runtime.
The current public API does not expose run discovery or a caller-supplied run ID,
so a process failure before `build()` returns a result cannot yet be resumed through
public contracts. Preserve the durable stores for a future recovery tool rather
than querying their private SQLite layout. To stop future work for a known run:

```python
service.cancel(run_id, scope)
```

Cancellation and deadlines preserve already committed immutable artifacts; they
do not claim that a synchronous native kernel can be forcibly interrupted.

## Inspect partial and complete results

`GraphBuildResult.status` is one of `succeeded`, `partial`, `failed`, or
`cancelled`. `completeness` is separately `complete` or `partial`. Inspect every
stage rather than inferring coverage from the top-level status:

```python
for stage in result.coverage:
    print(stage.stage_id, stage.status, stage.coverage, stage.issue_codes)

if result.graph_ref is not None:
    manifest = catalog.get(result.graph_ref.snapshot_id, scope)
    assert len(manifest.tables) == 17
```

All manifests include all 17 typed table roles, using valid empty Parquet tables
when necessary. Registration and successful publication remain inactive. A report
failure, unavailable local adapter, budget exhaustion, or rejected unit may produce
a partial snapshot with explicit coverage and issue codes. A partial snapshot is
inspectable but cannot activate.

## Activate explicitly

Activation is a separate compare-and-set operation:

```python
assert result.status == "succeeded"
assert result.completeness == "complete"
assert result.graph_ref is not None
active_ref, revision = catalog.active(scope)
new_revision = catalog.activate(
    result.graph_ref.snapshot_id,
    scope,
    expected_revision=revision,
)
```

Only a complete, validated, currently authorized snapshot can replace the pointer.
A competing activation raises `GraphActivationConflict`; reread `catalog.active`
and make an explicit operator decision. Building or registering a snapshot never
calls `activate`.

## Revocation and retention

When source access is revoked or authorized erasure removes evidence, update the
host policy first and durably invalidate dependants:

```python
catalog.mark_ineligible(
    revoked_evidence_ref,
    scope,
    reason_code="source_access_revoked",
)
```

If the active graph depends on that evidence, its active pointer is cleared and a
history event is appended. Artifacts, receipts, snapshots, and bindings remain
immutable records, but every read/reuse reauthorizes their evidence closure.
Retention/erasure policy decides whether protected bytes may remain; immutability
does not override deletion obligations. Do not delete individual catalog or
Parquet files behind the APIs, because that converts a governed revocation into
corruption.

## Retrieval binding

When `retrieval_runtime` is supplied, the build can publish graph-derived source,
entity, and report text into one exact existing text-index generation and persist
a `GraphRetrievalBinding`. Long text is split into exhaustive embedding ranges; it
is not silently truncated.

```python
if result.retrieval_binding_ref is not None:
    binding = retrieval_runtime.bindings.get(result.retrieval_binding_ref, scope)
    assert binding is not None
    print(binding.generation_refs, binding.embedding_spaces)
```

The binding pins exact validated generation manifests, embedding spaces, graph
snapshot, scope, and unit-manifest digest. Graph success does not depend on
retrieval success, and neither `GraphBuildService` nor `GraphRetrievalCatalog`
activates the retrieval provider's alias. If the product wants an active text
index, it must perform a separate explicit activation through the retrieval
generation service after validating the bound generation and product policy.

## Identity, schema, and runtime records

Retain these identities with every operated graph:

- graph manifest schema version 1 and each table's `DatasetSchema.sha256`;
- `GraphBuildProfile` fingerprint and all prompt/schema/profile versions;
- input and baseline snapshot IDs, scope, evidence inventory and authorization
  revision;
- model/tokenizer/BGE manifest digests, revisions, device, context/output limits,
  BGE role prefixes, preprocessing identity, exact `EmbeddingSpaceRef`, and
  generation profile;
- Louvain adapter ID/version, NetworkX package and Python runtime identity, closed
  partition profile, node/edge order, and projection fingerprint;
- graph snapshot, build run, protected artifact, receipt, retrieval binding,
  retrieval generation, index format, and validation identities.

The MVP uses `CommunityProfile` version 2 with a tagged
`LouvainPartitionProfile` version 1. NetworkX replay is supported only for the
recorded runtime and seed; cross-version identical partitions are not promised.

## Supported limits

The default identity pool is 2,000 entities and the development hard cap is
10,000. Candidate pair batches are capped at 2,000 and top-k defaults to 10.
Inference is capped at three provider calls and one repair per unit, with at most
two concurrent units. The default chunk core limit is 1,200 tokens. Community,
report, extraction, tokenizer-work, response, and deadline bounds are closed in
the accepted profile and must be recorded for each build.

These safeguards are not throughput or quality claims. The repository-only
operator procedure at `docs/development/document-graph-qualification.md` contains
the full initial budget table, real-adapter matrix, held-out evaluation rules, and
claim checklist; it is intentionally outside the published-site navigation until
a qualification record exists.

## Current exclusions

- The [native knowledge workspace](native-knowledge.md) composes production PDF,
  image OCR and audio preparation, preserving page/time locators. Its local
  acceptance does not establish general OCR/ASR or extraction quality.
- Scanned PDF preparation uses explicit page OCR; table structure, handwriting,
  arbitrary scene semantics and discontinuous cross-source reasoning are not
  implied by a successful text-evidence import.
- No Temporal Memory or Wiki read/write integration is performed.
- No Neo4j, dynamic graph query API, search-fusion service, implicit external API,
  telemetry, model download, or cloud fallback is introduced.
- Community membership expresses association, not truth. Diagnostics are inferred
  or hypothesized, require source evidence, preserve counter-evidence, and do not
  claim that a recommendation was executed or effective.
- Structural citation validation is not semantic entailment. Only independent
  held-out qualification can support a model-quality claim.
