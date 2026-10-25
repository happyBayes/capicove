# Module Boundaries

## Dependency direction

The permitted dependency direction is:

```text
interfaces (CLI / API / MCP)
        ↓
Tool Runtime or public application operations
        ↓
capability application operations
        ↓
canonical contracts and capability-owned ports
        ↑
private infrastructure and provider adapters
```

Dependency injection connects adapters to ports at the composition root. Canonical
contracts never import adapters or interface frameworks.

The capability that consumes an external service owns its port. Default
implementations remain private under that capability's `_adapters` package; separately
distributed implementations integrate through the plugin registry. See
[repository structure](repository-structure.md) for the package vocabulary.

## Boundary rules

1. `capicove.contracts` may use only the Python standard library and explicitly
   approved model/schema dependencies.
2. A capability imports another capability only through its public contracts.
3. Files or modules named `_internal` or beginning with `_` are private.
4. CLI, HTTP, and MCP modules translate transport input and output; they do not
   implement domain policy.
5. Storage, model, network, and execution providers implement ports owned by the
   capability that consumes them.
6. Optional packages are imported inside provider code, never at top-level package
   import time.
7. Plugin discovery happens through `capicove.plugins`, with explicitly selected
   model provider/installer groups owned by `capicove.model` under RFC-0023;
   other modules must not scan
   installed distributions independently.
8. Cross-module events use versioned canonical event schemas.
9. `capicove.runtime` may import public capability APIs to wire the application, but
   capabilities must not import the runtime composition root.
10. A capability package is introduced only with concrete behavior or a contract,
    tests, deliberate public exports, and boundary documentation.
11. Enterprise CLI and MCP tool calls enter through `capicove.tools`; transport
    adapters never invoke provider adapters directly.
12. Raw Episode, Step, and ToolCall records are append-only. Feedback, correction,
    reward, and evaluation updates create linked records.
13. `capicove.evolution` consumes trace and evaluation contracts; it never mutates
    source evidence, raw traces, or a released asset version.
14. Automatic promotion cannot expand the approved Tool Catalog or filesystem,
    network, credential, data, subprocess, privacy, security, or audit permissions.
15. `capicove.environments` owns replay, sandbox, shadow, and canary execution;
    synthesis code cannot mark its own candidate active.
16. `capicove.processing.video` owns media inspection, range writing, scene
    detection, byte-exact file deduplication, path authorization, and output-store
    ports; its `_adapters` package is private and optional packages are loaded only
    when selected. xxHash identifies candidates only; SHA-256 confirms duplicates.
17. `capicove video split` and `capicove_video_split` translate the same
    `capicove.video.split` Tool schema and preserve canonical status, warnings,
    errors, evidence references, and lineage.
18. Model-provided video Tool arguments cannot add subprocess flags or expand the
    runtime-configured filesystem roots and execution policy.
19. Video range writing is accurate transcoding by default. Stream copying is an
    explicit mode whose keyframe-alignment drift remains visible in configuration
    and lineage; every writer preserves selected audio and video tracks.
20. `capicove video filter` and `capicove_video_filter` translate the same
    `capicove.video.filter` Tool schema. They preserve `keep=false` as a successful
    decision and retain completed diagnostics when a later required evaluation
    fails.
21. Unified video-filter contracts, rules, ports, and pipeline use the
    capability-specific `video_filter_*` basenames. Provider adapters remain under
    `processing.video.filtering._adapters` and are never imported by canonical
    contracts.
22. Model-backed video filters resolve only explicit local assets or already-local
    cache entries. They disable remote code and network fallback; an absent model,
    vocabulary, package, or cache entry is reported as `backend_unavailable`.
23. `capicove.processing` owns source acceptance, decode/checkpoints, processing
    manifests, keyframe extraction, and target representations through its
    `MediaRepresentationEncoder` port. It does not import retrieval-private
    adapters.
24. `capicove.retrieval` owns `QueryEncoder`, candidate and index ports,
    `IndexGenerationCatalog`, fusion, temporal assembly, reranking, cursor
    semantics, projection, and evaluation. It consumes processing output only
    through public contracts and references.
25. One complete Capicove node owns one workspace and its sole index-generation
    catalog. CPU and GPU are provider choices inside that node; a query never
    selects another host or index owner.
26. Runtime may select `auto`, `cpu`, or `gpu` providers inside the current node.
    Automatic GPU-to-CPU fallback requires an exactly compatible embedding space
    and the same pinned index generation; explicit `gpu` never falls back.
27. `capicove.tools` is the policy and trace boundary for retrieval. Implemented
    transports translate the same Tool contracts and never invoke model,
    processing, retrieval, or index adapters directly. Stage 2 exposes
    `capicove.search` through Python, Tool, and MCP; the search CLI, evidence
    expansion, and ingest transports remain reserved.
28. Remote MCP client ingress is independent of execution selection. Listener,
    authentication, workspace, mount, model, provider, and index configuration is
    startup-owned and cannot be widened by Tool arguments.
29. Searchable units, processing manifests, and index-generation manifests are
    immutable. Activation uses an atomic alias and query lease; eviction may remove
    only policy-approved rebuildable payloads while retaining evidence, hashes,
    lineage, manifests, and lifecycle events.
30. Index synchronization, remote Retriever dispatch, cross-node generation
    mapping, and federated retrieval are excluded from the MVP. An enterprise
    federation extension requires a separate accepted RFC and ADR.
31. `capicove.traces` owns immutable Agent-operation event contracts,
    sanitization, recording, query ports, and feedback linkage. SQLite is a private
    adapter and never appears in canonical contracts.
32. `capicove.export` owns versioned portable JSONL packages, public schemas,
    content-addressed media copies, manifests, and streaming validation. It reads
    traces only through the public snapshot port.
33. `capicove.batch` owns only its execution observer port. Observer
    implementations, including trace adapters, live outside `capicove.batch` and
    do not reverse the dependency direction.
34. Raw Agent-operation records are append-only. Feedback and evaluation are linked
    records; an export filter selects complete Episodes rather than isolated
    matching records.
35. External Agent messages enter a trace only through an explicit host-supplied
    `TraceSession`. CLI and MCP cannot fabricate remote conversation messages and
    invoke the same Tool tracing wrapper when a recorder is configured.
36. Trace recording and export media copying are explicit local opt-ins. Neither
    capability downloads remote content or enables telemetry.
37. `capicove.batch` discovery, planning, execution, cache, and checkpoint contracts
    remain independent of media and database backends. Image/video adapters depend
    only on public canonical Tool contracts; the CLI composition root selects local
    handlers, an optional trace-side Tool invoker, and the private SQLite store
    lazily. The executor supplies the durable retry attempt number through its public
    attempt-aware operation port without importing trace contracts.
38. `capicove.contracts` owns the immutable `DatasetSnapshotRef`, dataset schema,
    lineage-edge values, and the public recursive schema codec. Dataset schema
    version 2 owns only arrays, string-keyed maps, and ordered structs; their
    recursive values, field order, nested nullability, and canonical hashes are
    backend-neutral and must not import PySpark or another execution SDK.
39. `capicove.processing` owns logical plans, processing runs, the dataset-catalog
    and run-store ports, the execution-backend port, and the Agent-readable
    `OperatorSpec` contract. The planner validates each operator's declared
    `input_count` and exact top-level parameter names. Storage and engine
    implementations remain private adapters or separately packaged plugins.
40. A Spark backend consumes and publishes only a committed DatasetSnapshotRef. It
    must not call Ray, accept arbitrary user code, or expose Spark DataFrames across
    the plugin boundary.
41. The version-1 Spark profile is bounded local JSONL/Parquet batch execution.
    Streaming, remote storage, Iceberg, Kubernetes, external packages, and
    credentials require separately accepted change control.
42. `capicove_execution_spark.operator_contracts` owns dependency-light closed
    Spark operator IDs, arity, and JSON parameter schemas. It and
    `SparkExecutionBackendPlugin.operator_specs()` must remain importable without
    importing PySpark, starting Java, creating a session, network access, or
    filesystem mutation.
43. `capicove_execution_spark.advanced_operators` owns reviewed PySpark expression
    construction and nested semantic/type checks for equality joins, global sort,
    ranking/lag/lead windows, and bounded aggregations. It may not accept SQL,
    arbitrary expressions, UDFs, caller code, paths, packages, JARs, remote
    locators, or runtime configuration.
44. The Spark I/O adapter owns recursive conversion and validation of schema version
    2 arrays, string-keyed maps, and ordered structs. It preserves schema metadata
    required to distinguish JSON from strings, rejects unsupported Spark types and
    map keys, and recursively validates required struct fields, array elements, and
    map values before accepting JSONL or Parquet data.
45. The local Spark profile remains synchronous and local-only. Active cancellation,
    nested mutation, `explode`, user-defined frames or aggregators, model-neutral
    Agent trace flattening, network access, and telemetry remain unimplemented and
    must not be advertised by plugin metadata or the operator registry.
46. `LocalProcessingWorkspace` is the public local publication/catalog operation
    used by execution plugins. It owns pending-publication crash recovery and is
    the only public route to private filesystem and SQLite adapters; separately
    distributed plugins must not import `capicove.processing._adapters`.
47. `capicove.wiki` owns the standalone local Wiki projection format, safe Markdown
    ingestion, deterministic lint/build, and its disposable private SQLite FTS5
    projection. Standalone means service-independent, not canonical truth: Wiki
    files are user-editable projections and reviewed interchange records rather
    than canonical Capicove knowledge or source evidence. It imports PyYAML only
    while parsing Wiki content and does not import Temporal Memory, graph databases,
    model SDKs, or provider retrieval implementations. A future Temporal Memory
    bridge may consume only public revisioned snapshots or submit explicitly
    reviewed records through public commands; it must not access memory-private
    adapters or introduce automatic bidirectional writes.
48. `capicove.contracts.documents` owns dependency-light document parsing values,
    coordinate semantics, hashes, receipts, requests, and results.
49. `capicove.documents` owns parser runtime ports and later parser orchestration.
50. Parser implementations depend on those public contracts and ports; core never
    imports an implementation distribution.
51. Document parsers do not acquire sources or models during `parse`.
52. `capicove.books` consumes only canonical document evidence and owns deterministic
    chapter structure, bounded Agent task contracts, append-only local workflow
    state, and strict publication orchestration. It does not own or invoke a model
    provider.
53. High-confidence sections pass without Agent review. Low-confidence structure
    work is bounded to resolvable local blocks and closed correction operations;
    summary tasks form an acyclic bottom-up graph.
54. Wiki v2 publication resolves every generated claim to immutable PDF page and
    block evidence. Deterministic L0-L2 validation is required before publication
    and during later Wiki lint/build operations.
55. `capicove.books` may call public Wiki publication operations but must not import
    Wiki-private SQLite adapters or treat the generated projection as source truth.

## Proposed reverse-image Stage 2 boundaries

These rules prepare the pending RFC-0002 amendment and ADR-0020 for review. They do
not authorize production implementation or activation until the code owner accepts
both records.

38. The calling Agent selects retrieval intent as `instance`, `concept`, or
    `balanced`. Retrieval executes only the declared semantic lanes plus the
    always-on content/perceptual lane and does not run a
    server-side intent classifier. Tool, MCP, and CLI adapters never translate
    intent into provider, model, or index parameters.
39. Every Python, Tool, and MCP image search receives a host-created
    `InvocationAuthorizationContext` that binds workspace, actor, operation,
    permissions, authorization scope, privacy class, and allowed source policy.
    Request fields may narrow it but cannot grant or widen authority.
40. Authorized local and mount sources are opened descriptor-relative beneath a
    startup-owned root with no-follow semantics (or an equivalent platform
    primitive). Authorization, identity/size checks, hashing, and decode use the
    same opened object; an adapter never authorizes a path and reopens it later.
41. The proposed `personal-lite` Stage 2 owns only the at-most-10,000-image local
    profile. Enterprise FAISS/MIH adapters and formal 30-million-image claims remain
    a separately reviewed Stage 3 and cannot expand Stage 2 dependencies or claims.

## Stable and experimental surfaces

### Document graph evidence foundation

`capicove.contracts.graph_build` owns immutable `GraphScope`, `SourceSpan`,
`InputSegment`, and `TextUnit` values. It reuses the public retrieval
`TextChunkLocator` and imports only canonical contracts and the standard library.
`capicove.graph_build.codec` owns closed, versioned evidence JSON and canonical
hash inputs. These imports do not initialize model, graph, database, or parser
runtimes. Source bounds and quote equality require an authorized representation
reader; structural validation alone does not prove evidence truth or grant access.

Graph evidence is not canonical Temporal Memory identity. This foundation performs
no memory/Wiki writes, source acquisition, model calls, graph analysis, or index
activation. See the accepted
[graph contract RFC](../rfcs/document-graph-pipeline-contracts.md) and
[snapshot boundary ADR](../adr/document-graph-snapshot-boundary.md).

`capicove.model` owns verified local asset selection, explicit provisioning and
bounded model factory leases. Providers compose public capability ports only;
canonical contracts never depend on the factory or model SDKs. Inference profiles
grant no installation authority. Model references must remain inside their leases.
The operator removed `capicove.model_management`; callers must use the new
namespace. Model lock bytes, receipts and indexes retain their existing formats.
See RFC-0023 and ADR-0027.

Version-2 model profiles additionally register explicitly opted-in HTTP VLM
endpoints without local asset roots. `capicove.model.vlm` owns the dependency-free
multimodal generation port; the generic OpenAI/vLLM adapter implements it.
HTTP/CLI composition consumes that port through factory leases. Endpoint model
revisions are operator declarations, while request and media hashes preserve the
exact submitted evidence. These profiles grant no model installation authority.

Public symbols are exported deliberately from a module's `__init__.py`. A symbol is
experimental only when its documentation and runtime warning say so. Everything
else follows the compatibility policy for the current release line.

Persistent identifiers, content hashes, event records, artifact manifests, and
lineage references receive compatibility review even during `0.x` development.

## Change ownership

The repository-wide owner in `.github/CODEOWNERS` reviews all changes until module
teams are established. As ownership expands, each capability directory will add a
specific team while retaining architecture review for contracts and security
boundaries.

Changes that create a circular import, duplicate a canonical model, or bypass a
port require redesign rather than an import exception.

## Temporal memory review candidate

[RFC-0009](../rfcs/0009-temporal-memory-contracts.md) and
[ADR-0015](../adr/0015-temporal-memory-sqlite-boundaries.md) are Accepted.
`source` owns immutable text Evidence and authorization; `knowledge` owns temporal
facts, ontology, identity, ledger and projections; `agent` owns admission and cited
context. Runtime injects public repository, Evidence, extraction and dense ports.
CLI/MCP enter through canonical memory Tools. The stdlib-only normalized RRF rule
is shared through `contracts.rank_fusion`; existing text fusion retains its public
API. No memory read imports optional image, model, or graph providers.

## Wiki maintenance foundation

Under accepted RFC-0031 and ADR-0035, `capicove.contracts.wiki_maintenance` owns
immutable structural values; `capicove.wiki.maintenance` owns the closed wire
codec and model-neutral generator ports. Structural decoding is not evidence
verification, reviewer authentication or publication authorization. Runtime owns
provider composition; Wiki does not import runtime, model SDKs or memory-private
adapters. W0 does not activate schema v3, migrate a workspace or publish changes.
