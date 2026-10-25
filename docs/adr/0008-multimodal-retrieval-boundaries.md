# ADR-0008: Establish Multimodal Retrieval Boundaries

- Status: Accepted
- Date: 2026-08-13
- Owners: Capicove maintainers

## Context

Capicove must support image-to-image search, keyframe retrieval, and temporal
video-segment retrieval through CLI and MCP. It must run quickly on a local CPU and
also use a personal GPU server without forcing an individual developer to operate
index replication, federation, or a separate GPU-worker control plane.

The capability spans media decoding and representation extraction, model
inference, persistent derived indexes, retrieval policy, evidence/lineage, Tool
security, and Agent-facing context limits. Without explicit boundaries, transport
adapters could accumulate provider logic, retrieval could import processing-private
implementations, or local and remote indexes could acquire incompatible identity
and generation semantics.

## Decision

Option A is the MVP architecture: one complete Capicove node owns one workspace
and its sole index-generation catalog. A node may be a local CPU installation or a
complete deployment on a GPU server. The GPU-server node owns evidence, processing
caches, model bundles, CPU/GPU providers, retrieval orchestration, index
generations, Tool Runtime, traces, and MCP service. A remote Agent selects that node
by connecting to authenticated MCP.

Execution profiles are `auto`, `cpu`, and `gpu` relative to the selected node. They
do not select a host. `auto` may fall back from GPU to a compatible CPU provider
only inside the same node and against the same pinned generation. Explicit `gpu`
fails when a compatible GPU provider is unavailable.

Capability ownership is:

- `capicove.contracts` owns transport-neutral retrieval identities and schemas and
  imports no media, storage, model, MCP, database, or accelerator SDK;
- `capicove.processing` owns source acceptance, decode/checkpoints, keyframe and
  temporal representation production, processing manifests, and derived caches;
- `capicove.retrieval` owns `QueryEncoder`, index reader/writer and generation
  catalog ports, candidate retrieval, fusion, temporal assembly, reranking,
  projection, cursor semantics, and evaluation;
- `capicove.tools` owns Tool specifications, permission/deadline enforcement,
  stable error mapping, trace capture/redaction, and evidence expansion;
- `capicove.runtime` owns provider/plugin composition, server-relative execution
  selection, credentials, client-ingress/privacy/retention policy, and active
  retrieval profiles; and
- `capicove.cli` and `capicove.mcp` perform transport translation only and never
  invoke model, index, processing, or provider adapters directly.

Processing-owned `MediaRepresentationEncoder` implementations produce target
representations. Retrieval-owned `QueryEncoder` implementations produce query
representations. A plugin distribution may provide both factories, but the two
capabilities depend only on their public ports and shared canonical identities;
they never import each other's private adapters.

Public search accepts text, image, or video-clip queries and image, keyframe, or
video-segment targets. Compact output is the Agent default; full trace data remains
internal and selective evidence expansion is a separate permission-checked Tool
operation.

Remote MCP client ingress is independent of CPU/GPU provider selection and uses the
closed modes `deny_remote`, `embedding_only`, `redacted_media`, and `raw_media`.
Non-loopback MCP is an explicit authenticated and encrypted startup choice. Tool
arguments cannot select listeners, credentials, workspace roots, models, providers,
or indexes.

Persistent source evidence and lineage are immutable. Processing layers and index
payloads are content-addressed and rebuildable. Each query pins generation aliases;
activation is atomic; eviction removes only eligible derived payloads while
retaining manifests, hashes, lineage, and events.

## Consequences

- Personal developers operate one Capicove workspace and one index owner rather
  than a federated retrieval control plane.
- Local CPU and GPU-server deployments expose the same canonical Tool and hit
  contracts.
- Text-to-keyframe semantic retrieval is mandatory for the default CPU profile;
  OCR and ASR are additional lanes rather than prerequisites.
- Models, indexes, databases, MCP, and accelerator runtimes remain replaceable
  private adapters or plugins behind capability-owned ports.
- A laptop workspace and a GPU-server workspace do not synchronize automatically.
  Moving a workspace is an explicit backup/restore or export/import operation.
- Server-side ingress, authentication, retention, and index administration remain
  explicit operational responsibilities.
- Cross-capability contracts, persistent formats, plugin metadata, and security
  boundaries require compatibility review under RFC-0002.

## Alternatives considered

- **Option B: local node plus stateless remote GPU worker.** Rejected for the MVP.
  It preserves one index owner but introduces worker deployment, authentication,
  transfer policy, retries, payload movement, diagnostics, and failure handling that
  are unnecessary when a personal GPU server can run the complete node.
- **Option C: federated multi-index retrieval.** Rejected for the MVP and reserved
  for an enterprise Post-MVP extension. Federated retrieval requires cross-workspace
  evidence identity and authorization, generation snapshots, score normalization,
  partial-failure semantics, data-residency controls, and operational monitoring.
  It requires a separate accepted RFC and ADR and cannot add fields, synchronizers,
  remote Retriever clients, mappings, or dependencies to the personal-developer
  MVP in anticipation of that work.
- **Transport-owned retrieval implementations.** Rejected because separate CLI and
  MCP logic would drift in policy, security, scoring, and context behavior.
- **One shared private model adapter imported by processing and retrieval.**
  Rejected because it violates capability ownership. Shared canonical bundle
  identity is allowed; private implementations remain capability-owned.
- **OCR/ASR-only text-to-video retrieval.** Rejected because it cannot retrieve
  visual scenes with no useful speech, subtitle, or on-screen text.

## References

- [RFC-0002: Versioned multimodal retrieval contracts](../rfcs/0002-multimodal-retrieval-contracts.md)
- Approved multimodal retrieval design (archived development reference)
- Implementation program (archived development reference)
- [Module boundaries](../architecture/module-boundaries.md)
