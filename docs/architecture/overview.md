# Capicove Architecture Overview

## Purpose

Capicove is a local-first infrastructure layer that turns heterogeneous source
material into immutable evidence, searchable representations, validated knowledge,
human-readable projections, and reusable agent memory.

The architecture starts as a modular monolith. Modules run in one process by
default but communicate through stable contracts so storage, workers, API surfaces,
and execution backends can be separated when scale requires it.

The canonical source and test layout is defined in
[repository structure](repository-structure.md). Capability packages are introduced
only with their first tested contract or behavior.

Implemented image mapping, filtering, difference, and perceptual deduplication
boundaries are documented in [image processing](image-processing.md).

## Canonical flow

```text
Source
  → Artifact
  → Segment
  → Representation
  → SearchableUnit
  → Evidence
  → FactCandidate
  → ValidatedFact
  → WikiProjection
  → AgentContext
  → ToolInvocation
  → EpisodeTrace and Feedback
  → EvolutionCandidate
  → Evaluation and Promotion
  → Versioned SOP / Workflow / Skill / Pitfall
  ↺ AgentContext
```

Every transition records lineage. Derived outputs may be rebuilt; source evidence
and accepted event history are immutable.

## Planned capability modules

| Module | Responsibility |
| --- | --- |
| `capicove.contracts` | Canonical models, identifiers, schemas, and public protocols |
| `capicove.source` | Source adapters, acquisition, immutable artifact versions |
| `capicove.processing` | Processors, execution requests, caching, checkpoints |
| `capicove.retrieval` | Indexes, retrievers, fusion, rerankers, query planning |
| `capicove.knowledge` | Entities, facts, ontology, validation, lineage |
| `capicove.wiki` | Deterministic human-readable projections |
| `capicove.tools` | Transport-neutral Tool Registry, invocation, policy, and middleware |
| `capicove.agent` | Evidence bundles, memory, cited answers, and feedback |
| `capicove.traces` | Immutable Episode and Step recording, queries, and feedback linkage |
| `capicove.environments` | Replay, sandbox, shadow, and canary environments |
| `capicove.evolution` | Trace mining, synthesis, evaluation, promotion, and version registry |
| `capicove.export` | Trace datasets and portable Capicove packages |
| `capicove.plugins` | Plugin discovery, compatibility, and diagnostics |
| `capicove.runtime` | Configuration, composition, logging setup, and diagnostics |
| `capicove.cli` | Human and automation-friendly command-line boundary |
| `capicove.api` | Optional HTTP boundary |
| `capicove.mcp` | Optional MCP boundary |

Only modules with implemented behavior are added to `src/capicove`; this table is a
boundary map, not a request to create empty packages. A new package follows the
admission checklist in [repository structure](repository-structure.md).

## Runtime shape

The first runtime uses a local filesystem content-addressed blob store, SQLite for
metadata and the event ledger, and in-process execution. Future adapters may add
object stores, relational databases, job queues, vector indexes, Ray, or Spark
without changing canonical models.

CLI and MCP enterprise tool surfaces call the same Tool Runtime, which delegates to
capability application operations. HTTP application surfaces use the same public
operations. Interfaces must not duplicate business rules or access storage
implementations directly.

Opt-in CLI and MCP recording shares one fail-closed Tool tracing boundary. Each
transport invocation becomes one complete Episode with sanitized arguments and the
canonical result; the no-recorder path remains a direct Tool call. Multi-step hosts use
`TraceSession` to preserve only visible messages and operations. The trace exporter
reads complete Episode snapshots through a storage-independent port and publishes
versioned, bounded JSONL shards with an integrity manifest. Reference and explicit
content-addressed copy modes remain separate from recording and never add
model-specific training tokens or preference labels.

The default RSI policy automatically creates and evaluates candidates but requires
human approval before activation. Administrators may explicitly enable sandbox,
shadow, and canary automation inside a pre-approved Tool Catalog and permission
envelope. Trust-boundary changes always return to supervised review.

## Reliability principles

- Artifact identity is stable and content is addressed by a cryptographic hash.
- External updates create a new artifact version.
- Processor cache keys include input hash, processor ID and version, and normalized
  configuration hash.
- Operations are idempotent and classify failures as retryable, permanent, or
  policy-rejected.
- Network access and telemetry are disabled unless explicitly configured.
- Logs carry correlation, artifact, processor, and trace identifiers without
  logging sensitive source content by default.

## Architecture change control

Public contracts, persistent formats, plugin groups, dependency direction, and
trust boundaries require an RFC and an Architecture Decision Record. See
[GOVERNANCE.md](../../GOVERNANCE.md) and [docs/adr](../adr).
