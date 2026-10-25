# Repository Structure

## Purpose

Capicove uses a capability-first modular monolith. Importable code lives under
`src/capicove`, tests reflect architectural risk, and optional integrations remain
outside canonical contracts.

This document is the authoritative guide for placing new source, test, plugin,
example, benchmark, and deployment files. See the
[architecture overview](overview.md) for the system flow and
[module boundaries](module-boundaries.md) for dependency rules.

## Current source tree

```text
src/capicove/
├── __about__.py
├── __init__.py
├── __main__.py
├── py.typed
├── contracts/
│   ├── __init__.py
│   ├── _validation.py
│   ├── documents.py
│   ├── evolution.py
│   ├── tools.py
│   └── traces.py
├── cli/
│   ├── __init__.py
│   ├── __main__.py
│   ├── app.py
│   ├── tracing.py
│   └── commands/
│       ├── __init__.py
│       ├── doctor.py
│       └── traces.py
├── export/
│   ├── __init__.py
│   ├── jsonl.py
│   ├── media.py
│   ├── models.py
│   ├── validation.py
│   └── schemas/
├── batch/
│   ├── __init__.py
│   ├── discovery.py
│   ├── planner.py
│   ├── executor.py
│   ├── tool_operations.py
│   └── _adapters/
├── traces/
│   ├── __init__.py
│   ├── content.py
│   ├── events.py
│   ├── models.py
│   ├── ports.py
│   ├── recorder.py
│   ├── sanitization.py
│   └── _adapters/
├── evolution/
│   ├── __init__.py
│   └── promotion.py
├── documents/
│   ├── __init__.py
│   └── ports.py
├── processing/
│   └── video/
│       ├── models.py
│       ├── ports.py
│       ├── source.py
│       ├── identity.py
│       ├── splitter.py
│       ├── filtering/
│       │   ├── video_filter_contracts.py
│       │   ├── video_filter_ports.py
│       │   ├── video_filter_rules.py
│       │   ├── video_filter_pipeline.py
│       │   └── _adapters/
│       ├── strategies/
│       └── _adapters/
├── mcp/
│   ├── video_filter.py
│   └── video_split.py
├── runtime/
    ├── __init__.py
│   └── diagnostics.py
└── tools/
    ├── __init__.py
    ├── errors.py
    ├── registry.py
    ├── video_filter.py
    └── video_split.py
```

`capicove.cli` is a thin interface. It parses input, dispatches public operations,
and renders output. `capicove.contracts` defines standard-library-only Tool, Trace,
and evolution values. `capicove.tools` resolves transport-neutral Tool metadata and
handlers. `capicove.processing.video` owns immutable split models, pure strategy
planners, a unified evidence-aware filter pipeline, capability-specific filter
contracts and ports, backend ports, orchestration, and private media/model adapters.
`capicove.mcp` maps the same Tool contract lazily behind the `mcp` extra.
`capicove.evolution` enforces supervised and automatic-canary transition policy.
`capicove.runtime` provides side-effect-free installation diagnostics and will
become the composition root as capabilities are implemented.

`capicove.traces` owns observable Agent-operation event contracts, sanitization,
recording, storage/query ports, and its private SQLite adapter. `capicove.export`
owns versioned JSONL packages, schemas, explicit media copying, and validation.
`capicove.batch` exposes a transport-neutral observer port; its trace adapter lives
in `capicove.traces`.

The console entry point remains `capicove.cli:main`. The package also supports
`python -m capicove` and `python -m capicove.cli`.

There is no importable top-level `capicove/` directory beside `src`. This ensures
local tests and tools resolve the installed `src` layout instead of accidentally
importing a repository-local shadow package.

## Capability map

The following package names are reserved boundaries. A reserved package appears in
Git only with its first tested contract or behavior.

| Package | Responsibility |
| --- | --- |
| `capicove.contracts` | Stable identifiers, hashes, Artifact, Evidence, lineage, and event envelopes |
| `capicove.source` | Acquisition, Source Adapter ports, and immutable artifact versions |
| `capicove.processing` | Processor ports, execution requests, cache identity, and checkpoints |
| `capicove.documents` | Document parser ports, orchestration, and provider-neutral translation |
| `capicove.retrieval` | Index and retriever ports, fusion, reranking, and query planning |
| `capicove.knowledge` | Entity, Fact, Ontology, validation, and contradiction state |
| `capicove.wiki` | Standalone local Markdown Wiki, validation, FTS projection, and future knowledge projections |
| `capicove.books` | Deterministic book structure, bounded Agent tasks, local workflow state, and publication orchestration |
| `capicove.tools` | Tool Registry, invocation lifecycle, policy, and middleware |
| `capicove.batch` | Deterministic directory discovery, planning, resumable execution, and Tool adapters |
| `capicove.agent` | Evidence bundles, cited context, and feedback |
| `capicove.traces` | Immutable Episode and Step recording, query, and feedback linkage |
| `capicove.environments` | Replay, sandbox, shadow, and canary execution |
| `capicove.evolution` | Trace mining, synthesis, evaluation, promotion, and version registry |
| `capicove.export` | Portable packages and trace dataset serialization |
| `capicove.plugins` | Discovery, compatibility, metadata, and availability diagnostics |
| `capicove.runtime` | Configuration, composition, logging setup, and diagnostics |
| `capicove.cli` | Human and automation-friendly command-line interface |
| `capicove.api` | Optional HTTP interface |
| `capicove.mcp` | Optional MCP interface |

`contracts`, `documents`, `books`, `tools`, `batch`, `processing`, `retrieval`, `wiki`,
`traces`, `export`, `mcp`, `evolution`, `cli`, and `runtime` are currently
implemented.
Every other entry remains a reserved boundary until its first tested contract or
behavior is merged.

The filtering package intentionally uses `video_filter_contracts.py`,
`video_filter_ports.py`, `video_filter_rules.py`, and `video_filter_pipeline.py`
instead of generic basenames. This keeps repository-wide searches, diagnostics,
coverage output, and future processing capabilities unambiguous.

The global `contracts` package is deliberately narrow. Ports such as `Processor`
and `Retriever` belong to the capabilities that consume them, not to a global
protocol collection.

## Capability package vocabulary

A capability may use these names when the files contain real responsibilities:

```text
<capability>/
├── __init__.py          # Deliberate public exports
├── models.py            # Capability-owned immutable values
├── ports.py             # Protocols consumed by this capability
├── service.py           # Application use cases
├── events.py            # Versioned capability events
├── _adapters/           # Private implementations of owned ports
└── _internal/           # Private helpers
```

This is vocabulary, not mandatory scaffolding. Keep a small capability in a small
number of files until a real responsibility requires a split.

Public symbols are exported deliberately from `__init__.py`. Underscored modules,
`_adapters`, and `_internal` have no cross-capability compatibility promise and must
not be imported by another capability.

## Adapter placement

The capability consuming an external service owns its port and keeps default
implementations private:

```text
source/
├── ports.py
└── _adapters/
    ├── filesystem_blobs.py
    └── sqlite_artifacts.py

retrieval/
├── ports.py
└── _adapters/
    ├── lexical.py
    └── vector.py
```

Databases, model providers, network clients, and execution backends never define
canonical types. Optional dependencies are imported only inside the adapter that
uses them. Runtime composition wires adapters to public ports.

## Test tree

```text
tests/
├── unit/                 # Mirrors implemented source packages
├── contract/             # Public and plugin compatibility
├── integration/          # Real local adapters and module seams
├── e2e/                  # Installed-package user journeys
└── fixtures/             # Minimal licensed, non-sensitive inputs
```

Only `tests/unit` is needed by the current implementation. Other tiers enter the
repository with their first real test; they are not preserved with empty markers.
See the [testing strategy](../development/testing.md) for markers and gates.

## Top-level directory admission

| Directory | Add when |
| --- | --- |
| `examples/` | The first example runs against the current public API and is tested or validated in CI |
| `benchmarks/` | The first reproducible benchmark defines data origin, metric, environment, and baseline |
| `plugins/` | The first independently packaged `capicove-*` plugin passes core contract tests |
| `deploy/` | The first deployment configuration is maintained and exercised by project automation |
| `scripts/` | A deterministic repository-maintenance operation cannot be expressed clearly in the Makefile |

Do not create speculative directories for providers, databases, execution engines,
or modalities. Documentation may reserve a boundary without adding it to Git.

## Package admission checklist

A new source package must include:

- one concrete and independently testable responsibility;
- tests at the appropriate tier;
- deliberate public exports;
- documented dependency and ownership boundaries;
- lazy loading for optional dependencies;
- an RFC or ADR when it changes a public contract, persistent format, plugin
  interface, security boundary, or dependency direction.

Empty directories, `.gitkeep`, placeholder classes, and unconditional
`NotImplementedError` implementations do not establish architecture.

Generated indexes, model weights, source corpora, user content, credentials, local
databases, and runtime state remain outside the source tree. Runtime state uses an
explicit user-configured data directory and is never written inside the installed
package.
