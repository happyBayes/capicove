# Plugin System

Capicove plugins extend providers and integrations without requiring edits to a
central registry or adding heavy dependencies to the core package.

## Naming

Plugin distributions use this form:

```text
capicove-<category>-<name>
```

Examples include `capicove-processor-whisper` and
`capicove-retriever-opensearch`. Python import names should use underscores.

## Entry-point groups

```text
capicove.source_adapters
capicove.processors
capicove.document_parsers
capicove.embedding_providers
capicove.retrievers
capicove.rerankers
capicove.execution_backends
capicove.web_search_providers
capicove.skill_packs
```

Document parser plugins follow this explicit lifecycle:

```text
list entry points -> explicitly load descriptor -> check availability -> explicitly create backend -> parse authorized source
```

Listing, descriptor loading, and availability checks must not download models,
open network connections, initialize accelerators, or inspect user documents.
Backend creation and parsing remain separate operations, and `parse` receives an
already-authorized source plus an explicitly installed model profile.

Execution backend entry points return a dependency-light plugin object. Listing
entry points never imports the implementation. Explicit loading validates metadata
and availability; creating a backend is a separate action:

```python
def spark_execution_backend_plugin() -> ExecutionBackendPlugin: ...
```

The version-1 Spark distribution is `capicove-execution-spark`, uses the entry-point
name `spark`, and keeps every PySpark import inside that distribution.

The Spark plugin's dependency-light `operator_contracts.py` publishes the complete
closed operator catalog. An Agent host may call
`SparkExecutionBackendPlugin.operator_specs()` to obtain immutable
`OperatorSpec` values before it creates a backend. Each specification exposes an
exact `input_count` and a closed JSON-compatible `parameter_schema`; discovery does
not import PySpark, start Java, create a Spark session, inspect user data, perform
network I/O, or mutate the filesystem. Core validates operator version, binding
arity, and exact top-level parameter names. The selected Spark implementation
performs nested validation and reviewed DataFrame expression construction.

The published Spark catalog includes unary project, filter, deterministic
drop-duplicates v2, group-count, and Agent-trace projection operators, plus a
two-input equality-only join, global sort, ranking/lag/lead window, and bounded
multi-aggregate operators. Their parameter schemas are the authoritative
machine-readable vocabulary: joins use only the reviewed join types; order entries
use `asc`/`desc` and `first`/`last`; lag/lead offsets and collection `max_items` are
bounded from 1 through 10,000; parameter and column arrays are bounded at 1,000
entries. A changed meaning requires a new operator version.

An entry-point name is a stable, lowercase identifier using ASCII letters, digits,
hyphens, and dots. Loading uses `importlib.metadata.entry_points` through the
Capicove plugin registry.

## Required plugin metadata

Each plugin exposes metadata containing:

- plugin ID and plugin version;
- compatible Capicove contract version range;
- implemented capability and input/output types;
- required optional packages and external binaries;
- CPU, memory, accelerator, and execution-backend requirements;
- network behavior and external service endpoints;
- privacy classification and data-retention behavior;
- determinism, cache, retry, and lineage policies;
- a side-effect-free availability check.

Importing a plugin must not download models, open a network connection, inspect user
content, start a process, or allocate an accelerator.

Execution plugins must identify whether they read or publish recursive dataset
schemas. The Spark plugin accepts schema version 2 arrays, string-keyed maps, and
ordered structs through its public snapshot contracts, preserves JSON-type metadata
inside nested Spark values, and validates recursive nullability before accepting
JSONL or Parquet snapshots. Snapshot publication remains local and immutable; a
multi-input execution appends lineage for every input snapshot.

## Compatibility

The core project will publish reusable contract tests. A plugin is compatible only
when it passes those tests for its declared contract range. Missing optional
dependencies produce an actionable availability result; they must not prevent
`import capicove` or unrelated plugins from working.

Plugin failures are isolated at the call boundary and preserve the originating
artifact, operation, and trace identifiers. Plugins never mutate source evidence.

## Security review

Plugins execute with the user's process permissions unless an execution backend
provides isolation. Installation is therefore a trust decision. Documentation must
identify network access, subprocess use, native code, model-code execution, and
credential requirements.

Changes to entry-point groups, metadata fields, or compatibility rules require an
RFC and ADR.

For the local Spark plugin, plugin metadata continues to declare no network access.
Its catalog and runtime reject remote execution, remote storage, arbitrary SQL or
expressions, Python/Scala/Java UDFs, caller code, package/JAR injection, and
arbitrary Spark configuration. Extending any of these trust boundaries requires
separate accepted governance.
