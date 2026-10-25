# 0013: Isolate Spark execution behind committed dataset snapshots

- Status: Accepted
- Date: 2026-08-24
- Owners: Capicove maintainers

## Context

Capicove needs Spark for large structured transformations and feature engineering,
while preserving a dependency-light local-first core and allowing Ray to own
multimodal and Agent workloads. Direct SDK coupling would leak execution types into
persistent contracts and make retries, publication, and lineage engine-specific.

## Decision

Capicove core owns standard-library dataset, plan, run, catalog, and lineage
contracts. This includes recursive schema version-2 identities and their public
closed JSON codec, plus `OperatorSpec.input_count` and immutable
`OperatorSpec.parameter_schema`. PySpark is provided only by the independently
packaged `capicove-execution-spark` plugin. Spark and other execution engines
exchange a committed DatasetSnapshotRef rather than runtime object references.

The Spark profile is bounded batch processing over local JSONL and Parquet. Its
dependency-light `operator_contracts.py` module publishes Agent-readable closed
operator metadata before PySpark is imported; `advanced_operators.py` owns
reviewed join, ordering, window, and aggregate expression construction and nested
semantic/type validation. The core planner validates binding availability, exact
top-level parameter names, and declared arity, while Spark validates referenced
columns, recursive Spark types, output layouts, and operator-specific bounds. The
plugin publishes immutable Parquet snapshots through a verified local staging
protocol.

## Consequences

- Installing or importing core Capicove does not require Java or PySpark.
- Spark failures map to Capicove run states without changing canonical types.
- Spark output is renamed atomically, then becomes catalog-visible only when the
  snapshot, lineage, and successful run transition commit in one SQLite
  transaction. A durable pending marker and public `LocalProcessingWorkspace`
  reconcile ambiguous commits and process death before later work is accepted.
- The separately distributed Spark plugin depends only on public processing
  operations and contracts, never core-private filesystem or SQLite adapters.
- The first backend is synchronous and does not advertise active cancellation.
- Schema version 1 scalar hashes remain unchanged. Schema version 2 represents
  only recursive arrays, string-keyed maps, and ordered structs, has a maximum
  depth of 16, and persists through the public schema codec in SQLite snapshot
  metadata. Recursive nullability, order, and metadata distinguishing nested JSON
  strings remain part of the snapshot contract; JSONL and Parquet reads validate
  required struct fields, array elements, and map values before publication.
- The closed Spark registry retains the existing unary operators and adds
  equality-only binary join, global sort, ranking/lag/lead window, and bounded
  multi-aggregate operators. A two-input join results in one lineage edge for each
  source snapshot. Collection output is capped at 10,000 items; caller-defined
  frames, nested mutation, and `explode` remain excluded.
- Agent discovery calls `operator_specs()` without importing PySpark, starting Java,
  creating a session, performing network I/O, or mutating the filesystem.
- SQL text, arbitrary expressions, Python/Scala/Java UDFs, caller code, remote
  masters, remote storage, package/JAR injection, arbitrary configuration, network
  access, and telemetry remain prohibited.
- Ray implementations must not call Spark internals, and Spark implementations
  must not call Ray.
- Streaming, Iceberg, object stores, Kubernetes, and remote credentials require
  separate designs, as do asynchronous cancellation and model-neutral trace
  flattening.

## Alternatives considered

### Put PySpark in the core distribution

Rejected because it makes a large optional runtime mandatory and violates lazy
dependency and canonical-contract boundaries.

### Invoke Ray from Spark UDFs

Rejected because it creates nested schedulers, unclear backpressure, ambiguous
retries, and incomplete lineage.

### Exchange in-memory engine objects

Rejected because Spark DataFrames and Ray object references are not stable,
persistent, or backend-neutral contracts.

## References

- [RFC-0007](../rfcs/0007-spark-data-processing-contracts.md)
- [ADR-0001](0001-python-modular-monolith.md)
- [ADR-0002](0002-plugin-entry-points.md)
