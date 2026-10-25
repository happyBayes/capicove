# RFC-0007: Spark Batch Data Processing Contracts

- Status: Accepted
- Date: 2026-08-24
- Owners: Capicove maintainers

## Summary

Capicove will expose backend-neutral contracts for immutable dataset snapshots,
closed logical processing plans, processing runs, execution backends, snapshot
catalogs, and lineage. The first distributed implementation is an independently
packaged Spark batch backend for local JSONL and Parquet data.

## Scope

Version 1 supports bounded batch processing only. It reads immutable JSONL or
Parquet snapshots, applies registered versioned operators, writes Parquet into a
private staging directory, verifies every output part, publishes a canonical
manifest, atomically commits a new local dataset version, and appends run and
lineage metadata.

Structured Streaming, Topics, Iceberg, object stores, Kubernetes submission,
arbitrary SQL, arbitrary Python UDFs, Ray invocation, training, and serving are
outside this version-1 operator profile.

## Canonical values

The standard-library-only public contracts are `DataFormat`, `SchemaField`,
`ComplexTypeKind`, `ComplexSchemaType`, `SchemaValueType`, `DatasetSchema`,
`DatasetSnapshotRef`, `LineageEdge`, `OperatorSpec`, `LogicalNode`,
`ProcessingPlan`, `ProcessingContext`, `ProcessingRun`, `BackendRunRef`,
`BackendRunStatus`, and `ProcessingExecution`. The public
`schema_value_type_to_json()` and `schema_value_type_from_json()` codec functions
encode and decode the closed recursive schema value representation.

Persistent identifiers use the prefixes `dataset:`, `snapshot:`, `plan:`, and
`run:`. Versions are positive integers. Digests are lowercase SHA-256 values.
Local version-1 locators are absolute `file://` URIs. Dataset schemas are closed,
ordered, and hashed using canonical UTF-8 JSON.

Schema version 1 continues to contain only the scalar logical types `boolean`,
`int64`, `float64`, `string`, `binary`, `timestamp_utc`, and `json`; its canonical
hash payload is unchanged byte-for-byte. Schema version 2 is required if and only
if a field contains one recursive value: `array`, a string-keyed `map`, or `struct`.
Arrays declare `element_type` and `contains_null`; maps declare `value_type` and
`value_contains_null`; structs declare a nonempty ordered tuple of uniquely named
fields. Inapplicable properties are rejected, and recursive depth is limited to
16. Complex types serialize as closed canonical JSON objects, while struct field
order and every nested nullability flag remain part of the schema identity.

Snapshot metadata persists the schema's public codec representation, including its
schema version, in the private SQLite adapter. Decoding rejects duplicate members
and undeclared fields before rebuilding the public immutable contracts. SQLite is
an implementation detail: neither its storage model nor its APIs are part of the
canonical schema contract.

## Planning and operators

A plan is a topologically ordered tuple of logical nodes. Each node names one
registered operator ID and positive version, existing input bindings, one new
output binding, and a closed JSON parameter object. The plan ID covers every input
snapshot ID, node, normalized parameter, output dataset ID, and operator version.

`OperatorSpec` declares a compatibility-defaulted `input_count` from 1 through 8
and an immutable, JSON-compatible `parameter_schema`. The published schemas use a
closed subset of JSON Schema 2020-12: `type` values `object`, `array`, `string`,
`integer`, `number`, `boolean`, and `null`; `enum`, `oneOf`, `required`,
`properties`, `items`, `minimum`, `maximum`, `minItems`, `maxItems`, and
`additionalProperties: false`. The
planner checks the exact arity, binding availability, exact top-level parameter
names, rejects operator versions declared non-deterministic, and computes the
deterministic plan identity; the Spark compiler owns nested semantic
and Spark-type validation. An Agent host can enumerate a backend's
`operator_specs()` catalog and inspect every `input_count` and
`parameter_schema` without importing PySpark, starting Java, or creating a Spark
session.

The Spark version-1 registry contains unary `capicove.data.project@1`,
`capicove.data.filter@1`, non-deterministic legacy
`capicove.data.drop-duplicates@1`, deterministic
`capicove.data.drop-duplicates@2`,
`capicove.data.group-count@1`, and `capicove.agent-trace.project@1`, plus the
following closed advanced operators:

- `capicove.data.drop-duplicates@2` requires nonempty key `columns` and explicit
  `order_by`. Byte-for-byte identical rows may collapse first; otherwise the
  ordering must uniquely decide a survivor within every key. Ambiguous ties fail
  the run instead of selecting a partition-dependent row.

- `capicove.data.join@1` accepts exactly two ordered inputs and equality-only
  `left_keys` and `right_keys`, `how` in `inner`, `left`, `right`, `full`,
  `left_semi`, or `left_anti`, and a canonical `right_prefix`. Key arrays are
  nonempty, unique, equal in length, and top-level; null keys never match. Standard
  joins retain left columns and coalesced left-named keys, prefix right non-key
  columns, and reject collisions. Semi and anti joins return only left columns.
- `capicove.data.sort@1` accepts one to 1,000 unique top-level `orders`, each
  exactly `column`, `direction` (`asc` or `desc`), and `nulls` (`first` or `last`).
  It performs global ordering; ties have no implicit secondary ordering.
- `capicove.data.window@1` accepts up to 1,000 `partition_by` columns, one to
  1,000 `order_by` entries using the sort vocabulary, and one to 1,000 functions.
  The function set is `row_number`, `rank`, `dense_rank`, `lag`, and `lead`.
  Lag and lead require an input column, a compatible finite JSON-scalar-or-null
  default, and an offset from 1 through 10,000. User-defined window frames are not
  accepted. `row_number`, `lag`, and `lead` require unique ordering within each
  partition; ambiguous ties fail. `rank` and `dense_rank` preserve tied semantics.
- `capicove.data.aggregate@1` accepts up to 1,000 grouping columns and one to
  1,000 named aggregations: `count`, `count_distinct`, `sum`, `avg`, `min`, `max`,
  `collect_list`, and `collect_set`. `sum` and `avg` require numeric input; `min`
  and `max` require reviewed recursively orderable input. Collection values and
  `collect_list` ordering use reviewed orderable scalars. `collect_list` has closed
  ordering and `collect_set` is value-sorted; both require `max_items` from 1
  through 10,000. Per-group ranking limits candidate rows before either collection
  aggregate materializes an array. Output names are bounded, unique, and may not
  collide with grouping columns. Input column names are exact top-level names;
  Spark metacharacters such as `*`, dots, and backticks are never reinterpreted.

Nested field mutation, `explode`, non-equality/cross/as-of joins, approximate or
user-defined aggregation, and model-neutral payload/media flattening remain
deferred. The profile does not interpret caller-provided SQL or import
caller-provided code.

## Run lifecycle

The canonical run model reserves these transitions for execution backends:

```text
submitted -> running | failed | cancelled
running   -> succeeded | failed | cancelled
```

Terminal states never transition. Repeating submission with the same run ID and
plan in the same catalog/output root is idempotent. Reusing a run ID with another
plan or output root in one backend instance fails. The backend-native run identity
contains a non-reversible catalog identifier so references cannot be confused
across roots. Error details use bounded stable codes and do not persist raw Spark
exceptions, source content, paths, or credentials.

The local Spark MVP executes synchronously: `submit` returns only after success or
failure. Active-run cancellation and asynchronous submission are not supported;
`CANCELLED` remains reserved for a later backend version.

## Publication and lineage

Publication ordering is:

```text
stage -> validate -> hash -> manifest -> atomic rename -> atomic metadata success
```

Before rename, publication writes a durable pending marker containing the run,
snapshot, and manifest identities. The metadata success transaction appends the
snapshot and lineage and transitions the run to `succeeded` together. A normal
success removes the marker. If the commit result is ambiguous, the backend reads
durable state before deciding whether to finalize or retract. A new
`LocalProcessingWorkspace` reconciles markers left by process death before
accepting work. It finalizes a durable success or retracts only the exact
hash-verified uncommitted version and marks its run failed. Inputs are never
overwritten, and a failed run leaves neither committed metadata nor a surviving
published version. Snapshot identity covers exact committed file hashes, schema
hash, processing fingerprint, and parent snapshot IDs. Every successful run appends
one lineage edge per input snapshot, including both ordered inputs of a join.
The metadata transaction requires the lineage output snapshot, run identity, and
unique parent set to match the committed snapshot and run exactly; disagreement
rolls back both snapshot and run mutations.
Cross-backend semantic parity compares
canonical rows and schema; it does not require byte-identical Parquet files.

Spark recursively translates canonical arrays to `ArrayType`, string-keyed maps to
`MapType(StringType(), value)`, and structs to ordered `StructType` values. The
inverse conversion rejects non-string map keys and unsupported Spark types. JSON
logical values use reviewed Spark metadata so a JSON string remains distinguishable
from an ordinary string even when nested inside a complex type.

JSONL always receives the declared recursive schema in FAILFAST mode. Parquet
reads only the manifest's exact verified part paths and rejects extra, nested,
duplicate, non-regular, or symbolic-linked Parquet entries before Spark reads it.
Spark may report Parquet fields as
physically nullable after a round trip. For both JSONL and Parquet, the adapter
compares recursive field order and logical types, then scans every canonical
non-nullable struct field, array element, and map value for missing/null values
before accepting a snapshot. Canonical nested nullability remains part of the
schema hash and is never silently weakened.

## Dependency and security policy

Canonical contracts and `capicove.processing` do not import PySpark. Spark lives in
the `capicove-execution-spark` plugin registered through
`capicove.execution_backends`. Plugin enumeration and operator-catalog inspection
are side-effect-free. Explicit loading may check dependency availability but does
not create a session until a run is submitted.

The local profile rejects remote URIs, arbitrary Spark configuration, external
packages, JARs, Spark submit environment injection, network access, telemetry,
SQL text, arbitrary expressions, and Python/Scala/Java UDFs. Operator parameters
cannot supply filesystem paths, packages, JARs, class/module names, environment
variables, remote locators, runtime configuration, or code.

Backend instances atomically bind a run ID to one output root, and local Spark
session creation, use, and shutdown are process-serialized because PySpark owns a
process-global `SparkContext`. Internal `.capicove`, SQLite, lock, and Spark
scratch paths reject symbolic-link traversal; scratch paths also reject commas so
Spark cannot reinterpret one configured path as multiple local directories.
It refuses to reuse an existing Spark context and validates the effective Spark
configuration against a positive key allowlist after startup. JVM launch uses a
`SparkConf` that does not load defaults and rejects submit, gateway, classpath,
Java-option, and Spark-config environment injection before starting a process.
Adding any of those capabilities requires separate change control.

## Compatibility

Adding an optional field or a new operator version is compatible. Changing field
meaning, ID derivation, canonical hashing, lifecycle transitions, publication
ordering, URI policy, plugin group, or trust boundary requires an RFC amendment or
new RFC and an ADR.

## References

- [ADR-0013](../adr/0013-spark-execution-and-snapshot-boundaries.md)
- [Architecture overview](../architecture/overview.md)
- [Module boundaries](../architecture/module-boundaries.md)
