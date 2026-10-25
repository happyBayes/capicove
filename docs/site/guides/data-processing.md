# Governed data processing

Capicove's public processing API describes deterministic data work independently of
Spark or another execution backend. Inputs and outputs are immutable dataset
snapshots; plans bind versioned logical operators to those snapshots; runs preserve
durable state and lineage.

## Dataset contracts

`DatasetSchema` is ordered and closed. Its supported logical types are `boolean`,
`int64`, `float64`, `string`, `binary`, `timestamp_utc`, and `json`. A
`DatasetSnapshotRef` identifies one committed local JSONL or Parquet version with a
manifest hash, schema, processing fingerprint, and parent snapshot IDs.

Snapshot URIs must be absolute `file:` URIs. Source evidence is read-only: a backend
publishes a new snapshot rather than modifying an input in place.

```python
from capicove.contracts import DatasetSchema, SchemaField

schema = DatasetSchema(
    (
        SchemaField("episode_id", "string", False),
        SchemaField("modality", "string", True),
    )
)
print(schema.sha256)
```

## Plans and runs

`ProcessingPlanner` validates that every operator version is registered, node
bindings are topologically valid, and parameters exactly match the operator
specification. It derives the plan ID and processing fingerprint from canonical
content, so the same logical inputs and nodes produce the same identity.

Runtime-owned values such as the run ID, output directory, and request time belong
to `ProcessingContext`; they do not alter the logical plan hash. Run states are
`submitted`, `running`, `succeeded`, `failed`, and `cancelled`. Successful runs bind
the output snapshot and lineage atomically; failed runs publish neither.

## Local workspace and recovery

`LocalProcessingWorkspace` provides the public local catalog and publication
boundary. It serializes execution per output root, stages private output, validates
it before publication, and commits snapshot, lineage, and successful run state
together. Reopening a workspace reconciles interrupted publications: committed
snapshots are finalized, while incomplete publications are retracted and their live
runs are marked failed.

Use `get()` or `latest()` to resolve committed snapshots and `get_run()` to inspect
durable run state. Treat `ProcessingRecoveryError` as a signal that workspace
metadata could not be reconciled safely and needs operator attention.

## Agent trace datasets

`trace_export_snapshot()` validates a version-1 Agent trace export and returns an
immutable JSONL snapshot suitable for a processing plan. Invalid manifests, hashes,
or empty exports are rejected before a backend sees the data.

```python
from pathlib import Path

from capicove.processing import trace_export_snapshot

source = trace_export_snapshot(
    Path("exports/trace-package"), dataset_id="dataset:agent-traces", version=1
)
```

This is an integration example: first create a valid export using the
[Agent trace export guide](agent-trace-export.md), then replace the path with that
export directory. `dataset_id` identifies your dataset; `version` is its positive
snapshot version. The return value is a `DatasetSnapshotRef`, not a processed
dataset. Missing or altered export files fail validation; do not repair their
hashes manually. Re-export the trusted trace source instead.

Continue with [local Spark execution](spark-execution.md) to run the reviewed Spark
operator set.
