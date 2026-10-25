# Local Spark execution

`capicove-execution-spark` is the optional, local Spark execution backend. It is a
Python plugin: Capicove does not currently expose a Spark CLI command.

## Requirements

This is a developer integration recipe. The sample constructs a backend only;
executing a task also requires an immutable input snapshot, reviewed operator
catalog, plan and execution context. Output is a published dataset snapshot, not
a modified input. Replace the scratch path with an authorized private directory.

- Python 3.11 or newer;
- the matching Capicove development release;
- PySpark 4.1 or 4.2;
- Java 17 available through an absolute `JAVA_HOME` or `PATH`.

For a repository checkout:

```bash
uv pip install -e . -e ./plugins/execution-spark
```

The plugin declares no network access. Spark sessions are created only after the
backend is explicitly selected, and only local masters from `local[1]` through
`local[8]` are accepted.

## Discover and create the backend

```python
from capicove.plugins import load_execution_backend_plugin

plugin = load_execution_backend_plugin("spark")
availability = plugin.availability()
if not availability.available:
    raise RuntimeError(availability.detail)

backend = plugin.create(
    {
        "master": "local[2]",
        "app_name": "capicove-local-batch",
        "driver_memory": "1g",
        "executor_memory": "1g",
        "executor_cores": 1,
        "shuffle_partitions": 2,
        "local_scratch_root": "/absolute/path/to/spark-scratch",
    }
)
```

Configuration is closed: unknown keys, non-local masters, environment-based Spark
submit injection, unapproved classpaths, and non-Java-17 runtimes are rejected.

## Reviewed operators

The current backend accepts a single input binding per node and implements these
version-1 operators:

| Operator | Parameters | Result |
| --- | --- | --- |
| `capicove.data.project` | `columns`, `strict` | Select columns, optionally tolerating missing columns |
| `capicove.data.filter` | `predicate` | Apply a closed comparison or null predicate |
| `capicove.data.drop-duplicates` | `columns` | Drop duplicate rows by selected columns |
| `capicove.data.group-count` | `columns`, `count_column` | Group rows and add a count column |
| `capicove.agent-trace.project` | `columns` | Select reviewed Agent trace columns |

Filter predicates support `eq`, `ne`, `lt`, `lte`, `gt`, `gte`, `in`, `is_null`,
and `is_not_null`. The compiler does not evaluate caller Python or arbitrary SQL.

Build a `ProcessingPlanner` catalog from the public `OperatorSpec` contract and the
operator IDs and exact parameters above, then call `backend.submit(plan, context)`
and `backend.poll(reference)`. Execution is synchronous in this release;
cancellation of an active run is not available.

## Publication guarantees

The backend verifies input manifests and hashes, writes versioned Parquet output,
validates the result, and publishes it through `LocalProcessingWorkspace`. Reusing
the same run ID with the same plan is idempotent. A runtime failure records
`SPARK_BATCH_EXECUTION_FAILED` and does not publish an output snapshot. Restarting
the workspace recovers interrupted staging or publication without mutating source
evidence.

See [governed data processing](data-processing.md) for snapshot, plan, lineage, and
recovery contracts.
