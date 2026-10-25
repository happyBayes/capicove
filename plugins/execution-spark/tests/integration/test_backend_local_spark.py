"""End-to-end local Spark batch backend tests."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest
from capicove_execution_spark.backend import SparkExecutionBackend
from capicove_execution_spark.compiler import spark_operator_specs
from capicove_execution_spark.io import SparkDatasetIO
from capicove_execution_spark.session import (
    SparkBackendConfiguration,
    create_spark_session,
)

from capicove.contracts.data import (
    ComplexSchemaType,
    ComplexTypeKind,
    DataFormat,
    DatasetSchema,
    DatasetSnapshotRef,
    SchemaField,
)
from capicove.processing._adapters.sqlite import SqliteProcessingStore
from capicove.processing.local import LocalProcessingWorkspace
from capicove.processing.models import (
    BackendRunRef,
    LogicalNode,
    ProcessingContext,
    ProcessingRun,
    RunState,
)
from capicove.processing.planner import ProcessingPlanner


pytestmark = pytest.mark.spark


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _snapshot(
    source: Path,
    payload: bytes,
    *,
    dataset_id: str = "dataset:trace-source",
    schema: DatasetSchema | None = None,
) -> DatasetSnapshotRef:
    return DatasetSnapshotRef(
        dataset_id=dataset_id,
        version=1,
        snapshot_id=f"snapshot:{_digest(payload)}",
        uri=source.resolve().as_uri(),
        data_format=DataFormat.JSONL,
        schema=schema
        or DatasetSchema(
            (
                SchemaField("episode_id", "string", True),
                SchemaField("record_type", "string", True),
                SchemaField("modality", "string", True),
            )
        ),
        manifest_sha256=_digest(b"source-manifest"),
        processing_fingerprint=_digest(b"source"),
        parent_snapshot_ids=(),
    )


def _node(
    node_id: str,
    operator_id: str,
    input_binding: str,
    output_binding: str,
    parameters: dict[str, object],
    *,
    operator_version: int = 1,
) -> LogicalNode:
    return LogicalNode(
        node_id=node_id,
        operator_id=operator_id,
        operator_version=operator_version,
        input_bindings=(input_binding,),
        output_binding=output_binding,
        parameters=parameters,
    )


def _advanced_node(
    node_id: str,
    operator_id: str,
    input_bindings: tuple[str, ...],
    output_binding: str,
    parameters: dict[str, object],
) -> LogicalNode:
    return LogicalNode(
        node_id=node_id,
        operator_id=operator_id,
        operator_version=1,
        input_bindings=input_bindings,
        output_binding=output_binding,
        parameters=parameters,
    )


def _configuration(tmp_path: Path) -> SparkBackendConfiguration:
    return SparkBackendConfiguration(
        master="local[2]",
        app_name="capicove-backend-test",
        local_scratch_root=(tmp_path / "spark").resolve(),
    )


def test_backend_publishes_advanced_two_input_plan_with_recursive_schema_and_lineage(
    tmp_path: Path,
) -> None:
    profile_type = ComplexSchemaType(
        ComplexTypeKind.STRUCT,
        fields=(
            SchemaField("tier", "string", False),
            SchemaField("level", "int64", False),
        ),
    )
    published_profile_type = ComplexSchemaType(
        ComplexTypeKind.STRUCT,
        fields=(
            SchemaField("tier", "string", True),
            SchemaField("level", "int64", True),
        ),
    )
    left_schema = DatasetSchema(
        (
            SchemaField("account_id", "string", False),
            SchemaField("position", "int64", False),
            SchemaField("amount", "int64", False),
            SchemaField("profile", profile_type, False),
        ),
        schema_version=2,
    )
    right_schema = DatasetSchema(
        (
            SchemaField("account_id", "string", False),
            SchemaField("region", "string", False),
        )
    )
    left_path = tmp_path / "left.jsonl"
    left_payload = (
        b'{"account_id":"account-a","position":2,"amount":10,'
        b'"profile":{"tier":"gold","level":3}}\n'
        b'{"account_id":"account-a","position":1,"amount":20,'
        b'"profile":{"tier":"gold","level":3}}\n'
        b'{"account_id":"account-b","position":1,"amount":7,'
        b'"profile":{"tier":"silver","level":1}}\n'
    )
    right_path = tmp_path / "right.jsonl"
    right_payload = (
        b'{"account_id":"account-a","region":"west"}\n{"account_id":"account-b","region":"east"}\n'
    )
    left_path.write_bytes(left_payload)
    right_path.write_bytes(right_payload)
    left = _snapshot(
        left_path,
        left_payload,
        dataset_id="dataset:advanced-left",
        schema=left_schema,
    )
    right = _snapshot(
        right_path,
        right_payload,
        dataset_id="dataset:advanced-right",
        schema=right_schema,
    )
    plan = ProcessingPlanner(spark_operator_specs()).build(
        input_bindings={"left": left, "right": right},
        nodes=(
            _advanced_node(
                "join_accounts",
                "capicove.data.join",
                ("left", "right"),
                "joined",
                {
                    "how": "inner",
                    "left_keys": ["account_id"],
                    "right_keys": ["account_id"],
                    "right_prefix": "right_",
                },
            ),
            _advanced_node(
                "rank_account_events",
                "capicove.data.window",
                ("joined",),
                "windowed",
                {
                    "partition_by": ["account_id"],
                    "order_by": [{"column": "position", "direction": "asc", "nulls": "last"}],
                    "functions": [{"function": "row_number", "output_column": "event_rank"}],
                },
            ),
            _advanced_node(
                "summarize_accounts",
                "capicove.data.aggregate",
                ("windowed",),
                "aggregated",
                {
                    "group_by": ["account_id", "right_region"],
                    "aggregations": [
                        {"function": "count", "output_column": "event_count"},
                        {
                            "function": "sum",
                            "input_column": "amount",
                            "output_column": "total_amount",
                        },
                        {
                            "function": "max",
                            "input_column": "event_rank",
                            "output_column": "last_rank",
                        },
                        {
                            "function": "max",
                            "input_column": "profile",
                            "output_column": "highest_profile",
                        },
                    ],
                },
            ),
            _advanced_node(
                "sort_summaries",
                "capicove.data.sort",
                ("aggregated",),
                "result",
                {
                    "orders": [
                        {"column": "total_amount", "direction": "desc", "nulls": "last"},
                        {"column": "account_id", "direction": "asc", "nulls": "last"},
                    ]
                },
            ),
        ),
        output_dataset_id="dataset:advanced-account-summary",
    )
    output_root = (tmp_path / "outputs").resolve()
    backend = SparkExecutionBackend(_configuration(tmp_path))

    reference = backend.submit(
        plan,
        ProcessingContext(
            run_id="run:advanced-two-input-1",
            output_root=output_root,
            requested_at=datetime.now(UTC),
        ),
    )

    status = backend.poll(reference)
    assert status.state is RunState.SUCCEEDED
    store = SqliteProcessingStore(output_root / ".capicove" / "processing.sqlite3")
    output = store.latest("dataset:advanced-account-summary")
    assert output is not None
    assert output.parent_snapshot_ids == (left.snapshot_id, right.snapshot_id)
    with sqlite3.connect(output_root / ".capicove" / "processing.sqlite3") as connection:
        row = connection.execute(
            "SELECT lineage_json FROM dataset_snapshots WHERE snapshot_id = ?",
            (output.snapshot_id,),
        ).fetchone()
    assert row is not None
    lineage = json.loads(row[0])
    assert len(lineage) == 2
    assert {edge["input_snapshot_id"] for edge in lineage} == {
        left.snapshot_id,
        right.snapshot_id,
    }
    assert left_path.is_file()
    assert right_path.is_file()
    assert left_path.read_bytes() == left_payload
    assert right_path.read_bytes() == right_payload
    assert output.schema == DatasetSchema(
        (
            SchemaField("account_id", "string", True),
            SchemaField("right_region", "string", True),
            SchemaField("event_count", "int64", False),
            SchemaField("total_amount", "int64", True),
            SchemaField("last_rank", "int64", True),
            SchemaField("highest_profile", published_profile_type, True),
        ),
        schema_version=2,
    )
    spark = create_spark_session(_configuration(tmp_path))
    try:
        rows = SparkDatasetIO(spark).read(output).orderBy("account_id").collect()
        assert [row.asDict(recursive=True) for row in rows] == [
            {
                "account_id": "account-a",
                "right_region": "west",
                "event_count": 2,
                "total_amount": 30,
                "last_rank": 2,
                "highest_profile": {"tier": "gold", "level": 3},
            },
            {
                "account_id": "account-b",
                "right_region": "east",
                "event_count": 1,
                "total_amount": 7,
                "last_rank": 1,
                "highest_profile": {"tier": "silver", "level": 1},
            },
        ]
    finally:
        spark.stop()


def test_backend_publishes_versioned_parquet_and_lineage_idempotently(
    tmp_path: Path,
) -> None:
    source = tmp_path / "trace.jsonl"
    payload = (
        b'{"episode_id":"episode-1","record_type":"tool.result","modality":"image"}\n'
        b'{"episode_id":"episode-1","record_type":"tool.result","modality":"image"}\n'
        b'{"episode_id":"episode-2","record_type":"message.observed","modality":"text"}\n'
    )
    source.write_bytes(payload)
    snapshot = _snapshot(source, payload)
    plan = ProcessingPlanner(spark_operator_specs()).build(
        input_bindings={"source": snapshot},
        nodes=(
            _node(
                "filter_results",
                "capicove.data.filter",
                "source",
                "filtered",
                {
                    "predicate": {
                        "column": "record_type",
                        "operator": "eq",
                        "value": "tool.result",
                    }
                },
            ),
            _node(
                "deduplicate",
                "capicove.data.drop-duplicates",
                "filtered",
                "unique",
                {
                    "columns": ["episode_id", "record_type", "modality"],
                    "order_by": [
                        {
                            "column": "episode_id",
                            "direction": "asc",
                            "nulls": "last",
                        }
                    ],
                },
                operator_version=2,
            ),
            _node(
                "group_count",
                "capicove.data.group-count",
                "unique",
                "result",
                {"columns": ["modality"], "count_column": "records"},
            ),
        ),
        output_dataset_id="dataset:trace-summary",
    )
    output_root = (tmp_path / "outputs").resolve()
    context = ProcessingContext(
        run_id="run:trace-summary-1",
        output_root=output_root,
        requested_at=datetime.now(UTC),
    )
    backend = SparkExecutionBackend(_configuration(tmp_path))

    reference = backend.submit(plan, context)
    repeated = backend.submit(plan, context)

    assert repeated == reference
    status = backend.poll(reference)
    assert status.state is RunState.SUCCEEDED
    assert source.read_bytes() == payload
    store = SqliteProcessingStore(output_root / ".capicove" / "processing.sqlite3")
    output = store.latest("dataset:trace-summary")
    assert output is not None
    assert output.version == 1
    assert output.parent_snapshot_ids == (snapshot.snapshot_id,)
    spark = create_spark_session(_configuration(tmp_path))
    try:
        rows = SparkDatasetIO(spark).read(output).collect()
        assert [row.asDict(recursive=True) for row in rows] == [{"modality": "image", "records": 1}]
        published_root = Path(unquote(urlsplit(output.uri).path))
        part = next((published_root / "data").glob("part-*.parquet"))
        part.write_bytes(part.read_bytes() + b"changed")
        with pytest.raises(ValueError, match=r"hash|size"):
            SparkDatasetIO(spark).read(output)
    finally:
        spark.stop()


def test_backend_records_runtime_failure_without_publishing_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "trace.jsonl"
    payload = b'{"episode_id":"episode-1","record_type":"tool.result","modality":"image"}\n'
    source.write_bytes(payload)
    snapshot = _snapshot(source, payload)
    plan = ProcessingPlanner(spark_operator_specs()).build(
        input_bindings={"source": snapshot},
        nodes=(
            _node(
                "missing_project",
                "capicove.data.project",
                "source",
                "result",
                {"columns": ["missing"], "strict": True},
            ),
        ),
        output_dataset_id="dataset:failed-output",
    )
    output_root = (tmp_path / "outputs").resolve()
    context = ProcessingContext(
        run_id="run:failed-1",
        output_root=output_root,
        requested_at=datetime.now(UTC),
    )
    backend = SparkExecutionBackend(_configuration(tmp_path))

    reference = backend.submit(plan, context)

    status = backend.poll(reference)
    assert status.state is RunState.FAILED
    assert status.detail == "SPARK_BATCH_EXECUTION_FAILED"
    store = SqliteProcessingStore(output_root / ".capicove" / "processing.sqlite3")
    assert store.latest("dataset:failed-output") is None
    assert source.read_bytes() == payload


@pytest.mark.parametrize("interrupted_state", (RunState.SUBMITTED, RunState.RUNNING))
def test_backend_recovers_execution_interrupted_before_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupted_state: RunState,
) -> None:
    source = tmp_path / "trace.jsonl"
    payload = b'{"episode_id":"episode-1","record_type":"tool.result","modality":"image"}\n'
    source.write_bytes(payload)
    plan = ProcessingPlanner(spark_operator_specs()).build(
        input_bindings={"source": _snapshot(source, payload)},
        nodes=(
            _node(
                "project",
                "capicove.data.project",
                "source",
                "result",
                {"columns": ["episode_id"], "strict": True},
            ),
        ),
        output_dataset_id="dataset:interrupted-output",
    )
    output_root = (tmp_path / "outputs").resolve()
    requested_at = datetime.now(UTC)
    context = ProcessingContext(
        run_id="run:interrupted-before-staging",
        output_root=output_root,
        requested_at=requested_at,
    )
    workspace = LocalProcessingWorkspace(output_root)
    workspace.create(
        ProcessingRun(
            run_id=context.run_id,
            plan_id=plan.plan_id,
            backend_id="capicove.execution.spark",
            state=RunState.SUBMITTED,
            created_at=requested_at,
            updated_at=requested_at,
        )
    )
    if interrupted_state is RunState.RUNNING:
        workspace.transition(
            context.run_id,
            RunState.SUBMITTED,
            RunState.RUNNING,
            native_run_id="spark-local:interrupted",
        )

    def fail_if_executed(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("an interrupted run must not be executed again")

    monkeypatch.setattr(SparkExecutionBackend, "_execute", fail_if_executed)
    backend = SparkExecutionBackend(_configuration(tmp_path))

    reference = backend.submit(plan, context)

    status = backend.poll(reference)
    assert status.state is RunState.FAILED
    assert status.detail == "RECOVERED_INCOMPLETE_EXECUTION"
    assert status.output_snapshot_id is None
    assert workspace.latest(plan.output_dataset_id) is None
    durable = workspace.get_run(context.run_id)
    assert durable is not None
    assert durable.native_run_id == reference.native_run_id
    recovered_at = durable.updated_at

    assert backend.submit(plan, context) == reference
    repeated = workspace.get_run(context.run_id)
    assert repeated is not None
    assert repeated.updated_at == recovered_at
    assert repeated.detail == "RECOVERED_INCOMPLETE_EXECUTION"


def test_backend_preserves_native_run_id_created_during_lock_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "trace.jsonl"
    payload = b'{"episode_id":"episode-1","record_type":"tool.result","modality":"image"}\n'
    source.write_bytes(payload)
    plan = ProcessingPlanner(spark_operator_specs()).build(
        input_bindings={"source": _snapshot(source, payload)},
        nodes=(
            _node(
                "project",
                "capicove.data.project",
                "source",
                "result",
                {"columns": ["episode_id"], "strict": True},
            ),
        ),
        output_dataset_id="dataset:concurrent-output",
    )
    output_root = (tmp_path / "outputs").resolve()
    requested_at = datetime.now(UTC)
    context = ProcessingContext(
        run_id="run:created-during-lock-acquisition",
        output_root=output_root,
        requested_at=requested_at,
    )
    original_execution_lock = LocalProcessingWorkspace.execution_lock

    @contextmanager
    def insert_peer_run(workspace: LocalProcessingWorkspace) -> Iterator[None]:
        with original_execution_lock(workspace):
            if workspace.get_run(context.run_id) is None:
                workspace.create(
                    ProcessingRun(
                        run_id=context.run_id,
                        plan_id=plan.plan_id,
                        backend_id="capicove.execution.spark",
                        state=RunState.SUBMITTED,
                        created_at=requested_at,
                        updated_at=requested_at,
                        native_run_id="spark-local:peer",
                    )
                )
            yield

    def fail_if_executed(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("a concurrently persisted run must not be executed again")

    monkeypatch.setattr(LocalProcessingWorkspace, "execution_lock", insert_peer_run)
    monkeypatch.setattr(SparkExecutionBackend, "_execute", fail_if_executed)
    backend = SparkExecutionBackend(_configuration(tmp_path))

    reference = backend.submit(plan, context)

    assert reference.native_run_id == "spark-local:peer"
    durable = LocalProcessingWorkspace(output_root).get_run(context.run_id)
    assert durable is not None
    assert durable.native_run_id == "spark-local:peer"
    assert durable.state is RunState.FAILED
    assert durable.detail == "RECOVERED_INCOMPLETE_EXECUTION"


def test_backend_preserves_success_when_spark_stop_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyspark.sql import SparkSession

    source = tmp_path / "trace.jsonl"
    payload = b'{"episode_id":"episode-1","record_type":"tool.result","modality":"image"}\n'
    source.write_bytes(payload)
    plan = ProcessingPlanner(spark_operator_specs()).build(
        input_bindings={"source": _snapshot(source, payload)},
        nodes=(
            _node(
                "project",
                "capicove.data.project",
                "source",
                "result",
                {"columns": ["episode_id"], "strict": True},
            ),
        ),
        output_dataset_id="dataset:stop-failure",
    )
    context = ProcessingContext(
        run_id="run:stop-failure",
        output_root=(tmp_path / "outputs").resolve(),
        requested_at=datetime.now(UTC),
    )
    original_stop = SparkSession.stop

    def stop_then_fail(session: SparkSession) -> None:
        original_stop(session)
        raise RuntimeError("simulated cleanup failure")

    monkeypatch.setattr(SparkSession, "stop", stop_then_fail)
    backend = SparkExecutionBackend(_configuration(tmp_path))

    reference = backend.submit(plan, context)

    assert backend.poll(reference).state is RunState.SUCCEEDED


def test_backend_rejects_reusing_run_id_with_another_output_root(tmp_path: Path) -> None:
    source = tmp_path / "trace.jsonl"
    payload = b'{"episode_id":"episode-1","record_type":"tool.result","modality":"image"}\n'
    source.write_bytes(payload)
    plan = ProcessingPlanner(spark_operator_specs()).build(
        input_bindings={"source": _snapshot(source, payload)},
        nodes=(
            _node(
                "project",
                "capicove.data.project",
                "source",
                "result",
                {"columns": ["episode_id"], "strict": True},
            ),
        ),
        output_dataset_id="dataset:projected",
    )
    backend = SparkExecutionBackend(_configuration(tmp_path))
    first = ProcessingContext(
        run_id="run:cross-root",
        output_root=(tmp_path / "first").resolve(),
        requested_at=datetime.now(UTC),
    )
    second = ProcessingContext(
        run_id="run:cross-root",
        output_root=(tmp_path / "second").resolve(),
        requested_at=datetime.now(UTC),
    )
    reference = backend.submit(plan, first)

    with pytest.raises(ValueError, match="output root"):
        backend.submit(plan, second)
    forged = BackendRunRef(reference.run_id, reference.backend_id, "spark-local:other-catalog")
    with pytest.raises(ValueError, match="native run identity"):
        backend.poll(forged)


def test_backend_retracts_published_files_when_metadata_commit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "trace.jsonl"
    payload = b'{"episode_id":"episode-1","record_type":"tool.result","modality":"image"}\n'
    source.write_bytes(payload)
    plan = ProcessingPlanner(spark_operator_specs()).build(
        input_bindings={"source": _snapshot(source, payload)},
        nodes=(
            _node(
                "project",
                "capicove.data.project",
                "source",
                "result",
                {"columns": ["episode_id"], "strict": True},
            ),
        ),
        output_dataset_id="dataset:commit-failure",
    )
    output_root = (tmp_path / "outputs").resolve()

    def fail_commit(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("credential=must-not-be-persisted")

    monkeypatch.setattr(LocalProcessingWorkspace, "commit_success", fail_commit)
    backend = SparkExecutionBackend(_configuration(tmp_path))
    reference = backend.submit(
        plan,
        ProcessingContext(
            run_id="run:commit-failure",
            output_root=output_root,
            requested_at=datetime.now(UTC),
        ),
    )

    assert backend.poll(reference).detail == "SPARK_BATCH_EXECUTION_FAILED"
    assert not (output_root / "commit-failure" / "v1").exists()


def test_backend_recovers_after_initial_publication_retract_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "trace.jsonl"
    payload = b'{"episode_id":"episode-1","record_type":"tool.result","modality":"image"}\n'
    source.write_bytes(payload)
    plan = ProcessingPlanner(spark_operator_specs()).build(
        input_bindings={"source": _snapshot(source, payload)},
        nodes=(
            _node(
                "project",
                "capicove.data.project",
                "source",
                "result",
                {"columns": ["episode_id"], "strict": True},
            ),
        ),
        output_dataset_id="dataset:retract-recovery",
    )
    output_root = (tmp_path / "outputs").resolve()

    def fail_commit(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("simulated metadata commit failure")

    monkeypatch.setattr(LocalProcessingWorkspace, "commit_success", fail_commit)
    monkeypatch.setattr(LocalProcessingWorkspace, "retract", lambda *args: False)
    backend = SparkExecutionBackend(_configuration(tmp_path))

    reference = backend.submit(
        plan,
        ProcessingContext(
            run_id="run:retract-recovery",
            output_root=output_root,
            requested_at=datetime.now(UTC),
        ),
    )

    status = backend.poll(reference)
    assert status.state is RunState.FAILED
    assert status.detail == "RECOVERED_INCOMPLETE_PUBLICATION"
    assert not (output_root / "retract-recovery" / "v1").exists()


def test_backend_keeps_successful_snapshot_when_commit_returns_ambiguously(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "trace.jsonl"
    payload = b'{"episode_id":"episode-1","record_type":"tool.result","modality":"image"}\n'
    source.write_bytes(payload)
    plan = ProcessingPlanner(spark_operator_specs()).build(
        input_bindings={"source": _snapshot(source, payload)},
        nodes=(
            _node(
                "project",
                "capicove.data.project",
                "source",
                "result",
                {"columns": ["episode_id"], "strict": True},
            ),
        ),
        output_dataset_id="dataset:ambiguous-commit",
    )
    output_root = (tmp_path / "outputs").resolve()
    real_commit = LocalProcessingWorkspace.commit_success

    def commit_then_raise(*args: object, **kwargs: object) -> None:
        real_commit(*args, **kwargs)  # type: ignore[arg-type]
        raise RuntimeError("ambiguous database return")

    monkeypatch.setattr(LocalProcessingWorkspace, "commit_success", commit_then_raise)
    backend = SparkExecutionBackend(_configuration(tmp_path))
    reference = backend.submit(
        plan,
        ProcessingContext(
            run_id="run:ambiguous-commit",
            output_root=output_root,
            requested_at=datetime.now(UTC),
        ),
    )

    status = backend.poll(reference)
    assert status.state is RunState.SUCCEEDED
    assert status.output_snapshot_id is not None
    assert (output_root / "ambiguous-commit" / "v1" / "manifest.json").is_file()
    assert not (output_root / "ambiguous-commit" / "v1" / ".capicove-pending.json").exists()


def test_backend_preserves_json_expression_metadata_in_published_snapshot(
    tmp_path: Path,
) -> None:
    left_path = tmp_path / "json-left.jsonl"
    left_payload = b'{"key":"{\\"id\\":1}","position":1}\n{"key":"{\\"id\\":2}","position":2}\n'
    right_path = tmp_path / "json-right.jsonl"
    right_payload = b'{"key":"{\\"id\\":1}"}\n{"key":"{\\"id\\":2}"}\n'
    left_path.write_bytes(left_payload)
    right_path.write_bytes(right_payload)
    left = _snapshot(
        left_path,
        left_payload,
        dataset_id="dataset:json-left",
        schema=DatasetSchema(
            (
                SchemaField("key", "json", False),
                SchemaField("position", "int64", False),
            )
        ),
    )
    right = _snapshot(
        right_path,
        right_payload,
        dataset_id="dataset:json-right",
        schema=DatasetSchema((SchemaField("key", "json", False),)),
    )
    plan = ProcessingPlanner(spark_operator_specs()).build(
        input_bindings={"left": left, "right": right},
        nodes=(
            _advanced_node(
                "join_json",
                "capicove.data.join",
                ("left", "right"),
                "joined",
                {
                    "how": "inner",
                    "left_keys": ["key"],
                    "right_keys": ["key"],
                    "right_prefix": "right_",
                },
            ),
            _advanced_node(
                "window_json",
                "capicove.data.window",
                ("joined",),
                "windowed",
                {
                    "partition_by": [],
                    "order_by": [{"column": "position", "direction": "asc", "nulls": "last"}],
                    "functions": [
                        {
                            "function": "lag",
                            "output_column": "previous_key",
                            "input_column": "key",
                            "offset": 1,
                            "default": None,
                        },
                        {
                            "function": "lead",
                            "output_column": "next_key",
                            "input_column": "key",
                            "offset": 1,
                            "default": None,
                        },
                    ],
                },
            ),
            _advanced_node(
                "collect_json",
                "capicove.data.aggregate",
                ("windowed",),
                "result",
                {
                    "group_by": ["key", "previous_key", "next_key"],
                    "aggregations": [
                        {
                            "function": "collect_list",
                            "input_column": "key",
                            "order_by": [
                                {
                                    "column": "position",
                                    "direction": "asc",
                                    "nulls": "last",
                                }
                            ],
                            "max_items": 10,
                            "output_column": "listed_keys",
                        },
                        {
                            "function": "collect_set",
                            "input_column": "key",
                            "max_items": 10,
                            "output_column": "unique_keys",
                        },
                    ],
                },
            ),
        ),
        output_dataset_id="dataset:json-expression-metadata",
    )
    output_root = (tmp_path / "json-outputs").resolve()
    backend = SparkExecutionBackend(_configuration(tmp_path))

    reference = backend.submit(
        plan,
        ProcessingContext(
            run_id="run:json-expression-metadata",
            output_root=output_root,
            requested_at=datetime.now(UTC),
        ),
    )

    assert backend.poll(reference).state is RunState.SUCCEEDED
    output = SqliteProcessingStore(output_root / ".capicove" / "processing.sqlite3").latest(
        "dataset:json-expression-metadata"
    )
    assert output is not None
    nullable_json_array = ComplexSchemaType(
        ComplexTypeKind.ARRAY,
        element_type="json",
        contains_null=True,
    )
    nonnull_json_array = ComplexSchemaType(
        ComplexTypeKind.ARRAY,
        element_type="json",
        contains_null=False,
    )
    assert output.schema == DatasetSchema(
        (
            SchemaField("key", "json", True),
            SchemaField("previous_key", "json", True),
            SchemaField("next_key", "json", True),
            SchemaField("listed_keys", nullable_json_array, False),
            SchemaField("unique_keys", nonnull_json_array, False),
        ),
        schema_version=2,
    )
    spark = create_spark_session(_configuration(tmp_path))
    try:
        rows = SparkDatasetIO(spark).read(output).orderBy("key").collect()
        assert [row.asDict(recursive=True) for row in rows] == [
            {
                "key": '{"id":1}',
                "previous_key": None,
                "next_key": '{"id":2}',
                "listed_keys": ['{"id":1}'],
                "unique_keys": ['{"id":1}'],
            },
            {
                "key": '{"id":2}',
                "previous_key": '{"id":1}',
                "next_key": None,
                "listed_keys": ['{"id":2}'],
                "unique_keys": ['{"id":2}'],
            },
        ]
    finally:
        spark.stop()


def test_backend_parentage_is_sorted_and_independent_of_mapping_insertion(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "a-source.jsonl"
    first_payload = b'{"episode_id":"a","record_type":"event","modality":"text"}\n'
    second_path = tmp_path / "z-source.jsonl"
    second_payload = b'{"episode_id":"z","record_type":"event","modality":"text"}\n'
    first_path.write_bytes(first_payload)
    second_path.write_bytes(second_payload)
    first = _snapshot(first_path, first_payload, dataset_id="dataset:a-source")
    second = _snapshot(second_path, second_payload, dataset_id="dataset:z-source")
    nodes = (
        _node(
            "project",
            "capicove.data.project",
            "a_source",
            "result",
            {"columns": ["episode_id"], "strict": True},
        ),
    )
    planner = ProcessingPlanner(spark_operator_specs())
    forward = planner.build(
        input_bindings={"z_source": second, "a_source": first},
        nodes=nodes,
        output_dataset_id="dataset:deterministic-parentage",
    )
    reversed_plan = planner.build(
        input_bindings={"a_source": first, "z_source": second},
        nodes=nodes,
        output_dataset_id="dataset:deterministic-parentage",
    )
    assert forward.plan_id == reversed_plan.plan_id
    assert forward.processing_fingerprint == reversed_plan.processing_fingerprint

    observed_parentage: list[tuple[str, ...]] = []
    observed_lineage: list[tuple[str, ...]] = []
    for index, plan in enumerate((forward, reversed_plan), start=1):
        output_root = (tmp_path / f"parentage-output-{index}").resolve()
        backend = SparkExecutionBackend(_configuration(tmp_path))
        reference = backend.submit(
            plan,
            ProcessingContext(
                run_id=f"run:deterministic-parentage-{index}",
                output_root=output_root,
                requested_at=datetime.now(UTC),
            ),
        )
        assert backend.poll(reference).state is RunState.SUCCEEDED
        store = SqliteProcessingStore(output_root / ".capicove" / "processing.sqlite3")
        output = store.latest("dataset:deterministic-parentage")
        assert output is not None
        observed_parentage.append(output.parent_snapshot_ids)
        with sqlite3.connect(output_root / ".capicove" / "processing.sqlite3") as connection:
            row = connection.execute(
                "SELECT lineage_json FROM dataset_snapshots WHERE snapshot_id = ?",
                (output.snapshot_id,),
            ).fetchone()
        assert row is not None
        observed_lineage.append(tuple(edge["input_snapshot_id"] for edge in json.loads(row[0])))

    expected = (first.snapshot_id, second.snapshot_id)
    assert observed_parentage == [expected, expected]
    assert observed_lineage == [expected, expected]


def test_backend_self_join_publishes_one_parent_and_one_lineage_edge(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "self-join.jsonl"
    payload = (
        b'{"episode_id":"one","record_type":"event","modality":"text"}\n'
        b'{"episode_id":"two","record_type":"event","modality":"image"}\n'
    )
    source_path.write_bytes(payload)
    source = _snapshot(source_path, payload, dataset_id="dataset:self-join-source")
    plan = ProcessingPlanner(spark_operator_specs()).build(
        input_bindings={"left": source, "right": source},
        nodes=(
            _advanced_node(
                "self_join",
                "capicove.data.join",
                ("left", "right"),
                "result",
                {
                    "how": "inner",
                    "left_keys": ["episode_id"],
                    "right_keys": ["episode_id"],
                    "right_prefix": "right_",
                },
            ),
        ),
        output_dataset_id="dataset:self-join-output",
    )
    output_root = (tmp_path / "self-join-output").resolve()
    backend = SparkExecutionBackend(_configuration(tmp_path))

    reference = backend.submit(
        plan,
        ProcessingContext(
            run_id="run:self-join",
            output_root=output_root,
            requested_at=datetime.now(UTC),
        ),
    )

    assert backend.poll(reference).state is RunState.SUCCEEDED
    output = SqliteProcessingStore(output_root / ".capicove" / "processing.sqlite3").latest(
        "dataset:self-join-output"
    )
    assert output is not None
    assert output.parent_snapshot_ids == (source.snapshot_id,)
    with sqlite3.connect(output_root / ".capicove" / "processing.sqlite3") as connection:
        row = connection.execute(
            "SELECT lineage_json FROM dataset_snapshots WHERE snapshot_id = ?",
            (output.snapshot_id,),
        ).fetchone()
    assert row is not None
    lineage = json.loads(row[0])
    assert [edge["input_snapshot_id"] for edge in lineage] == [source.snapshot_id]


def test_backend_stages_before_session_start_and_recovers_process_death(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "process-death.jsonl"
    payload = b'{"episode_id":"one","record_type":"event","modality":"text"}\n'
    source_path.write_bytes(payload)
    plan = ProcessingPlanner(spark_operator_specs()).build(
        input_bindings={"source": _snapshot(source_path, payload)},
        nodes=(
            _node(
                "project",
                "capicove.data.project",
                "source",
                "result",
                {"columns": ["episode_id"], "strict": True},
            ),
        ),
        output_dataset_id="dataset:process-death-output",
    )
    output_root = (tmp_path / "process-death-output").resolve()
    context = ProcessingContext(
        run_id="run:process-death",
        output_root=output_root,
        requested_at=datetime.now(UTC),
    )

    def stop_process(_: SparkBackendConfiguration) -> None:
        raise SystemExit("simulated process death")

    monkeypatch.setattr("capicove_execution_spark.backend.create_spark_session", stop_process)
    backend = SparkExecutionBackend(_configuration(tmp_path))

    with pytest.raises(SystemExit, match="process death"):
        backend.submit(plan, context)

    staging_root = output_root / ".staging"
    staged = tuple(staging_root.iterdir())
    assert len(staged) == 1
    assert (staged[0] / ".capicove-pending.json").is_file()
    running = SqliteProcessingStore(output_root / ".capicove" / "processing.sqlite3").get_run(
        context.run_id
    )
    assert running is not None and running.state is RunState.RUNNING

    reconciled = LocalProcessingWorkspace(output_root).get_run(context.run_id)

    assert reconciled is not None
    assert reconciled.state is RunState.FAILED
    assert reconciled.detail == "RECOVERED_INCOMPLETE_PUBLICATION"
    assert tuple(staging_root.iterdir()) == ()
