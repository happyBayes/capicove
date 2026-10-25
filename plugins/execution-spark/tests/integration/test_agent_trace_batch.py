"""End-to-end Spark batch test for a validated Agent trace export."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from capicove_execution_spark.backend import SparkExecutionBackend
from capicove_execution_spark.compiler import spark_operator_specs
from capicove_execution_spark.io import SparkDatasetIO
from capicove_execution_spark.session import (
    SparkBackendConfiguration,
    create_spark_session,
)

from capicove.export import TraceExporter
from capicove.processing import LogicalNode, ProcessingContext, ProcessingPlanner
from capicove.processing._adapters.sqlite import SqliteProcessingStore
from capicove.processing.trace_dataset import trace_export_snapshot
from capicove.traces import (
    EpisodeCompleted,
    EpisodeStarted,
    EpisodeStatus,
    TraceRecord,
    TraceRecordType,
    TraceSelectionSummary,
)
from capicove.traces.codec import trace_record_sha256


pytestmark = pytest.mark.spark
NOW = datetime(2026, 8, 24, 12, tzinfo=UTC)


def _records() -> tuple[TraceRecord, ...]:
    started = TraceRecord(
        schema_version=1,
        record_type=TraceRecordType.EPISODE_STARTED,
        record_version=1,
        record_id="record-1",
        episode_id="episode-spark",
        trace_id="trace-spark",
        sequence=0,
        occurred_at=NOW,
        payload=EpisodeStarted(
            goal="Build a governed Spark dataset.",
            workspace_id="workspace-1",
            actor_id="actor-1",
            environment_id="environment-1",
            environment_version=1,
            tool_catalog_sha256="a" * 64,
            active_asset_versions={},
        ),
    )
    return (
        started,
        TraceRecord(
            schema_version=1,
            record_type=TraceRecordType.EPISODE_COMPLETED,
            record_version=1,
            record_id="record-2",
            episode_id="episode-spark",
            trace_id="trace-spark",
            sequence=1,
            occurred_at=NOW,
            payload=EpisodeCompleted(EpisodeStatus.SUCCEEDED),
            previous_record_sha256=trace_record_sha256(started),
        ),
    )


class _TraceSnapshot:
    summary = TraceSelectionSummary(1, 0, 0)

    def iter_records(self) -> Iterator[TraceRecord]:
        yield from _records()

    def __enter__(self) -> _TraceSnapshot:
        return self

    def __exit__(self, *args: object) -> None:
        del args


class _TraceReader:
    def open_snapshot(self, query: object, /) -> _TraceSnapshot:
        del query
        return _TraceSnapshot()


def _configuration(tmp_path: Path) -> SparkBackendConfiguration:
    return SparkBackendConfiguration(
        master="local[2]",
        app_name="capicove-agent-trace-batch-test",
        local_scratch_root=(tmp_path / "spark").resolve(),
    )


def test_validated_trace_export_projects_governed_columns_to_parquet(
    tmp_path: Path,
) -> None:
    export = (tmp_path / "trace-export").resolve()
    TraceExporter(
        _TraceReader(),  # type: ignore[arg-type]
        clock=lambda: NOW,
        export_id_factory=lambda: "spark-trace-export",
    ).export(export)
    source = trace_export_snapshot(export, dataset_id="dataset:agent-traces", version=1)
    node = LogicalNode(
        node_id="project_trace",
        operator_id="capicove.agent-trace.project",
        operator_version=1,
        input_bindings=("source",),
        output_binding="result",
        parameters={
            "columns": [
                "episode_id",
                "record_type",
                "sequence",
                "record_id",
                "previous_record_sha256",
            ]
        },
    )
    plan = ProcessingPlanner(spark_operator_specs()).build(
        input_bindings={"source": source},
        nodes=(node,),
        output_dataset_id="dataset:agent-trace-flat",
    )
    output_root = (tmp_path / "outputs").resolve()
    backend = SparkExecutionBackend(_configuration(tmp_path))

    reference = backend.submit(
        plan,
        ProcessingContext(
            run_id="run:agent-trace-flat-1",
            output_root=output_root,
            requested_at=NOW,
        ),
    )

    assert backend.poll(reference).state.value == "succeeded"
    output = SqliteProcessingStore(output_root / ".capicove" / "processing.sqlite3").latest(
        "dataset:agent-trace-flat"
    )
    assert output is not None
    spark = create_spark_session(_configuration(tmp_path))
    try:
        rows = SparkDatasetIO(spark).read(output).orderBy("sequence").collect()
        assert [row.asDict(recursive=True) for row in rows] == [
            {
                "episode_id": "episode-spark",
                "record_type": "episode.started",
                "sequence": 0,
                "record_id": "record-1",
                "previous_record_sha256": None,
            },
            {
                "episode_id": "episode-spark",
                "record_type": "episode.completed",
                "sequence": 1,
                "record_id": "record-2",
                "previous_record_sha256": trace_record_sha256(_records()[0]),
            },
        ]
    finally:
        spark.stop()


def test_agent_trace_projection_composes_with_advanced_operators_without_payload(
    tmp_path: Path,
) -> None:
    export = (tmp_path / "trace-export").resolve()
    TraceExporter(
        _TraceReader(),  # type: ignore[arg-type]
        clock=lambda: NOW,
        export_id_factory=lambda: "spark-trace-advanced-export",
    ).export(export)
    source = trace_export_snapshot(export, dataset_id="dataset:agent-traces", version=1)
    plan = ProcessingPlanner(spark_operator_specs()).build(
        input_bindings={"source": source},
        nodes=(
            LogicalNode(
                node_id="project_trace",
                operator_id="capicove.agent-trace.project",
                operator_version=1,
                input_bindings=("source",),
                output_binding="projected",
                parameters={"columns": ["episode_id", "record_type", "sequence"]},
            ),
            LogicalNode(
                node_id="sort_trace",
                operator_id="capicove.data.sort",
                operator_version=1,
                input_bindings=("projected",),
                output_binding="sorted",
                parameters={
                    "orders": [{"column": "sequence", "direction": "asc", "nulls": "last"}]
                },
            ),
            LogicalNode(
                node_id="rank_trace",
                operator_id="capicove.data.window",
                operator_version=1,
                input_bindings=("sorted",),
                output_binding="windowed",
                parameters={
                    "partition_by": ["episode_id"],
                    "order_by": [{"column": "sequence", "direction": "asc", "nulls": "last"}],
                    "functions": [{"function": "row_number", "output_column": "record_position"}],
                },
            ),
            LogicalNode(
                node_id="summarize_trace",
                operator_id="capicove.data.aggregate",
                operator_version=1,
                input_bindings=("windowed",),
                output_binding="aggregated",
                parameters={
                    "group_by": ["record_type"],
                    "aggregations": [
                        {"function": "count", "output_column": "record_count"},
                        {
                            "function": "max",
                            "input_column": "record_position",
                            "output_column": "last_position",
                        },
                    ],
                },
            ),
            LogicalNode(
                node_id="sort_summary",
                operator_id="capicove.data.sort",
                operator_version=1,
                input_bindings=("aggregated",),
                output_binding="result",
                parameters={
                    "orders": [
                        {"column": "record_count", "direction": "desc", "nulls": "last"},
                        {"column": "record_type", "direction": "asc", "nulls": "last"},
                    ]
                },
            ),
        ),
        output_dataset_id="dataset:agent-trace-advanced-summary",
    )
    output_root = (tmp_path / "outputs").resolve()
    backend = SparkExecutionBackend(_configuration(tmp_path))

    reference = backend.submit(
        plan,
        ProcessingContext(
            run_id="run:agent-trace-advanced-1",
            output_root=output_root,
            requested_at=NOW,
        ),
    )

    assert backend.poll(reference).state.value == "succeeded"
    output = SqliteProcessingStore(output_root / ".capicove" / "processing.sqlite3").latest(
        "dataset:agent-trace-advanced-summary"
    )
    assert output is not None
    assert tuple(field.name for field in output.schema.fields) == (
        "record_type",
        "record_count",
        "last_position",
    )
    assert "payload" not in {field.name for field in output.schema.fields}
    spark = create_spark_session(_configuration(tmp_path))
    try:
        rows = SparkDatasetIO(spark).read(output).orderBy("record_type").collect()
        assert [row.asDict(recursive=True) for row in rows] == [
            {
                "record_type": "episode.completed",
                "record_count": 1,
                "last_position": 2,
            },
            {
                "record_type": "episode.started",
                "record_count": 1,
                "last_position": 1,
            },
        ]
        assert all("payload" not in row.asDict(recursive=True) for row in rows)
    finally:
        spark.stop()
