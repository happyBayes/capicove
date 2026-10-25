"""Integration tests for governed Agent trace export dataset snapshots."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from capicove.export import TraceExporter
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


NOW = datetime(2026, 8, 24, 12, tzinfo=UTC)
SHA256 = "a" * 64


def _records() -> tuple[TraceRecord, ...]:
    started = TraceRecord(
        schema_version=1,
        record_type=TraceRecordType.EPISODE_STARTED,
        record_version=1,
        record_id="record-1",
        episode_id="episode-1",
        trace_id="trace-1",
        sequence=0,
        occurred_at=NOW,
        payload=EpisodeStarted(
            goal="Test Spark batch input.",
            workspace_id="workspace-1",
            actor_id="actor-1",
            environment_id="environment-1",
            environment_version=1,
            tool_catalog_sha256=SHA256,
            active_asset_versions={},
        ),
    )
    completed = TraceRecord(
        schema_version=1,
        record_type=TraceRecordType.EPISODE_COMPLETED,
        record_version=1,
        record_id="record-2",
        episode_id="episode-1",
        trace_id="trace-1",
        sequence=1,
        occurred_at=NOW,
        payload=EpisodeCompleted(EpisodeStatus.SUCCEEDED),
        previous_record_sha256=trace_record_sha256(started),
    )
    return started, completed


class _Snapshot:
    summary = TraceSelectionSummary(1, 0, 0)

    def iter_records(self) -> Iterator[TraceRecord]:
        yield from _records()

    def __enter__(self) -> _Snapshot:
        return self

    def __exit__(self, *args: object) -> None:
        del args


class _Reader:
    def open_snapshot(self, query: object, /) -> _Snapshot:
        del query
        return _Snapshot()


def _export(path: Path) -> None:
    TraceExporter(
        _Reader(),  # type: ignore[arg-type]
        clock=lambda: NOW,
        export_id_factory=lambda: "trace-export-1",
    ).export(path)


def test_valid_trace_export_becomes_hashed_jsonl_snapshot(tmp_path: Path) -> None:
    export = (tmp_path / "trace-export").resolve()
    _export(export)

    snapshot = trace_export_snapshot(export, dataset_id="dataset:agent-traces", version=1)

    assert snapshot.uri == export.as_uri()
    assert snapshot.data_format.value == "jsonl"
    assert tuple(field.name for field in snapshot.schema.fields) == (
        "schema_version",
        "record_type",
        "record_version",
        "record_id",
        "episode_id",
        "trace_id",
        "sequence",
        "occurred_at",
        "previous_record_sha256",
    )


def test_tampered_trace_export_is_rejected_before_spark(tmp_path: Path) -> None:
    export = (tmp_path / "trace-export").resolve()
    _export(export)
    shard = export / "records-00000.jsonl"
    shard.write_bytes(shard.read_bytes() + b"{}\n")

    with pytest.raises(ValueError, match="validation"):
        trace_export_snapshot(export, dataset_id="dataset:agent-traces", version=1)
