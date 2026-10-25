"""Integration tests for one-pass sharded JSONL trace export."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from capicove.export import (
    TraceExporter,
    TraceExportExistsError,
    TraceExportOptions,
    TraceExportRecordTooLargeError,
)
from capicove.traces import (
    EpisodeCompleted,
    EpisodeStarted,
    EpisodeStatus,
    TraceRecord,
    TraceRecordType,
    TraceSelectionSummary,
)
from capicove.traces.codec import canonical_record_bytes, trace_record_sha256


SHA256 = "a" * 64
NOW = datetime(2026, 8, 18, 12, tzinfo=UTC)


def _records(episode_count: int = 2) -> tuple[TraceRecord, ...]:
    records: list[TraceRecord] = []
    for episode_index in range(episode_count):
        episode_id = f"episode-{episode_index}"
        started = TraceRecord(
            schema_version=1,
            record_type=TraceRecordType.EPISODE_STARTED,
            record_version=1,
            record_id=f"record-{episode_index}-0",
            episode_id=episode_id,
            trace_id=f"trace-{episode_index}",
            sequence=0,
            occurred_at=NOW + timedelta(minutes=episode_index),
            payload=EpisodeStarted(
                goal="Export this Episode.",
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
            record_id=f"record-{episode_index}-1",
            episode_id=episode_id,
            trace_id=f"trace-{episode_index}",
            sequence=1,
            occurred_at=NOW + timedelta(minutes=episode_index, seconds=1),
            payload=EpisodeCompleted(EpisodeStatus.SUCCEEDED),
            previous_record_sha256=trace_record_sha256(started),
        )
        records.extend((started, completed))
    return tuple(records)


class Snapshot:
    """Expose one fixed single-pass record iterator."""

    def __init__(self, records: tuple[TraceRecord, ...], episodes: int) -> None:
        self._records = records
        self.summary = TraceSelectionSummary(episodes, 0, 0)
        self.iterations = 0

    def iter_records(self) -> Iterator[TraceRecord]:
        self.iterations += 1
        if self.iterations > 1:
            raise AssertionError("records were requested twice")
        yield from self._records

    def __enter__(self) -> Snapshot:
        return self

    def __exit__(self, *args: object) -> None:
        del args


class Reader:
    """Capture exactly one snapshot request."""

    def __init__(self, snapshot: Snapshot) -> None:
        self.snapshot = snapshot
        self.opens = 0

    def open_snapshot(self, query: object, /) -> Snapshot:
        del query
        self.opens += 1
        if self.opens > 1:
            raise AssertionError("snapshot opened twice")
        return self.snapshot


def _exporter(reader: Reader) -> TraceExporter:
    return TraceExporter(
        reader,  # type: ignore[arg-type]
        clock=lambda: NOW,
        export_id_factory=lambda: "export-1",
    )


def test_export_streams_canonical_lines_into_bounded_shards_and_exact_manifest(
    tmp_path: Path,
) -> None:
    records = _records()
    snapshot = Snapshot(records, 2)
    reader = Reader(snapshot)
    output = tmp_path / "export"

    manifest = _exporter(reader).export(
        output,
        options=TraceExportOptions(max_records_per_shard=2),
    )

    assert reader.opens == 1
    assert snapshot.iterations == 1
    assert tuple(path.name for path in sorted(output.glob("records-*.jsonl"))) == (
        "records-00000.jsonl",
        "records-00001.jsonl",
    )
    exported_lines = [
        line
        for shard in sorted(output.glob("records-*.jsonl"))
        for line in shard.read_bytes().splitlines(keepends=True)
    ]
    assert exported_lines == [canonical_record_bytes(record) + b"\n" for record in records]
    assert manifest.record_count == 4
    assert manifest.included_episode_count == 2
    assert manifest.record_type_counts == {
        "episode.completed": 2,
        "episode.started": 2,
    }
    assert sum(shard.record_count for shard in manifest.shards) == 4
    for shard in manifest.shards:
        payload = (output / shard.path).read_bytes()
        assert shard.size_bytes == len(payload)
        assert shard.sha256 == hashlib.sha256(payload).hexdigest()

    raw_manifest = (output / "manifest.json").read_bytes()
    assert raw_manifest.endswith(b"\n")
    assert json.loads(raw_manifest)["export_id"] == "export-1"
    schema = output / "schemas" / "capicove-agent-trace-v1.schema.json"
    assert schema.is_file()
    assert manifest.schema_sha256 == hashlib.sha256(schema.read_bytes()).hexdigest()


def test_byte_threshold_rotates_before_writing_the_next_record(tmp_path: Path) -> None:
    records = _records(1)
    first_size = len(canonical_record_bytes(records[0])) + 1
    output = tmp_path / "export"

    manifest = _exporter(Reader(Snapshot(records, 1))).export(
        output,
        options=TraceExportOptions(max_bytes_per_shard=first_size),
    )

    assert len(manifest.shards) == 2
    assert tuple(shard.record_count for shard in manifest.shards) == (1, 1)


def test_existing_output_and_oversized_record_fail_without_modification_or_publication(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "marker"
    marker.write_text("unchanged", encoding="utf-8")
    with pytest.raises(TraceExportExistsError):
        _exporter(Reader(Snapshot(_records(1), 1))).export(existing)
    assert marker.read_text(encoding="utf-8") == "unchanged"

    output = tmp_path / "too-large"
    with pytest.raises(TraceExportRecordTooLargeError):
        _exporter(Reader(Snapshot(_records(1), 1))).export(
            output,
            options=TraceExportOptions(max_record_bytes=10),
        )
    assert not output.exists()
    assert not tuple(tmp_path.glob(".too-large.tmp-*"))


def test_empty_selection_publishes_only_schema_and_zero_count_manifest(tmp_path: Path) -> None:
    output = tmp_path / "empty"

    manifest = _exporter(Reader(Snapshot((), 0))).export(output)

    assert manifest.record_count == 0
    assert manifest.shards == ()
    assert not (output / "media").exists()
    assert not tuple(output.glob("records-*.jsonl"))
    assert (output / "manifest.json").is_file()
    assert (output / manifest.schema_path).is_file()
