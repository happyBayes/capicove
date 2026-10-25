"""Scale gate for single-pass, bounded trace export and validation."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

import capicove.export.jsonl as jsonl_module
from capicove.export import TraceExporter, TraceExportOptions, validate_trace_export
from capicove.traces import (
    EpisodeCompleted,
    EpisodeStarted,
    EpisodeStatus,
    MessageObserved,
    MessageRole,
    TextContent,
    TracePayload,
    TraceReader,
    TraceReadSnapshot,
    TraceRecord,
    TraceRecordType,
    TraceSelectionSummary,
)
from capicove.traces.codec import canonical_record_bytes, trace_record_sha256


pytestmark = pytest.mark.integration
_NOW = datetime(2026, 8, 18, 12, tzinfo=UTC)
_RECORD_COUNT = 100_000
_MAX_RECORD_BYTES = 4_096


class _SinglePassSnapshot:
    """Generate records only after the exporter encoded the prior yield."""

    summary = TraceSelectionSummary(1, 0, 0)

    def __init__(self, record_count: int) -> None:
        self.record_count = record_count
        self.iterations = 0
        self.encoded_count = 0
        self.prior_encoded = True

    def iter_records(self) -> Iterator[TraceRecord]:
        self.iterations += 1
        if self.iterations != 1:
            raise AssertionError("the trace iterator was requested more than once")
        previous_sha256: str | None = None
        for sequence in range(self.record_count):
            if not self.prior_encoded:
                raise AssertionError("the next record was requested before encoding the prior one")
            record = _record(sequence, previous_sha256, self.record_count)
            self.prior_encoded = False
            yield record
            if not self.prior_encoded:
                raise AssertionError("the exporter advanced without encoding the yielded record")
            previous_sha256 = trace_record_sha256(record)

    def __enter__(self) -> _SinglePassSnapshot:
        return self

    def __exit__(self, *args: object) -> None:
        del args


class _SingleSnapshotReader:
    """Refuse snapshot rewind or reopening."""

    def __init__(self, snapshot: _SinglePassSnapshot) -> None:
        self.snapshot = snapshot
        self.opens = 0

    def open_snapshot(self, query: object, /) -> TraceReadSnapshot:
        del query
        self.opens += 1
        if self.opens != 1:
            raise AssertionError("the exporter opened its snapshot more than once")
        return cast(TraceReadSnapshot, self.snapshot)


@pytest.mark.parametrize(
    ("record_count", "records_per_shard"),
    [
        pytest.param(11, 3, id="functional"),
        pytest.param(
            _RECORD_COUNT,
            10_000,
            marks=[pytest.mark.slow, pytest.mark.qualification],
            id="capacity",
        ),
    ],
)
def test_export_streams_records_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_count: int,
    records_per_shard: int,
) -> None:
    snapshot = _SinglePassSnapshot(record_count)
    reader = _SingleSnapshotReader(snapshot)
    original_encoder = canonical_record_bytes

    def acknowledge(record: TraceRecord) -> bytes:
        encoded = original_encoder(record)
        snapshot.encoded_count += 1
        snapshot.prior_encoded = True
        return encoded

    monkeypatch.setattr(jsonl_module, "canonical_record_bytes", acknowledge)
    output = (tmp_path / "export").resolve()
    manifest = TraceExporter(
        cast(TraceReader, reader),
        clock=lambda: _NOW,
        export_id_factory=lambda: f"export-{record_count}",
    ).export(
        output,
        options=TraceExportOptions(
            max_records_per_shard=records_per_shard,
            max_record_bytes=_MAX_RECORD_BYTES,
        ),
    )

    assert reader.opens == 1
    assert snapshot.iterations == 1
    assert snapshot.encoded_count == record_count
    assert manifest.record_count == record_count
    shard_count = (record_count + records_per_shard - 1) // records_per_shard
    assert len(manifest.shards) == shard_count
    assert tuple(shard.record_count for shard in manifest.shards) == tuple(
        min(records_per_shard, record_count - ordinal * records_per_shard)
        for ordinal in range(shard_count)
    )
    assert tuple(shard.path.name for shard in manifest.shards) == tuple(
        f"records-{index:05d}.jsonl" for index in range(shard_count)
    )
    for shard in manifest.shards:
        path = output / shard.path
        digest = hashlib.sha256()
        observed_count = 0
        with path.open("rb") as stream:
            for line in stream:
                observed_count += 1
                assert len(line) - 1 <= _MAX_RECORD_BYTES
                digest.update(line)
        assert observed_count == shard.record_count
        assert digest.hexdigest() == shard.sha256

    report = validate_trace_export(output, max_record_bytes=_MAX_RECORD_BYTES)
    assert report.valid, report.issues
    assert report.records_checked == record_count
    assert report.episodes_checked == 1


def _record(sequence: int, previous_sha256: str | None, record_count: int) -> TraceRecord:
    payload: TracePayload
    if sequence == 0:
        payload = EpisodeStarted(
            "Exercise bounded streaming export.",
            "workspace-1",
            "actor-1",
            "local",
            1,
            "a" * 64,
            {},
        )
        record_type = TraceRecordType.EPISODE_STARTED
    elif sequence == record_count - 1:
        payload = EpisodeCompleted(EpisodeStatus.SUCCEEDED)
        record_type = TraceRecordType.EPISODE_COMPLETED
    else:
        payload = MessageObserved(MessageRole.ASSISTANT, (TextContent("x"),))
        record_type = TraceRecordType.MESSAGE_OBSERVED
    return TraceRecord(
        schema_version=1,
        record_type=record_type,
        record_version=1,
        record_id=f"record-{sequence}",
        episode_id="episode-scale",
        trace_id="trace-scale",
        sequence=sequence,
        occurred_at=_NOW,
        payload=payload,
        previous_record_sha256=previous_sha256,
    )
