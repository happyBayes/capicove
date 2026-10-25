"""Tests for immutable trace export options, shards, and manifests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import cast

import pytest

from capicove.export.models import (
    MediaMode,
    TraceExportManifest,
    TraceExportOptions,
    TraceExportShard,
    decode_trace_export_manifest,
    encode_trace_export_manifest,
)


SHA256 = "a" * 64


def _manifest(**changes: object) -> TraceExportManifest:
    shard = TraceExportShard(PurePosixPath("records-00000.jsonl"), 1, 10, SHA256)
    values: dict[str, object] = {
        "format_version": 1,
        "schema_version": 1,
        "export_id": "export-1",
        "created_at": datetime(2026, 8, 18, tzinfo=UTC),
        "options": TraceExportOptions(),
        "shards": (shard,),
        "record_count": 1,
        "record_type_counts": {"episode.started": 1},
        "modality_counts": {},
        "tool_status_counts": {},
        "episode_status_counts": {},
        "included_episode_count": 1,
        "incomplete_episode_count": 1,
        "omitted_restricted_episode_count": 0,
        "media_count": 0,
        "media_size_bytes": 0,
        "missing_media_count": 0,
        "media_layout": "sha256-v1",
        "schema_path": PurePosixPath("schemas/capicove-agent-trace-v1.schema.json"),
        "schema_sha256": SHA256,
    }
    values.update(changes)
    return TraceExportManifest(**values)  # type: ignore[arg-type]


def test_export_option_defaults_and_manifest_round_trip_are_exact() -> None:
    options = TraceExportOptions()
    assert options.max_records_per_shard == 100_000
    assert options.max_bytes_per_shard == 536_870_912
    assert options.max_record_bytes == 16_777_216
    assert options.media_mode is MediaMode.REFERENCE
    assert not options.query.include_incomplete
    assert not options.query.include_restricted

    manifest = _manifest()
    assert decode_trace_export_manifest(encode_trace_export_manifest(manifest)) == manifest


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TraceExportOptions(max_records_per_shard=True),
        lambda: TraceExportOptions(max_records_per_shard=0),
        lambda: TraceExportOptions(max_records_per_shard=10_000_001),
        lambda: TraceExportOptions(max_bytes_per_shard=0),
        lambda: TraceExportOptions(max_bytes_per_shard=8 * 1024**3 + 1),
        lambda: TraceExportOptions(media_mode=cast(MediaMode, "inline")),
        lambda: TraceExportShard(PurePosixPath("/absolute.jsonl"), 1, 1, SHA256),
        lambda: TraceExportShard(PurePosixPath("../escape.jsonl"), 1, 1, SHA256),
        lambda: _manifest(record_count=-1),
        lambda: _manifest(record_count=2),
        lambda: _manifest(
            shards=(
                TraceExportShard(PurePosixPath("same.jsonl"), 1, 1, SHA256),
                TraceExportShard(PurePosixPath("same.jsonl"), 1, 1, SHA256),
            ),
            record_count=2,
            record_type_counts={"episode.started": 2},
        ),
        lambda: _manifest(
            shards=(TraceExportShard(PurePosixPath("records-00001.jsonl"), 1, 1, SHA256),)
        ),
    ],
)
def test_export_values_reject_invalid_limits_paths_counts_and_duplicates(
    factory: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()  # type: ignore[operator]


def test_manifest_count_mappings_are_defensively_frozen() -> None:
    counts = {"episode.started": 1}
    manifest = _manifest(record_type_counts=counts)
    counts["message.observed"] = 2

    assert dict(manifest.record_type_counts) == {"episode.started": 1}
    with pytest.raises(TypeError):
        manifest.record_type_counts["changed"] = 1  # type: ignore[index]
