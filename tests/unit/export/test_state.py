"""Tests for disk-backed export media identity aggregation."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import pytest

from capicove.export._state import ExportAggregation, ExportStateError
from capicove.export.validation import _IssueCollector, _ValidationState
from capicove.traces import (
    EpisodeFeedback,
    EpisodeStarted,
    MediaLocator,
    MediaLocatorKind,
    MediaReference,
    TraceModality,
    TraceRecord,
    TraceRecordType,
)
from capicove.traces.codec import trace_record_sha256


_NOW = datetime(2026, 8, 20, tzinfo=UTC)
_PAYLOAD = b"same-media"
_SHA256 = hashlib.sha256(_PAYLOAD).hexdigest()


def _reference(
    media_id: str,
    *,
    media_type: str = "image/png",
    locator: str = "inputs/media.png",
) -> MediaReference:
    modality = TraceModality.VIDEO if media_type.startswith("video/") else TraceModality.IMAGE
    return MediaReference(
        media_id,
        modality,
        media_type,
        _SHA256,
        len(_PAYLOAD),
        evidence_ref=f"sha256:{_SHA256}",
        locator=MediaLocator(MediaLocatorKind.WORKSPACE_RELATIVE, locator),
    )


def _records(
    first_reference: MediaReference,
    second_reference: MediaReference,
) -> tuple[TraceRecord, TraceRecord]:
    first = TraceRecord(
        schema_version=1,
        record_type=TraceRecordType.EPISODE_STARTED,
        record_version=1,
        record_id="record-0",
        episode_id="episode-1",
        trace_id="trace-1",
        sequence=0,
        occurred_at=_NOW,
        payload=EpisodeStarted(
            "Aggregate media.",
            "workspace",
            "actor",
            "environment",
            1,
            "a" * 64,
            {},
        ),
        media=(first_reference,),
    )
    second = TraceRecord(
        schema_version=1,
        record_type=TraceRecordType.EPISODE_FEEDBACK,
        record_version=1,
        record_id="record-1",
        episode_id="episode-1",
        trace_id="trace-1",
        sequence=1,
        occurred_at=_NOW,
        payload=EpisodeFeedback("note", {}, "test"),
        media=(second_reference,),
        previous_record_sha256=trace_record_sha256(first),
    )
    return first, second


def test_export_aggregation_rejects_any_media_id_identity_change(tmp_path: Path) -> None:
    first, changed = _records(
        _reference("media-1"),
        _reference("media-1", locator="other/media.png"),
    )
    state = ExportAggregation(tmp_path / "state.sqlite3")
    try:
        state.observe(first)
        with pytest.raises(ExportStateError, match="media ID"):
            state.observe(changed)
    finally:
        state.close()


def test_export_aggregation_rejects_one_hash_with_multiple_media_types(
    tmp_path: Path,
) -> None:
    first, conflicting = _records(
        _reference("media-image"),
        _reference("media-video", media_type="video/mp4", locator="inputs/media.mp4"),
    )
    state = ExportAggregation(tmp_path / "state.sqlite3")
    try:
        state.observe(first)
        with pytest.raises(ExportStateError, match="media hash"):
            state.observe(conflicting)
    finally:
        state.close()


def test_validator_reports_full_media_and_content_identity_conflicts(
    tmp_path: Path,
) -> None:
    first, changed = _records(
        _reference("media-1"),
        _reference("media-1", locator="other/media.png"),
    )
    issues = _IssueCollector()
    state = _ValidationState(tmp_path / "validation.sqlite3", issues)
    try:
        state.observe(first, PurePosixPath("records.jsonl"), 1)
        state.observe(changed, PurePosixPath("records.jsonl"), 2)
    finally:
        state.close()
    assert "MEDIA_IDENTITY_CONFLICT" in {issue.code for issue in issues.issues}

    first, conflicting = _records(
        _reference("media-image"),
        _reference("media-video", media_type="video/mp4", locator="inputs/media.mp4"),
    )
    issues = _IssueCollector()
    state = _ValidationState(tmp_path / "content-validation.sqlite3", issues)
    try:
        state.observe(first, PurePosixPath("records.jsonl"), 1)
        state.observe(conflicting, PurePosixPath("records.jsonl"), 2)
    finally:
        state.close()
    assert "MEDIA_CONTENT_IDENTITY_CONFLICT" in {issue.code for issue in issues.issues}
