"""Integration tests for streaming trace package validation and tamper detection."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import pytest

import capicove.export.validation as export_validation
from capicove.contracts import ToolStatus
from capicove.export import (
    TraceExporter,
    TraceExportOptions,
    TraceExportShard,
    decode_trace_export_manifest,
    encode_trace_export_manifest,
    validate_trace_export,
)
from capicove.traces import (
    ArtifactProduced,
    EpisodeCompleted,
    EpisodeStarted,
    EpisodeStatus,
    MediaContent,
    MediaReference,
    MessageObserved,
    MessageRole,
    ToolCallObserved,
    ToolResultObserved,
    TraceModality,
    TraceQuery,
    TraceRecord,
    TraceRecordType,
)
from capicove.traces.codec import (
    canonical_json_bytes,
    trace_record_sha256,
)

from .test_jsonl_export import Reader, Snapshot, _records


NOW = datetime(2026, 8, 18, tzinfo=UTC)


def _export(tmp_path: Path, *, incomplete: bool = False) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    records = _records(1)
    if incomplete:
        records = records[:1]
    output = tmp_path / "export"
    TraceExporter(
        Reader(Snapshot(records, 1)),  # type: ignore[arg-type]
        clock=lambda: NOW,
        export_id_factory=lambda: "validation-export",
    ).export(
        output,
        options=TraceExportOptions(
            query=TraceQuery(include_incomplete=incomplete),
        ),
    )
    return output


@pytest.mark.skipif(os.name != "posix", reason="POSIX no-follow directory handles")
def test_validate_rejects_symlinked_export_root(tmp_path: Path) -> None:
    genuine = _export(tmp_path / "source")
    alias = tmp_path / "alias"
    alias.symlink_to(genuine, target_is_directory=True)

    report = validate_trace_export(alias)

    assert not report.valid
    assert any(issue.code == "EXPORT_UNSAFE_PATH" for issue in report.issues)
    assert str(genuine) not in str(report.issues)


@pytest.mark.skipif(os.name != "posix", reason="POSIX no-follow directory handles")
def test_validate_keeps_pinned_export_after_root_is_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = _export(tmp_path / "selected")
    foreign = _export(tmp_path / "foreign", incomplete=True)
    moved = tmp_path / "selected-moved"
    original = export_validation._ExportRoot.is_regular
    swapped = False

    def swap_before_manifest_check(
        root: export_validation._ExportRoot, relative: PurePosixPath
    ) -> bool:
        nonlocal swapped
        if root.path == selected and relative == PurePosixPath("manifest.json") and not swapped:
            selected.rename(moved)
            selected.symlink_to(foreign, target_is_directory=True)
            swapped = True
        return original(root, relative)

    monkeypatch.setattr(export_validation._ExportRoot, "is_regular", swap_before_manifest_check)

    report = validate_trace_export(selected)

    assert swapped
    assert report.valid
    assert report.records_checked == 2


@pytest.mark.skipif(os.name != "posix", reason="POSIX no-follow file handles")
def test_validate_rejects_shard_swapped_to_external_symlink_after_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = _export(tmp_path / "selected")
    shard = next(export.glob("records-*.jsonl"))
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(shard.read_bytes())
    original = export_validation._ExportRoot.is_regular
    swapped = False

    def swap_after_regular_check(
        root: export_validation._ExportRoot, relative: PurePosixPath
    ) -> bool:
        nonlocal swapped
        regular = original(root, relative)
        if root.path == export and relative.name == shard.name and regular and not swapped:
            shard.unlink()
            shard.symlink_to(outside)
            swapped = True
        return regular

    monkeypatch.setattr(export_validation._ExportRoot, "is_regular", swap_after_regular_check)

    report = validate_trace_export(export)

    assert swapped
    assert not report.valid
    assert any(issue.code == "SHARD_READ_FAILED" for issue in report.issues)


@pytest.mark.skipif(os.name != "posix", reason="POSIX no-follow directory handles")
def test_inspect_uses_one_pinned_export_after_root_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from capicove.export import inspect_trace_export

    selected = _export(tmp_path / "selected")
    foreign = _export(tmp_path / "foreign", incomplete=True)
    moved = tmp_path / "selected-moved"
    original = export_validation._ExportRoot.is_regular
    swapped = False

    def swap_before_validation(
        root: export_validation._ExportRoot, relative: PurePosixPath
    ) -> bool:
        nonlocal swapped
        if root.path == selected and relative == PurePosixPath("manifest.json") and not swapped:
            selected.rename(moved)
            selected.symlink_to(foreign, target_is_directory=True)
            swapped = True
        return original(root, relative)

    monkeypatch.setattr(export_validation._ExportRoot, "is_regular", swap_before_validation)

    manifest, report = inspect_trace_export(selected, verify_media=False)

    assert swapped
    assert manifest.record_count == 2
    assert report.valid
    assert report.records_checked == 2


def test_inspect_reports_manifest_changed_between_metadata_and_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from capicove.export import inspect_trace_export

    export = _export(tmp_path / "selected")
    manifest_path = export / "manifest.json"
    original = export_validation._ExportRoot.is_regular
    changed = False

    def change_before_validation(
        root: export_validation._ExportRoot, relative: PurePosixPath
    ) -> bool:
        nonlocal changed
        if root.path == export and relative == PurePosixPath("manifest.json") and not changed:
            manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
            changed = True
        return original(root, relative)

    monkeypatch.setattr(export_validation._ExportRoot, "is_regular", change_before_validation)

    manifest, report = inspect_trace_export(export)

    assert changed
    assert manifest.record_count == 2
    assert not report.valid
    assert any(issue.code == "MANIFEST_CHANGED" for issue in report.issues)


def _export_records(
    tmp_path: Path,
    records: tuple[TraceRecord, ...],
    *,
    include_incomplete: bool = False,
) -> Path:
    output = tmp_path / "export"
    TraceExporter(
        Reader(Snapshot(records, 1)),  # type: ignore[arg-type]
        clock=lambda: NOW,
        export_id_factory=lambda: "semantic-validation-export",
    ).export(
        output,
        options=TraceExportOptions(
            query=TraceQuery(include_incomplete=include_incomplete),
        ),
    )
    return output


def _episode_with_middle(
    payload: ArtifactProduced | MessageObserved | ToolCallObserved | ToolResultObserved,
    *,
    request_id: str | None = None,
    parent_record_id: str | None = None,
    media: tuple[MediaReference, ...] = (),
) -> tuple[TraceRecord, ...]:
    started = _records(1)[0]
    middle = TraceRecord(
        schema_version=1,
        record_type=(
            TraceRecordType.MESSAGE_OBSERVED
            if isinstance(payload, MessageObserved)
            else TraceRecordType.TOOL_CALL
            if isinstance(payload, ToolCallObserved)
            else TraceRecordType.ARTIFACT_PRODUCED
            if isinstance(payload, ArtifactProduced)
            else TraceRecordType.TOOL_RESULT
        ),
        record_version=1,
        record_id="record-middle",
        episode_id=started.episode_id,
        trace_id=started.trace_id,
        sequence=1,
        occurred_at=NOW,
        payload=payload,
        request_id=request_id,
        parent_record_id=parent_record_id,
        media=media,
        previous_record_sha256=trace_record_sha256(started),
    )
    completed = TraceRecord(
        schema_version=1,
        record_type=TraceRecordType.EPISODE_COMPLETED,
        record_version=1,
        record_id="record-completed",
        episode_id=started.episode_id,
        trace_id=started.trace_id,
        sequence=2,
        occurred_at=NOW,
        payload=EpisodeCompleted(EpisodeStatus.SUCCEEDED),
        previous_record_sha256=trace_record_sha256(middle),
    )
    return started, middle, completed


def _rewrite_shard_and_manifest(output: Path, lines: list[bytes]) -> None:
    shard_path = output / "records-00000.jsonl"
    payload = b"".join(lines)
    shard_path.write_bytes(payload)
    manifest_path = output / "manifest.json"
    manifest = decode_trace_export_manifest(json.loads(manifest_path.read_bytes()))
    shard = TraceExportShard(
        manifest.shards[0].path,
        len(lines),
        len(payload),
        hashlib.sha256(payload).hexdigest(),
    )
    updated = replace(
        manifest,
        shards=(shard,),
        record_count=len(lines),
        record_type_counts={
            "episode.started": 1,
            "episode.completed": max(0, len(lines) - 1),
        },
    )
    manifest_path.write_bytes(canonical_json_bytes(encode_trace_export_manifest(updated)) + b"\n")


def test_validator_accepts_valid_export_and_warns_for_allowed_incomplete_episode(
    tmp_path: Path,
) -> None:
    complete = validate_trace_export(_export(tmp_path / "complete"))
    incomplete = validate_trace_export(_export(tmp_path / "incomplete", incomplete=True))

    assert complete.valid
    assert complete.records_checked == 2
    assert complete.episodes_checked == 1
    assert incomplete.valid
    assert {issue.code for issue in incomplete.issues} == {"INCOMPLETE_EPISODE"}


def test_validator_accepts_dangling_call_in_explicitly_incomplete_episode(
    tmp_path: Path,
) -> None:
    records = _episode_with_middle(
        ToolCallObserved("capicove.test.tool", 1, {}, "a" * 64, 1),
        request_id="request-1",
    )[:2]

    report = validate_trace_export(_export_records(tmp_path, records, include_incomplete=True))

    assert report.valid
    assert {issue.code for issue in report.issues} == {"INCOMPLETE_EPISODE"}


def test_validator_reports_malformed_json_with_shard_and_line_and_hash_tamper(
    tmp_path: Path,
) -> None:
    output = _export(tmp_path)
    shard = output / "records-00000.jsonl"
    shard.write_bytes(shard.read_bytes() + b"{bad json}\n")

    report = validate_trace_export(output)

    codes = {issue.code for issue in report.issues}
    assert "INVALID_JSON" in codes
    assert "SHARD_SHA256_MISMATCH" in codes
    invalid = next(issue for issue in report.issues if issue.code == "INVALID_JSON")
    assert invalid.shard is not None
    assert invalid.line == 3
    assert not report.valid


def test_validator_rejects_manifest_before_reading_an_unbounded_payload(
    tmp_path: Path,
) -> None:
    output = _export(tmp_path)
    (output / "manifest.json").write_bytes(b" " * 65)

    report = validate_trace_export(output, max_manifest_bytes=64)

    assert {issue.code for issue in report.issues} == {"MANIFEST_TOO_LARGE"}
    assert not report.valid


def test_validator_detects_noncanonical_json_sequence_and_predecessor_tampering(
    tmp_path: Path,
) -> None:
    output = _export(tmp_path)
    shard = output / "records-00000.jsonl"
    values = [json.loads(line) for line in shard.read_bytes().splitlines()]
    values[1]["sequence"] = 3
    values[1]["previous_record_sha256"] = "b" * 64
    lines = [
        json.dumps(value, ensure_ascii=False, sort_keys=False).encode() + b"\n" for value in values
    ]
    _rewrite_shard_and_manifest(output, lines)

    report = validate_trace_export(output)

    codes = {issue.code for issue in report.issues}
    assert "NON_CANONICAL_JSON" in codes
    assert "SEQUENCE_GAP" in codes
    assert "PREDECESSOR_MISMATCH" in codes
    assert not report.valid


def test_validator_requires_episode_start_at_sequence_zero(tmp_path: Path) -> None:
    started = TraceRecord(
        schema_version=1,
        record_type=TraceRecordType.EPISODE_STARTED,
        record_version=1,
        record_id="record-started",
        episode_id="episode-start-offset",
        trace_id="trace-start-offset",
        sequence=1,
        occurred_at=NOW,
        payload=EpisodeStarted(
            goal="Validate this Episode.",
            workspace_id="workspace-1",
            actor_id="actor-1",
            environment_id="environment-1",
            environment_version=1,
            tool_catalog_sha256="a" * 64,
            active_asset_versions={},
        ),
        previous_record_sha256="b" * 64,
    )
    completed = TraceRecord(
        schema_version=1,
        record_type=TraceRecordType.EPISODE_COMPLETED,
        record_version=1,
        record_id="record-completed",
        episode_id=started.episode_id,
        trace_id=started.trace_id,
        sequence=2,
        occurred_at=NOW,
        payload=EpisodeCompleted(EpisodeStatus.SUCCEEDED),
        previous_record_sha256=trace_record_sha256(started),
    )

    report = validate_trace_export(_export_records(tmp_path, (started, completed)))

    assert "EPISODE_START_SEQUENCE" in {issue.code for issue in report.issues}


def test_validator_requires_one_trace_identity_per_episode(tmp_path: Path) -> None:
    output = _export(tmp_path)
    shard = output / "records-00000.jsonl"
    values = [json.loads(line) for line in shard.read_bytes().splitlines()]
    values[1]["trace_id"] = "different-trace"
    _rewrite_shard_and_manifest(
        output,
        [canonical_json_bytes(value) + b"\n" for value in values],
    )

    report = validate_trace_export(output)

    assert "TRACE_ID_MISMATCH" in {issue.code for issue in report.issues}


def test_validator_requires_tool_result_to_parent_its_call_record(tmp_path: Path) -> None:
    started = _records(1)[0]
    call = TraceRecord(
        schema_version=1,
        record_type=TraceRecordType.TOOL_CALL,
        record_version=1,
        record_id="record-call",
        episode_id=started.episode_id,
        trace_id=started.trace_id,
        sequence=1,
        occurred_at=NOW,
        request_id="request-1",
        payload=ToolCallObserved("capicove.test.tool", 1, {}, "a" * 64, 1),
        previous_record_sha256=trace_record_sha256(started),
    )
    result = TraceRecord(
        schema_version=1,
        record_type=TraceRecordType.TOOL_RESULT,
        record_version=1,
        record_id="record-result",
        episode_id=started.episode_id,
        trace_id=started.trace_id,
        sequence=2,
        occurred_at=NOW,
        request_id="request-1",
        parent_record_id="wrong-parent",
        payload=ToolResultObserved(
            "capicove.test.tool",
            1,
            ToolStatus.SUCCEEDED,
            "b" * 64,
            result={},
        ),
        previous_record_sha256=trace_record_sha256(call),
    )
    completed = TraceRecord(
        schema_version=1,
        record_type=TraceRecordType.EPISODE_COMPLETED,
        record_version=1,
        record_id="record-completed",
        episode_id=started.episode_id,
        trace_id=started.trace_id,
        sequence=3,
        occurred_at=NOW,
        payload=EpisodeCompleted(EpisodeStatus.SUCCEEDED),
        previous_record_sha256=trace_record_sha256(result),
    )

    report = validate_trace_export(_export_records(tmp_path, (started, call, result, completed)))

    assert "TOOL_RESULT_PARENT_MISMATCH" in {issue.code for issue in report.issues}


def test_validator_reconciles_message_media_blocks_with_record_references(
    tmp_path: Path,
) -> None:
    records = _episode_with_middle(MessageObserved(MessageRole.USER, (MediaContent("media-1"),)))

    report = validate_trace_export(_export_records(tmp_path, records))

    assert "MEDIA_CONTENT_REFERENCE_MISMATCH" in {issue.code for issue in report.issues}


def test_validator_requires_artifact_payload_to_name_its_attached_media(
    tmp_path: Path,
) -> None:
    records = _episode_with_middle(ArtifactProduced("artifact:orphan", "media-orphan"))

    report = validate_trace_export(_export_records(tmp_path, records))

    assert "ARTIFACT_MEDIA_REFERENCE_MISMATCH" in {issue.code for issue in report.issues}


def test_validator_requires_artifact_and_media_derivation_to_match(
    tmp_path: Path,
) -> None:
    media = MediaReference(
        "media-child",
        TraceModality.VIDEO,
        "video/mp4",
        "c" * 64,
        5,
        derived_from=("media-parent",),
    )
    records = _episode_with_middle(
        ArtifactProduced(
            "artifact:child",
            media.media_id,
            derived_from=("different-parent",),
        ),
        media=(media,),
    )

    report = validate_trace_export(_export_records(tmp_path, records))

    assert "ARTIFACT_DERIVATION_MISMATCH" in {issue.code for issue in report.issues}


def test_validator_detects_schema_hash_and_unexpected_file_tampering(tmp_path: Path) -> None:
    output = _export(tmp_path)
    (output / "schemas" / "capicove-agent-trace-v1.schema.json").write_text("{}", encoding="utf-8")
    (output / "unexpected.txt").write_text("unexpected", encoding="utf-8")

    report = validate_trace_export(output)

    codes = {issue.code for issue in report.issues}
    assert "SCHEMA_SHA256_MISMATCH" in codes
    assert "UNEXPECTED_FILE" in codes
    assert not report.valid


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (
            lambda record: record["payload"].__setitem__("goal", "/Users/alice/private/file.txt"),
            "PRIVACY_ABSOLUTE_PATH",
        ),
        (
            lambda record: record["payload"].__setitem__("goal", "Bearer abcdefghijklmnop"),
            "PRIVACY_CREDENTIAL",
        ),
        (
            lambda record: record["payload"].__setitem__("goal", "data:image/png;base64,aGVsbG8="),
            "PRIVACY_INLINE_BASE64",
        ),
        (
            lambda record: record["payload"]["active_asset_versions"].__setitem__("token", 1),
            "PRIVACY_SECRET_KEY",
        ),
    ],
)
def test_validator_rejects_privacy_boundary_violations(
    tmp_path: Path,
    mutate: object,
    expected_code: str,
) -> None:
    """Catch packages that are structurally valid but leak sensitive values."""
    output = _export(tmp_path)
    shard = output / "records-00000.jsonl"
    values = [json.loads(line) for line in shard.read_bytes().splitlines()]
    mutate(values[0])  # type: ignore[operator]
    lines = [canonical_json_bytes(value) + b"\n" for value in values]
    _rewrite_shard_and_manifest(output, lines)

    report = validate_trace_export(output)

    assert expected_code in {issue.code for issue in report.issues}
    assert not report.valid
