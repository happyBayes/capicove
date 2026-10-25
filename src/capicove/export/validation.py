"""Bounded streaming validation and tamper detection for trace export packages."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import BinaryIO, Protocol, cast

from capicove.export.media import media_reference_identity_sha256
from capicove.export.models import (
    MediaMode,
    TraceExportManifest,
    decode_trace_export_manifest,
    encode_trace_export_manifest,
)
from capicove.traces import (
    ArtifactProduced,
    EpisodeCompleted,
    EpisodeFeedback,
    EpisodeStarted,
    MediaContent,
    MessageObserved,
    ToolCallObserved,
    ToolResultObserved,
    TraceRecord,
)
from capicove.traces.codec import (
    canonical_json_bytes,
    canonical_record_bytes,
    decode_trace_record,
    trace_record_sha256,
)
from capicove.traces.sanitization import TracePrivacyError, validate_trace_privacy


_ISSUE_LIMIT = 1_000
_EXTENSIONS = {
    "audio/wav": ".wav",
    "video/mp4": ".mp4",
    "image/png": ".png",
    "image/jpeg": ".jpg",
}


class ValidationSeverity(StrEnum):
    """Classify whether one validation issue invalidates a package."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class TraceValidationIssue:
    """Describe one bounded machine-readable package validation issue."""

    severity: ValidationSeverity
    code: str
    message: str
    shard: PurePosixPath | None = None
    line: int | None = None
    episode_id: str | None = None
    record_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.severity, ValidationSeverity):
            raise ValueError("severity must be a ValidationSeverity")
        _require_non_blank("code", self.code)
        _require_non_blank("message", self.message)
        if self.shard is not None and (
            not isinstance(self.shard, PurePosixPath)
            or self.shard.is_absolute()
            or ".." in self.shard.parts
        ):
            raise ValueError("shard must be a safe relative path")
        if self.line is not None and (
            isinstance(self.line, bool) or not isinstance(self.line, int) or self.line <= 0
        ):
            raise ValueError("line must be a positive integer")
        for field_name in ("episode_id", "record_id"):
            value = getattr(self, field_name)
            if value is not None:
                _require_non_blank(field_name, value)


@dataclass(frozen=True, slots=True)
class TraceValidationReport:
    """Return bounded issues and checked-object counts for one package."""

    valid: bool
    records_checked: int
    episodes_checked: int
    media_checked: int
    issues: tuple[TraceValidationIssue, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise ValueError("valid must be a boolean")
        for field_name in ("records_checked", "episodes_checked", "media_checked"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if not all(isinstance(issue, TraceValidationIssue) for issue in self.issues):
            raise ValueError("issues must contain TraceValidationIssue values")
        has_error = any(issue.severity is ValidationSeverity.ERROR for issue in self.issues)
        if self.valid == has_error:
            raise ValueError("valid must be false exactly when an error issue exists")


def inspect_trace_export(
    export_directory: Path,
    /,
    *,
    verify_media: bool = False,
    required_workspace_id: str | None = None,
) -> tuple[TraceExportManifest, TraceValidationReport]:
    """Read manifest metadata and validation under one pinned export root."""
    if not isinstance(export_directory, Path) or not export_directory.is_absolute():
        raise ValueError("trace export directory must be absolute")
    if type(verify_media) is not bool:
        raise ValueError("verify_media must be a boolean")
    if required_workspace_id is not None and (
        not isinstance(required_workspace_id, str) or not required_workspace_id.strip()
    ):
        raise ValueError("required_workspace_id must be a nonblank string")
    try:
        with _ExportRoot(export_directory) as root:
            manifest, payload = _read_open_manifest(root)
            report = _validate_open_export(
                root,
                verify_media=verify_media,
                max_record_bytes=16_777_216,
                max_manifest_bytes=4_194_304,
                expected_manifest_bytes=payload,
                required_workspace_id=required_workspace_id,
            )
            return manifest, report
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        raise ValueError("trace export inspection failed") from None


def _read_open_manifest(root: _ExportRoot) -> tuple[TraceExportManifest, bytes]:
    """Read one bounded manifest from an already pinned export root."""
    with root.open_regular(PurePosixPath("manifest.json")) as stream:
        payload = stream.read(4_194_305)
    if len(payload) > 4_194_304:
        raise ValueError("manifest is too large")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("manifest must be an object")
    return decode_trace_export_manifest(value), payload


def validate_trace_export(
    export_directory: Path,
    /,
    *,
    verify_media: bool = True,
    max_record_bytes: int = 16_777_216,
    max_manifest_bytes: int = 4_194_304,
) -> TraceValidationReport:
    """Validate one local export without following links or retaining unbounded state."""
    if not isinstance(export_directory, Path) or not export_directory.is_absolute():
        raise ValueError("export_directory must be absolute")
    if not isinstance(verify_media, bool):
        raise ValueError("verify_media must be a boolean")
    if (
        isinstance(max_record_bytes, bool)
        or not isinstance(max_record_bytes, int)
        or max_record_bytes <= 0
    ):
        raise ValueError("max_record_bytes must be a positive integer")
    if (
        isinstance(max_manifest_bytes, bool)
        or not isinstance(max_manifest_bytes, int)
        or max_manifest_bytes <= 0
    ):
        raise ValueError("max_manifest_bytes must be a positive integer")
    try:
        with _ExportRoot(export_directory) as root:
            return _validate_open_export(
                root,
                verify_media=verify_media,
                max_record_bytes=max_record_bytes,
                max_manifest_bytes=max_manifest_bytes,
            )
    except OSError:
        issues = _IssueCollector()
        issues.error("EXPORT_UNSAFE_PATH", "export directory is unavailable or unsafe")
        return issues.report(0, 0, 0)


def _validate_open_export(
    root: _ExportRoot,
    *,
    verify_media: bool,
    max_record_bytes: int,
    max_manifest_bytes: int,
    expected_manifest_bytes: bytes | None = None,
    required_workspace_id: str | None = None,
) -> TraceValidationReport:
    """Validate package bytes beneath one pinned local export directory."""
    issues = _IssueCollector()
    if not root.is_regular(PurePosixPath("manifest.json")):
        issues.error("MANIFEST_MISSING", "manifest.json is missing or is not a regular file")
        return issues.report(0, 0, 0)
    try:
        with root.open_regular(PurePosixPath("manifest.json")) as stream:
            manifest_bytes = stream.read(max_manifest_bytes + 1)
        if expected_manifest_bytes is not None and manifest_bytes != expected_manifest_bytes:
            issues.error("MANIFEST_CHANGED", "manifest changed during inspection")
            return issues.report(0, 0, 0)
        if len(manifest_bytes) > max_manifest_bytes:
            issues.error(
                "MANIFEST_TOO_LARGE",
                "manifest.json exceeds the configured byte limit",
            )
            return issues.report(0, 0, 0)
        manifest_value = json.loads(manifest_bytes)
        if not isinstance(manifest_value, dict):
            raise ValueError("manifest must be an object")
        manifest = decode_trace_export_manifest(manifest_value)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        issues.error("INVALID_MANIFEST", "manifest.json is not a valid version 1 manifest")
        return issues.report(0, 0, 0)
    canonical_manifest = canonical_json_bytes(encode_trace_export_manifest(manifest)) + b"\n"
    if manifest_bytes != canonical_manifest:
        issues.error("NON_CANONICAL_MANIFEST", "manifest.json is not canonical JSON plus LF")

    expected_files = {PurePosixPath("manifest.json"), manifest.schema_path}
    if not root.is_regular(manifest.schema_path):
        issues.error("SCHEMA_MISSING", "the declared trace schema is missing")
    else:
        schema_hash = root.hash_file(manifest.schema_path)[0]
        if schema_hash != manifest.schema_sha256:
            issues.error("SCHEMA_SHA256_MISMATCH", "the declared trace schema hash does not match")

    record_type_counts: Counter[str] = Counter()
    modality_counts: Counter[str] = Counter()
    tool_status_counts: Counter[str] = Counter()
    episode_status_counts: Counter[str] = Counter()
    records_checked = 0
    with tempfile.TemporaryDirectory(prefix="capicove-trace-validation-") as state_directory:
        state = _ValidationState(Path(state_directory) / "state.sqlite3", issues)
        try:
            for shard in manifest.shards:
                expected_files.add(shard.path)
                if not root.is_regular(shard.path):
                    issues.error(
                        "SHARD_MISSING",
                        "a declared JSONL shard is missing",
                        shard=shard.path,
                    )
                    continue
                actual_count = 0
                actual_size = 0
                digest = hashlib.sha256()
                try:
                    with root.open_regular(shard.path) as stream:
                        for line_number, line, line_size, oversized in _bounded_lines(
                            stream, max_record_bytes + 1, digest
                        ):
                            actual_count += 1
                            actual_size += line_size
                            if oversized:
                                issues.error(
                                    "RECORD_TOO_LARGE",
                                    "a JSONL record exceeds the configured byte limit",
                                    shard=shard.path,
                                    line=line_number,
                                )
                                continue
                            raw = line[:-1] if line.endswith(b"\n") else line
                            if not line.endswith(b"\n"):
                                issues.error(
                                    "MISSING_LINE_FEED",
                                    "a JSONL record is missing its trailing LF",
                                    shard=shard.path,
                                    line=line_number,
                                )
                            try:
                                decoded_json = json.loads(raw)
                                if not isinstance(decoded_json, dict):
                                    raise ValueError("record must be an object")
                            except (UnicodeError, json.JSONDecodeError, ValueError):
                                issues.error(
                                    "INVALID_JSON",
                                    "a JSONL line is not a valid JSON object",
                                    shard=shard.path,
                                    line=line_number,
                                )
                                continue
                            try:
                                validate_trace_privacy(decoded_json)
                            except TracePrivacyError as error:
                                issues.error(
                                    error.code,
                                    str(error),
                                    shard=shard.path,
                                    line=line_number,
                                    episode_id=cast(
                                        str | None,
                                        decoded_json.get("episode_id"),
                                    ),
                                    record_id=cast(
                                        str | None,
                                        decoded_json.get("record_id"),
                                    ),
                                )
                            try:
                                record = decode_trace_record(decoded_json)
                            except ValueError:
                                issues.error(
                                    "INVALID_RECORD",
                                    "a JSONL line violates the version 1 record contract",
                                    shard=shard.path,
                                    line=line_number,
                                )
                                continue
                            if canonical_record_bytes(record) != raw:
                                issues.error(
                                    "NON_CANONICAL_JSON",
                                    "a JSONL record does not use canonical encoding",
                                    shard=shard.path,
                                    line=line_number,
                                    episode_id=record.episode_id,
                                    record_id=record.record_id,
                                )
                            if (
                                required_workspace_id is not None
                                and isinstance(record.payload, EpisodeStarted)
                                and record.payload.workspace_id != required_workspace_id
                            ):
                                issues.error(
                                    "WORKSPACE_SCOPE_MISMATCH",
                                    "an Episode belongs to another workspace",
                                )
                            records_checked += 1
                            record_type_counts[record.record_type.value] += 1
                            modality_counts.update(item.value for item in record.modalities)
                            if isinstance(record.payload, ToolResultObserved):
                                tool_status_counts[record.payload.status.value] += 1
                            if isinstance(record.payload, EpisodeCompleted):
                                episode_status_counts[record.payload.status.value] += 1
                            state.observe(record, shard.path, line_number)
                except OSError:
                    issues.error(
                        "SHARD_READ_FAILED",
                        "a declared JSONL shard could not be read",
                        shard=shard.path,
                    )
                    continue
                if actual_count != shard.record_count:
                    issues.error(
                        "SHARD_RECORD_COUNT_MISMATCH",
                        "a shard record count does not match its manifest entry",
                        shard=shard.path,
                    )
                if actual_size != shard.size_bytes:
                    issues.error(
                        "SHARD_SIZE_MISMATCH",
                        "a shard byte size does not match its manifest entry",
                        shard=shard.path,
                    )
                if digest.hexdigest() != shard.sha256:
                    issues.error(
                        "SHARD_SHA256_MISMATCH",
                        "a shard digest does not match its manifest entry",
                        shard=shard.path,
                    )

            episodes_checked = state.finalize_episodes(manifest)
            media_checked = state.finalize_media(
                root,
                manifest,
                expected_files,
                verify_media=verify_media,
            )
            state.finalize_derivations()
        finally:
            state.close()

    _compare_count("RECORD_COUNT_MISMATCH", records_checked, manifest.record_count, issues)
    _compare_mapping(
        "RECORD_TYPE_COUNTS_MISMATCH", record_type_counts, manifest.record_type_counts, issues
    )
    _compare_mapping("MODALITY_COUNTS_MISMATCH", modality_counts, manifest.modality_counts, issues)
    _compare_mapping(
        "TOOL_STATUS_COUNTS_MISMATCH", tool_status_counts, manifest.tool_status_counts, issues
    )
    _compare_mapping(
        "EPISODE_STATUS_COUNTS_MISMATCH",
        episode_status_counts,
        manifest.episode_status_counts,
        issues,
    )
    _compare_count(
        "EPISODE_COUNT_MISMATCH", episodes_checked, manifest.included_episode_count, issues
    )
    root.find_unexpected_files(expected_files, issues)
    return issues.report(records_checked, episodes_checked, media_checked)


class _ValidationState:
    """Keep cross-record semantic state in a private temporary SQLite database."""

    def __init__(self, path: Path, issues: _IssueCollector) -> None:
        self._issues = issues
        self._connection = sqlite3.connect(path)
        self._connection.executescript(
            """
            CREATE TABLE episodes (
                episode_id TEXT PRIMARY KEY,
                last_sequence INTEGER NOT NULL,
                last_hash TEXT,
                trace_id TEXT NOT NULL,
                started INTEGER NOT NULL,
                completed INTEGER NOT NULL
            );
            CREATE TABLE record_ids (record_id TEXT PRIMARY KEY);
            CREATE TABLE open_calls (
                episode_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                tool_id TEXT NOT NULL,
                tool_version INTEGER NOT NULL,
                call_record_id TEXT NOT NULL,
                PRIMARY KEY (episode_id, request_id)
            );
            CREATE TABLE media (
                media_id TEXT PRIMARY KEY,
                identity_sha256 TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                media_type TEXT NOT NULL,
                missing INTEGER NOT NULL
            );
            CREATE TABLE media_hashes (
                sha256 TEXT PRIMARY KEY,
                size_bytes INTEGER NOT NULL,
                media_type TEXT NOT NULL
            );
            CREATE TABLE derivations (
                child_id TEXT NOT NULL,
                parent_id TEXT NOT NULL,
                PRIMARY KEY (child_id, parent_id)
            );
            """
        )

    def observe(self, record: TraceRecord, shard: PurePosixPath, line: int) -> None:
        """Validate lifecycle, chain, Tool pairing, and media identity for one record."""
        try:
            self._connection.execute(
                "INSERT INTO record_ids (record_id) VALUES (?)", (record.record_id,)
            )
        except sqlite3.IntegrityError:
            self._issues.error(
                "DUPLICATE_RECORD_ID",
                "record_id appears more than once",
                shard=shard,
                line=line,
                episode_id=record.episode_id,
                record_id=record.record_id,
            )
        episode = self._connection.execute(
            """
            SELECT last_sequence, last_hash, trace_id, started, completed
            FROM episodes WHERE episode_id = ?
            """,
            (record.episode_id,),
        ).fetchone()
        if episode is None:
            episode = (-1, None, record.trace_id, 0, 0)
            self._connection.execute(
                """
                INSERT INTO episodes (
                    episode_id, last_sequence, last_hash, trace_id, started, completed
                )
                VALUES (?, -1, NULL, ?, 0, 0)
                """,
                (record.episode_id, record.trace_id),
            )
        expected_sequence = episode[0] + 1
        if record.sequence != expected_sequence:
            self._issues.error(
                "SEQUENCE_GAP",
                "Episode sequence is not contiguous",
                shard=shard,
                line=line,
                episode_id=record.episode_id,
                record_id=record.record_id,
            )
        if record.trace_id != episode[2]:
            self._issues.error(
                "TRACE_ID_MISMATCH",
                "records in one Episode must share one trace_id",
                shard=shard,
                line=line,
                episode_id=record.episode_id,
                record_id=record.record_id,
            )
        if record.previous_record_sha256 != episode[1]:
            self._issues.error(
                "PREDECESSOR_MISMATCH",
                "record predecessor does not match the prior canonical record",
                shard=shard,
                line=line,
                episode_id=record.episode_id,
                record_id=record.record_id,
            )
        if episode[4] and not isinstance(record.payload, EpisodeFeedback):
            self._issues.error(
                "EVENT_AFTER_COMPLETION",
                "an ordinary event appears after Episode completion",
                shard=shard,
                line=line,
                episode_id=record.episode_id,
                record_id=record.record_id,
            )
        started = episode[3]
        completed = episode[4]
        if isinstance(record.payload, EpisodeStarted):
            if record.sequence != 0:
                self._issues.error(
                    "EPISODE_START_SEQUENCE",
                    "Episode start must be the sequence-zero record",
                    shard=shard,
                    line=line,
                    episode_id=record.episode_id,
                    record_id=record.record_id,
                )
            if started:
                self._issues.error(
                    "DUPLICATE_EPISODE_START",
                    "Episode contains more than one start record",
                    shard=shard,
                    line=line,
                    episode_id=record.episode_id,
                )
            started = 1
        if isinstance(record.payload, EpisodeCompleted):
            if completed:
                self._issues.error(
                    "DUPLICATE_EPISODE_COMPLETION",
                    "Episode contains more than one completion record",
                    shard=shard,
                    line=line,
                    episode_id=record.episode_id,
                )
            completed = 1
        if isinstance(record.payload, ToolCallObserved):
            try:
                self._connection.execute(
                    """
                    INSERT INTO open_calls (
                        episode_id, request_id, tool_id, tool_version, call_record_id
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        record.episode_id,
                        record.request_id,
                        record.payload.tool_id,
                        record.payload.tool_version,
                        record.record_id,
                    ),
                )
            except sqlite3.IntegrityError:
                self._issues.error(
                    "DUPLICATE_TOOL_CALL",
                    "request_id has more than one open Tool call",
                    shard=shard,
                    line=line,
                    episode_id=record.episode_id,
                )
        if isinstance(record.payload, ToolResultObserved):
            call = self._connection.execute(
                """
                SELECT tool_id, tool_version, call_record_id FROM open_calls
                WHERE episode_id = ? AND request_id = ?
                """,
                (record.episode_id, record.request_id),
            ).fetchone()
            if call is None:
                self._issues.error(
                    "UNMATCHED_TOOL_RESULT",
                    "Tool result has no matching open call",
                    shard=shard,
                    line=line,
                    episode_id=record.episode_id,
                )
            else:
                if call[:2] != (record.payload.tool_id, record.payload.tool_version):
                    self._issues.error(
                        "TOOL_IDENTITY_MISMATCH",
                        "Tool result identity differs from its call",
                        shard=shard,
                        line=line,
                        episode_id=record.episode_id,
                    )
                if record.parent_record_id != call[2]:
                    self._issues.error(
                        "TOOL_RESULT_PARENT_MISMATCH",
                        "Tool result parent_record_id must name its call record",
                        shard=shard,
                        line=line,
                        episode_id=record.episode_id,
                        record_id=record.record_id,
                    )
                self._connection.execute(
                    "DELETE FROM open_calls WHERE episode_id = ? AND request_id = ?",
                    (record.episode_id, record.request_id),
                )
        record_hash = trace_record_sha256(record)
        self._connection.execute(
            """
            UPDATE episodes
            SET last_sequence = ?, last_hash = ?, started = ?, completed = ?
            WHERE episode_id = ?
            """,
            (record.sequence, record_hash, started, completed, record.episode_id),
        )
        if isinstance(record.payload, MessageObserved):
            content_media_ids = {
                block.media_id
                for block in record.payload.content
                if isinstance(block, MediaContent)
            }
            reference_media_ids = {reference.media_id for reference in record.media}
            if content_media_ids != reference_media_ids:
                self._issues.error(
                    "MEDIA_CONTENT_REFERENCE_MISMATCH",
                    "message media blocks and record media references must match",
                    shard=shard,
                    line=line,
                    episode_id=record.episode_id,
                    record_id=record.record_id,
                )
        for reference in record.media:
            existing = self._connection.execute(
                """
                SELECT identity_sha256, sha256, size_bytes, media_type, missing
                FROM media WHERE media_id = ?
                """,
                (reference.media_id,),
            ).fetchone()
            identity = (
                media_reference_identity_sha256(reference),
                reference.sha256,
                reference.size_bytes,
                reference.media_type,
                int(reference.locator is None),
            )
            if existing is None:
                self._connection.execute(
                    """
                    INSERT INTO media (
                        media_id, identity_sha256, sha256, size_bytes, media_type, missing
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (reference.media_id, *identity),
                )
            elif existing != identity:
                self._issues.error(
                    "MEDIA_IDENTITY_CONFLICT",
                    "one media_id has conflicting immutable identity",
                    shard=shard,
                    line=line,
                    episode_id=record.episode_id,
                )
            content_identity = (reference.size_bytes, reference.media_type)
            existing_content = self._connection.execute(
                "SELECT size_bytes, media_type FROM media_hashes WHERE sha256 = ?",
                (reference.sha256,),
            ).fetchone()
            if existing_content is None:
                self._connection.execute(
                    """
                    INSERT INTO media_hashes (sha256, size_bytes, media_type)
                    VALUES (?, ?, ?)
                    """,
                    (reference.sha256, *content_identity),
                )
            elif existing_content != content_identity:
                self._issues.error(
                    "MEDIA_CONTENT_IDENTITY_CONFLICT",
                    "one media hash has conflicting size or media type",
                    shard=shard,
                    line=line,
                    episode_id=record.episode_id,
                    record_id=record.record_id,
                )
            for parent in reference.derived_from:
                self._connection.execute(
                    "INSERT OR IGNORE INTO derivations (child_id, parent_id) VALUES (?, ?)",
                    (reference.media_id, parent),
                )
        if isinstance(record.payload, ArtifactProduced):
            attached = tuple(
                reference
                for reference in record.media
                if reference.media_id == record.payload.media_id
            )
            if len(attached) != 1:
                self._issues.error(
                    "ARTIFACT_MEDIA_REFERENCE_MISMATCH",
                    "artifact payload must name one attached media reference",
                    shard=shard,
                    line=line,
                    episode_id=record.episode_id,
                    record_id=record.record_id,
                )
            elif record.payload.derived_from != attached[0].derived_from:
                self._issues.error(
                    "ARTIFACT_DERIVATION_MISMATCH",
                    "artifact and attached media derivation parents must match",
                    shard=shard,
                    line=line,
                    episode_id=record.episode_id,
                    record_id=record.record_id,
                )

    def finalize_episodes(self, manifest: TraceExportManifest) -> int:
        """Validate Episode start/completion and dangling Tool calls."""
        rows = tuple(
            self._connection.execute("SELECT episode_id, started, completed FROM episodes")
        )
        for episode_id, started, completed in rows:
            if not started:
                self._issues.error(
                    "MISSING_EPISODE_START",
                    "Episode has no start record",
                    episode_id=episode_id,
                )
            if not completed:
                if manifest.options.query.include_incomplete:
                    self._issues.warning(
                        "INCOMPLETE_EPISODE",
                        "Episode has no completion record and is explicitly included",
                        episode_id=episode_id,
                    )
                else:
                    self._issues.error(
                        "INCOMPLETE_EPISODE",
                        "Episode has no completion record but incomplete export is disabled",
                        episode_id=episode_id,
                    )
        for (episode_id,) in self._connection.execute(
            "SELECT DISTINCT open_calls.episode_id "
            "FROM open_calls JOIN episodes "
            "ON episodes.episode_id = open_calls.episode_id "
            "WHERE episodes.completed = 1"
        ):
            self._issues.error(
                "DANGLING_TOOL_CALL",
                "Episode ends with an unmatched Tool call",
                episode_id=episode_id,
            )
        return len(rows)

    def finalize_media(
        self,
        export_root: _ExportRoot,
        manifest: TraceExportManifest,
        expected_files: set[PurePosixPath],
        *,
        verify_media: bool,
    ) -> int:
        """Validate manifest media aggregates and optional copied bytes."""
        media_count, missing_count = self._connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(missing), 0) FROM media"
        ).fetchone()
        media_size = self._connection.execute(
            "SELECT COALESCE(SUM(size_bytes), 0) FROM (SELECT sha256, MAX(size_bytes) size_bytes FROM media GROUP BY sha256)"
        ).fetchone()[0]
        _compare_count("MEDIA_COUNT_MISMATCH", media_count, manifest.media_count, self._issues)
        _compare_count("MEDIA_SIZE_MISMATCH", media_size, manifest.media_size_bytes, self._issues)
        expected_missing = 0 if manifest.options.media_mode is MediaMode.COPY else missing_count
        _compare_count(
            "MISSING_MEDIA_COUNT_MISMATCH",
            expected_missing,
            manifest.missing_media_count,
            self._issues,
        )
        checked_hashes: set[str] = set()
        if manifest.options.media_mode is not MediaMode.COPY:
            return 0
        for sha256, size_bytes, media_type in self._connection.execute(
            "SELECT sha256, MAX(size_bytes), MIN(media_type) FROM media GROUP BY sha256"
        ):
            extension = _EXTENSIONS.get(media_type, ".bin")
            relative = PurePosixPath("media", "sha256", sha256[:2], f"{sha256}{extension}")
            expected_files.add(relative)
            if not verify_media:
                continue
            if not export_root.is_regular(relative):
                self._issues.error("MEDIA_MISSING", "copied media is missing or not a regular file")
                continue
            actual_hash, actual_size = export_root.hash_file(relative)
            if actual_size != size_bytes:
                self._issues.error("MEDIA_SIZE_MISMATCH", "copied media byte size does not match")
            if actual_hash != sha256:
                self._issues.error("MEDIA_SHA256_MISMATCH", "copied media digest does not match")
            checked_hashes.add(sha256)
        return len(checked_hashes)

    def finalize_derivations(self) -> None:
        """Reject unknown derivation parents and graph cycles."""
        for child_id, _parent_id in self._connection.execute(
            """
            SELECT child_id, parent_id FROM derivations
            WHERE parent_id NOT IN (SELECT media_id FROM media)
            """
        ):
            self._issues.error(
                "UNKNOWN_DERIVATION_PARENT",
                f"media derivation parent for {child_id} is not present",
            )
        cycle = self._connection.execute(
            """
            WITH RECURSIVE reach(origin, node) AS (
                SELECT child_id, parent_id FROM derivations
                UNION
                SELECT reach.origin, derivations.parent_id
                FROM reach JOIN derivations ON derivations.child_id = reach.node
            )
            SELECT 1 FROM reach WHERE origin = node LIMIT 1
            """
        ).fetchone()
        if cycle is not None:
            self._issues.error("DERIVATION_CYCLE", "media derivation graph contains a cycle")

    def close(self) -> None:
        self._connection.close()


class _IssueCollector:
    """Cap retained issues while continuing integrity calculations."""

    def __init__(self) -> None:
        self.issues: list[TraceValidationIssue] = []
        self._limit_reported = False

    def error(self, code: str, message: str, **location: object) -> None:
        self._add(ValidationSeverity.ERROR, code, message, location)

    def warning(self, code: str, message: str, **location: object) -> None:
        self._add(ValidationSeverity.WARNING, code, message, location)

    def report(
        self, records_checked: int, episodes_checked: int, media_checked: int
    ) -> TraceValidationReport:
        valid = not any(issue.severity is ValidationSeverity.ERROR for issue in self.issues)
        return TraceValidationReport(
            valid,
            records_checked,
            episodes_checked,
            media_checked,
            tuple(self.issues),
        )

    def _add(
        self,
        severity: ValidationSeverity,
        code: str,
        message: str,
        location: dict[str, object],
    ) -> None:
        if len(self.issues) < _ISSUE_LIMIT:
            self.issues.append(
                TraceValidationIssue(
                    severity,
                    code,
                    message,
                    shard=cast(PurePosixPath | None, location.get("shard")),
                    line=cast(int | None, location.get("line")),
                    episode_id=cast(str | None, location.get("episode_id")),
                    record_id=cast(str | None, location.get("record_id")),
                )
            )
        elif not self._limit_reported:
            self._limit_reported = True
            self.issues.append(
                TraceValidationIssue(
                    ValidationSeverity.ERROR,
                    "ISSUE_LIMIT_REACHED",
                    "validation issue retention limit was reached",
                )
            )


def _bounded_lines(
    stream: BinaryIO,
    maximum_line_bytes: int,
    digest: _Digest,
) -> _BoundedLineIterator:
    return _BoundedLineIterator(stream, maximum_line_bytes, digest)


class _Digest(Protocol):
    def update(self, payload: bytes, /) -> object:
        """Incorporate one byte chunk into a streaming digest."""


class _BoundedLineIterator:
    def __init__(self, stream: BinaryIO, maximum: int, digest: _Digest) -> None:
        self._stream = stream
        self._maximum = maximum
        self._digest = digest
        self._line = 0

    def __iter__(self) -> _BoundedLineIterator:
        return self

    def __next__(self) -> tuple[int, bytes, int, bool]:
        first = self._stream.readline(self._maximum + 1)
        if not first:
            raise StopIteration
        self._line += 1
        self._digest.update(first)
        retained = first
        size_bytes = len(first)
        if not first.endswith(b"\n"):
            while True:
                chunk = self._stream.readline(self._maximum + 1)
                if not chunk:
                    break
                self._digest.update(chunk)
                size_bytes += len(chunk)
                if chunk.endswith(b"\n"):
                    break
        oversized = size_bytes > self._maximum
        return self._line, retained, size_bytes, oversized


def _compare_count(
    code: str,
    actual: int,
    expected: int,
    issues: _IssueCollector,
) -> None:
    if actual != expected:
        issues.error(code, "an aggregate count does not match the manifest")


def _compare_mapping(
    code: str,
    actual: Counter[str],
    expected: Mapping[str, int],
    issues: _IssueCollector,
) -> None:
    if dict(actual) != dict(expected):
        issues.error(code, "an aggregate count mapping does not match the manifest")


class _ExportRoot:
    """Pin a POSIX export directory and open only regular children beneath it."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._descriptor: int | None = None

    def __enter__(self) -> _ExportRoot:
        if os.name != "posix":
            return self
        if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
            raise OSError("secure export directory APIs are unavailable")
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        descriptor = os.open(self.path.anchor, flags)
        try:
            for component in self.path.parts[1:]:
                child = os.open(component, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
        except Exception:
            os.close(descriptor)
            raise
        self._descriptor = descriptor
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None

    def open_regular(self, relative: PurePosixPath) -> BinaryIO:
        """Open one regular package file without traversing a POSIX link."""
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise OSError("invalid export member path")
        if os.name != "posix":
            path = self.path.joinpath(*relative.parts)
            if not stat.S_ISREG(path.lstat().st_mode):
                raise OSError("export member is not a regular file")
            return path.open("rb")
        assert self._descriptor is not None
        directory_fd = os.dup(self._descriptor)
        try:
            for component in relative.parts[:-1]:
                child = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
                os.close(directory_fd)
                directory_fd = child
            file_fd = os.open(
                relative.parts[-1],
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory_fd,
            )
        finally:
            os.close(directory_fd)
        try:
            if not stat.S_ISREG(os.fstat(file_fd).st_mode):
                raise OSError("export member is not a regular file")
            return os.fdopen(file_fd, "rb")
        except Exception:
            os.close(file_fd)
            raise

    def is_regular(self, relative: PurePosixPath) -> bool:
        """Check a member through the same no-follow path used for reading."""
        try:
            with self.open_regular(relative):
                return True
        except OSError:
            return False

    def hash_file(self, relative: PurePosixPath) -> tuple[str, int]:
        """Hash one opened member without reopening it through an unsafe path."""
        digest = hashlib.sha256()
        size_bytes = 0
        with self.open_regular(relative) as stream:
            while chunk := stream.read(1_048_576):
                digest.update(chunk)
                size_bytes += len(chunk)
        return digest.hexdigest(), size_bytes

    def find_unexpected_files(self, expected: set[PurePosixPath], issues: _IssueCollector) -> None:
        """Scan the pinned package tree without traversing symbolic links."""
        try:
            if self._descriptor is None:
                for path in self.path.rglob("*"):
                    observed = path.lstat()
                    if not stat.S_ISDIR(observed.st_mode):
                        relative = PurePosixPath(path.relative_to(self.path).as_posix())
                        if relative not in expected:
                            issues.error(
                                "UNEXPECTED_FILE", "the export contains an undeclared file"
                            )
                return
            for directory, child_directories, files, directory_fd in os.fwalk(
                ".", dir_fd=self._descriptor, follow_symlinks=False
            ):
                base = PurePosixPath() if directory == "." else PurePosixPath(directory)
                for name in sorted((*child_directories, *files)):
                    try:
                        observed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    except OSError:
                        issues.error("UNREADABLE_PATH", "an export path could not be inspected")
                        continue
                    if not stat.S_ISDIR(observed.st_mode) and base / name not in expected:
                        issues.error("UNEXPECTED_FILE", "the export contains an undeclared file")
        except OSError:
            issues.error("EXPORT_SCAN_FAILED", "the export directory could not be scanned")


def _require_non_blank(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be blank")
