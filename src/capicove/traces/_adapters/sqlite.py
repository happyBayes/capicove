"""Private append-only SQLite implementation of the trace store port."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
from collections.abc import Callable, Iterator
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, cast

from capicove.traces.codec import (
    canonical_json_bytes,
    canonical_record_bytes,
    decode_trace_record,
    encode_trace_draft,
    trace_record_sha256,
)
from capicove.traces.events import (
    EpisodeCompleted,
    EpisodeFeedback,
    EpisodeStarted,
    ToolCallObserved,
    ToolResultObserved,
)
from capicove.traces.models import TraceRecord, TraceRecordDraft
from capicove.traces.ports import (
    DEFAULT_TRACE_QUERY,
    TraceConflictError,
    TraceLifecycleError,
    TracePostCommitPrivacyError,
    TraceQuery,
    TraceSelectionSummary,
    TraceStoreError,
)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS trace_episodes (
    episode_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    privacy_class TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    terminal_status TEXT
);

CREATE TABLE IF NOT EXISTS trace_records (
    episode_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    idempotency_key TEXT NOT NULL,
    draft_sha256 TEXT NOT NULL,
    record_id TEXT NOT NULL UNIQUE,
    record_type TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    modality_keys TEXT NOT NULL,
    tool_id TEXT,
    tool_status TEXT,
    privacy_class TEXT NOT NULL,
    record_json TEXT NOT NULL,
    record_sha256 TEXT NOT NULL,
    PRIMARY KEY (episode_id, sequence),
    UNIQUE (episode_id, idempotency_key),
    FOREIGN KEY (episode_id) REFERENCES trace_episodes(episode_id)
);

CREATE INDEX IF NOT EXISTS trace_records_occurred
    ON trace_records (occurred_at, episode_id, sequence);
CREATE INDEX IF NOT EXISTS trace_records_type ON trace_records (record_type);
CREATE INDEX IF NOT EXISTS trace_records_tool ON trace_records (tool_id);
CREATE INDEX IF NOT EXISTS trace_records_tool_status ON trace_records (tool_status);
"""


class SqliteTraceStore:
    """Persist immutable trace records in one explicitly configured SQLite file."""

    def __init__(self, database_path: Path) -> None:
        if not isinstance(database_path, Path) or not database_path.is_absolute():
            raise ValueError("database_path must be absolute")
        self._database_path = database_path
        try:
            if os.name == "nt":
                self._validate_windows_parent()
            else:
                parent_existed = database_path.parent.exists()
                database_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                if not parent_existed:
                    database_path.parent.chmod(0o700)
                parent_stat = database_path.parent.lstat()
                if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
                    raise TraceStoreError("trace store parent must be a regular directory")
                if stat.S_IMODE(parent_stat.st_mode) & 0o077:
                    raise TraceStoreError("trace store parent permissions must be owner-only")
            self._prepare_database_file()
            with closing(self._connect()) as connection:
                version = cast(
                    sqlite3.Row,
                    connection.execute("PRAGMA user_version").fetchone(),
                )[0]
                if version not in (0, 1):
                    raise TraceStoreError("trace store schema version is unsupported")
                connection.executescript(_SCHEMA)
                connection.execute("PRAGMA user_version = 1")
            if os.name == "nt":
                self._validate_windows_files()
        except TraceStoreError:
            raise
        except OSError as error:
            raise TraceStoreError("trace store parent could not be prepared") from error
        except sqlite3.Error as error:
            raise TraceStoreError("trace store initialization failed") from error

    def append(
        self,
        draft: TraceRecordDraft,
        /,
        *,
        idempotency_key: str,
    ) -> TraceRecord:
        """Append one record transactionally or return its idempotent predecessor."""
        if not isinstance(draft, TraceRecordDraft):
            raise ValueError("draft must be a TraceRecordDraft")
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise ValueError("idempotency_key must not be blank")
        draft_sha256 = _idempotency_draft_sha256(draft)
        connection = self._connect()
        pending_error: BaseException | None = None
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT record_json
                FROM trace_records
                WHERE episode_id = ? AND idempotency_key = ?
                """,
                (draft.episode_id, idempotency_key),
            ).fetchone()
            if existing is not None:
                existing_record = _decode_json_record(existing["record_json"])
                if _idempotency_record_sha256(existing_record) != draft_sha256:
                    raise TraceConflictError(
                        "idempotency key already identifies different trace content"
                    )
                record = _decode_json_record(existing["record_json"])
                if os.name == "nt":
                    self._validate_windows_files()
                connection.execute("COMMIT")
                return record

            episode = connection.execute(
                """
                SELECT trace_id, completed_at, terminal_status
                FROM trace_episodes
                WHERE episode_id = ?
                """,
                (draft.episode_id,),
            ).fetchone()
            if isinstance(draft.payload, EpisodeStarted):
                if episode is not None:
                    raise TraceLifecycleError("Episode has already started")
                connection.execute(
                    """
                    INSERT INTO trace_episodes (
                        episode_id, trace_id, privacy_class, started_at,
                        completed_at, terminal_status
                    ) VALUES (?, ?, ?, ?, NULL, NULL)
                    """,
                    (
                        draft.episode_id,
                        draft.trace_id,
                        draft.privacy_class.value,
                        _timestamp(draft.occurred_at),
                    ),
                )
            else:
                if episode is None:
                    raise TraceLifecycleError("Episode must start before ordinary records")
                if episode["trace_id"] != draft.trace_id:
                    raise TraceLifecycleError("record trace_id must match its Episode")
                completed = episode["completed_at"] is not None
                if isinstance(draft.payload, EpisodeFeedback):
                    if not completed:
                        raise TraceLifecycleError("feedback requires a completed Episode")
                elif completed:
                    raise TraceLifecycleError("only feedback may follow Episode completion")

            prior = connection.execute(
                """
                SELECT sequence, record_sha256
                FROM trace_records
                WHERE episode_id = ?
                ORDER BY sequence DESC
                LIMIT 1
                """,
                (draft.episode_id,),
            ).fetchone()
            sequence = 0 if prior is None else cast(int, prior["sequence"]) + 1
            if isinstance(draft.payload, EpisodeStarted) and sequence != 0:
                raise TraceLifecycleError("Episode start must be sequence zero")
            if not isinstance(draft.payload, EpisodeStarted) and sequence == 0:
                raise TraceLifecycleError("Episode start record is missing")
            previous_hash = None if prior is None else cast(str, prior["record_sha256"])
            identity = f"{draft.episode_id}\0{sequence}\0{idempotency_key}".encode()
            record_id = f"record:{hashlib.sha256(identity).hexdigest()}"
            record = TraceRecord(
                schema_version=1,
                record_type=draft.record_type,
                record_version=1,
                record_id=record_id,
                episode_id=draft.episode_id,
                trace_id=draft.trace_id,
                sequence=sequence,
                occurred_at=draft.occurred_at,
                payload=draft.payload,
                modalities=draft.modalities,
                step_id=draft.step_id,
                request_id=draft.request_id,
                parent_record_id=draft.parent_record_id,
                media=draft.media,
                lineage=draft.lineage,
                warnings=draft.warnings,
                error=draft.error,
                privacy_class=draft.privacy_class,
                previous_record_sha256=previous_hash,
            )
            record_json = canonical_record_bytes(record).decode("utf-8")
            record_sha256 = trace_record_sha256(record)
            tool_id = None
            tool_status = None
            if isinstance(draft.payload, (ToolCallObserved, ToolResultObserved)):
                tool_id = draft.payload.tool_id
            if isinstance(draft.payload, ToolResultObserved):
                tool_status = draft.payload.status.value
            modality_keys = f",{','.join(sorted(item.value for item in draft.modalities))},"
            connection.execute(
                """
                INSERT INTO trace_records (
                    episode_id, sequence, idempotency_key, draft_sha256,
                    record_id, record_type, occurred_at, modality_keys,
                    tool_id, tool_status, privacy_class, record_json, record_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    draft.episode_id,
                    sequence,
                    idempotency_key,
                    draft_sha256,
                    record_id,
                    draft.record_type.value,
                    _timestamp(draft.occurred_at),
                    modality_keys,
                    tool_id,
                    tool_status,
                    draft.privacy_class.value,
                    record_json,
                    record_sha256,
                ),
            )
            if isinstance(draft.payload, EpisodeCompleted):
                connection.execute(
                    """
                    UPDATE trace_episodes
                    SET completed_at = ?, terminal_status = ?
                    WHERE episode_id = ?
                    """,
                    (
                        _timestamp(draft.occurred_at),
                        draft.payload.status.value,
                        draft.episode_id,
                    ),
                )
            if os.name == "nt":
                self._validate_windows_files()
            connection.execute("COMMIT")
            return record
        except (TraceConflictError, TraceLifecycleError) as error:
            pending_error = error
            _rollback(connection)
            raise
        except sqlite3.IntegrityError as error:
            pending_error = TraceConflictError("trace append conflicts with persisted state")
            _rollback(connection)
            raise pending_error from error
        except sqlite3.Error as error:
            pending_error = TraceStoreError("trace append failed")
            _rollback(connection)
            raise pending_error from error
        except Exception as error:
            pending_error = error
            _rollback(connection)
            raise
        except BaseException as error:
            pending_error = error
            raise
        finally:
            close_error: BaseException | None = None
            try:
                connection.close()
            except BaseException as error:
                close_error = error
            privacy_error: TraceStoreError | None = None
            if os.name == "nt":
                try:
                    self._validate_windows_files()
                except TraceStoreError as error:
                    privacy_error = error
            if pending_error is not None:
                if close_error is not None:
                    pending_error.add_note(f"trace append connection close failed: {close_error}")
                if privacy_error is not None:
                    pending_error.add_note(
                        "Windows trace privacy recheck failed after SQLite closed"
                    )
            elif close_error is not None:
                if not isinstance(close_error, Exception):
                    if privacy_error is not None:
                        close_error.add_note(
                            "Windows trace privacy recheck failed after SQLite closed"
                        )
                    raise close_error
                if os.name != "nt":
                    raise close_error
                failure = TraceStoreError(
                    "trace append may have persisted; retry only with the same idempotency_key"
                )
                if privacy_error is not None:
                    failure.add_note("Windows trace privacy recheck failed after SQLite closed")
                raise failure from close_error
            elif privacy_error is not None:
                raise TracePostCommitPrivacyError(
                    "trace append may have persisted; retry only with the same idempotency_key"
                ) from privacy_error

    def get_episode(self, episode_id: str, /) -> tuple[TraceRecord, ...]:
        """Return one immutable Episode ordered by sequence."""
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id must not be blank")
        pending_error: BaseException | None = None
        try:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    """
                    SELECT record_json
                    FROM trace_records
                    WHERE episode_id = ?
                    ORDER BY sequence
                    """,
                    (episode_id,),
                )
                records = tuple(_decode_json_record(row["record_json"]) for row in rows)
            return records
        except sqlite3.Error as error:
            pending_error = TraceStoreError("trace read failed")
            raise pending_error from error
        except BaseException as error:
            pending_error = error
            raise
        finally:
            if os.name == "nt":
                try:
                    self._validate_windows_files()
                except TraceStoreError:
                    if pending_error is not None:
                        pending_error.add_note(
                            "Windows trace privacy recheck failed after SQLite closed"
                        )
                    else:
                        raise

    def open_snapshot(
        self,
        query: TraceQuery = DEFAULT_TRACE_QUERY,
        /,
    ) -> _SqliteTraceSnapshot:
        """Open one consistent read snapshot and select whole Episodes."""
        if not isinstance(query, TraceQuery):
            raise ValueError("query must be a TraceQuery")
        if os.name == "nt":
            self._validate_windows_files()
        connection: sqlite3.Connection | None = None
        try:
            uri = f"{self._database_path.as_uri()}?mode=ro"
            connection = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=5.0)
            _configure_connection(connection, writable=False)
            if os.name == "nt":
                self._validate_windows_files()
            connection.execute("BEGIN")
            snapshot = _SqliteTraceSnapshot(
                connection,
                query,
                on_close=self._validate_windows_files if os.name == "nt" else None,
            )
            if os.name == "nt":
                self._validate_windows_files()
            return snapshot
        except sqlite3.Error as error:
            if connection is not None:
                connection.close()
            failure = TraceStoreError("trace snapshot failed")
            if os.name == "nt":
                try:
                    self._validate_windows_files()
                except TraceStoreError:
                    failure.add_note("Windows trace privacy recheck failed after SQLite closed")
            raise failure from error
        except BaseException as error:
            if connection is not None:
                connection.close()
            if os.name == "nt":
                try:
                    self._validate_windows_files()
                except TraceStoreError:
                    error.add_note("Windows trace privacy recheck failed after SQLite closed")
            raise

    def _connect(self) -> sqlite3.Connection:
        if os.name == "nt":
            self._validate_windows_files()
        connection = sqlite3.connect(
            self._database_path,
            isolation_level=None,
            timeout=5.0,
        )
        try:
            _configure_connection(connection, writable=True)
            self._enforce_private_file_permissions()
        except BaseException:
            connection.close()
            raise
        return connection

    def _prepare_database_file(self) -> None:
        """Create or validate the database path without following a link."""
        if os.name == "nt":
            self._validate_windows_files()
            try:
                self._database_path.lstat()
            except FileNotFoundError:
                flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
                descriptor = os.open(self._database_path, flags, 0o600)
                os.close(descriptor)
            self._validate_windows_files()
            return
        try:
            existing = self._database_path.lstat()
        except FileNotFoundError:
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(self._database_path, flags, 0o600)
            os.close(descriptor)
            return
        if stat.S_ISLNK(existing.st_mode) or not stat.S_ISREG(existing.st_mode):
            raise TraceStoreError("trace store database must be a regular file")
        self._database_path.chmod(0o600)

    def _enforce_private_file_permissions(self) -> None:
        """Keep the database and SQLite journal sidecars owner-readable only."""
        if os.name == "nt":
            self._validate_windows_files()
            return
        for path in (
            self._database_path,
            Path(f"{self._database_path}-wal"),
            Path(f"{self._database_path}-shm"),
            Path(f"{self._database_path}-journal"),
        ):
            try:
                path.chmod(0o600, follow_symlinks=False)
            except FileNotFoundError:
                continue

    def _validate_windows_parent(self) -> None:
        from capicove._windows_files import validate_windows_path

        try:
            validate_windows_path(
                self._database_path.parent,
                directory=True,
                private_acl=True,
                require_private_inheritance=True,
            )
        except (OSError, ValueError) as error:
            raise TraceStoreError(
                "trace store requires an existing private Windows directory"
            ) from error

    def _validate_windows_files(self) -> None:
        from capicove._windows_files import validate_windows_file_target

        for path in (
            self._database_path,
            Path(f"{self._database_path}-wal"),
            Path(f"{self._database_path}-shm"),
            Path(f"{self._database_path}-journal"),
        ):
            try:
                validate_windows_file_target(path)
            except (OSError, ValueError) as error:
                raise TraceStoreError("trace store requires private Windows files") from error


class _SqliteTraceSnapshot:
    """Own one SQLite read transaction and its selected Episode set."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        query: TraceQuery,
        *,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self._connection = connection
        self._on_close = on_close
        self._closed = False
        self._iterated = False
        conditions, parameters = _selection_conditions(query)
        where_clause = " AND ".join(conditions) if conditions else "1 = 1"
        restricted = (
            "EXISTS (SELECT 1 FROM trace_records restricted "
            "WHERE restricted.episode_id = e.episode_id "
            "AND restricted.privacy_class = 'restricted')"
        )
        connection.execute(
            "CREATE TEMP TABLE selected_trace_episodes (episode_id TEXT PRIMARY KEY)"
        )
        selection_conditions = list(conditions)
        selection_parameters = list(parameters)
        if not query.include_restricted:
            selection_conditions.append(f"NOT {restricted}")
        selected_where = " AND ".join(selection_conditions) if selection_conditions else "1 = 1"
        # Clauses are fixed internally; filter values remain DB-API parameters.
        connection.execute(
            f"""
            INSERT INTO selected_trace_episodes (episode_id)
            SELECT e.episode_id
            FROM trace_episodes e
            WHERE {selected_where}
            """,  # nosec B608
            tuple(selection_parameters),
        )
        counts = cast(
            sqlite3.Row,
            connection.execute(
                """
                SELECT
                    COUNT(*) AS selected_count,
                    COALESCE(SUM(
                        CASE
                            WHEN e.completed_at IS NULL OR e.terminal_status = 'incomplete'
                            THEN 1 ELSE 0
                        END
                    ), 0) AS incomplete_count
                FROM selected_trace_episodes selected
                JOIN trace_episodes e ON e.episode_id = selected.episode_id
                """
            ).fetchone(),
        )
        omitted_restricted = 0
        if not query.include_restricted:
            omitted_restricted = cast(
                sqlite3.Row,
                # Clauses are fixed internally; filter values remain DB-API parameters.
                connection.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM trace_episodes e
                    WHERE {where_clause} AND {restricted}
                    """,  # nosec B608
                    tuple(parameters),
                ).fetchone(),
            )[0]
        self._summary = TraceSelectionSummary(
            selected_episode_count=counts["selected_count"],
            selected_incomplete_episode_count=counts["incomplete_count"],
            omitted_restricted_episode_count=omitted_restricted,
        )

    @property
    def summary(self) -> TraceSelectionSummary:
        """Return counts from the same fixed selection as the iterator."""
        return self._summary

    def iter_records(self) -> Iterator[TraceRecord]:
        """Decode selected records once in Episode and sequence order."""
        if self._closed:
            raise TraceStoreError("trace snapshot is closed")
        if self._iterated:
            raise TraceStoreError("trace snapshot records are single-pass")
        self._iterated = True
        try:
            cursor = self._connection.execute(
                """
                SELECT records.record_json
                FROM selected_trace_episodes selected
                JOIN trace_records records ON records.episode_id = selected.episode_id
                ORDER BY records.episode_id, records.sequence
                """
            )
            for row in cursor:
                yield _decode_json_record(row["record_json"])
        except sqlite3.Error as error:
            raise TraceStoreError("trace snapshot iteration failed") from error

    def __enter__(self) -> _SqliteTraceSnapshot:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, traceback
        if exc_value is None:
            self.close()
            return
        try:
            self.close()
        except TraceStoreError as error:
            exc_value.add_note(f"Windows trace privacy recheck failed after SQLite closed: {error}")
        except BaseException as error:
            exc_value.add_note(f"trace snapshot close failed: {error}")

    def close(self) -> None:
        """Release the fixed read transaction and connection."""
        if self._closed:
            return
        self._closed = True
        pending_error: BaseException | None = None
        try:
            self._connection.execute("ROLLBACK")
        except BaseException as error:
            pending_error = error
        try:
            self._connection.close()
        except BaseException as error:
            if pending_error is None:
                pending_error = error
            else:
                pending_error.add_note(f"trace snapshot connection close failed: {error}")
        if self._on_close is not None:
            try:
                self._on_close()
            except TraceStoreError as error:
                if pending_error is None:
                    raise
                pending_error.add_note(
                    f"Windows trace privacy recheck failed after SQLite closed: {error}"
                )
        if pending_error is not None:
            raise pending_error


def _selection_conditions(query: TraceQuery) -> tuple[list[str], list[object]]:
    conditions: list[str] = []
    parameters: list[object] = []
    if query.episode_ids is not None:
        placeholders = ",".join("?" for _ in query.episode_ids)
        conditions.append(f"e.episode_id IN ({placeholders})")
        parameters.extend(sorted(query.episode_ids))
    if query.started_after is not None:
        conditions.append("e.started_at >= ?")
        parameters.append(_timestamp(query.started_after))
    if query.completed_before is not None:
        conditions.append("e.completed_at IS NOT NULL AND e.completed_at <= ?")
        parameters.append(_timestamp(query.completed_before))
    if not query.include_incomplete:
        conditions.append("e.completed_at IS NOT NULL AND e.terminal_status <> 'incomplete'")
    if query.modalities is not None:
        matches = []
        for modality in sorted(item.value for item in query.modalities):
            matches.append("records.modality_keys LIKE ?")
            parameters.append(f"%,{modality},%")
        # This joins only fixed LIKE placeholders; values remain bound parameters.
        conditions.append(
            "EXISTS (SELECT 1 FROM trace_records records WHERE records.episode_id = e.episode_id "
            f"AND ({' OR '.join(matches)}))"  # nosec B608
        )
    if query.tool_ids is not None:
        placeholders = ",".join("?" for _ in query.tool_ids)
        # Only generated question-mark placeholders enter SQL; values remain bound.
        conditions.append(
            "EXISTS (SELECT 1 FROM trace_records records WHERE records.episode_id = e.episode_id "
            f"AND records.tool_id IN ({placeholders}))"  # nosec B608
        )
        parameters.extend(sorted(query.tool_ids))
    if query.tool_statuses is not None:
        values = sorted(item.value for item in query.tool_statuses)
        placeholders = ",".join("?" for _ in values)
        # Only generated question-mark placeholders enter SQL; values remain bound.
        conditions.append(
            "EXISTS (SELECT 1 FROM trace_records records WHERE records.episode_id = e.episode_id "
            f"AND records.tool_status IN ({placeholders}))"  # nosec B608
        )
        parameters.extend(values)
    if query.episode_statuses is not None:
        statuses = sorted(
            item.value for item in query.episode_statuses if item.value != "incomplete"
        )
        status_conditions: list[str] = []
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            status_conditions.append(f"e.terminal_status IN ({placeholders})")
            parameters.extend(statuses)
        if any(item.value == "incomplete" for item in query.episode_statuses):
            status_conditions.append("e.completed_at IS NULL OR e.terminal_status = 'incomplete'")
        conditions.append(f"({' OR '.join(status_conditions)})")
    return conditions, parameters


def _configure_connection(connection: sqlite3.Connection, *, writable: bool) -> None:
    connection.row_factory = sqlite3.Row
    if os.name == "nt":
        # TEMP_STORE=0 can override the pragma and spill snapshot data to disk.
        options = {str(row[0]) for row in connection.execute("PRAGMA compile_options")}
        if not options.intersection({"TEMP_STORE=1", "TEMP_STORE=2", "TEMP_STORE=3"}):
            raise TraceStoreError("Windows trace store requires SQLite in-memory temporary storage")
        connection.execute("PRAGMA temp_store = MEMORY")
        setting = connection.execute("PRAGMA temp_store").fetchone()
        if setting is None or setting[0] != 2:
            raise TraceStoreError("Windows trace store requires SQLite in-memory temporary storage")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    if writable:
        connection.execute("PRAGMA journal_mode = WAL")


def _decode_json_record(value: str) -> TraceRecord:
    try:
        decoded = cast(dict[str, Any], json.loads(value))
        return decode_trace_record(decoded)
    except (json.JSONDecodeError, UnicodeError, ValueError) as error:
        raise TraceStoreError("persisted trace record is invalid") from error


def _timestamp(value: datetime) -> str:
    if value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _idempotency_draft_sha256(draft: TraceRecordDraft) -> str:
    encoded = encode_trace_draft(draft)
    encoded.pop("occurred_at")
    return hashlib.sha256(canonical_json_bytes(encoded)).hexdigest()


def _idempotency_record_sha256(record: TraceRecord) -> str:
    return _idempotency_draft_sha256(
        TraceRecordDraft(
            episode_id=record.episode_id,
            trace_id=record.trace_id,
            occurred_at=record.occurred_at,
            payload=record.payload,
            modalities=record.modalities,
            step_id=record.step_id,
            request_id=record.request_id,
            parent_record_id=record.parent_record_id,
            media=record.media,
            lineage=record.lineage,
            warnings=record.warnings,
            error=record.error,
            privacy_class=record.privacy_class,
        )
    )


def _rollback(connection: sqlite3.Connection) -> None:
    try:
        connection.execute("ROLLBACK")
    except sqlite3.Error:
        pass
