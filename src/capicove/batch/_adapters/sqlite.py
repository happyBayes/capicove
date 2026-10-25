"""SQLite checkpoint and cache adapter for local batch execution."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from capicove.batch.cache import CacheConflictError, ResultCache
from capicove.batch.checkpoint import (
    BatchCheckpoint,
    CheckpointStore,
    CheckpointTransitionError,
    require_checkpoint_transition,
)
from capicove.batch.models import BatchFailure, BatchPlan, BatchState


_SCHEMA_VERSION = 1
_CACHE_KEY_PATTERN = re.compile(r"^cache:[0-9a-f]{64}$")


class SqliteBatchStoreError(RuntimeError):
    """Report an unavailable or incompatible SQLite batch store."""


class SqliteBatchStore(CheckpointStore, ResultCache):
    """Persist batch progress and JSON-compatible results in local SQLite."""

    def __init__(self, database_path: Path, *, busy_timeout_seconds: float = 5.0) -> None:
        if not database_path.is_absolute():
            raise ValueError("database_path must be absolute")
        if busy_timeout_seconds <= 0:
            raise ValueError("busy_timeout_seconds must be positive")
        self._database_path = database_path
        self._busy_timeout_ms = round(busy_timeout_seconds * 1000)
        try:
            database_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        except OSError as error:
            raise SqliteBatchStoreError("batch database directory is unavailable") from error
        self._initialize()

    @property
    def database_path(self) -> Path:
        """Return the configured local store path for operator-owned state."""
        return self._database_path

    def prepare(self, plan: BatchPlan) -> None:
        """Insert missing plan items and recover interrupted running items."""
        now = _timestamp()
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO batch_runs (
                    plan_id, operation_id, operation_version,
                    parameters_sha256, provider_fingerprint, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(plan_id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (
                    plan.plan_id,
                    plan.operation.operation_id,
                    plan.operation.operation_version,
                    plan.operation.parameters_sha256,
                    plan.operation.provider_fingerprint,
                    now,
                    now,
                ),
            )
            for planned in plan.items:
                item = planned.item
                inserted = connection.execute(
                    """
                    INSERT INTO batch_items (
                        plan_id, item_id, cache_key, source_sha256, modality,
                        relative_path, media_type, size_bytes, state, attempts,
                        failure_code, failure_message, failure_retryable, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, NULL, NULL, ?)
                    ON CONFLICT(plan_id, item_id) DO NOTHING
                    """,
                    (
                        plan.plan_id,
                        item.item_id,
                        planned.cache_key,
                        item.sha256,
                        item.modality.value,
                        item.relative_path.as_posix(),
                        item.media_type,
                        item.size_bytes,
                        BatchState.PENDING.value,
                        now,
                    ),
                )
                if inserted.rowcount == 1:
                    self._insert_event(
                        connection,
                        BatchCheckpoint(
                            plan_id=plan.plan_id,
                            item_id=item.item_id,
                            cache_key=planned.cache_key,
                            source_sha256=item.sha256,
                            state=BatchState.PENDING,
                        ),
                        now,
                    )

            interrupted = connection.execute(
                """
                SELECT * FROM batch_items
                WHERE plan_id = ? AND state = ?
                ORDER BY modality, relative_path, item_id
                """,
                (plan.plan_id, BatchState.RUNNING.value),
            ).fetchall()
            for row in interrupted:
                recovered = BatchCheckpoint(
                    plan_id=str(row["plan_id"]),
                    item_id=str(row["item_id"]),
                    cache_key=str(row["cache_key"]),
                    source_sha256=str(row["source_sha256"]),
                    state=BatchState.PENDING,
                    attempts=int(row["attempts"]),
                )
                connection.execute(
                    """
                    UPDATE batch_items
                    SET state = ?, failure_code = NULL, failure_message = NULL,
                        failure_retryable = NULL, updated_at = ?
                    WHERE plan_id = ? AND item_id = ? AND state = ?
                    """,
                    (
                        recovered.state.value,
                        now,
                        recovered.plan_id,
                        recovered.item_id,
                        BatchState.RUNNING.value,
                    ),
                )
                self._insert_event(connection, recovered, now)

    def load(self, plan_id: str, item_id: str) -> BatchCheckpoint:
        """Load one prepared checkpoint."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM batch_items WHERE plan_id = ? AND item_id = ?",
                (plan_id, item_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"checkpoint is unavailable: {item_id}")
        return _checkpoint_from_row(row)

    def update(
        self,
        plan_id: str,
        item_id: str,
        *,
        expected_states: frozenset[BatchState],
        state: BatchState,
        attempts: int,
        failure: BatchFailure | None = None,
    ) -> BatchCheckpoint:
        """Apply one state transition under an immediate transaction."""
        if not expected_states:
            raise ValueError("expected_states must not be empty")
        now = _timestamp()
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM batch_items WHERE plan_id = ? AND item_id = ?",
                (plan_id, item_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"checkpoint is unavailable: {item_id}")
            current = _checkpoint_from_row(row)
            if current.state not in expected_states:
                expected = ", ".join(sorted(value.value for value in expected_states))
                raise CheckpointTransitionError(
                    f"checkpoint state {current.state.value} did not match expected {expected}"
                )
            require_checkpoint_transition(current.state, state)
            checkpoint = BatchCheckpoint(
                plan_id=plan_id,
                item_id=item_id,
                cache_key=current.cache_key,
                source_sha256=current.source_sha256,
                state=state,
                attempts=attempts,
                failure=failure,
            )
            failure_values = _failure_values(failure)
            updated = connection.execute(
                """
                UPDATE batch_items
                SET state = ?, attempts = ?, failure_code = ?, failure_message = ?,
                    failure_retryable = ?, updated_at = ?
                WHERE plan_id = ? AND item_id = ? AND state = ?
                """,
                (
                    checkpoint.state.value,
                    checkpoint.attempts,
                    *failure_values,
                    now,
                    plan_id,
                    item_id,
                    current.state.value,
                ),
            )
            if updated.rowcount != 1:
                raise CheckpointTransitionError("checkpoint changed concurrently")
            self._insert_event(connection, checkpoint, now)
        return checkpoint

    def list(self, plan_id: str) -> tuple[BatchCheckpoint, ...]:
        """Return checkpoints in deterministic plan order."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM batch_items
                WHERE plan_id = ?
                ORDER BY modality, relative_path, item_id
                """,
                (plan_id,),
            ).fetchall()
        return tuple(_checkpoint_from_row(row) for row in rows)

    def history(self, plan_id: str, item_id: str) -> tuple[BatchCheckpoint, ...]:
        """Return append-only checkpoint observations."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM batch_events
                WHERE plan_id = ? AND item_id = ?
                ORDER BY event_id
                """,
                (plan_id, item_id),
            ).fetchall()
        return tuple(_checkpoint_from_row(row) for row in rows)

    def get(self, cache_key: str) -> Mapping[str, object] | None:
        """Return a decoded cached result without mutating cache state."""
        _require_cache_key(cache_key)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT result_json FROM batch_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        decoded = json.loads(str(row["result_json"]))
        if not isinstance(decoded, dict):
            raise SqliteBatchStoreError("cached result is not a JSON object")
        return decoded

    def put(self, cache_key: str, result: Mapping[str, object]) -> None:
        """Store one canonical JSON result without overwriting a conflict."""
        _require_cache_key(cache_key)
        try:
            encoded = json.dumps(
                dict(result),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("cached result must be JSON-compatible") from error
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT result_json FROM batch_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if existing is not None:
                if str(existing["result_json"]) != encoded:
                    raise CacheConflictError("cache key already contains a different result")
                return
            connection.execute(
                "INSERT INTO batch_cache (cache_key, result_json, created_at) VALUES (?, ?, ?)",
                (cache_key, encoded, _timestamp()),
            )

    def evict(self, cache_key: str) -> None:
        """Remove one cached result after operation-specific validation fails."""
        _require_cache_key(cache_key)
        with self._transaction() as connection:
            connection.execute(
                "DELETE FROM batch_cache WHERE cache_key = ?",
                (cache_key,),
            )

    def _initialize(self) -> None:
        try:
            with self._connect() as connection:
                version = int(connection.execute("PRAGMA user_version").fetchone()[0])
                if version not in {0, _SCHEMA_VERSION}:
                    raise SqliteBatchStoreError(
                        f"unsupported batch database schema version: {version}"
                    )
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS batch_runs (
                        plan_id TEXT PRIMARY KEY,
                        operation_id TEXT NOT NULL,
                        operation_version TEXT NOT NULL,
                        parameters_sha256 TEXT NOT NULL,
                        provider_fingerprint TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS batch_items (
                        plan_id TEXT NOT NULL,
                        item_id TEXT NOT NULL,
                        cache_key TEXT NOT NULL,
                        source_sha256 TEXT NOT NULL,
                        modality TEXT NOT NULL,
                        relative_path TEXT NOT NULL,
                        media_type TEXT NOT NULL,
                        size_bytes INTEGER NOT NULL,
                        state TEXT NOT NULL,
                        attempts INTEGER NOT NULL,
                        failure_code TEXT,
                        failure_message TEXT,
                        failure_retryable INTEGER,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (plan_id, item_id),
                        FOREIGN KEY (plan_id) REFERENCES batch_runs(plan_id)
                    );
                    CREATE TABLE IF NOT EXISTS batch_events (
                        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        plan_id TEXT NOT NULL,
                        item_id TEXT NOT NULL,
                        cache_key TEXT NOT NULL,
                        source_sha256 TEXT NOT NULL,
                        state TEXT NOT NULL,
                        attempts INTEGER NOT NULL,
                        failure_code TEXT,
                        failure_message TEXT,
                        failure_retryable INTEGER,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (plan_id, item_id)
                            REFERENCES batch_items(plan_id, item_id)
                    );
                    CREATE TABLE IF NOT EXISTS batch_cache (
                        cache_key TEXT PRIMARY KEY,
                        result_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    """
                )
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        except sqlite3.Error as error:
            raise SqliteBatchStoreError("batch database initialization failed") from error

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._database_path,
            timeout=self._busy_timeout_ms / 1000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection,
        checkpoint: BatchCheckpoint,
        timestamp: str,
    ) -> None:
        failure_values = _failure_values(checkpoint.failure)
        connection.execute(
            """
            INSERT INTO batch_events (
                plan_id, item_id, cache_key, source_sha256, state, attempts,
                failure_code, failure_message, failure_retryable, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                checkpoint.plan_id,
                checkpoint.item_id,
                checkpoint.cache_key,
                checkpoint.source_sha256,
                checkpoint.state.value,
                checkpoint.attempts,
                *failure_values,
                timestamp,
            ),
        )


def _checkpoint_from_row(row: sqlite3.Row) -> BatchCheckpoint:
    state = BatchState(str(row["state"]))
    failure = None
    if row["failure_code"] is not None:
        failure = BatchFailure(
            code=str(row["failure_code"]),
            message=str(row["failure_message"]),
            retryable=bool(row["failure_retryable"]),
        )
    return BatchCheckpoint(
        plan_id=str(row["plan_id"]),
        item_id=str(row["item_id"]),
        cache_key=str(row["cache_key"]),
        source_sha256=str(row["source_sha256"]),
        state=state,
        attempts=int(row["attempts"]),
        failure=failure,
    )


def _failure_values(failure: BatchFailure | None) -> tuple[str | None, str | None, int | None]:
    if failure is None:
        return None, None, None
    return failure.code, failure.message, int(failure.retryable)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _require_cache_key(cache_key: str) -> None:
    if _CACHE_KEY_PATTERN.fullmatch(cache_key) is None:
        raise ValueError("cache_key must be a cache-prefixed lowercase SHA-256 hash")
