"""Private append-only SQLite processing metadata store."""

from __future__ import annotations

import json
import sqlite3
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from capicove.contracts import schema_value_type_from_json, schema_value_type_to_json
from capicove.contracts.data import (
    DataFormat,
    DatasetSchema,
    DatasetSnapshotRef,
    LineageEdge,
    SchemaField,
)
from capicove.processing.models import ProcessingRun, RunState


_SCHEMA = """
CREATE TABLE IF NOT EXISTS dataset_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    snapshot_json TEXT NOT NULL,
    lineage_json TEXT NOT NULL,
    UNIQUE (dataset_id, version)
);
CREATE INDEX IF NOT EXISTS dataset_snapshots_latest
    ON dataset_snapshots (dataset_id, version DESC);

CREATE TABLE IF NOT EXISTS processing_runs (
    run_id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL,
    backend_id TEXT NOT NULL,
    state TEXT NOT NULL,
    run_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS processing_runs_state ON processing_runs (state);
"""


class ProcessingStoreError(RuntimeError):
    """Report bounded local processing-store failures."""


class SnapshotConflictError(ProcessingStoreError):
    """Report reuse of snapshot identity for different content."""


class RunConflictError(ProcessingStoreError):
    """Report reuse of run identity for different content."""


class RunTransitionError(ProcessingStoreError):
    """Report a stale or illegal run-state transition."""


class LineageConflictError(ProcessingStoreError):
    """Report lineage that disagrees with its snapshot or processing run."""


class SqliteProcessingStore:
    """Persist immutable snapshots and processing runs in one SQLite file."""

    def __init__(self, database_path: Path) -> None:
        if not isinstance(database_path, Path) or not database_path.is_absolute():
            raise ValueError("database_path must be an absolute Path")
        self._database_path = database_path
        try:
            _reject_symlink_components(database_path)
            database_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            _reject_symlink_components(database_path)
        except OSError as error:
            raise ProcessingStoreError("processing store path initialization failed") from error
        try:
            with self._connect() as connection:
                version = cast(int, connection.execute("PRAGMA user_version").fetchone()[0])
                if version not in (0, 1):
                    raise ProcessingStoreError("processing store schema version is unsupported")
                connection.executescript(_SCHEMA)
                connection.execute("PRAGMA user_version = 1")
        except sqlite3.Error as error:
            raise ProcessingStoreError("processing store initialization failed") from error

    def get(self, snapshot_id: str, /) -> DatasetSnapshotRef | None:
        """Return one immutable snapshot by ID."""
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT snapshot_json FROM dataset_snapshots WHERE snapshot_id = ?",
                    (snapshot_id,),
                ).fetchone()
        except sqlite3.Error as error:
            raise ProcessingStoreError("snapshot read failed") from error
        return None if row is None else _decode_snapshot(row["snapshot_json"])

    def latest(self, dataset_id: str, /) -> DatasetSnapshotRef | None:
        """Return the highest committed dataset version."""
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT snapshot_json FROM dataset_snapshots
                    WHERE dataset_id = ? ORDER BY version DESC LIMIT 1
                    """,
                    (dataset_id,),
                ).fetchone()
        except sqlite3.Error as error:
            raise ProcessingStoreError("latest snapshot read failed") from error
        return None if row is None else _decode_snapshot(row["snapshot_json"])

    def commit(
        self,
        snapshot: DatasetSnapshotRef,
        lineage: tuple[LineageEdge, ...],
        /,
    ) -> None:
        """Append one snapshot and lineage idempotently."""
        snapshot_json = _snapshot_json(snapshot)
        lineage_json = _lineage_json(tuple(lineage))
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            _validate_lineage(snapshot, lineage)
            existing = connection.execute(
                "SELECT snapshot_json, lineage_json FROM dataset_snapshots WHERE snapshot_id = ?",
                (snapshot.snapshot_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["snapshot_json"] != snapshot_json
                    or existing["lineage_json"] != lineage_json
                ):
                    raise SnapshotConflictError("snapshot ID identifies different content")
                connection.execute("COMMIT")
                return
            connection.execute(
                """
                INSERT INTO dataset_snapshots (
                    snapshot_id, dataset_id, version, snapshot_json, lineage_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    snapshot.snapshot_id,
                    snapshot.dataset_id,
                    snapshot.version,
                    snapshot_json,
                    lineage_json,
                ),
            )
            connection.execute("COMMIT")
        except (LineageConflictError, SnapshotConflictError):
            _rollback(connection)
            raise
        except sqlite3.IntegrityError as error:
            _rollback(connection)
            raise SnapshotConflictError("dataset version is already committed") from error
        except sqlite3.Error as error:
            _rollback(connection)
            raise ProcessingStoreError("snapshot commit failed") from error
        finally:
            connection.close()

    def create(self, run: ProcessingRun, /) -> None:
        """Create one processing run idempotently."""
        run_json = _run_json(run)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT run_json FROM processing_runs WHERE run_id = ?",
                (run.run_id,),
            ).fetchone()
            if existing is not None:
                if existing["run_json"] != run_json:
                    raise RunConflictError("run ID identifies different content")
                connection.execute("COMMIT")
                return
            connection.execute(
                """
                INSERT INTO processing_runs (run_id, plan_id, backend_id, state, run_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run.run_id, run.plan_id, run.backend_id, run.state.value, run_json),
            )
            connection.execute("COMMIT")
        except RunConflictError:
            _rollback(connection)
            raise
        except sqlite3.Error as error:
            _rollback(connection)
            raise ProcessingStoreError("run creation failed") from error
        finally:
            connection.close()

    def commit_success(
        self,
        run_id: str,
        snapshot: DatasetSnapshotRef,
        lineage: tuple[LineageEdge, ...],
        /,
        *,
        native_run_id: str,
    ) -> ProcessingRun:
        """Atomically append a snapshot and transition its running run."""
        snapshot_json = _snapshot_json(snapshot)
        lineage_json = _lineage_json(tuple(lineage))
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            _validate_lineage(snapshot, lineage, run_id=run_id)
            existing_snapshot = connection.execute(
                "SELECT snapshot_json, lineage_json FROM dataset_snapshots WHERE snapshot_id = ?",
                (snapshot.snapshot_id,),
            ).fetchone()
            if existing_snapshot is None:
                connection.execute(
                    """
                    INSERT INTO dataset_snapshots (
                        snapshot_id, dataset_id, version, snapshot_json, lineage_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot.snapshot_id,
                        snapshot.dataset_id,
                        snapshot.version,
                        snapshot_json,
                        lineage_json,
                    ),
                )
            elif (
                existing_snapshot["snapshot_json"] != snapshot_json
                or existing_snapshot["lineage_json"] != lineage_json
            ):
                raise SnapshotConflictError("snapshot ID identifies different content")
            row = connection.execute(
                "SELECT run_json FROM processing_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise RunTransitionError("processing run does not exist")
            current = _decode_run(row["run_json"])
            if current.state is not RunState.RUNNING:
                raise RunTransitionError("processing run state changed concurrently")
            updated = ProcessingRun(
                run_id=current.run_id,
                plan_id=current.plan_id,
                backend_id=current.backend_id,
                state=RunState.SUCCEEDED,
                created_at=current.created_at,
                updated_at=max(datetime.now(UTC), current.updated_at),
                native_run_id=native_run_id,
                output_snapshot_id=snapshot.snapshot_id,
            )
            connection.execute(
                "UPDATE processing_runs SET state = ?, run_json = ? WHERE run_id = ?",
                (updated.state.value, _run_json(updated), run_id),
            )
            connection.execute("COMMIT")
            return updated
        except (LineageConflictError, RunTransitionError, SnapshotConflictError):
            _rollback(connection)
            raise
        except sqlite3.IntegrityError as error:
            _rollback(connection)
            raise SnapshotConflictError("dataset version is already committed") from error
        except sqlite3.Error as error:
            _rollback(connection)
            raise ProcessingStoreError("successful run commit failed") from error
        finally:
            connection.close()

    def get_run(self, run_id: str, /) -> ProcessingRun | None:
        """Return one processing run by ID."""
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT run_json FROM processing_runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
        except sqlite3.Error as error:
            raise ProcessingStoreError("run read failed") from error
        return None if row is None else _decode_run(row["run_json"])

    def transition(
        self,
        run_id: str,
        expected: RunState,
        target: RunState,
        /,
        *,
        native_run_id: str | None = None,
        output_snapshot_id: str | None = None,
        detail: str | None = None,
    ) -> ProcessingRun:
        """Compare-and-set one legal processing-run transition."""
        legal = {
            RunState.SUBMITTED: frozenset({RunState.RUNNING, RunState.FAILED, RunState.CANCELLED}),
            RunState.RUNNING: frozenset({RunState.SUCCEEDED, RunState.FAILED, RunState.CANCELLED}),
        }
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT run_json FROM processing_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise RunTransitionError("processing run does not exist")
            current = _decode_run(row["run_json"])
            if current.state is not expected:
                raise RunTransitionError("processing run state changed concurrently")
            if target not in legal.get(current.state, frozenset()):
                raise RunTransitionError("processing run transition is illegal")
            updated = ProcessingRun(
                run_id=current.run_id,
                plan_id=current.plan_id,
                backend_id=current.backend_id,
                state=target,
                created_at=current.created_at,
                updated_at=max(datetime.now(UTC), current.updated_at),
                native_run_id=native_run_id or current.native_run_id,
                output_snapshot_id=output_snapshot_id,
                detail=detail,
            )
            connection.execute(
                "UPDATE processing_runs SET state = ?, run_json = ? WHERE run_id = ?",
                (updated.state.value, _run_json(updated), run_id),
            )
            connection.execute("COMMIT")
            return updated
        except RunTransitionError:
            _rollback(connection)
            raise
        except sqlite3.Error as error:
            _rollback(connection)
            raise ProcessingStoreError("run transition failed") from error
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        _reject_symlink_components(self._database_path)
        uri = self._database_path.as_uri() + "?nofollow=1"
        connection = sqlite3.connect(uri, timeout=5.0, uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection


def _reject_symlink_components(path: Path) -> None:
    """Reject every existing symbolic-link component in an absolute path."""
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ProcessingStoreError("processing store path contains a symbolic link")
        if current != path and not stat.S_ISDIR(mode):
            raise ProcessingStoreError("processing store parent is not a directory")


def _snapshot_json(snapshot: DatasetSnapshotRef) -> str:
    value = {
        "data_format": snapshot.data_format.value,
        "dataset_id": snapshot.dataset_id,
        "manifest_sha256": snapshot.manifest_sha256,
        "parent_snapshot_ids": list(snapshot.parent_snapshot_ids),
        "processing_fingerprint": snapshot.processing_fingerprint,
        "schema": {
            "fields": [
                {
                    "logical_type": schema_value_type_to_json(field.logical_type),
                    "name": field.name,
                    "nullable": field.nullable,
                }
                for field in snapshot.schema.fields
            ],
            "schema_version": snapshot.schema.schema_version,
        },
        "snapshot_id": snapshot.snapshot_id,
        "uri": snapshot.uri,
        "version": snapshot.version,
    }
    return _canonical_json(value)


def _decode_snapshot(payload: str) -> DatasetSnapshotRef:
    try:
        value = _require_closed_object(
            json.loads(payload, object_pairs_hook=_reject_duplicate_members),
            {
                "data_format",
                "dataset_id",
                "manifest_sha256",
                "parent_snapshot_ids",
                "processing_fingerprint",
                "schema",
                "snapshot_id",
                "uri",
                "version",
            },
        )
        schema_value = _require_closed_object(value["schema"], {"fields", "schema_version"})
        fields = _require_list(schema_value["fields"])
        schema = DatasetSchema(
            tuple(_decode_schema_field(field) for field in fields),
            schema_version=_require_integer(schema_value["schema_version"]),
        )
        parents = _require_list(value["parent_snapshot_ids"])
        return DatasetSnapshotRef(
            dataset_id=_require_string(value["dataset_id"]),
            version=_require_integer(value["version"]),
            snapshot_id=_require_string(value["snapshot_id"]),
            uri=_require_string(value["uri"]),
            data_format=DataFormat(_require_string(value["data_format"])),
            schema=schema,
            manifest_sha256=_require_string(value["manifest_sha256"]),
            processing_fingerprint=_require_string(value["processing_fingerprint"]),
            parent_snapshot_ids=tuple(_require_string(parent) for parent in parents),
        )
    except (json.JSONDecodeError, KeyError, RecursionError, TypeError, ValueError) as error:
        raise ProcessingStoreError("snapshot metadata is invalid") from error


def _reject_duplicate_members(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build one decoded JSON object while rejecting duplicate member names."""
    value: dict[str, object] = {}
    for key, member in pairs:
        if key in value:
            raise ValueError("snapshot object contains duplicate members")
        value[key] = member
    return value


def _require_closed_object(value: object, expected_keys: set[str]) -> dict[str, object]:
    """Return a JSON object with exactly the declared persistent fields."""
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("persisted object has unsupported fields")
    return cast(dict[str, object], value)


def _require_list(value: object) -> list[object]:
    """Return one JSON array value."""
    if not isinstance(value, list):
        raise ValueError("persisted value must be a list")
    return value


def _require_string(value: object) -> str:
    """Return one JSON string value."""
    if not isinstance(value, str):
        raise ValueError("persisted value must be a string")
    return value


def _require_integer(value: object) -> int:
    """Return one strict JSON integer value."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("persisted value must be an integer")
    return value


def _decode_schema_field(value: object) -> SchemaField:
    """Decode one closed persisted canonical schema field."""
    field = _require_closed_object(value, {"logical_type", "name", "nullable"})
    nullable = field["nullable"]
    if not isinstance(nullable, bool):
        raise ValueError("nullable must be a boolean")
    return SchemaField(
        _require_string(field["name"]),
        schema_value_type_from_json(field["logical_type"]),
        nullable,
    )


def _lineage_json(lineage: tuple[LineageEdge, ...]) -> str:
    value = [
        {
            "input_snapshot_id": edge.input_snapshot_id,
            "operator_ids": list(edge.operator_ids),
            "output_snapshot_id": edge.output_snapshot_id,
            "run_id": edge.run_id,
        }
        for edge in lineage
    ]
    return _canonical_json(value)


def _validate_lineage(
    snapshot: DatasetSnapshotRef,
    lineage: tuple[LineageEdge, ...],
    *,
    run_id: str | None = None,
) -> None:
    """Require exact, unique parent edges for one output and optional run."""
    inputs = tuple(edge.input_snapshot_id for edge in lineage)
    if len(inputs) != len(set(inputs)):
        raise LineageConflictError("lineage contains duplicate parent edges")
    if set(inputs) != set(snapshot.parent_snapshot_ids):
        raise LineageConflictError("lineage parent set does not match the snapshot")
    if any(edge.output_snapshot_id != snapshot.snapshot_id for edge in lineage):
        raise LineageConflictError("lineage output does not match the snapshot")
    observed_run_ids = {edge.run_id for edge in lineage}
    if run_id is not None:
        if any(edge.run_id != run_id for edge in lineage):
            raise LineageConflictError("lineage run does not match the processing run")
    elif len(observed_run_ids) > 1:
        raise LineageConflictError("lineage edges must identify one processing run")


def _run_json(run: ProcessingRun) -> str:
    return _canonical_json(
        {
            "backend_id": run.backend_id,
            "created_at": run.created_at.isoformat(),
            "detail": run.detail,
            "native_run_id": run.native_run_id,
            "output_snapshot_id": run.output_snapshot_id,
            "plan_id": run.plan_id,
            "run_id": run.run_id,
            "state": run.state.value,
            "updated_at": run.updated_at.isoformat(),
        }
    )


def _decode_run(payload: str) -> ProcessingRun:
    value = json.loads(payload)
    return ProcessingRun(
        run_id=value["run_id"],
        plan_id=value["plan_id"],
        backend_id=value["backend_id"],
        state=RunState(value["state"]),
        created_at=datetime.fromisoformat(value["created_at"]),
        updated_at=datetime.fromisoformat(value["updated_at"]),
        native_run_id=value["native_run_id"],
        output_snapshot_id=value["output_snapshot_id"],
        detail=value["detail"],
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _rollback(connection: sqlite3.Connection) -> None:
    try:
        connection.execute("ROLLBACK")
    except sqlite3.Error:
        pass
