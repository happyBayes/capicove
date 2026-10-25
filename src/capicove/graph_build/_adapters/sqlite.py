"""Durable compound-scope receipt claims; never an exactly-once provider guarantee."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from capicove.contracts.graph_build import GraphScope
from capicove.graph_build._adapters.artifacts import (
    ArtifactStoreError,
    _private_directory,
    _private_stat,
)
from capicove.graph_build.codec import canonical_bytes
from capicove.graph_build.ports import AccessGate, ProtectedArtifacts
from capicove.graph_build.receipts import (
    InferenceReceipt,
    ReceiptConflictError,
    ReceiptStoreError,
    require_content_id,
)


_SCHEMA = """
CREATE TABLE inference_units (
    workspace_ref TEXT NOT NULL,
    scope_ref TEXT NOT NULL,
    unit_key TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('claimed', 'succeeded', 'failed', 'cancelled')),
    owner_attempt TEXT NOT NULL,
    lease_until REAL,
    attempts_json TEXT NOT NULL,
    receipt_json TEXT,
    receipt_sha256 TEXT,
    PRIMARY KEY (workspace_ref, scope_ref, unit_key),
    CHECK ((status = 'claimed' AND lease_until IS NOT NULL AND receipt_json IS NULL
            AND receipt_sha256 IS NULL) OR
           (status <> 'claimed' AND lease_until IS NULL AND receipt_json IS NOT NULL
            AND receipt_sha256 IS NOT NULL))
)
"""
_COLUMNS = (
    "workspace_ref",
    "scope_ref",
    "unit_key",
    "status",
    "owner_attempt",
    "lease_until",
    "attempts_json",
    "receipt_json",
    "receipt_sha256",
)


def _stat_database_file(directory: int, name: str, *, sidecar: bool) -> os.stat_result | None:
    """Recheck one unlinked sidecar without relaxing live-file safeguards.

    A concurrent SQLite commit can unlink a journal between path lookup and
    metadata collection. Zero links describes that detached inode, not a hard
    link. Recheck its name once: absence is valid only for optional sidecars,
    and any replacement still needs owner/type/mode and single-link validation.
    Repeated churn fails closed rather than retrying indefinitely.
    """
    try:
        info = os.stat(name, dir_fd=directory, follow_symlinks=False)
        _private_stat(info)
        if sidecar and info.st_nlink == 0:
            info = os.stat(name, dir_fd=directory, follow_symlinks=False)
            _private_stat(info)
        return info
    except FileNotFoundError:
        if sidecar:
            return None
        raise


class SqliteReceiptStore:
    """Serialize durable state transitions using BEGIN IMMEDIATE per operation.

    Leases use Unix seconds, so restarts and independent workers share one time
    domain. The host must provide a trustworthy clock. Every accepted attempt ID
    is retained to prevent fencing-token reuse, even following failed completion.
    The explicit database parent must be owner-only and free of symlink ancestors.
    SQLite itself opens paths: that host-owned directory must not be concurrently
    replaced by another process running as the same OS user.
    """

    def __init__(
        self,
        database_path: Path,
        gate: AccessGate,
        artifacts: ProtectedArtifacts,
        *,
        clock: Callable[[], float] = time.time,
        busy_timeout_seconds: float = 5.0,
    ) -> None:
        if (
            not isinstance(database_path, Path)
            or not database_path.is_absolute()
            or ".." in database_path.parts
        ):
            raise ValueError("database_path must be an absolute Path without traversal")
        _finite(busy_timeout_seconds)
        if busy_timeout_seconds <= 0:
            raise ValueError("busy timeout must be positive")
        self._path = database_path
        self._gate = gate
        self._artifacts = artifacts
        self._clock = clock
        self._timeout = busy_timeout_seconds
        try:
            with _private_directory(database_path.parent, create=True) as directory:
                try:
                    descriptor = os.open(
                        database_path.name,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600,
                        dir_fd=directory,
                    )
                except FileExistsError:
                    pass
                else:
                    os.close(descriptor)
                    os.fsync(directory)
        except (ArtifactStoreError, OSError) as error:
            raise ReceiptStoreError("receipt database path is unsafe or unavailable") from error
        with self._transaction(initialize=True):
            pass

    @contextmanager
    def _transaction(self, *, initialize: bool = False) -> Iterator[sqlite3.Connection]:
        connection = None
        ready = False
        try:
            with _private_directory(self._path.parent) as directory:
                self._verify_files(directory)
                before = os.stat(self._path.name, dir_fd=directory, follow_symlinks=False)
                connection = sqlite3.connect(
                    self._path.as_uri() + "?mode=rw",
                    uri=True,
                    timeout=self._timeout,
                    isolation_level=None,
                )
                connection.row_factory = sqlite3.Row
                after = os.stat(self._path.name, dir_fd=directory, follow_symlinks=False)
                if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                    raise ReceiptStoreError("receipt database changed during open")
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute("BEGIN IMMEDIATE")
                self._schema(connection, initialize=initialize)
                ready = True
                yield connection
                connection.commit()
        except PermissionError:
            raise
        except (sqlite3.Error, OSError, ArtifactStoreError) as error:
            if ready and isinstance(error, ArtifactStoreError):
                raise
            raise ReceiptStoreError(
                "receipt database is unavailable, unsafe, or corrupt"
            ) from error
        finally:
            if connection is not None:
                if connection.in_transaction:
                    connection.rollback()
                connection.close()

    def _verify_files(self, directory: int) -> None:
        for suffix in ("", "-journal", "-wal", "-shm"):
            info = _stat_database_file(directory, self._path.name + suffix, sidecar=bool(suffix))
            if info is None:
                continue
            if info.st_nlink != 1:
                raise ReceiptStoreError("receipt database files must not be hard-linked")

    @staticmethod
    def _schema(connection: sqlite3.Connection, *, initialize: bool) -> None:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        objects = connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master"
        ).fetchall()
        if version == 0 and not objects and initialize:
            connection.execute(_SCHEMA)
            connection.execute("PRAGMA user_version=1")
            objects = connection.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master"
            ).fetchall()
        elif version != 1:
            raise ReceiptStoreError("unsupported receipt schema")
        # This private version supports only its exact table definition and the
        # implicit compound-PK index. Reject extra triggers, indexes and views,
        # plus changes to constraints/types that column-name checks cannot detect.
        expected_objects = {
            ("table", "inference_units", "inference_units", _SCHEMA.strip()),
            ("index", "sqlite_autoindex_inference_units_1", "inference_units", None),
        }
        if {tuple(row) for row in objects} != expected_objects:
            raise ReceiptStoreError("receipt schema contains unsupported definitions or objects")
        columns = connection.execute("PRAGMA table_info(inference_units)").fetchall()
        if tuple(row[1] for row in columns) != _COLUMNS or tuple(row[5] for row in columns) != (
            1,
            2,
            3,
            0,
            0,
            0,
            0,
            0,
            0,
        ):
            raise ReceiptStoreError("receipt schema is corrupt")

    def _authorize(self, scope: GraphScope, key: str) -> tuple[str, str, str]:
        self._gate.require_scope(scope)
        require_content_id(key, "receipt")
        return scope.workspace_ref, scope.scope_ref, key

    def _now(self) -> float:
        now = self._clock()
        _finite(now)
        return now

    def claim(self, scope: GraphScope, key: str, attempt_id: str, lease_until: float) -> bool:
        """Atomically claim a unit; a reused attempt ID can never acquire a lease."""
        identity = self._authorize(scope, key)
        _attempt(attempt_id)
        _finite(lease_until)
        with self._transaction() as connection:
            now = self._now()
            if lease_until <= now:
                raise ValueError("lease must expire in the future")
            row = self._row(connection, identity)
            attempts: list[str] = []
            if row is not None:
                attempts = _attempts(row)
                receipt = self._receipt(scope, key, row)
                if receipt is not None and receipt.status == "succeeded":
                    return False
                if attempt_id in attempts or (
                    row["lease_until"] is not None and row["lease_until"] > now
                ):
                    return False
            if lease_until <= self._now():
                raise ValueError("lease expired during authorization")
            attempts.append(attempt_id)
            connection.execute(
                """INSERT INTO inference_units VALUES (?, ?, ?, 'claimed', ?, ?, ?, NULL, NULL)
                ON CONFLICT(workspace_ref, scope_ref, unit_key) DO UPDATE SET
                    status='claimed', owner_attempt=excluded.owner_attempt,
                    lease_until=excluded.lease_until, attempts_json=excluded.attempts_json,
                    receipt_json=NULL, receipt_sha256=NULL""",
                (*identity, attempt_id, lease_until, canonical_bytes(attempts).decode()),
            )
            return True

    def complete(
        self,
        scope: GraphScope,
        key: str,
        attempt_id: str,
        receipt: InferenceReceipt,
        *,
        success_guard: Callable[[], None] | None = None,
    ) -> None:
        """Accept only a current owner and a verified durable artifact."""
        identity = self._authorize(scope, key)
        _attempt(attempt_id)
        if not isinstance(receipt, InferenceReceipt) or receipt.key != key:
            raise ReceiptConflictError("receipt key does not match claim")
        with self._transaction() as connection:
            row = self._row(connection, identity)
            if row is None or row["owner_attempt"] != attempt_id:
                raise ReceiptConflictError("attempt does not own this claim")
            existing = self._receipt(scope, key, row)
            if existing is not None:
                if existing == receipt:
                    return
                raise ReceiptConflictError("completion conflicts with accepted receipt")
            self._verify_artifact(scope, receipt)
            if row["lease_until"] <= self._now():
                raise ReceiptConflictError("attempt lease has expired")
            encoded = canonical_bytes({"schema_version": 1, "receipt": asdict(receipt)})
            parameters = (
                receipt.status,
                encoded.decode(),
                hashlib.sha256(encoded).hexdigest(),
                *identity,
            )
            if receipt.status == "succeeded" and success_guard is not None:
                success_guard()
            connection.execute(
                """UPDATE inference_units SET status=?, lease_until=NULL,
                    receipt_json=?, receipt_sha256=?
                WHERE workspace_ref=? AND scope_ref=? AND unit_key=?""",
                parameters,
            )

    def get(self, scope: GraphScope, key: str) -> InferenceReceipt | None:
        """Read terminal metadata only after checking current artifact access/digest."""
        identity = self._authorize(scope, key)
        with self._transaction() as connection:
            row = self._row(connection, identity)
            return None if row is None else self._receipt(scope, key, row)

    @staticmethod
    def _row(connection: sqlite3.Connection, identity: tuple[str, str, str]) -> sqlite3.Row | None:
        row: sqlite3.Row | None = connection.execute(
            "SELECT * FROM inference_units WHERE workspace_ref=? AND scope_ref=? AND unit_key=?",
            identity,
        ).fetchone()
        return row

    def _receipt(self, scope: GraphScope, key: str, row: sqlite3.Row) -> InferenceReceipt | None:
        try:
            _attempts(row)
            if row["status"] == "claimed":
                _finite(row["lease_until"])
                if row["receipt_json"] is not None or row["receipt_sha256"] is not None:
                    raise ValueError("claim has terminal payload")
                return None
            if row["lease_until"] is not None:
                raise ValueError("terminal receipt has lease")
            encoded = row["receipt_json"].encode("utf-8")
            if hashlib.sha256(encoded).hexdigest() != row["receipt_sha256"]:
                raise ValueError("receipt digest mismatch")
            value = json.loads(encoded)
            if (
                not isinstance(value, dict)
                or set(value) != {"schema_version", "receipt"}
                or type(value["schema_version"]) is not int
                or value["schema_version"] != 1
            ):
                raise ValueError("invalid receipt envelope")
            if canonical_bytes(value) != encoded:
                raise ValueError("noncanonical receipt")
            receipt = InferenceReceipt(**value["receipt"])
            if receipt.key != key or receipt.status != row["status"]:
                raise ValueError("receipt identity mismatch")
        except (ValueError, TypeError, KeyError, AttributeError, RecursionError) as error:
            raise ReceiptStoreError("stored receipt is corrupt") from error
        self._verify_artifact(scope, receipt)
        return receipt

    def _verify_artifact(self, scope: GraphScope, receipt: InferenceReceipt) -> None:
        payload = self._artifacts.read(scope, receipt.result_artifact_ref)
        if hashlib.sha256(payload).hexdigest() != receipt.result_digest:
            raise ReceiptStoreError("receipt result digest mismatch")


def _finite(value: float) -> None:
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError("time values must be finite numbers")


def _attempt(value: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError("attempt ID must be a nonblank bounded string")


def _attempts(row: sqlite3.Row) -> list[str]:
    try:
        attempts = json.loads(row["attempts_json"])
        if not isinstance(attempts, list) or not attempts:
            raise ValueError("invalid attempt history")
        for attempt in attempts:
            _attempt(attempt)
        if len(set(attempts)) != len(attempts) or row["owner_attempt"] != attempts[-1]:
            raise ValueError("invalid claim owner")
        return attempts
    except (ValueError, TypeError, KeyError, RecursionError) as error:
        raise ReceiptStoreError("stored claim is corrupt") from error


_GRAPH_SCHEMA = (
    """CREATE TABLE graph_snapshots (
        workspace_ref TEXT NOT NULL, scope_ref TEXT NOT NULL, snapshot_id TEXT NOT NULL,
        manifest BLOB NOT NULL, manifest_sha256 TEXT NOT NULL,
        validation_sha256 TEXT NOT NULL, evidence_json TEXT NOT NULL,
        authorization_revision INTEGER NOT NULL CHECK (authorization_revision >= 0),
        state_sha256 TEXT NOT NULL,
        PRIMARY KEY (workspace_ref, scope_ref, snapshot_id))""",
    """CREATE TABLE graph_evidence (
        workspace_ref TEXT NOT NULL, scope_ref TEXT NOT NULL, evidence_ref TEXT NOT NULL,
        snapshot_id TEXT NOT NULL,
        PRIMARY KEY (workspace_ref, scope_ref, evidence_ref, snapshot_id))""",
    """CREATE TABLE graph_ineligible (
        workspace_ref TEXT NOT NULL, scope_ref TEXT NOT NULL, evidence_ref TEXT NOT NULL,
        reason_code TEXT NOT NULL,
        PRIMARY KEY (workspace_ref, scope_ref, evidence_ref))""",
    """CREATE TABLE graph_active (
        workspace_ref TEXT NOT NULL, scope_ref TEXT NOT NULL, snapshot_id TEXT,
        revision INTEGER NOT NULL CHECK (revision >= 0),
        PRIMARY KEY (workspace_ref, scope_ref))""",
    """CREATE TABLE graph_history (
        workspace_ref TEXT NOT NULL, scope_ref TEXT NOT NULL, revision INTEGER NOT NULL,
        snapshot_id TEXT, reason_code TEXT NOT NULL,
        PRIMARY KEY (workspace_ref, scope_ref, revision))""",
)


class SqliteSnapshotStore:
    """Private scoped metadata transactions, sharing the receipt path safeguards.

    This is a separate database schema from inference receipts. Both adapters
    reject unknown SQLite definitions; neither stores graph table row payloads.
    """

    def __init__(self, database_path: Path) -> None:
        if (
            not isinstance(database_path, Path)
            or not database_path.is_absolute()
            or ".." in database_path.parts
        ):
            raise ValueError("database_path must be an absolute Path without traversal")
        self._path = database_path
        self._timeout = 5.0
        with _private_directory(database_path.parent, create=True) as directory:
            try:
                descriptor = os.open(
                    database_path.name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=directory,
                )
            except FileExistsError:
                pass
            else:
                os.close(descriptor)
                os.fsync(directory)
        with self.transaction(initialize=True):
            pass

    @contextmanager
    def transaction(self, *, initialize: bool = False) -> Iterator[sqlite3.Connection]:
        connection = None
        try:
            with _private_directory(self._path.parent) as directory:
                for suffix in ("", "-journal", "-wal", "-shm"):
                    info = _stat_database_file(
                        directory, self._path.name + suffix, sidecar=bool(suffix)
                    )
                    if info is None:
                        continue
                    if info.st_nlink != 1:
                        raise ValueError("graph database must not be hard-linked")
                before = os.stat(self._path.name, dir_fd=directory, follow_symlinks=False)
                connection = sqlite3.connect(
                    self._path.as_uri() + "?mode=rw",
                    uri=True,
                    timeout=self._timeout,
                    isolation_level=None,
                )
                connection.row_factory = sqlite3.Row
                after = os.stat(self._path.name, dir_fd=directory, follow_symlinks=False)
                if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                    raise ValueError("graph database changed during open")
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute("BEGIN IMMEDIATE")
                self._schema(connection, initialize)
                yield connection
                connection.commit()
        finally:
            if connection is not None:
                if connection.in_transaction:
                    connection.rollback()
                connection.close()

    @staticmethod
    def _schema(connection: sqlite3.Connection, initialize: bool) -> None:
        objects = connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master"
        ).fetchall()
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if initialize and version == 0 and not objects:
            for statement in _GRAPH_SCHEMA:
                connection.execute(statement)
            connection.execute("PRAGMA user_version=1")
            version = 1
            objects = connection.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master"
            ).fetchall()
        expected: set[tuple[str, str, str, str | None]] = set()
        for statement in _GRAPH_SCHEMA:
            name = statement.split()[2]
            expected.add(("table", name, name, statement))
            expected.add(("index", "sqlite_autoindex_" + name + "_1", name, None))
        if version != 1 or {tuple(row) for row in objects} != expected:
            raise ValueError("graph database schema is unsupported or corrupt")


_BINDING_SCHEMA = tuple(
    f"""CREATE TABLE {name} (
        workspace_ref TEXT NOT NULL, scope_ref TEXT NOT NULL, {identity} TEXT NOT NULL,
        payload BLOB NOT NULL, sha256 TEXT NOT NULL,
        PRIMARY KEY (workspace_ref, scope_ref, {identity}))"""
    for name, identity in (
        ("graph_retrieval_bindings", "binding_id"),
        ("graph_index_attempts", "attempt_ref"),
    )
)


class SqliteBindingStore(SqliteSnapshotStore):
    """Separate binding/attempt database using the same private path safeguards."""

    @staticmethod
    def _schema(connection: sqlite3.Connection, initialize: bool) -> None:
        objects = connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master"
        ).fetchall()
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if initialize and version == 0 and not objects:
            for statement in _BINDING_SCHEMA:
                connection.execute(statement)
            connection.execute("PRAGMA user_version=1")
            version = 1
            objects = connection.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master"
            ).fetchall()
        expected: set[tuple[str, str, str, str | None]] = set()
        for statement in _BINDING_SCHEMA:
            name = statement.split()[2]
            expected.add(("table", name, name, statement))
            expected.add(("index", "sqlite_autoindex_" + name + "_1", name, None))
        if version != 1 or {tuple(row) for row in objects} != expected:
            raise ValueError("binding database schema is unsupported or corrupt")
