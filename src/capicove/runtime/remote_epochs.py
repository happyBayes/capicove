"""Persistent workspace revocation permits for local single-store deployments.

Every protected effect must execute inside a permit. Permits are concurrent read
transactions holding a SQLite SHARED lock; revocation commits its epoch change
only after acquiring the EXCLUSIVE lock, so it is acknowledged once prior permits
finish, and its pending lock stops new permits from starting meanwhile. The store
must stay in rollback-journal mode (never WAL), where these lock semantics hold.
With an external revocation journal (kept outside the restorable store), opening
the store reapplies every recorded revocation, so restoring an older backup cannot
revive a revoked epoch. This coordinator does not undo external effects and is not
a distributed lock.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class EpochStore:
    """Serialize revocation and bounded effects through SQLite transactions."""

    def __init__(self, path: Path, *, revocations: Path | None = None) -> None:
        self._path = path
        self._revocations = revocations
        with self._connection() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS workspace_epochs "
                "(workspace TEXT PRIMARY KEY, epoch INTEGER NOT NULL CHECK(epoch >= 0))"
            )
        if revocations is not None:
            self._reapply(self._journal())

    def _journal(self) -> dict[str, int]:
        """Read the external revocation journal; malformed records fail closed."""
        assert self._revocations is not None
        if self._revocations.is_symlink():
            raise RuntimeError("Epoch revocation journal must not be a symlink")
        if not self._revocations.exists():
            self._revocations.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            return {}
        latest: dict[str, int] = {}
        for line in self._revocations.read_bytes().splitlines():
            try:
                record = json.loads(line)
                workspace, epoch = record["workspace"], record["epoch"]
            except (ValueError, KeyError, TypeError):
                raise RuntimeError("Epoch revocation journal is corrupt") from None
            if (
                not isinstance(workspace, str)
                or type(epoch) is not int
                or epoch < 1
                or set(record) != {"workspace", "epoch"}
            ):
                raise RuntimeError("Epoch revocation journal is corrupt")
            latest[workspace] = max(epoch, latest.get(workspace, 0))
        return latest

    def _reapply(self, latest: dict[str, int]) -> None:
        """Restoring an older store can only move epochs forward, never back."""
        for workspace, epoch in latest.items():
            with self._transaction(workspace) as (connection, current):
                if current < epoch:
                    connection.execute(
                        "UPDATE workspace_epochs SET epoch = ? WHERE workspace = ?",
                        (epoch, workspace),
                    )

    def _record(self, workspace: str, epoch: int) -> None:
        assert self._revocations is not None
        line = json.dumps({"workspace": workspace, "epoch": epoch}, separators=(",", ":"))
        descriptor = os.open(
            self._revocations, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600
        )
        with os.fdopen(descriptor, "ab") as handle:
            handle.write(line.encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())

    @contextmanager
    def _connection(self, timeout: float = 2) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._path, timeout=timeout, isolation_level=None)
        try:
            connection.execute("PRAGMA synchronous=FULL")
            mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
            if str(mode).lower() == "wal":
                raise RuntimeError("Epoch store must not use WAL journaling")
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(
        self, workspace: str, *, timeout: float = 2
    ) -> Iterator[tuple[sqlite3.Connection, int]]:
        if not isinstance(workspace, str) or not workspace.strip() or len(workspace) > 2048:
            raise ValueError("A bounded workspace binding is required")
        with self._connection(timeout) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "INSERT OR IGNORE INTO workspace_epochs(workspace, epoch) VALUES (?, 0)",
                    (workspace,),
                )
                row = connection.execute(
                    "SELECT epoch FROM workspace_epochs WHERE workspace = ?", (workspace,)
                ).fetchone()
                if row is None or type(row[0]) is not int or row[0] < 0:
                    raise RuntimeError("Invalid persisted workspace epoch")
                yield connection, row[0]
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    def _read(self, workspace: str) -> int | None:
        if not isinstance(workspace, str) or not workspace.strip() or len(workspace) > 2048:
            raise ValueError("A bounded workspace binding is required")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT epoch FROM workspace_epochs WHERE workspace = ?", (workspace,)
            ).fetchone()
        if row is None:
            return None
        if type(row[0]) is not int or row[0] < 0:
            raise RuntimeError("Invalid persisted workspace epoch")
        return int(row[0])

    def _exists(self, workspace: str) -> bool:
        return self._read(workspace) is not None

    def current(self, workspace: str) -> int:
        """Read a current epoch without a stale availability cache."""
        epoch = self._read(workspace)
        if epoch is not None:
            return epoch
        with self._transaction(workspace) as (_, created):
            return created

    def revoke(self, workspace: str) -> int:
        """Advance the epoch and acknowledge after existing permits finish."""
        # Waiting for in-flight permits (each send is bounded) needs a longer timeout.
        with self._transaction(workspace, timeout=30) as (connection, epoch):
            if epoch >= 2**63 - 1:
                raise RuntimeError("Workspace epoch exhausted")
            updated = epoch + 1
            if self._revocations is not None:
                # Journal first: a crash before the commit only over-revokes.
                self._record(workspace, updated)
            connection.execute(
                "UPDATE workspace_epochs SET epoch = ? WHERE workspace = ?",
                (updated, workspace),
            )
        return updated

    @contextmanager
    def permit(self, workspace: str, expected_epoch: int, *, phase: str) -> Iterator[None]:
        """Fence one effect; callers must bound its duration and never nest permits."""
        if type(expected_epoch) is not int or expected_epoch < 0:
            raise ValueError("A nonnegative expected epoch is required")
        if phase not in {"start", "send", "commit", "release"}:
            raise ValueError("Unknown protected effect phase")
        if not self._exists(workspace):
            self.current(workspace)  # Create the row outside the shared fence.
        with self._connection() as connection:
            connection.execute("BEGIN DEFERRED")
            try:
                # The first read takes a SHARED lock that is held until COMMIT.
                row = connection.execute(
                    "SELECT epoch FROM workspace_epochs WHERE workspace = ?", (workspace,)
                ).fetchone()
                if row is None or type(row[0]) is not int or row[0] < 0:
                    raise RuntimeError("Invalid persisted workspace epoch")
                if row[0] != expected_epoch:
                    raise PermissionError("Invocation epoch has been revoked")
                yield
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
