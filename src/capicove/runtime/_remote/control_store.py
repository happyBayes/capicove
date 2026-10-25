"""Workspace-owned transactional ledger for remote operation admission.

One SQLite file holds operations, their append-only state history and the audit
outbox. Every transition is an immediate transaction, so idempotency-key
reservation, quota checks and state compare-and-swap are atomic across threads
and processes that share the file.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS operations (operation_id TEXT PRIMARY KEY,"
    " owner TEXT NOT NULL, application TEXT NOT NULL, tenant TEXT NOT NULL,"
    " authority TEXT NOT NULL,"
    " tool_id TEXT NOT NULL, tool_version INTEGER NOT NULL, kind TEXT NOT NULL,"
    " idempotency_key TEXT NOT NULL, input_sha256 TEXT NOT NULL, arguments TEXT NOT NULL,"
    " ceiling TEXT NOT NULL, state TEXT NOT NULL, fence INTEGER NOT NULL DEFAULT 0,"
    " worker TEXT, lease_until REAL, output TEXT, created REAL NOT NULL,"
    " UNIQUE(owner, tool_id, tool_version, kind, idempotency_key))",
    "CREATE TABLE IF NOT EXISTS history (sequence INTEGER PRIMARY KEY AUTOINCREMENT,"
    " operation_id TEXT NOT NULL, state TEXT NOT NULL, at REAL NOT NULL, reason TEXT)",
    "CREATE TABLE IF NOT EXISTS outbox (sequence INTEGER PRIMARY KEY AUTOINCREMENT,"
    " operation_id TEXT NOT NULL, event TEXT NOT NULL, at REAL NOT NULL,"
    " drained INTEGER NOT NULL DEFAULT 0)",
    "CREATE TABLE IF NOT EXISTS settings (name TEXT PRIMARY KEY, value TEXT NOT NULL)",
)


class ControlStore:
    """Open the private ledger and run each change in ``BEGIN IMMEDIATE``."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        if self._path.is_symlink():
            raise ValueError("control store must not be a symlink")
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not self._path.exists():
            try:
                os.close(
                    os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                )
            except FileExistsError:
                pass
        with self.transaction() as connection:
            for statement in _SCHEMA:
                connection.execute(statement)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Yield one connection inside an immediate transaction."""
        connection = sqlite3.connect(self._path, timeout=60, isolation_level=None)
        try:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        finally:
            connection.close()
