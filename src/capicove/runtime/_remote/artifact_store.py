"""Private SQLite ledger and owner-only files for remote artifact uploads.

The ledger is the only authority for upload, chunk, artifact and quota state.
Content files are written before the ledger transaction that makes them
reachable, so an interrupted write leaves at most unreachable private bytes.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS uploads (upload_id TEXT PRIMARY KEY, owner TEXT NOT NULL,"
    " application TEXT NOT NULL, idempotency_key TEXT NOT NULL, size INTEGER NOT NULL,"
    " sha256 TEXT NOT NULL, media_type TEXT NOT NULL, created REAL NOT NULL,"
    " expires REAL NOT NULL, state TEXT NOT NULL, artifact_id TEXT,"
    " UNIQUE(owner, idempotency_key))",
    "CREATE TABLE IF NOT EXISTS chunks (upload_id TEXT NOT NULL, idx INTEGER NOT NULL,"
    " sha256 TEXT NOT NULL, size INTEGER NOT NULL, PRIMARY KEY(upload_id, idx))",
    "CREATE TABLE IF NOT EXISTS artifacts (artifact_id TEXT PRIMARY KEY, owner TEXT NOT NULL,"
    " application TEXT NOT NULL, upload_id TEXT NOT NULL UNIQUE, sha256 TEXT NOT NULL,"
    " size INTEGER NOT NULL, media_type TEXT NOT NULL, version INTEGER NOT NULL,"
    " lineage TEXT NOT NULL, created REAL NOT NULL, state TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS quotas (application TEXT PRIMARY KEY, tenant TEXT NOT NULL,"
    " reserved INTEGER NOT NULL CHECK(reserved >= 0), stored INTEGER NOT NULL CHECK(stored >= 0))",
)


def private_directory(path: Path) -> Path:
    """Create or verify an owner-only real directory (never a symlink)."""
    if path.is_symlink():
        raise ValueError("artifact store paths must not be symlinks")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    status = path.lstat()
    if not path.is_dir() or status.st_uid != os.geteuid() or status.st_mode & 0o077:
        raise ValueError("artifact store directories must be owner-only real directories")
    return path


def write_private(path: Path, data: bytes) -> None:
    """Atomically create or replace an owner-only file, durable before return."""
    temporary = path.with_name(path.name + ".tmp-" + os.urandom(8).hex())
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_private(path: Path, offset: int = 0, length: int | None = None) -> bytes:
    """Read an owner-only regular file without following symlinks."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        handle.seek(offset)
        return handle.read() if length is None else handle.read(length)


class ArtifactLedger:
    """Serialize every state transition through immediate SQLite transactions."""

    def __init__(self, path: Path) -> None:
        self._path = path
        if not path.exists():
            os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600))
        with self.transaction() as connection:
            for statement in _SCHEMA:
                connection.execute(statement)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Yield one connection inside ``BEGIN IMMEDIATE``; commit or roll back."""
        connection = sqlite3.connect(self._path, timeout=30, isolation_level=None)
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
