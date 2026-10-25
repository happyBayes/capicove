"""Private transactional storage for native knowledge generations and evidence."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

from capicove.runtime.media_handoff import open_private_directory


class KnowledgeStore:
    """Pin the storage directory and serialize short state transitions in SQLite."""

    def __init__(self, root: Path, identity: str, *, read_only: bool = False) -> None:
        self.read_only = read_only
        self.fd = open_private_directory(root, create=not read_only)
        self.root = Path(f"/proc/{os.getpid()}/fd/{self.fd}")
        self.db: sqlite3.Connection
        try:
            for name in (
                "knowledge.sqlite",
                "knowledge.sqlite-journal",
                "knowledge.sqlite-wal",
                "knowledge.sqlite-shm",
            ):
                try:
                    info = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_nlink != 1
                    or info.st_uid != os.getuid()
                    or info.st_mode & 0o077
                ):
                    raise ValueError("unsafe_knowledge_store")
            if read_only:
                uri = "file:" + quote(str(self.root / "knowledge.sqlite"), safe="/") + "?mode=ro"
                self.db = sqlite3.connect(uri, uri=True, timeout=10)
                self.db.row_factory = sqlite3.Row
                if (
                    self.db.execute("PRAGMA user_version").fetchone()[0] != 1
                    or self.get("identity") != identity
                ):
                    raise ValueError("knowledge_workspace_mismatch")
                return
            descriptor = os.open(
                "knowledge.sqlite", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600, dir_fd=self.fd
            )
            os.close(descriptor)
            self.db = sqlite3.connect(self.root / "knowledge.sqlite", timeout=10)
            self.db.row_factory = sqlite3.Row
            with self.transaction():
                version = self.db.execute("PRAGMA user_version").fetchone()[0]
                if version not in (0, 1):
                    raise ValueError("unsupported_knowledge_store")
                self.db.execute(
                    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                self.db.execute(
                    "CREATE TABLE IF NOT EXISTS versions (id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
                )
                self.db.execute(
                    "CREATE TABLE IF NOT EXISTS sources (source TEXT PRIMARY KEY, version TEXT NOT NULL)"
                )
                self.db.execute(
                    "CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)"
                )
                self.db.execute("PRAGMA user_version=1")
                existing = self.get("identity")
                if existing is not None and existing != identity:
                    raise ValueError("knowledge_workspace_mismatch")
                self.set("identity", identity)
                if self.get("revision") is None:
                    self.set("revision", 0)
                    self.set("generation", 0)
                    self.set("graph", None)
        except BaseException:
            if hasattr(self, "db"):
                self.db.close()
            os.close(self.fd)
            raise

    def close(self) -> None:
        """Release database and pinned filesystem handles."""
        self.db.close()
        os.close(self.fd)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Protect compare-and-swap publication without locking model inference."""
        self.db.execute("BEGIN" if self.read_only else "BEGIN IMMEDIATE")
        try:
            yield
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def get(self, key: str) -> Any:
        """Read a small typed state value."""
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: Any) -> None:
        """Write state inside the caller's transaction."""
        self.db.execute(
            "INSERT INTO meta VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value, ensure_ascii=False, allow_nan=False)),
        )

    def records(self) -> list[dict[str, Any]]:
        """Read only currently published source versions in stable order."""
        from capicove.runtime.native_knowledge import digest

        records = []
        for row in self.db.execute(
            "SELECT v.payload,v.id FROM sources s JOIN versions v ON s.version=v.id ORDER BY s.source"
        ):
            payload = json.loads(row[0])
            if (
                payload.get("version") != row[1]
                or digest({key: value for key, value in payload.items() if key != "version"})
                != row[1]
            ):
                raise ValueError("corrupt_knowledge_version")
            records.append(payload)
        return records

    def preserve(self, source: Path, expected: str) -> None:
        """Publish original bytes only after a complete, durable verified copy."""
        temporary = "pending-" + secrets.token_hex(16)
        name = "source-" + expected
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400, dir_fd=self.fd
        )
        try:
            checksum = hashlib.sha256()
            with os.fdopen(descriptor, "wb") as output, source.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    checksum.update(block)
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
            if checksum.hexdigest() != expected:
                raise ValueError("staged_source_changed")
            try:
                os.link(
                    temporary, name, src_dir_fd=self.fd, dst_dir_fd=self.fd, follow_symlinks=False
                )
            except FileExistsError:
                existing = os.open(
                    name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.fd
                )
                with os.fdopen(existing, "rb") as stream:
                    info = os.fstat(stream.fileno())
                    if (
                        not stat.S_ISREG(info.st_mode)
                        or info.st_nlink != 1
                        or hashlib.file_digest(stream, "sha256").hexdigest() != expected
                    ):
                        raise ValueError("stored_source_corrupted") from None
            os.fsync(self.fd)
        finally:
            os.unlink(temporary, dir_fd=self.fd)

    def run(self, payload: dict[str, Any]) -> None:
        """Retain append-only operational outcomes without raw exceptions."""
        with self.transaction():
            self.db.execute("INSERT INTO runs(payload) VALUES (?)", (json.dumps(payload),))

    def last_run(self) -> dict[str, Any] | None:
        """Return the latest completed attempt."""
        row = self.db.execute("SELECT payload FROM runs ORDER BY id DESC LIMIT 1").fetchone()
        return json.loads(row[0]) if row else None
