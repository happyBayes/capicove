"""SQLite journal deletion must not weaken protected-path validation."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, closing
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from capicove.graph_build._adapters import sqlite as adapter
from capicove.graph_build._adapters.artifacts import (
    ArtifactStoreError,
    FilesystemProtectedArtifacts,
)
from capicove.graph_build.receipts import ReceiptStoreError
from tests.integration.graph_build.test_receipts import Gate


pytestmark = pytest.mark.integration
Transaction = Callable[[], AbstractContextManager[sqlite3.Connection]]


@pytest.fixture(params=["snapshot", "receipt"])
def store(tmp_path: Path, request: pytest.FixtureRequest) -> tuple[Path, Transaction]:
    path = tmp_path / "db" / "graph.sqlite"
    if request.param == "snapshot":
        return path, adapter.SqliteSnapshotStore(path).transaction
    gate = Gate()
    artifacts = FilesystemProtectedArtifacts(tmp_path / "artifacts", gate, evidence_refs=("a",))
    return path, adapter.SqliteReceiptStore(path, gate, artifacts)._transaction


@pytest.mark.parametrize(
    "replacement", ["absent", "private", "hardlink", "symlink", "public", "fifo"]
)
def test_commit_during_journal_lookup(
    store: tuple[Path, Transaction], monkeypatch: pytest.MonkeyPatch, replacement: str
) -> None:
    path, transaction = store
    journal = path.with_name(path.name + "-journal")
    original_stat = os.stat
    observations: list[int] = []
    external = path.parent / "external"
    external.write_bytes(b"unrelated private content")
    external.chmod(0o600)
    with closing(sqlite3.connect(path, isolation_level=None)) as writer:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("PRAGMA user_version=2")
        writer.execute("PRAGMA user_version=1")
        assert journal.exists()

        def stat_during_commit(name: Any, *args: Any, **kwargs: Any) -> os.stat_result:
            if name == journal.name and not observations:
                # Pin the inode at path lookup, then let SQLite unlink it before
                # metadata is returned. Do not invent a platform stat_result.
                descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=kwargs["dir_fd"])
                try:
                    writer.commit()
                    result = os.fstat(descriptor)
                    assert result.st_nlink == 0
                    observations.append(result.st_nlink)
                finally:
                    os.close(descriptor)
                if replacement == "hardlink":
                    os.link(external, journal)
                elif replacement == "symlink":
                    journal.symlink_to(external)
                elif replacement == "fifo":
                    os.mkfifo(journal, mode=0o600)
                elif replacement in ("private", "public"):
                    journal.write_bytes(b"")
                    journal.chmod(0o600 if replacement == "private" else 0o644)
                return result
            return original_stat(name, *args, **kwargs)

        monkeypatch.setattr(os, "stat", stat_during_commit)
        if replacement in ("absent", "private"):
            with transaction() as connection:
                assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        else:
            connect = Mock(side_effect=AssertionError("unsafe path reached SQLite"))
            monkeypatch.setattr(sqlite3, "connect", connect)
            with pytest.raises((ValueError, ArtifactStoreError, ReceiptStoreError)):
                with transaction():
                    pytest.fail("unsafe replacement must be rejected")
            connect.assert_not_called()
        assert observations == [0]
        assert external.read_bytes() == b"unrelated private content"


@pytest.mark.parametrize("suffix", ["", "-journal", "-wal", "-shm"])
def test_real_hardlinks_are_rejected_before_sqlite_open(
    store: tuple[Path, Transaction], monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    path, transaction = store
    target = path.with_name(path.name + suffix)
    if suffix:
        target.write_bytes(b"")
        target.chmod(0o600)
    alias = path.parent / "alias"
    os.link(target, alias)
    assert target.stat().st_nlink == 2
    connect = Mock(side_effect=AssertionError("hardlink reached SQLite"))
    monkeypatch.setattr(sqlite3, "connect", connect)
    with pytest.raises((ValueError, ReceiptStoreError), match="hard-linked"):
        with transaction():
            pytest.fail("hardlink must be rejected")
    connect.assert_not_called()


@pytest.mark.parametrize("suffix, expected_lookups", [("", 1), ("-journal", 2)])
def test_unlinked_metadata_rechecks_are_bounded_and_main_database_is_required(
    store: tuple[Path, Transaction],
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
    expected_lookups: int,
) -> None:
    path, transaction = store
    detached = path.parent / "detached"
    detached.write_bytes(b"")
    detached.chmod(0o600)
    original_stat = os.stat
    lookups = 0
    with detached.open("rb") as stream:
        detached.unlink()
        unlinked = os.fstat(stream.fileno())
        assert unlinked.st_nlink == 0

        def stat_unlinked(name: Any, *args: Any, **kwargs: Any) -> os.stat_result:
            nonlocal lookups
            if name == path.name + suffix:
                lookups += 1
                return unlinked
            return original_stat(name, *args, **kwargs)

        monkeypatch.setattr(os, "stat", stat_unlinked)
        connect = Mock(side_effect=AssertionError("unlinked file reached SQLite"))
        monkeypatch.setattr(sqlite3, "connect", connect)
        with pytest.raises((ValueError, ReceiptStoreError), match="hard-linked"):
            with transaction():
                pytest.fail("unresolved metadata must fail closed")
        connect.assert_not_called()
        assert lookups == expected_lookups


def test_independent_stores_preserve_concurrent_transactions_and_rollback(tmp_path: Path) -> None:
    path = tmp_path / "db" / "graph.sqlite"
    stores = (adapter.SqliteSnapshotStore(path), adapter.SqliteSnapshotStore(path))
    with stores[0].transaction() as connection:
        connection.execute("INSERT INTO graph_active VALUES ('w', 's', NULL, 0)")

    def increment(store: adapter.SqliteSnapshotStore) -> None:
        for _ in range(32):
            with store.transaction() as connection:
                connection.execute("UPDATE graph_active SET revision = revision + 1")

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(increment, stores))
    with pytest.raises(RuntimeError, match="interrupt"):
        with stores[0].transaction() as connection:
            connection.execute("UPDATE graph_active SET revision = 1000")
            raise RuntimeError("interrupt")
    with adapter.SqliteSnapshotStore(path).transaction() as connection:
        assert connection.execute("SELECT revision FROM graph_active").fetchone()[0] == 64
