"""Windows trace files must be private before SQLite opens any path."""

from __future__ import annotations

import csv
import io
import os
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

import capicove.traces._adapters.sqlite as sqlite_store
from capicove.traces import (
    EpisodeStarted,
    TraceLifecycleError,
    TracePostCommitPrivacyError,
    TraceRecordDraft,
    TraceStoreError,
)
from capicove.traces._adapters.sqlite import SqliteTraceStore


def test_importing_sqlite_store_does_not_require_windows_helpers() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import builtins; "
            "original_import = builtins.__import__; "
            "builtins.__import__ = lambda name, *args, **kwargs: "
            "(_ for _ in ()).throw(ModuleNotFoundError(name)) "
            "if name == 'capicove._windows_files' "
            "else original_import(name, *args, **kwargs); "
            "import capicove.traces._adapters.sqlite",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_windows_sqlite_temporary_data_stays_in_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sqlite_store, "os", SimpleNamespace(name="nt"))
    connection = sqlite3.connect(":memory:")
    try:
        sqlite_store._configure_connection(connection, writable=False)
        assert connection.execute("PRAGMA temp_store").fetchone()[0] == 2
        connection.execute("CREATE TEMP TABLE selected_trace_episodes (episode_id TEXT)")
        connection.execute("INSERT INTO selected_trace_episodes VALUES ('episode-1')")
        assert connection.execute("SELECT episode_id FROM selected_trace_episodes").fetchone()[
            0
        ] == ("episode-1")
    finally:
        connection.close()


def test_windows_sidecars_are_checked_before_sqlite_connect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from capicove import _windows_files

    store = object.__new__(SqliteTraceStore)
    store._database_path = tmp_path / "trace.sqlite"
    checked: list[Path] = []

    def reject_sidecar(path: Path) -> None:
        checked.append(path)
        if path.name.endswith("-shm"):
            raise ValueError("reparse point")

    monkeypatch.setattr(sqlite_store, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(_windows_files, "validate_windows_file_target", reject_sidecar)
    with (
        patch.object(sqlite3, "connect", side_effect=AssertionError("SQLite opened")),
        pytest.raises(TraceStoreError, match="private"),
    ):
        store._connect()
    assert [path.name for path in checked] == [
        "trace.sqlite",
        "trace.sqlite-wal",
        "trace.sqlite-shm",
    ]


def test_windows_snapshot_rechecks_files_after_connection_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SqliteTraceStore(tmp_path / "trace.sqlite")
    checks = 0

    def check_files() -> None:
        nonlocal checks
        checks += 1

    monkeypatch.setattr(sqlite_store, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(store, "_validate_windows_files", check_files)
    with store.open_snapshot() as snapshot:
        assert tuple(snapshot.iter_records()) == ()
        checks_before_close = checks
    assert checks == checks_before_close + 1


def test_windows_snapshot_postcheck_does_not_mask_body_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SqliteTraceStore(tmp_path / "trace.sqlite")
    checks = 0

    def fail_on_close() -> None:
        nonlocal checks
        checks += 1
        if checks == 4:
            raise TraceStoreError("fixture privacy failure")

    monkeypatch.setattr(sqlite_store, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(store, "_validate_windows_files", fail_on_close)
    with pytest.raises(ValueError, match="body failed") as caught:
        with store.open_snapshot():
            raise ValueError("body failed")
    assert any("privacy recheck failed" in note for note in caught.value.__notes__)


def _start_draft() -> TraceRecordDraft:
    return TraceRecordDraft(
        episode_id="episode-1",
        trace_id="trace-1",
        occurred_at=datetime.now(UTC),
        payload=EpisodeStarted(
            goal="Test Windows trace privacy.",
            workspace_id="workspace-1",
            actor_id="actor-1",
            environment_id="environment-1",
            environment_version=1,
            tool_catalog_sha256="a" * 64,
            active_asset_versions={},
        ),
    )


def test_windows_append_reports_ambiguous_commit_if_postcheck_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "trace.sqlite"
    store = SqliteTraceStore(database)
    draft = _start_draft()
    checks = 0

    def fail_after_commit() -> None:
        nonlocal checks
        checks += 1
        if checks == 4:
            raise TraceStoreError("fixture privacy failure")

    with monkeypatch.context() as windows:
        windows.setattr(sqlite_store, "os", SimpleNamespace(name="nt"))
        windows.setattr(store, "_validate_windows_files", fail_after_commit)
        with pytest.raises(TracePostCommitPrivacyError, match="may have persisted"):
            store.append(draft, idempotency_key="episode.started")

    retried = SqliteTraceStore(database).append(draft, idempotency_key="episode.started")
    assert SqliteTraceStore(database).get_episode("episode-1") == (retried,)


@pytest.mark.parametrize(
    ("close_error", "expected_type"),
    [
        (sqlite3.OperationalError("fixture close failure"), TraceStoreError),
        (KeyboardInterrupt(), KeyboardInterrupt),
    ],
)
def test_windows_append_rechecks_privacy_even_if_connection_close_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    close_error: BaseException,
    expected_type: type[BaseException],
) -> None:
    store = SqliteTraceStore(tmp_path / "trace.sqlite")
    real_connect = store._connect
    closed = False
    checked_after_close = False

    class FailingCloseConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self.connection = connection

        def execute(self, *args: Any) -> sqlite3.Cursor:
            return self.connection.execute(*args)

        def close(self) -> None:
            nonlocal closed
            self.connection.close()
            closed = True
            raise close_error

    def check_files() -> None:
        nonlocal checked_after_close
        if closed:
            checked_after_close = True
            raise TraceStoreError("fixture privacy failure")

    with monkeypatch.context() as windows:
        windows.setattr(sqlite_store, "os", SimpleNamespace(name="nt"))
        windows.setattr(store, "_validate_windows_files", check_files)
        windows.setattr(store, "_connect", lambda: FailingCloseConnection(real_connect()))
        with pytest.raises(expected_type) as caught:
            store.append(_start_draft(), idempotency_key="episode.started")
    assert checked_after_close
    if isinstance(close_error, Exception):
        assert "may have persisted" in str(caught.value)
        assert caught.value.__cause__ is close_error
    else:
        assert caught.value is close_error
    assert any("privacy recheck failed" in note for note in caught.value.__notes__)


def test_windows_append_close_failure_does_not_mask_body_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SqliteTraceStore(tmp_path / "trace.sqlite")
    draft = _start_draft()
    store.append(draft, idempotency_key="episode.started")
    real_connect = store._connect
    closed = False
    checked_after_close = False

    class FailingCloseConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self.connection = connection

        def execute(self, *args: Any) -> sqlite3.Cursor:
            return self.connection.execute(*args)

        def close(self) -> None:
            nonlocal closed
            self.connection.close()
            closed = True
            raise sqlite3.OperationalError("fixture close failure")

    def check_files() -> None:
        nonlocal checked_after_close
        if closed:
            checked_after_close = True

    with monkeypatch.context() as windows:
        windows.setattr(sqlite_store, "os", SimpleNamespace(name="nt"))
        windows.setattr(store, "_validate_windows_files", check_files)
        windows.setattr(store, "_connect", lambda: FailingCloseConnection(real_connect()))
        with pytest.raises(TraceLifecycleError, match="already started") as caught:
            store.append(draft, idempotency_key="duplicate-start")
    assert checked_after_close
    assert any("connection close failed" in note for note in caught.value.__notes__)


def test_windows_append_reports_postcheck_failure_inside_outer_exception_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SqliteTraceStore(tmp_path / "trace.sqlite")
    checks = 0

    def fail_after_commit() -> None:
        nonlocal checks
        checks += 1
        if checks == 4:
            raise TraceStoreError("fixture privacy failure")

    with monkeypatch.context() as windows:
        windows.setattr(sqlite_store, "os", SimpleNamespace(name="nt"))
        windows.setattr(store, "_validate_windows_files", fail_after_commit)
        try:
            raise ValueError("outer")
        except ValueError:
            with pytest.raises(TraceStoreError, match="may have persisted"):
                store.append(_start_draft(), idempotency_key="episode.started")


def test_windows_postcheck_does_not_mask_existing_append_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SqliteTraceStore(tmp_path / "trace.sqlite")
    draft = _start_draft()
    store.append(draft, idempotency_key="episode.started")
    checks = 0

    def fail_after_rollback() -> None:
        nonlocal checks
        checks += 1
        if checks == 3:
            raise TraceStoreError("fixture privacy failure")

    with monkeypatch.context() as windows:
        windows.setattr(sqlite_store, "os", SimpleNamespace(name="nt"))
        windows.setattr(store, "_validate_windows_files", fail_after_rollback)
        with pytest.raises(TraceLifecycleError, match="already started") as caught:
            store.append(draft, idempotency_key="duplicate-start")
    assert any("privacy recheck failed" in note for note in caught.value.__notes__)


def test_windows_read_reports_postcheck_failure_inside_outer_exception_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SqliteTraceStore(tmp_path / "trace.sqlite")
    checks = 0

    def fail_after_read() -> None:
        nonlocal checks
        checks += 1
        if checks == 3:
            raise TraceStoreError("fixture privacy failure")

    with monkeypatch.context() as windows:
        windows.setattr(sqlite_store, "os", SimpleNamespace(name="nt"))
        windows.setattr(store, "_validate_windows_files", fail_after_read)
        try:
            raise ValueError("outer")
        except ValueError:
            with pytest.raises(TraceStoreError, match="fixture privacy failure"):
                store.get_episode("missing")


def test_windows_snapshot_close_reports_postcheck_failure_inside_outer_exception_handler(
    tmp_path: Path,
) -> None:
    store = SqliteTraceStore(tmp_path / "trace.sqlite")
    snapshot = store.open_snapshot()

    def fail_on_close() -> None:
        raise TraceStoreError("fixture privacy failure")

    snapshot._on_close = fail_on_close
    try:
        raise ValueError("outer")
    except ValueError:
        with pytest.raises(TraceStoreError, match="fixture privacy failure"):
            snapshot.close()


def _private_windows_directory(tmp_path: Path) -> Path:
    directory = tmp_path / "private-traces"
    directory.mkdir()
    user = subprocess.run(
        ["whoami", "/user", "/fo", "csv", "/nh"],
        check=True,
        capture_output=True,
        text=True,
    )
    sid = next(csv.reader(io.StringIO(user.stdout)))[1]
    for arguments in (
        ["/grant:r", f"*{sid}:(OI)(CI)F"],
        ["/inheritance:r"],
        ["/setowner", f"*{sid}"],
    ):
        result = subprocess.run(
            ["icacls", str(directory), *arguments], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
    return directory


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows ACLs")
def test_windows_trace_store_uses_private_directory_and_database(tmp_path: Path) -> None:
    from capicove._windows_files import validate_windows_path

    directory = _private_windows_directory(tmp_path)
    database = directory / "trace.sqlite"
    store = SqliteTraceStore(database)

    assert database.is_file()
    assert store.get_episode("missing") == ()
    validate_windows_path(directory, directory=True, private_acl=True)
    validate_windows_path(database, directory=False, private_acl=True)
    connection = store._connect()
    try:
        assert connection.execute("PRAGMA temp_store").fetchone()[0] == 2
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("CREATE TABLE trace_windows_fixture (value INTEGER)")
        connection.execute("INSERT INTO trace_windows_fixture VALUES (1)")
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{database}{suffix}")
            assert sidecar.is_file()
            validate_windows_path(sidecar, directory=False, private_acl=True)
        connection.execute("ROLLBACK")
    finally:
        connection.close()


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows reparse points")
def test_windows_trace_store_rejects_reparse_sidecar_before_sqlite_open(tmp_path: Path) -> None:
    directory = _private_windows_directory(tmp_path)
    database = directory / "trace.sqlite"
    SqliteTraceStore(database)
    target = directory / "target"
    target.write_bytes(b"fixture")
    sidecar = Path(f"{database}-wal")
    sidecar.unlink(missing_ok=True)
    sidecar.symlink_to(target)

    with (
        patch.object(sqlite3, "connect", side_effect=AssertionError("SQLite opened")),
        pytest.raises(TraceStoreError, match="private"),
    ):
        SqliteTraceStore(database)
