"""MCP tracing must not retain connections after operations or startup failure."""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from capicove.traces._adapters.sqlite import SqliteTraceStore


def test_trace_connections_close_after_init_and_read(tmp_path: Path) -> None:
    connections: list[sqlite3.Connection] = []
    connect = sqlite3.connect

    def tracked_connect(
        database: Path, *, isolation_level: None, timeout: float
    ) -> sqlite3.Connection:
        connection = connect(database, isolation_level=isolation_level, timeout=timeout)
        connections.append(connection)
        return connection

    with patch("capicove.traces._adapters.sqlite.sqlite3.connect", tracked_connect):
        store = SqliteTraceStore(tmp_path / "trace.sqlite")
        assert store.get_episode("missing") == ()
    assert len(connections) == 2
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")


def test_trace_connection_closes_when_setup_fails(tmp_path: Path) -> None:
    connection = sqlite3.connect(":memory:")
    with (
        patch("capicove.traces._adapters.sqlite.sqlite3.connect", return_value=connection),
        patch(
            "capicove.traces._adapters.sqlite._configure_connection",
            side_effect=RuntimeError("setup failed"),
        ),
        pytest.raises(RuntimeError, match="setup failed"),
    ):
        SqliteTraceStore(tmp_path / "trace.sqlite")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
