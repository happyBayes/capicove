"""Policy replacement and revocation must persist atomically across restarts."""

import importlib
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from capicove.contracts.remote import AuthorityKey, PrincipalKey
from capicove.runtime.remote_dispatch import WorkspaceGrant


def grant(subject: str = "alice") -> WorkspaceGrant:
    authority = AuthorityKey(
        PrincipalKey("https://issuer.test", "tenant", "app", "human", subject),
        "workspace",
        "grant",
        "https://mcp.test/mcp",
    )
    return WorkspaceGrant(
        authority, "production", frozenset({"read"}), frozenset({"read"}), frozenset({"read"})
    )


def store(path: Path) -> Any:
    assert importlib.util.find_spec("capicove.runtime._remote.policy_store") is not None
    return importlib.import_module("capicove.runtime._remote.policy_store").PolicyStore(path)


def test_grants_persist_with_server_assigned_monotonic_epoch(tmp_path: Path) -> None:
    path = tmp_path / "policy.sqlite"
    policy = store(path)
    original = grant()
    enrolled = policy.replace_grants("workspace", (original,), expected_epoch=0)
    assert enrolled[0].epoch == 1
    assert original.epoch == 0
    reopened = store(path)
    assert (
        reopened.grant_for(original.authority.principal_key, original.authority.audience)
        == enrolled[0]
    )
    assert reopened.current("workspace") == 1
    with pytest.raises(PermissionError):
        reopened.grant_for(grant("bob").authority.principal_key, original.authority.audience)


def test_revoked_policy_cannot_reappear_after_restart(tmp_path: Path) -> None:
    path = tmp_path / "policy.sqlite"
    policy = store(path)
    original = grant()
    policy.replace_grants("workspace", (original,), expected_epoch=0)
    assert policy.revoke("workspace") == 2
    reopened = store(path)
    with pytest.raises(PermissionError):
        reopened.grant_for(original.authority.principal_key, original.authority.audience)
    with pytest.raises(PermissionError):
        reopened.replace_grants("workspace", (original,), expected_epoch=0)
    assert reopened.current("workspace") == 2


def test_write_failure_rolls_back_grants_and_epoch_together(tmp_path: Path) -> None:
    path = tmp_path / "policy.sqlite"
    policy = store(path)
    original = grant()
    enrolled = policy.replace_grants("workspace", (original,), expected_epoch=0)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TRIGGER fail_bob BEFORE INSERT ON remote_grants WHEN NEW.principal_json LIKE '%bob%' BEGIN SELECT RAISE(ABORT, 'fixture write failure'); END"
        )
    with pytest.raises(sqlite3.DatabaseError):
        policy.replace_grants(
            "workspace",
            (replace(original, epoch=1), replace(grant("bob"), epoch=1)),
            expected_epoch=1,
        )
    assert policy.current("workspace") == 1
    assert (
        policy.grant_for(original.authority.principal_key, original.authority.audience)
        == enrolled[0]
    )


def test_replacement_rejects_foreign_workspace_without_state_change(tmp_path: Path) -> None:
    policy = store(tmp_path / "policy.sqlite")
    with pytest.raises(ValueError):
        policy.replace_grants("other", (grant(),), expected_epoch=0)
    assert policy.current("workspace") == 0


def test_future_schema_is_rejected_before_initializing_tables(tmp_path: Path) -> None:
    path = tmp_path / "future.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE remote_policy_schema (id INTEGER PRIMARY KEY, version INTEGER NOT NULL)"
        )
        connection.execute("INSERT INTO remote_policy_schema VALUES (1,2)")
    path.chmod(0o600)
    with pytest.raises(ValueError):
        store(path)
    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert tables == {"remote_policy_schema"}


def test_unreadable_replacement_cannot_commit_or_advance_epoch(tmp_path: Path) -> None:
    path = tmp_path / "policy.sqlite"
    policy = store(path)
    original = policy.replace_grants("workspace", (grant(),), expected_epoch=0)[0]
    oversized = replace(
        original, application=frozenset(f"permission_{index}" for index in range(257))
    )
    with pytest.raises(ValueError):
        policy.replace_grants("workspace", (oversized,), expected_epoch=1)
    assert policy.current("workspace") == 1
    assert (
        policy.grant_for(original.authority.principal_key, original.authority.audience) == original
    )
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM remote_policy_history").fetchone() == (1,)


def test_empty_snapshot_revokes_every_grant_atomically(tmp_path: Path) -> None:
    path = tmp_path / "policy.sqlite"
    policy = store(path)
    original = policy.replace_grants("workspace", (grant(),), expected_epoch=0)[0]
    assert policy.replace_grants("workspace", (), expected_epoch=1) == ()
    assert policy.current("workspace") == 2
    reopened = store(path)
    with pytest.raises(PermissionError):
        reopened.grant_for(original.authority.principal_key, original.authority.audience)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM remote_grants").fetchone() == (0,)
        assert connection.execute(
            "SELECT snapshot FROM remote_policy_history WHERE epoch=2"
        ).fetchone() == ("[]",)
