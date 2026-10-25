"""Workspace grant snapshots atomically sharing the remote epoch database.

This store is an operator control-plane primitive. Remote administration must
compose it with fresh authorization, approval and audit admission. An external
restore/revocation anchor is still required to detect an older database backup.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
from dataclasses import asdict, replace
from pathlib import Path

from capicove.contracts.remote import AuthorityKey, PrincipalKey
from capicove.runtime.remote_config import _check_private_acl, _permission_set
from capicove.runtime.remote_dispatch import WorkspaceGrant
from capicove.runtime.remote_epochs import EpochStore


_LIMIT = 262144


def _private_database(path: Path) -> None:
    """Create or verify private state without following any path component."""
    if (
        os.name != "posix"
        or not hasattr(os, "O_NOFOLLOW")
        or not path.is_absolute()
        or ".." in path.parts
    ):
        raise ValueError("Private policy state requires an absolute POSIX path")
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    descriptor: int | None = None
    try:
        _check_private_acl(directory)
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
            _check_private_acl(directory)
            metadata = os.fstat(directory)
            if metadata.st_uid not in {0, os.geteuid()} or (
                metadata.st_mode & 0o022
                and not (metadata.st_uid == 0 and metadata.st_mode & stat.S_ISVTX)
            ):
                raise ValueError("Private policy state has an unsafe parent")
        descriptor = os.open(
            path.name,
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
            0o600,
            dir_fd=directory,
        )
        metadata = os.fstat(descriptor)
        _check_private_acl(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
            or metadata.st_nlink != 1
        ):
            raise ValueError("Policy state must be an owner-private regular file")
    except OSError as error:
        raise ValueError("Private policy state cannot be opened") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)


def _principal(key: PrincipalKey) -> str:
    return json.dumps(asdict(key), sort_keys=True, separators=(",", ":"))


def _encode(grant: WorkspaceGrant) -> str:
    for permissions in (grant.application, grant.user, grant.workspace):
        _permission_set(permissions)
    value = {
        "schema_version": 1,
        "authority": asdict(grant.authority),
        "environment": grant.environment_id,
        "application": sorted(grant.application),
        "user": sorted(grant.user),
        "workspace": sorted(grant.workspace),
        "epoch": grant.epoch,
    }
    result = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(result.encode()) > _LIMIT:
        raise ValueError("One grant exceeds its storage bound")
    return result


def _decode(payload: str) -> WorkspaceGrant:
    if len(payload.encode()) > _LIMIT:
        raise ValueError("Persisted grant exceeds its storage bound")
    value = json.loads(payload)
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "schema_version",
            "authority",
            "environment",
            "application",
            "user",
            "workspace",
            "epoch",
        }
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
    ):
        raise ValueError("Invalid persisted grant schema")
    raw_authority = value["authority"]
    if (
        not isinstance(raw_authority, dict)
        or set(raw_authority) != {"principal_key", "workspace_binding_id", "grant_id", "audience"}
        or not isinstance(raw_authority["principal_key"], dict)
    ):
        raise ValueError("Invalid persisted grant authority")
    principal = PrincipalKey(**raw_authority["principal_key"])
    authority = AuthorityKey(
        principal,
        raw_authority["workspace_binding_id"],
        raw_authority["grant_id"],
        raw_authority["audience"],
    )
    for key in ("application", "user", "workspace"):
        if not isinstance(value[key], list) or len(value[key]) > 256:
            raise ValueError("Invalid persisted permission set")
    grant = WorkspaceGrant(
        authority,
        value["environment"],
        frozenset(value["application"]),
        frozenset(value["user"]),
        frozenset(value["workspace"]),
        value["epoch"],
    )
    if _encode(grant) != payload:
        raise ValueError("Persisted grant is not canonical")
    return grant


class PolicyStore(EpochStore):
    """Replace complete workspace grant snapshots with one monotonic epoch commit."""

    def __init__(self, path: Path) -> None:
        _private_database(path)
        with sqlite3.connect(path, timeout=2) as connection:
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if tables - {
                "workspace_epochs",
                "remote_policy_schema",
                "remote_grants",
                "remote_policy_history",
            }:
                raise ValueError("Policy state contains unrelated tables")
            if "remote_policy_schema" in tables and connection.execute(
                "SELECT version FROM remote_policy_schema WHERE id=1"
            ).fetchone() != (1,):
                raise ValueError("Unsupported remote policy store version")
        super().__init__(path)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS remote_policy_schema (id INTEGER PRIMARY KEY CHECK(id=1), version INTEGER NOT NULL)"
                )
                connection.execute("INSERT OR IGNORE INTO remote_policy_schema VALUES (1,1)")
                if connection.execute(
                    "SELECT version FROM remote_policy_schema WHERE id=1"
                ).fetchone() != (1,):
                    raise ValueError("Unsupported remote policy store version")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS remote_grants (principal_json TEXT NOT NULL, audience TEXT NOT NULL, workspace TEXT NOT NULL, payload TEXT NOT NULL, sha256 TEXT NOT NULL, PRIMARY KEY(principal_json,audience))"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS remote_policy_history (workspace TEXT NOT NULL, epoch INTEGER NOT NULL, snapshot TEXT NOT NULL, sha256 TEXT NOT NULL, PRIMARY KEY(workspace,epoch))"
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    def replace_grants(
        self, workspace: str, grants: tuple[WorkspaceGrant, ...], *, expected_epoch: int
    ) -> tuple[WorkspaceGrant, ...]:
        """Commit a trusted operator snapshot and invalidate every prior permit."""
        if (
            not isinstance(grants, tuple)
            or len(grants) > 1024
            or type(expected_epoch) is not int
            or not 0 <= expected_epoch < 2**63 - 1
        ):
            raise ValueError("A bounded workspace policy snapshot and epoch are required")
        identities: set[tuple[PrincipalKey, str]] = set()
        for grant in grants:
            if (
                not isinstance(grant, WorkspaceGrant)
                or grant.authority.workspace_binding_id != workspace
                or grant.epoch != expected_epoch
            ):
                raise ValueError("Grant workspace or epoch differs from the replacement")
            key = grant.authority.principal_key, grant.authority.audience
            if key in identities:
                raise ValueError("Duplicate endpoint principal grant")
            identities.add(key)
        updated = tuple(replace(grant, epoch=expected_epoch + 1) for grant in grants)
        snapshot = json.dumps(
            [json.loads(_encode(grant)) for grant in updated], sort_keys=True, separators=(",", ":")
        )
        if len(snapshot.encode()) > _LIMIT:
            raise ValueError("Workspace policy snapshot exceeds its byte bound")
        with self._transaction(workspace) as (connection, epoch):
            if epoch != expected_epoch:
                raise PermissionError("Workspace policy epoch changed")
            connection.execute("DELETE FROM remote_grants WHERE workspace=?", (workspace,))
            for grant in updated:
                payload = _encode(grant)
                connection.execute(
                    "INSERT INTO remote_grants VALUES (?,?,?,?,?)",
                    (
                        _principal(grant.authority.principal_key),
                        grant.authority.audience,
                        workspace,
                        payload,
                        hashlib.sha256(payload.encode()).hexdigest(),
                    ),
                )
            connection.execute(
                "INSERT INTO remote_policy_history VALUES (?,?,?,?)",
                (
                    workspace,
                    expected_epoch + 1,
                    snapshot,
                    hashlib.sha256(snapshot.encode()).hexdigest(),
                ),
            )
            connection.execute(
                "UPDATE workspace_epochs SET epoch=? WHERE workspace=?",
                (expected_epoch + 1, workspace),
            )
        return updated

    def grant_for(self, principal: PrincipalKey, audience: str) -> WorkspaceGrant:
        """Return a current exact-principal grant, without leaking foreign existence."""
        if (
            not isinstance(principal, PrincipalKey)
            or not isinstance(audience, str)
            or not 1 <= len(audience) <= 2048
        ):
            raise PermissionError("Workspace grant unavailable")
        identity = _principal(principal), audience
        try:
            with self._connection() as connection:
                located = connection.execute(
                    "SELECT workspace FROM remote_grants WHERE principal_json=? AND audience=?",
                    identity,
                ).fetchone()
            if located is None:
                raise PermissionError("Workspace grant unavailable")
            with self._transaction(located[0]) as (connection, epoch):
                row = connection.execute(
                    "SELECT workspace,payload,sha256 FROM remote_grants WHERE principal_json=? AND audience=?",
                    identity,
                ).fetchone()
                if (
                    row is None
                    or row[0] != located[0]
                    or hashlib.sha256(row[1].encode()).hexdigest() != row[2]
                ):
                    raise PermissionError("Workspace grant unavailable")
                grant = _decode(row[1])
                if (
                    grant.authority.principal_key != principal
                    or grant.authority.audience != audience
                    or grant.authority.workspace_binding_id != located[0]
                    or grant.epoch != epoch
                ):
                    raise PermissionError("Workspace grant unavailable")
                return grant
        except Exception:
            raise PermissionError("Workspace grant unavailable") from None
