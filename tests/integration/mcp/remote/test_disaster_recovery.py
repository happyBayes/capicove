"""R6.04 restore of a complete control-plane backup never revives revoked authority."""

from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path

import pytest

from capicove.contracts.remote import AuthorityKey, PrincipalKey
from capicove.runtime.remote_approvals import ApprovalService, ReviewerClaims
from capicove.runtime.remote_artifacts import ArtifactError, ArtifactService
from capicove.runtime.remote_audit import AuditLog
from capicove.runtime.remote_epochs import EpochStore
from capicove.runtime.remote_operations import OperationError, OperationService, OperationTarget
from capicove.runtime.remote_recovery import TombstoneJournal


ADMIN = "https://mcp.test/admin"
SCHEMA = {"type": "object", "additionalProperties": False, "properties": {}}
PERMISSIONS = frozenset({"operation:submit", "notes:write"})


def _authority(subject: str = "alice", audience: str = ADMIN) -> AuthorityKey:
    return AuthorityKey(
        PrincipalKey("https://issuer.test", "tenant", "console", "human", subject),
        "workspace-1",
        "grant-1",
        audience,
    )


def test_restored_epoch_store_cannot_revive_a_revoked_grant(tmp_path: Path) -> None:
    journal = tmp_path / "durable" / "revocations.jsonl"
    store = tmp_path / "control" / "epochs.sqlite3"
    store.parent.mkdir(mode=0o700)
    epochs = EpochStore(store, revocations=journal)
    assert epochs.current("workspace-1") == 0
    shutil.copy(store, tmp_path / "epochs-backup.sqlite3")
    assert epochs.revoke("workspace-1") == 1
    shutil.copy(tmp_path / "epochs-backup.sqlite3", store)
    restored = EpochStore(store, revocations=journal)
    assert restored.current("workspace-1") == 1
    with pytest.raises(PermissionError):
        with restored.permit("workspace-1", 0, phase="start"):
            pytest.fail("a stale backup revived the revoked epoch")
    lines = journal.read_text().splitlines()
    journal.write_text("\n".join([*lines, '{"workspace":"workspace-1","epoch":"x"}']) + "\n")
    with pytest.raises(RuntimeError, match="revocation journal"):
        EpochStore(store, revocations=journal)


def test_full_backup_restore_keeps_integrity_and_current_revocations(tmp_path: Path) -> None:
    durable = tmp_path / "durable"
    journal = TombstoneJournal(durable / "tombstones.jsonl")
    control = tmp_path / "control"
    control.mkdir(mode=0o700)

    def open_all() -> tuple[EpochStore, ArtifactService, OperationService, ApprovalService]:
        epochs = EpochStore(control / "epochs.sqlite3", revocations=durable / "revocations.jsonl")
        artifacts = ArtifactService(control / "artifacts", tombstones=journal)
        target = OperationTarget(
            "capicove.model.install",
            1,
            "mutation",
            SCHEMA,
            frozenset({"notes:write"}),
            dict,
            admin=True,
            requires_approval=True,
        )
        operations = OperationService(
            control / "operations.sqlite3",
            (target,),
            admin_audience=ADMIN,
            audit=AuditLog(control / "audit.jsonl"),
        )
        approvals = ApprovalService(
            operations, admin_audience=ADMIN, current_pins=dict, tombstones=journal
        )
        return epochs, artifacts, operations, approvals

    epochs, artifacts, operations, approvals = open_all()
    alice = _authority()
    payloads = {name: (name * 50).encode() for name in ("keep", "drop")}
    references = {}
    for name, data in payloads.items():
        status = artifacts.begin(
            alice,
            size=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            media_type="application/octet-stream",
            idempotency_key=name,
        )
        artifacts.write_chunk(alice, status.upload_id, 0, data)
        references[name] = artifacts.commit(alice, status.upload_id)
    plan = approvals.plan(alice, PERMISSIONS, "capicove.model.install", 1, {})
    approvals.review(
        _authority("bob"),
        frozenset({"approval:review"}),
        ReviewerClaims(time.time() - 5, frozenset({"mfa"})),
        plan.approval_id,
        plan.plan_sha256,
        approve=True,
    )
    assert operations.drain(timeout_seconds=0.1) == {"active": 0, "queued": 0}
    backup = tmp_path / "backup"
    shutil.copytree(control, backup)
    audit_count = AuditLog(control / "audit.jsonl").verify()

    # After the backup: delete one artifact, void the plan, revoke the workspace.
    artifacts.delete(alice, references["drop"].artifact_id)
    approvals.void(plan.approval_id, reason="operator withdrew the plan")
    epochs.revoke("workspace-1")

    shutil.rmtree(control)
    shutil.copytree(backup, control)
    epochs, artifacts, operations, approvals = open_all()
    assert epochs.current("workspace-1") == 1
    kept = artifacts.read_chunk(alice, references["keep"].artifact_id, 0, 4096)
    assert hashlib.sha256(kept).hexdigest() == references["keep"].sha256
    with pytest.raises(ArtifactError):
        artifacts.stat(alice, references["drop"].artifact_id)
    assert approvals.status(plan.approval_id).state == "void"
    with pytest.raises(OperationError):
        approvals.execute(alice, PERMISSIONS, plan.approval_id, "after-restore")
    assert AuditLog(control / "audit.jsonl").verify() == audit_count
    # The restored store keeps the drained admission state until an operator resumes.
    with pytest.raises(OperationError) as error:
        operations.submit(alice, PERMISSIONS, "capicove.model.install", 1, {}, "x")
    assert error.value.code in {"draining", "approval_required"}


def test_launch_wires_the_revocation_journal_into_the_endpoint(tmp_path: Path) -> None:
    import json

    from capicove.runtime import remote_launch
    from tests.integration.mcp.remote.test_launch import _private, _setup

    journal = tmp_path / "durable" / "revocations.jsonl"
    launch_path = _setup(tmp_path, revocation_journal=str(journal))
    launch = remote_launch.load_launch(launch_path)
    assert launch.revocation_journal == journal
    launch.epoch_store.parent.mkdir(mode=0o700, parents=True)
    EpochStore(launch.epoch_store).current("workspace-1")
    backup = tmp_path / "epochs-backup.sqlite3"
    shutil.copy(launch.epoch_store, backup)
    EpochStore(launch.epoch_store, revocations=journal).revoke("workspace-1")
    shutil.copy(backup, launch.epoch_store)
    assert EpochStore(launch.epoch_store).current("workspace-1") == 0
    remote_launch.compose_remote_app(launch)
    assert EpochStore(launch.epoch_store).current("workspace-1") == 1
    _private(launch_path, {**json.loads(launch_path.read_text()), "revocation_journal": "rel/j"})
    with pytest.raises(ValueError):
        remote_launch.load_launch(launch_path)
