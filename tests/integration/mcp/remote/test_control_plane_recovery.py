"""R3.05 aggregate limits, audit durability, restore gating, retention and drain."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import pytest

from capicove.contracts.remote import AuthorityKey, PrincipalKey
from capicove.runtime.remote_artifacts import ArtifactError, ArtifactService
from capicove.runtime.remote_audit import AuditLog
from capicove.runtime.remote_operations import OperationError, OperationService, OperationTarget
from capicove.runtime.remote_recovery import TombstoneJournal


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"text": {"type": "string", "maxLength": 32}},
}
PERMISSIONS = frozenset({"operation:submit", "notes:write"})


def _authority(application: str = "app-a", subject: str = "alice") -> AuthorityKey:
    return AuthorityKey(
        PrincipalKey("https://issuer.test", "tenant", application, "human", subject),
        "workspace-1",
        "grant-1",
        "https://mcp.test/mcp",
    )


def _operations(root: Path, **options: Any) -> OperationService:
    target = OperationTarget(
        "capicove.notes.append", 1, "mutation", SCHEMA, frozenset({"notes:write"}), dict
    )
    return OperationService(root / "control.sqlite3", (target,), **options)


def _upload(service: ArtifactService, authority: AuthorityKey, data: bytes, key: str) -> Any:
    status = service.begin(
        authority,
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        media_type="application/octet-stream",
        idempotency_key=key,
    )
    service.write_chunk(authority, status.upload_id, 0, data)
    return service.commit(authority, status.upload_id)


def test_aggregate_tenant_limits_prevent_multi_application_evasion(tmp_path: Path) -> None:
    service = _operations(tmp_path, max_queued_per_application=5, max_queued_per_tenant=6)
    for index in range(4):
        service.submit(
            _authority("app-a"), PERMISSIONS, "capicove.notes.append", 1, {}, f"a{index}"
        )
    service.submit(_authority("app-b"), PERMISSIONS, "capicove.notes.append", 1, {}, "b0")
    service.submit(_authority("app-c"), PERMISSIONS, "capicove.notes.append", 1, {}, "c0")
    with pytest.raises(OperationError) as error:
        service.submit(_authority("app-d"), PERMISSIONS, "capicove.notes.append", 1, {}, "d0")
    assert error.value.code == "quota_exceeded" and service.count() == 6
    artifacts = ArtifactService(
        tmp_path / "artifacts", application_quota_bytes=100, tenant_quota_bytes=150
    )
    _upload(artifacts, _authority("app-a"), b"a" * 90, "a")
    with pytest.raises(ArtifactError) as artifact_error:
        artifacts.begin(
            _authority("app-b"),
            size=90,
            sha256="0" * 64,
            media_type="text/plain",
            idempotency_key="b",
        )
    assert artifact_error.value.code == "quota_exceeded"


def test_audit_failure_rejects_mutations_but_cancellation_works(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit" / "audit.jsonl")
    service = _operations(tmp_path, audit=audit)
    queued = service.submit(_authority(), PERMISSIONS, "capicove.notes.append", 1, {}, "q")
    assert audit.verify() == 2
    # Simulate an audit-device failure that also applies to root: the log path
    # temporarily resolves to a directory, so every append raises OSError.
    saved = audit.path.read_bytes()
    audit.path.unlink()
    audit.path.mkdir()
    try:
        assert not audit.writable()
        with pytest.raises(OperationError) as error:
            service.submit(_authority(), PERMISSIONS, "capicove.notes.append", 1, {}, "x")
        assert error.value.code == "audit_unavailable"
        assert service.count() == 1
        cancelled = service.cancel(_authority(), PERMISSIONS, queued.operation_id)
        assert cancelled.state == "cancelled"
    finally:
        audit.path.rmdir()
        audit.path.write_bytes(saved)
        audit.path.chmod(0o600)
    # The cancellation event waited in the outbox and is exported exactly once.
    assert service.drain_audit() == 1
    assert audit.verify() == 3


def test_audit_chain_detects_tampering(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    for index in range(5):
        audit.append({"operation_id": f"operation:{index}", "event": "queued"})
    checkpoint = audit.checkpoint()
    assert audit.verify() == 5 and len(checkpoint) == 64
    lines = audit.path.read_bytes().splitlines(keepends=True)
    lines[2] = lines[2].replace(b"queued", b"forged")
    audit.path.write_bytes(b"".join(lines))
    with pytest.raises(ValueError, match="audit chain"):
        audit.verify()
    audit.path.write_bytes(b"".join(lines[:2] + lines[3:]))
    with pytest.raises(ValueError, match="audit chain"):
        audit.verify()


def test_restore_cannot_resurrect_deleted_artifacts_or_void_approvals(tmp_path: Path) -> None:
    from capicove.runtime.remote_approvals import ApprovalService, ReviewerClaims

    journal = TombstoneJournal(tmp_path / "tombstones" / "journal.jsonl")
    store = tmp_path / "artifacts"
    artifacts = ArtifactService(store, tombstones=journal)
    alice = _authority()
    reference = _upload(artifacts, alice, b"to be deleted", "k")
    backup = tmp_path / "backup"
    shutil.copytree(store, backup)
    artifacts.delete(alice, reference.artifact_id)
    shutil.rmtree(store)
    shutil.copytree(backup, store)
    restored = ArtifactService(store, tombstones=journal)
    with pytest.raises(ArtifactError) as error:
        restored.stat(alice, reference.artifact_id)
    assert error.value.code == "not_found"
    assert not any(path.is_file() for path in (store / "artifacts").iterdir())
    assert restored.usage(alice) == {"reserved": 0, "stored": 0}

    admin = "https://mcp.test/admin"
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
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    operations = OperationService(control / "control.sqlite3", (target,), admin_audience=admin)
    approvals = ApprovalService(
        operations, admin_audience=admin, current_pins=dict, tombstones=journal
    )
    requester = AuthorityKey(alice.principal_key, "workspace-1", "grant-1", admin)
    reviewer = AuthorityKey(
        PrincipalKey("https://issuer.test", "tenant", "console", "human", "bob"),
        "workspace-1",
        "grant-1",
        admin,
    )
    plan = approvals.plan(requester, PERMISSIONS, "capicove.model.install", 1, {})
    import time

    approvals.review(
        reviewer,
        frozenset({"approval:review"}),
        ReviewerClaims(time.time() - 10, frozenset({"mfa"})),
        plan.approval_id,
        plan.plan_sha256,
        approve=True,
    )
    shutil.copy(control / "control.sqlite3", tmp_path / "control-backup.sqlite3")
    approvals.void(plan.approval_id, reason="operator revoked the plan")
    shutil.copy(tmp_path / "control-backup.sqlite3", control / "control.sqlite3")
    reopened = ApprovalService(
        OperationService(control / "control.sqlite3", (target,), admin_audience=admin),
        admin_audience=admin,
        current_pins=dict,
        tombstones=journal,
    )
    assert reopened.status(plan.approval_id).state == "void"
    with pytest.raises(OperationError):
        reopened.execute(requester, PERMISSIONS, plan.approval_id, "after-restore")


def test_retention_purges_only_expired_unreferenced_store_objects(tmp_path: Path) -> None:
    clock = [1_000.0]
    service = ArtifactService(tmp_path / "store", clock=lambda: clock[0])
    alice = _authority()
    old = _upload(service, alice, b"old", "old")
    clock[0] += 100
    new = _upload(service, alice, b"new", "new")
    pending = service.begin(
        alice, size=5, sha256="0" * 64, media_type="text/plain", idempotency_key="pending"
    )
    outside = tmp_path / "canonical-source.md"
    outside.write_text("never touched")
    clock[0] += 24 * 3600 + 1
    assert service.purge(retention_seconds=24 * 3600 + 50) == {"artifacts": 1, "uploads": 1}
    assert [reference.artifact_id for reference in service.list_artifacts(alice)] == [
        new.artifact_id
    ]
    with pytest.raises(ArtifactError):
        service.stat(alice, old.artifact_id)
    assert service.status(alice, pending.upload_id).state == "expired"
    assert service.usage(alice) == {"reserved": 0, "stored": 3}
    assert outside.read_text() == "never touched"


def test_drain_stops_new_work_while_cancellation_remains_available(tmp_path: Path) -> None:
    service = _operations(tmp_path)
    first = service.submit(_authority(), PERMISSIONS, "capicove.notes.append", 1, {}, "1")
    second = service.submit(_authority(), PERMISSIONS, "capicove.notes.append", 1, {}, "2")
    claim = service.claim("worker", lease_seconds=60)
    assert claim is not None and claim.operation_id == first.operation_id
    report = service.drain(timeout_seconds=0.2)
    assert report == {"active": 1, "queued": 1}
    assert service.claim("worker", lease_seconds=60) is None
    with pytest.raises(OperationError) as error:
        service.submit(_authority(), PERMISSIONS, "capicove.notes.append", 1, {}, "3")
    assert error.value.code == "draining"
    assert service.cancel(_authority(), PERMISSIONS, second.operation_id).state == "cancelled"
    service.complete(claim.operation_id, claim.fence, {})
    assert service.status(_authority(), PERMISSIONS, first.operation_id).state == "succeeded"
