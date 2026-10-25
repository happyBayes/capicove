"""R3.02 durable operations: idempotent admission, fenced execution and recovery."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from capicove.contracts.remote import AuthorityKey, PrincipalKey
from capicove.runtime.remote_operations import (
    OperationError,
    OperationService,
    OperationTarget,
)


def _authority(subject: str = "alice", audience: str = "https://mcp.test/mcp") -> AuthorityKey:
    return AuthorityKey(
        PrincipalKey("https://issuer.test", "tenant", "app-a", "human", subject),
        "workspace-1",
        "grant-1",
        audience,
    )


class Effects:
    """An independent observer of committed side effects."""

    def __init__(self) -> None:
        self.calls: list[Mapping[str, Any]] = []
        self.lock = threading.Lock()

    def handler(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.calls.append(dict(arguments))
            return {"count": len(self.calls)}


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["text"],
    "properties": {"text": {"type": "string", "maxLength": 64}},
}


def _service(root: Path, effects: Effects, **options: Any) -> OperationService:
    targets = (
        OperationTarget(
            "capicove.notes.append",
            1,
            "mutation",
            SCHEMA,
            frozenset({"notes:write"}),
            effects.handler,
        ),
        OperationTarget(
            "capicove.admin.reindex",
            1,
            "mutation",
            SCHEMA,
            frozenset({"admin:write"}),
            effects.handler,
            admin=True,
        ),
    )
    return OperationService(root / "control.sqlite3", targets, **options)


PERMISSIONS = frozenset({"operation:submit", "notes:write", "admin:write"})


def test_one_hundred_simultaneous_same_key_submits_admit_one_effect(tmp_path: Path) -> None:
    effects = Effects()
    references: list[str] = []
    errors: list[BaseException] = []
    start = threading.Barrier(100)

    def submit() -> None:
        # Each thread uses its own service instance and SQLite connections.
        service = _service(tmp_path, effects)
        start.wait()
        try:
            reference = service.submit(
                _authority(), PERMISSIONS, "capicove.notes.append", 1, {"text": "hello"}, "key-1"
            )
            references.append(reference.operation_id)
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=submit) for _ in range(100)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(references) == 100 and len(set(references)) == 1
    service = _service(tmp_path, effects)
    claimed = service.claim("worker-1", lease_seconds=30)
    assert claimed is not None and service.claim("worker-2", lease_seconds=30) is None
    service.complete(claimed.operation_id, claimed.fence, service.execute(claimed))
    assert len(effects.calls) == 1
    result = service.result(_authority(), PERMISSIONS, references[0])
    assert result == {"state": "succeeded", "output": {"count": 1}}
    with pytest.raises(OperationError) as error:
        service.submit(
            _authority(), PERMISSIONS, "capicove.notes.append", 1, {"text": "changed"}, "key-1"
        )
    assert error.value.code == "idempotency_conflict"
    other = service.submit(
        _authority("bob"), PERMISSIONS, "capicove.notes.append", 1, {"text": "hello"}, "key-1"
    )
    assert other.operation_id != references[0]
    with pytest.raises(OperationError) as error:
        service.result(_authority("bob"), PERMISSIONS, references[0])
    assert error.value.code == "not_found"


@pytest.mark.parametrize(
    ("tool_id", "arguments", "permissions", "audience", "code"),
    [
        ("capicove.operation.submit", {"text": "x"}, PERMISSIONS, None, "unavailable"),
        ("capicove.approval.approve", {"text": "x"}, PERMISSIONS, None, "unavailable"),
        ("capicove.hidden.tool", {"text": "x"}, PERMISSIONS, None, "unavailable"),
        ("capicove.admin.reindex", {"text": "x"}, PERMISSIONS, None, "unavailable"),
        (
            "capicove.notes.append",
            {"text": "x", "provider": "https://evil"},
            PERMISSIONS,
            None,
            "invalid_input",
        ),
        ("capicove.notes.append", {"text": "x" * 65}, PERMISSIONS, None, "invalid_input"),
        (
            "capicove.notes.append",
            {"text": "x"},
            frozenset({"notes:write"}),
            None,
            "permission_denied",
        ),
        (
            "capicove.notes.append",
            {"text": "x"},
            frozenset({"operation:submit"}),
            None,
            "permission_denied",
        ),
    ],
)
def test_forbidden_targets_and_forged_inputs_have_zero_effect(
    tmp_path: Path,
    tool_id: str,
    arguments: dict[str, Any],
    permissions: frozenset[str],
    audience: str | None,
    code: str,
) -> None:
    effects = Effects()
    service = _service(tmp_path, effects)
    with pytest.raises(OperationError) as error:
        service.submit(_authority(), permissions, tool_id, 1, arguments, "key")
    assert error.value.code == code
    assert service.count() == 0 and service.claim("worker", lease_seconds=5) is None
    assert effects.calls == []


def test_admin_targets_require_the_admin_audience(tmp_path: Path) -> None:
    effects = Effects()
    service = _service(tmp_path, effects, admin_audience="https://mcp.test/admin")
    admin = _authority(audience="https://mcp.test/admin")
    reference = service.submit(admin, PERMISSIONS, "capicove.admin.reindex", 1, {"text": "x"}, "a")
    assert reference.state == "queued"
    with pytest.raises(OperationError):
        service.submit(admin, PERMISSIONS, "capicove.notes.append", 1, {"text": "x"}, "n")


def test_crash_points_reconcile_without_duplicate_or_fabricated_effects(tmp_path: Path) -> None:
    effects = Effects()
    clock = [1000.0]
    service = _service(tmp_path, effects, clock=lambda: clock[0])
    queued = service.submit(
        _authority(), PERMISSIONS, "capicove.notes.append", 1, {"text": "a"}, "q"
    )
    started = service.submit(
        _authority(), PERMISSIONS, "capicove.notes.append", 1, {"text": "b"}, "s"
    )
    # Crash after admission, before start: still queued, runs exactly once later.
    first = service.claim("worker-1", lease_seconds=10)
    assert first is not None and first.operation_id == queued.operation_id
    # Crash after start, before completion: the worker vanishes holding a lease.
    second = service.claim("worker-1", lease_seconds=10)
    assert second is not None and second.operation_id == started.operation_id
    output = service.execute(second)
    assert len(effects.calls) == 1
    clock[0] += 11
    restarted = _service(tmp_path, effects, clock=lambda: clock[0])
    assert restarted.recover() == {"requeued": 0, "outcome_unknown": 2}
    states = {
        reference.operation_id: restarted.status(
            _authority(), PERMISSIONS, reference.operation_id
        ).state
        for reference in (queued, started)
    }
    assert set(states.values()) == {"outcome_unknown"}
    # Unknown outcomes never auto-retry, and a stale fence cannot publish.
    assert restarted.claim("worker-2", lease_seconds=10) is None
    with pytest.raises(OperationError) as error:
        restarted.complete(second.operation_id, second.fence, output)
    assert error.value.code == "stale_fence"
    resolved = restarted.reconcile(
        started.operation_id, "succeeded", output, evidence="effect store shows count 1"
    )
    assert resolved.state == "succeeded"
    history = restarted.history(started.operation_id)
    assert [entry["state"] for entry in history][-2:] == ["outcome_unknown", "succeeded"]
    assert len(effects.calls) == 1


def test_queued_work_requeues_after_restart_and_cancellation_is_truthful(tmp_path: Path) -> None:
    effects = Effects()
    service = _service(tmp_path, effects)
    a = service.submit(_authority(), PERMISSIONS, "capicove.notes.append", 1, {"text": "a"}, "a")
    b = service.submit(_authority(), PERMISSIONS, "capicove.notes.append", 1, {"text": "b"}, "b")
    restarted = _service(tmp_path, effects)
    assert restarted.recover() == {"requeued": 0, "outcome_unknown": 0}
    assert restarted.cancel(_authority(), PERMISSIONS, b.operation_id).state == "cancelled"
    claimed = restarted.claim("worker", lease_seconds=30)
    assert claimed is not None and claimed.operation_id == a.operation_id
    requested = restarted.cancel(_authority(), PERMISSIONS, a.operation_id)
    assert requested.state == "cancellation_requested"
    restarted.complete(claimed.operation_id, claimed.fence, restarted.execute(claimed))
    # A cancellation racing a committed success reports the observed success.
    assert restarted.status(_authority(), PERMISSIONS, a.operation_id).state == "succeeded"
    assert restarted.cancel(_authority(), PERMISSIONS, a.operation_id).state == "succeeded"
    assert restarted.claim("worker", lease_seconds=30) is None
    assert len(effects.calls) == 1


def test_revoked_authority_cannot_start_or_publish(tmp_path: Path) -> None:
    effects = Effects()
    revoked: set[AuthorityKey] = set()

    def authorize(authority: AuthorityKey) -> None:
        if authority in revoked:
            raise PermissionError("revoked")

    service = _service(tmp_path, effects, authorize=authorize)
    queued = service.submit(
        _authority(), PERMISSIONS, "capicove.notes.append", 1, {"text": "a"}, "a"
    )
    revoked.add(_authority())
    assert service.claim("worker", lease_seconds=10) is None
    revoked.clear()
    state = service.status(_authority(), PERMISSIONS, queued.operation_id).state
    assert state == "cancelled"
    assert effects.calls == []


def test_application_queue_limit_rejects_before_admission(tmp_path: Path) -> None:
    effects = Effects()
    service = _service(tmp_path, effects, max_queued_per_application=3)
    for index in range(3):
        service.submit(
            _authority(), PERMISSIONS, "capicove.notes.append", 1, {"text": str(index)}, f"k{index}"
        )
    with pytest.raises(OperationError) as error:
        service.submit(
            _authority("bob"), PERMISSIONS, "capicove.notes.append", 1, {"text": "x"}, "k"
        )
    assert error.value.code == "quota_exceeded"
    assert service.count() == 3
