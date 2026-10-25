"""Claimed operations execute once in an isolated worker and publish under their fence."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from capicove.contracts.remote import AuthorityKey, PrincipalKey
from capicove.runtime.remote_execution import IsolatedExecutor
from capicove.runtime.remote_operations import OperationService, OperationTarget
from capicove.runtime.remote_workers import WorkerDomain


pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or os.geteuid() != 0,
    reason="isolated execution requires Linux and a root supervisor",
)
REPOSITORY = Path(__file__).resolve().parents[4]
REGISTRY = "tests.remote_support.worker_operations"
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string", "maxLength": 32},
        "text": {"type": "string", "maxLength": 64},
        "marker": {"type": "string", "maxLength": 64},
    },
}
PERMISSIONS = frozenset({"operation:submit", "notes:write"})


def _authority() -> AuthorityKey:
    return AuthorityKey(
        PrincipalKey("https://issuer.test", "tenant", "app", "human", "alice"),
        "workspace-1",
        "grant-1",
        "https://mcp.test/mcp",
    )


def _world(tmp_path: Path) -> tuple[OperationService, WorkerDomain, IsolatedExecutor]:

    def refuse(arguments: object) -> dict[str, object]:
        raise AssertionError("isolated targets must never run in the gateway process")

    targets = tuple(
        OperationTarget(
            f"capicove.notes.{name}", 1, "mutation", SCHEMA, frozenset({"notes:write"}), refuse
        )
        for name in ("write", "crash", "spawn-and-hang")
    )
    service = OperationService(tmp_path / "control.sqlite3", targets)
    tmp_path.chmod(0o755)
    domain = WorkerDomain.provision("tenant-a", tmp_path / "domains", uid=65534, gid=65534)
    executor = IsolatedExecutor(
        service,
        domain,
        {
            "capicove.notes.write": "write",
            "capicove.notes.crash": "crash",
            "capicove.notes.spawn-and-hang": "spawn_and_hang",
        },
        registry=REGISTRY,
        import_roots=(REPOSITORY,),
    )
    return service, domain, executor


def test_claimed_operation_runs_once_in_its_domain_and_publishes(tmp_path: Path) -> None:
    service, domain, executor = _world(tmp_path)
    reference = service.submit(
        _authority(),
        PERMISSIONS,
        "capicove.notes.write",
        1,
        {"name": "note.txt", "text": "isolated"},
        "k1",
    )
    outcome = executor.run_next("worker-1", deadline_seconds=20)
    assert outcome is not None
    assert outcome.operation_id == reference.operation_id and outcome.state == "succeeded"
    assert (domain.scratch / "note.txt").read_text() == "isolated"
    assert (domain.scratch / "note.txt").stat().st_uid == 65534
    result = service.result(_authority(), PERMISSIONS, reference.operation_id)
    assert result == {"state": "succeeded", "output": {"written": True}}
    assert executor.run_next("worker-1", deadline_seconds=20) is None


def test_reported_failures_fail_while_crashes_and_timeouts_stay_unknown(tmp_path: Path) -> None:
    service, _, executor = _world(tmp_path)
    reported = service.submit(
        _authority(), PERMISSIONS, "capicove.notes.write", 1, {}, "missing-fields"
    )
    crashed = service.submit(_authority(), PERMISSIONS, "capicove.notes.crash", 1, {}, "crash")
    hung = service.submit(
        _authority(),
        PERMISSIONS,
        "capicove.notes.spawn-and-hang",
        1,
        {"marker": f"capicove-op-hang-{os.getpid()}"},
        "hang",
    )
    first = executor.run_next("worker-1", deadline_seconds=20)
    assert first is not None
    assert first.operation_id == reported.operation_id and first.state == "failed"
    # A crash or timeout may follow partial effects, so the outcome is unknown.
    second = executor.run_next("worker-1", deadline_seconds=20)
    assert second is not None
    assert second.operation_id == crashed.operation_id and second.state == "outcome_unknown"
    third = executor.run_next("worker-1", deadline_seconds=2)
    assert third is not None
    assert third.operation_id == hung.operation_id and third.state == "outcome_unknown"
    # Unknown outcomes are never claimed again automatically.
    assert executor.run_next("worker-1", deadline_seconds=2) is None
    history = [entry["state"] for entry in service.history(hung.operation_id)]
    assert history[-2:] == ["running", "outcome_unknown"]
