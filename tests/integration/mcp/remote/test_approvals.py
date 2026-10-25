"""R3.03 admin plan/review/execute: distinct fresh human review bound to exact pins."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from capicove.contracts.remote import AuthorityKey, PrincipalKey
from capicove.runtime.remote_approvals import ApprovalService, ReviewerClaims
from capicove.runtime.remote_operations import OperationError, OperationService, OperationTarget


ADMIN = "https://mcp.test/admin"
APP = "https://mcp.test/mcp"
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["model"],
    "properties": {"model": {"type": "string", "maxLength": 64}},
}


def _authority(
    subject: str, *, application: str = "admin-console", audience: str = ADMIN, kind: str = "human"
) -> AuthorityKey:
    return AuthorityKey(
        PrincipalKey("https://issuer.test", "tenant", application, kind, subject),  # type: ignore[arg-type]
        "workspace-1",
        "grant-1",
        audience,
    )


class World:
    def __init__(self, root: Path) -> None:
        self.now = 10_000.0
        self.pins: dict[str, str] = {
            "policy_revision": "7",
            "model_revision": "sha256:abc",
            "egress_profile": "none",
            "active_pointer": "generation-3",
        }
        self.effects: list[Mapping[str, Any]] = []
        target = OperationTarget(
            "capicove.model.install",
            1,
            "mutation",
            SCHEMA,
            frozenset({"model:install"}),
            lambda arguments: self.effects.append(dict(arguments)) or {"installed": True},  # type: ignore[func-returns-value]
            admin=True,
            requires_approval=True,
        )
        self.operations = OperationService(
            root / "control.sqlite3", (target,), admin_audience=ADMIN, clock=lambda: self.now
        )
        self.approvals = ApprovalService(
            self.operations,
            admin_audience=ADMIN,
            current_pins=lambda: dict(self.pins),
            clock=lambda: self.now,
        )

    def fresh(self, subject: str = "reviewer") -> ReviewerClaims:
        return ReviewerClaims(auth_time=self.now - 30, methods=frozenset({"pwd", "mfa"}))


PERMISSIONS = frozenset({"operation:submit", "model:install"})
REVIEW = frozenset({"approval:review"})


def _plan(world: World) -> Any:
    return world.approvals.plan(
        _authority("alice"), PERMISSIONS, "capicove.model.install", 1, {"model": "m1"}
    )


def test_distinct_fresh_human_review_then_single_execution(tmp_path: Path) -> None:
    world = World(tmp_path)
    plan = _plan(world)
    assert plan.state == "pending"
    with pytest.raises(OperationError) as error:
        world.operations.submit(
            _authority("alice"), PERMISSIONS, "capicove.model.install", 1, {"model": "m1"}, "k"
        )
    assert error.value.code == "approval_required"
    approved = world.approvals.review(
        _authority("bob"), REVIEW, world.fresh(), plan.approval_id, plan.plan_sha256, approve=True
    )
    assert approved.state == "approved" and approved.reviewer == "bob"
    service = _authority("deployer", application="automation", kind="service")
    first = world.approvals.execute(service, PERMISSIONS, plan.approval_id, "exec-1")
    assert first.state == "queued"
    assert world.approvals.execute(service, PERMISSIONS, plan.approval_id, "exec-1") == first
    with pytest.raises(OperationError) as error:
        world.approvals.execute(service, PERMISSIONS, plan.approval_id, "exec-2")
    assert error.value.code == "approval_consumed"
    claim = world.operations.claim("worker", lease_seconds=30)
    assert claim is not None
    world.operations.complete(claim.operation_id, claim.fence, world.operations.execute(claim))
    assert world.effects == [{"model": "m1"}]
    states = [entry["state"] for entry in world.approvals.history(plan.approval_id)]
    assert states == ["pending", "approved", "reserved", "consumed"]


@pytest.mark.parametrize(
    "case",
    [
        "self_second_client",
        "service_reviewer",
        "wrong_audience",
        "no_mfa",
        "stale_mfa",
        "missing_permission",
        "changed_digest",
    ],
)
def test_invalid_reviews_are_rejected_without_state_change(tmp_path: Path, case: str) -> None:
    world = World(tmp_path)
    plan = _plan(world)
    reviewer = _authority("bob")
    claims = world.fresh()
    permissions = REVIEW
    digest = plan.plan_sha256
    if case == "self_second_client":
        reviewer = _authority("alice", application="other-client")
    elif case == "service_reviewer":
        reviewer = _authority("robot", kind="service")
    elif case == "wrong_audience":
        reviewer = _authority("bob", audience=APP)
    elif case == "no_mfa":
        claims = ReviewerClaims(auth_time=world.now - 30, methods=frozenset({"pwd"}))
    elif case == "stale_mfa":
        claims = ReviewerClaims(auth_time=world.now - 3600, methods=frozenset({"mfa"}))
    elif case == "missing_permission":
        permissions = frozenset({"operation:submit"})
    elif case == "changed_digest":
        digest = "0" * 64
    with pytest.raises(OperationError):
        world.approvals.review(
            reviewer, permissions, claims, plan.approval_id, digest, approve=True
        )
    assert world.approvals.status(plan.approval_id).state == "pending"


@pytest.mark.parametrize("change", ["policy_revision", "model_revision", "active_pointer"])
def test_stale_plans_are_voided_and_never_recycled(tmp_path: Path, change: str) -> None:
    world = World(tmp_path)
    plan = _plan(world)
    world.approvals.review(
        _authority("bob"), REVIEW, world.fresh(), plan.approval_id, plan.plan_sha256, approve=True
    )
    world.pins[change] = "changed"
    with pytest.raises(OperationError) as error:
        world.approvals.execute(_authority("alice"), PERMISSIONS, plan.approval_id, "x")
    assert error.value.code == "approval_stale"
    assert world.approvals.status(plan.approval_id).state == "void"
    world.pins[change] = "restored-but-different"
    with pytest.raises(OperationError):
        world.approvals.execute(_authority("alice"), PERMISSIONS, plan.approval_id, "y")
    assert world.operations.count() == 0 and world.effects == []


def test_expired_review_window_and_rejected_plans_cannot_execute(tmp_path: Path) -> None:
    world = World(tmp_path)
    plan = _plan(world)
    world.now += 24 * 3600 + 1
    with pytest.raises(OperationError) as error:
        world.approvals.review(
            _authority("bob"),
            REVIEW,
            world.fresh(),
            plan.approval_id,
            plan.plan_sha256,
            approve=True,
        )
    assert error.value.code == "approval_expired"
    second = _plan(world)
    world.approvals.review(
        _authority("bob"),
        REVIEW,
        world.fresh(),
        second.approval_id,
        second.plan_sha256,
        approve=False,
    )
    with pytest.raises(OperationError):
        world.approvals.execute(_authority("alice"), PERMISSIONS, second.approval_id, "z")
    assert world.operations.count() == 0


def test_concurrent_executions_reserve_exactly_one_operation(tmp_path: Path) -> None:
    world = World(tmp_path)
    plan = _plan(world)
    world.approvals.review(
        _authority("bob"), REVIEW, world.fresh(), plan.approval_id, plan.plan_sha256, approve=True
    )
    results: list[str] = []
    gate = threading.Barrier(20)

    def run() -> None:
        approvals = ApprovalService(
            OperationService(
                tmp_path / "control.sqlite3",
                world.operations._targets.values(),  # type: ignore[arg-type]
                admin_audience=ADMIN,
                clock=lambda: world.now,
            ),
            admin_audience=ADMIN,
            current_pins=lambda: dict(world.pins),
            clock=lambda: world.now,
        )
        gate.wait()
        try:
            results.append(
                approvals.execute(
                    _authority("alice"), PERMISSIONS, plan.approval_id, "same"
                ).operation_id
            )
        except OperationError as error:
            results.append(error.code)

    threads = [threading.Thread(target=run) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(set(results)) == 1 and results[0].startswith("operation:")
    assert world.operations.count() == 1
