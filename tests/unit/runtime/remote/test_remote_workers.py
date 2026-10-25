"""Fail-closed admission and envelope verification of isolated remote workers."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from capicove.runtime import remote_workers
from capicove.runtime._remote.worker_entry import canonical, sign
from capicove.runtime.remote_workers import WorkerDomain, WorkerError, run_isolated


def _deadline() -> datetime:
    return datetime.now(UTC) + timedelta(seconds=5)


def _domain(tmp_path: Path) -> WorkerDomain:
    return WorkerDomain(
        name="tenant", uid=65534, gid=65534, root=tmp_path, scratch=tmp_path / "tenant"
    )


@pytest.mark.parametrize(
    ("name", "uid", "gid"),
    [
        ("Upper", 1000, 1000),
        ("", 1000, 1000),
        ("a/b", 1000, 1000),
        ("ok", 0, 1000),
        ("ok", 1000, 0),
    ],
)
def test_provision_rejects_unsafe_names_and_privileged_identities(
    tmp_path: Path, name: str, uid: int, gid: int
) -> None:
    with pytest.raises(ValueError):
        WorkerDomain.provision(name, tmp_path / "root", uid=uid, gid=gid)


def test_provision_rejects_shared_writable_roots(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    root.chmod(0o777)
    with pytest.raises(ValueError):
        WorkerDomain.provision("tenant", root, uid=os.geteuid() or 1000, gid=os.getegid() or 1000)


def test_unknown_operations_and_registries_fail_before_spawning(tmp_path: Path) -> None:
    for operation, registry in (
        ("os.system", remote_workers.DEFAULT_REGISTRY),
        ("probe", "os; import subprocess"),
        ("probe", "json"),
    ):
        with pytest.raises(WorkerError) as error:
            run_isolated(_domain(tmp_path), operation, {}, deadline=_deadline(), registry=registry)
        assert error.value.code == "unknown_operation"


def test_isolation_is_unavailable_without_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    with pytest.raises(WorkerError) as error:
        run_isolated(_domain(tmp_path), "noop", {}, deadline=_deadline())
    assert error.value.code == "isolation_unavailable"


def test_oversized_requests_and_naive_deadlines_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(WorkerError) as error:
        run_isolated(_domain(tmp_path), "noop", {"blob": "x" * 600_000}, deadline=_deadline())
    assert error.value.code == "invalid_input"
    with pytest.raises(ValueError):
        remote_workers._seconds_left(datetime.now())


def _envelope(key: bytes, **changes: object) -> dict[str, object]:
    envelope: dict[str, object] = {
        "fence": "f",
        "operation": "noop",
        "status": "succeeded",
        "output": {},
    }
    envelope.update(changes)
    envelope["mac"] = sign(key, envelope)
    return envelope


def test_envelopes_must_be_exact_signed_and_request_bound() -> None:
    key = b"k" * 32
    good = canonical(_envelope(key))
    assert remote_workers._verify(good, operation="noop", fence="f", key=key) == ("succeeded", {})
    tampered = _envelope(key)
    tampered["output"] = {"forged": True}
    cases = [
        b"not json",
        json.dumps([1]).encode(),
        canonical({**_envelope(key), "extra": 1}),
        canonical(tampered),
        canonical(_envelope(b"x" * 32)),
        canonical(_envelope(key, fence="replayed")),
        canonical(_envelope(key, operation="probe")),
        canonical(_envelope(key, status="maybe")),
        canonical(_envelope(key, output=[1])),
    ]
    for data in cases:
        with pytest.raises(WorkerError) as error:
            remote_workers._verify(data, operation="noop", fence="f", key=key)
        assert error.value.code == "invalid_envelope", data
