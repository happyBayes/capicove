"""Linux isolated remote workers: domains, egress, descendants and IPC (R3.04).

These run real namespaced subprocesses as distinct unprivileged identities. Every
assertion is an observation of the operating system (file access, sockets,
/proc membership, peer credentials), never a reported verdict of the worker.
"""

from __future__ import annotations

import os
import socket
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest


if TYPE_CHECKING:
    from capicove.runtime.remote_workers import WorkerDomain

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or os.geteuid() != 0,
    reason="R3.04 Linux isolation qualification requires Linux and a root supervisor",
)


REPOSITORY = Path(__file__).resolve().parents[4]
REGISTRY = "tests.remote_support.worker_operations"


def _deadline(seconds: float = 20) -> datetime:
    return datetime.now(UTC) + timedelta(seconds=seconds)


def _run(domain: WorkerDomain, operation: str, payload: dict[str, object], **options: Any) -> Any:
    from capicove.runtime.remote_workers import run_isolated

    return run_isolated(
        domain,
        operation,
        payload,
        registry=REGISTRY,
        import_roots=(REPOSITORY,),
        deadline=options.pop("deadline", _deadline()),
        **options,
    )


@pytest.fixture
def domains(tmp_path: Path) -> tuple[WorkerDomain, WorkerDomain]:
    from capicove.runtime.remote_workers import WorkerDomain

    tmp_path.chmod(0o755)
    first = WorkerDomain.provision("tenant-a", tmp_path / "domains", uid=65534, gid=65534)
    second = WorkerDomain.provision("tenant-b", tmp_path / "domains", uid=65533, gid=65533)
    return first, second


def _survivors(marker: str) -> list[str]:
    deadline = time.monotonic() + 5
    while True:
        found = []
        for entry in Path("/proc").iterdir():
            if entry.name.isdigit():
                try:
                    if marker.encode() in (entry / "cmdline").read_bytes():
                        found.append(entry.name)
                except OSError:
                    continue
        if not found or time.monotonic() > deadline:
            return found
        time.sleep(0.05)


def test_worker_runs_as_its_domain_identity_without_network_or_parent_secrets(
    domains: tuple[WorkerDomain, WorkerDomain], monkeypatch: pytest.MonkeyPatch
) -> None:
    first, _ = domains
    monkeypatch.setenv("CAPICOVE_PARENT_SECRET", "must-not-leak")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        result = _run(first, "probe", {"port": port})
    assert result.peer_uid == first.uid and result.output["uid"] == first.uid
    assert result.output["gid"] == first.gid
    assert result.output["pid"] == 1, "worker must be PID 1 of its own PID namespace"
    assert result.output["loopback_connect"] == "refused_or_unreachable"
    assert result.output["external_connect"] == "refused_or_unreachable"
    assert "CAPICOVE_PARENT_SECRET" not in result.output["environment"]
    assert result.output["home_writable"] is True
    assert result.output["root_writable"] is False
    security = result.output["security"]
    for field in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"):
        assert int(security[field], 16) == 0, field
    assert security["NoNewPrivs"] == "1"


def test_domains_cannot_read_each_other(domains: tuple[WorkerDomain, WorkerDomain]) -> None:
    first, second = domains
    secret = first.scratch / "secret.txt"
    written = _run(first, "write", {"name": "secret.txt", "text": "a-only"})
    assert written.output["written"] is True and secret.read_text() == "a-only"
    assert secret.stat().st_uid == first.uid and secret.stat().st_mode & 0o077 == 0
    for path in (str(secret), "/work/secret.txt", str(first.scratch)):
        attempt = _run(second, "read", {"path": path})
        assert attempt.output["status"] in {"absent", "denied"}, path
    own = _run(first, "read", {"path": "/work/secret.txt"})
    assert own.output == {"status": "ok", "text": "a-only"}
    assert (second.scratch.stat().st_mode & 0o777) == 0o700


def test_timeout_kills_every_descendant_of_the_operation(
    domains: tuple[WorkerDomain, WorkerDomain],
) -> None:
    from capicove.runtime.remote_workers import WorkerError

    first, _ = domains
    marker = f"capicove-descendant-{os.getpid()}-{time.monotonic_ns()}"
    with pytest.raises(WorkerError) as error:
        _run(first, "spawn_and_hang", {"marker": marker}, deadline=_deadline(2))
    assert error.value.code == "deadline_exceeded"
    assert _survivors(marker) == []


def test_forged_oversized_or_replayed_envelopes_are_rejected(
    domains: tuple[WorkerDomain, WorkerDomain],
) -> None:
    from capicove.runtime.remote_workers import WorkerError

    first, _ = domains
    for operation, code in (
        ("forge_envelope", "invalid_envelope"),
        ("oversized_output", "output_too_large"),
        ("crash", "worker_failed"),
    ):
        with pytest.raises(WorkerError) as error:
            _run(first, operation, {})
        assert error.value.code == code, operation


def test_unknown_operations_and_unsafe_domains_fail_before_spawning(tmp_path: Path) -> None:
    from capicove.runtime.remote_workers import WorkerDomain, WorkerError

    tmp_path.chmod(0o755)
    domain = WorkerDomain.provision("tenant-c", tmp_path / "d", uid=65532, gid=65532)
    with pytest.raises(WorkerError) as error:
        _run(domain, "os.system", {})
    assert error.value.code == "unknown_operation"
    with pytest.raises(ValueError):
        WorkerDomain.provision("tenant-root", tmp_path / "d", uid=0, gid=0)
    with pytest.raises(ValueError):
        WorkerDomain.provision("../escape", tmp_path / "d", uid=65531, gid=65531)


@pytest.mark.skipif(not Path("/usr/bin/ffmpeg").exists(), reason="system FFmpeg is not installed")
def test_ffmpeg_runs_inside_the_sandbox_and_dies_with_its_operation(
    domains: tuple[WorkerDomain, WorkerDomain],
) -> None:
    from capicove.runtime.remote_workers import WorkerError

    first, _ = domains
    assert _run(first, "ffmpeg", {}).output == {"returncode": 0}
    marker = f"capicove-ffmpeg-{os.getpid()}-{time.monotonic_ns()}"
    with pytest.raises(WorkerError) as error:
        _run(first, "ffmpeg", {"hang": True, "marker": marker}, deadline=_deadline(3))
    assert error.value.code == "deadline_exceeded"
    assert _survivors(marker) == []


@pytest.fixture
def cgroup_root() -> Path:
    configured = os.environ.get("CAPICOVE_WORKER_CGROUP_ROOT")
    if not configured:
        pytest.skip("set CAPICOVE_WORKER_CGROUP_ROOT to a delegated cgroup v2 directory")
    return Path(configured)


def test_cgroup_limits_bound_processes_and_memory(
    domains: tuple[WorkerDomain, WorkerDomain], cgroup_root: Path
) -> None:
    from capicove.runtime.remote_workers import CgroupLimits, WorkerError

    first, _ = domains
    limits = CgroupLimits(cgroup_root, pids_max=16, memory_max=256 * 1024 * 1024)
    forked = _run(first, "fork_many", {"count": 100}, cgroup=limits)
    assert forked.output["spawned"] < 16 and forked.output["failure"] == "BlockingIOError"
    small = _run(first, "allocate", {"bytes": 32 * 1024 * 1024}, cgroup=limits)
    assert small.output["allocated"] == 32 * 1024 * 1024
    with pytest.raises(WorkerError) as error:
        _run(first, "allocate", {"bytes": 768 * 1024 * 1024}, cgroup=limits)
    assert error.value.code == "worker_failed"
    assert [
        entry.name for entry in cgroup_root.iterdir() if entry.name.startswith("capicove-op-")
    ] == []
