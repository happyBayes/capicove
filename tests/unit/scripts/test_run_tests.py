"""scripts/run_tests.py: portable temporary storage, budgets, and cleanup."""

from __future__ import annotations

import importlib
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest


runner = importlib.import_module("scripts.run_tests")
REPOSITORY = Path(__file__).resolve().parents[3]
SCRIPT = REPOSITORY / "scripts" / "run_tests.py"
GIB = 1024**3


# Temporary-directory selection.


def test_explicit_parent_wins(tmp_path: Path) -> None:
    chosen = runner.choose_temp_parent(
        override=str(tmp_path),
        platform="linux",
        shm=tmp_path / "shm",
        fallback=tmp_path / "fallback",
        free_bytes=lambda _: 100 * GIB,
        shm_min_bytes=8 * GIB,
    )
    assert chosen == tmp_path


@pytest.mark.parametrize(
    ("platform", "shm_free", "expected"),
    [
        ("linux", 9 * GIB, "shm"),
        ("linux", 3 * GIB, "fallback"),
        ("linux", None, "fallback"),
        ("darwin", 100 * GIB, "fallback"),
        ("win32", 100 * GIB, "fallback"),
    ],
)
def test_ram_backed_storage_is_used_only_on_linux_with_enough_space(
    tmp_path: Path, platform: str, shm_free: int | None, expected: str
) -> None:
    chosen = runner.choose_temp_parent(
        override=None,
        platform=platform,
        shm=tmp_path / "shm",
        fallback=tmp_path / "fallback",
        free_bytes=lambda path: shm_free if path.name == "shm" else 100 * GIB,
        shm_min_bytes=8 * GIB,
    )
    assert chosen == tmp_path / expected


def test_default_fallback_honours_platform_temp_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TEMP" if sys.platform == "win32" else "TMPDIR", str(tmp_path))
    assert runner.system_temp_dir() == tmp_path


# Worker budget.


@pytest.mark.parametrize(
    ("cpus", "memory", "disk", "expected"),
    [
        (8, 64 * GIB, 64 * GIB, 8),
        (8, 3 * GIB, 64 * GIB, 3),
        (8, None, 64 * GIB, 8),
        (8, 64 * GIB, 5 * GIB, 3),
        (8, GIB // 2, GIB, 1),
    ],
)
def test_automatic_workers_fit_cpu_memory_and_disk(
    cpus: int, memory: int | None, disk: int, expected: int
) -> None:
    assert (
        runner.worker_count("auto", cpus=cpus, available_memory=memory, free_disk=disk) == expected
    )


def test_explicit_worker_count_is_used_verbatim() -> None:
    assert runner.worker_count("2", cpus=64, available_memory=GIB, free_disk=GIB) == 2
    with pytest.raises(ValueError, match="workers"):
        runner.worker_count("zero", cpus=4, available_memory=None, free_disk=GIB)


# Stale-run cleanup.


def _run_directory(
    parent: Path,
    name: str,
    *,
    pid: int,
    state: str,
    created: datetime,
    host: str = "this-host",
    extra: Path | None = None,
) -> Path:
    directory = parent / f"capicove-pytest.{name}"
    (directory / "base").mkdir(parents=True)
    record = {
        "pid": pid,
        "host": host,
        "state": state,
        "created": created.isoformat(),
        "extra_dirs": [str(extra)] if extra is not None else [],
    }
    (directory / "run.json").write_text(json.dumps(record))
    return directory


def test_sweep_removes_crashed_runs_and_old_failures_but_keeps_live_and_recent(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 10, 8, 12, tzinfo=UTC)
    extra = tmp_path / "elsewhere" / "wheels"
    extra.mkdir(parents=True)
    crashed = _run_directory(tmp_path, "crashed", pid=11, state="running", created=now, extra=extra)
    live = _run_directory(tmp_path, "live", pid=22, state="running", created=now)
    foreign = _run_directory(
        tmp_path, "foreign", pid=11, state="running", created=now, host="other-host"
    )
    old_failure = _run_directory(
        tmp_path, "old", pid=11, state="failed", created=now - timedelta(hours=2)
    )
    new_failure = _run_directory(tmp_path, "new", pid=11, state="failed", created=now)
    legacy_old = tmp_path / "capicove-pytest.legacy1"
    legacy_old.mkdir()
    os.utime(legacy_old, (0, 0))
    legacy_recent = tmp_path / "capicove-pytest.legacy2"
    legacy_recent.mkdir()
    unrelated = tmp_path / "something-else"
    unrelated.mkdir()

    removed = runner.sweep_stale_runs(
        tmp_path, alive=lambda pid: pid == 22, host="this-host", keep_failed=1, now=now
    )

    assert set(removed) == {crashed, old_failure, legacy_old}
    assert not crashed.exists() and not extra.exists() and not old_failure.exists()
    assert live.exists() and foreign.exists() and new_failure.exists()
    assert legacy_recent.exists() and unrelated.exists()


def test_incomplete_removal_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "stubborn"
    target.mkdir()

    def refuse(path: Path, **_: object) -> None:
        raise PermissionError("in use")

    monkeypatch.setattr(runner.shutil, "rmtree", refuse)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)

    assert runner.remove_tree(target) == [str(target)]


def test_process_liveness_detects_this_process_and_a_reaped_child() -> None:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(30)
    assert runner.process_alive(os.getpid())
    assert not runner.process_alive(child.pid)


# The runner end to end.


def _suite(directory: Path, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "test_sample.py").write_text(textwrap.dedent(body))
    return directory


def _run_runner(parent: Path, suite: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            *(sys.executable, str(SCRIPT), "--tmp-parent", str(parent), "--workers", "1"),
            *extra,
            "--",
            *("-c", str(REPOSITORY / "pyproject.toml"), "--rootdir", str(suite)),
            *("-p", "no:cacheprovider", "-q", str(suite / "test_sample.py")),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=suite,
    )


def _runs(parent: Path) -> list[Path]:
    return sorted(parent.glob("capicove-pytest.*"))


PASS_AND_FAIL = """
    import os
    import tempfile
    from pathlib import Path

    def test_passes(tmp_path):
        (tmp_path / "artifact").write_text("passed")
        run = Path(tempfile.gettempdir()).parent
        assert run.name.startswith("capicove-pytest.")
        assert Path(os.environ["CAPICOVE_TEST_WHEEL_ROOT"]).is_dir()

    def test_fails(tmp_path):
        (tmp_path / "artifact").write_text("failed")
        assert False
"""
PASSING = """
def test_passes(tmp_path):
    (tmp_path / "artifact").write_text("passed")
"""


@pytest.mark.slow
def test_failed_run_keeps_only_failure_evidence_and_passing_run_cleans_up(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()

    failed = _run_runner(parent, _suite(tmp_path / "failing", PASS_AND_FAIL))

    assert failed.returncode == 1, failed.stdout + failed.stderr
    (kept,) = _runs(parent)
    assert str(kept) in failed.stderr
    assert json.loads((kept / "run.json").read_text())["state"] == "failed"
    assert (kept / "base" / "test_fails0" / "artifact").read_text() == "failed"
    assert not (kept / "base" / "test_passes0").exists()

    passed = _run_runner(parent, _suite(tmp_path / "passing", PASSING))

    assert passed.returncode == 0, passed.stdout + passed.stderr
    assert _runs(parent) == [kept]

    again = _run_runner(parent, _suite(tmp_path / "failing", PASS_AND_FAIL))

    assert again.returncode == 1
    (newest,) = _runs(parent)
    assert newest != kept


@pytest.mark.slow
def test_runner_refuses_a_parent_without_enough_free_space(tmp_path: Path) -> None:
    suite = _suite(tmp_path / "suite", PASSING)

    refused = _run_runner(tmp_path, suite, "--min-free-gib", "1000000")

    assert refused.returncode == 2
    assert "free" in refused.stderr and "--tmp-parent" in refused.stderr
    assert _runs(tmp_path) == []


@pytest.mark.slow
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal delivery")
def test_terminated_runner_stops_pytest_and_keeps_its_evidence(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    marker = tmp_path / "started"
    suite = _suite(
        tmp_path / "suite",
        f"""
        import time
        from pathlib import Path

        def test_waits(tmp_path):
            Path({str(marker)!r}).write_text(str(__import__("os").getpid()))
            time.sleep(120)
        """,
    )
    process = subprocess.Popen(
        [
            *(sys.executable, str(SCRIPT), "--tmp-parent", str(parent), "--workers", "1", "--"),
            *("-c", str(REPOSITORY / "pyproject.toml"), "--rootdir", str(suite)),
            *("-p", "no:cacheprovider", "-q", str(suite / "test_sample.py")),
        ],
        cwd=suite,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 60
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert marker.exists()
    test_pid = int(marker.read_text())

    process.send_signal(signal.SIGTERM)
    _, stderr = process.communicate(timeout=60)

    assert process.returncode != 0
    assert not runner.process_alive(test_pid)
    (kept,) = _runs(parent)
    assert json.loads((kept / "run.json").read_text())["state"] == "failed"
    assert str(kept) in stderr
