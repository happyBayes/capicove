"""Run pytest with bounded, portable temporary storage.

The full suite writes fixture data, installed-wheel environments, and subprocess
temporary files. This runner gives each run one private directory, sizes the xdist
worker count from CPU, memory, and free disk, and cleans up the same way on Linux,
macOS, and Windows:

* A run that passes deletes its directory. Passing tests already release their
  ``tmp_path`` as they finish (``tmp_path_retention_policy = "failed"``).
* A run that fails or is interrupted keeps its directory as evidence, prints its
  path, and only the newest ``--keep-failed`` such runs are retained.
* Runs left behind by a crash or hard kill are removed by the next run.

Usage::

    python scripts/run_tests.py [--tmp-parent DIR] [--workers auto|N] -- [pytest args]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import FrameType


GIB = 1024**3
RUN_PREFIX = "capicove-pytest."
WHEEL_PREFIX = "capicove-pytest-wheels."
SHM = Path("/dev/shm")
SHM_MIN_BYTES = 8 * GIB
MIN_FREE_GIB = 2.0
WORKER_MEMORY_BYTES = GIB
WORKER_DISK_BYTES = GIB
DISK_RESERVE_BYTES = 2 * GIB
LEGACY_AGE = timedelta(hours=24)
STOP_GRACE_SECONDS = 30
_PREFIX = "capicove-tests:"


def _say(message: str) -> None:
    print(f"{_PREFIX} {message}", file=sys.stderr, flush=True)


# Platform facts.


def system_temp_dir() -> Path:
    """Return the platform temporary directory, honouring TMPDIR, TEMP, and TMP."""
    for name in ("TMPDIR", "TEMP", "TMP"):
        value = os.environ.get(name)
        if value and Path(value).is_dir():
            return Path(value)
    return Path(tempfile.gettempdir())


def free_bytes(path: Path) -> int | None:
    """Return free bytes on the filesystem holding ``path``, or None if unavailable."""
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return None


def available_memory() -> int | None:
    """Return available physical memory in bytes where the OS reports it."""
    if sys.platform == "win32":
        import ctypes

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
        return None
    try:
        with open("/proc/meminfo", encoding="ascii") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    # macOS does not expose available memory without extra tools; skip the cap.
    return None


def process_alive(pid: int) -> bool:
    """Return whether a process with ``pid`` is running."""
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def uv_cache_dir() -> Path | None:
    """Return uv's cache directory, or None when uv is unavailable."""
    try:
        completed = subprocess.run(
            ["uv", "cache", "dir"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = completed.stdout.strip()
    return Path(value) if completed.returncode == 0 and value else None


def _same_device(first: Path, second: Path) -> bool:
    try:
        return first.stat().st_dev == second.stat().st_dev
    except OSError:
        return False


# Decisions.


def choose_temp_parent(
    *,
    override: str | None,
    platform: str,
    shm: Path,
    fallback: Path,
    free_bytes: Callable[[Path], int | None],
    shm_min_bytes: int,
) -> Path:
    """Pick the run directory's parent: explicit, then RAM-backed, then disk.

    RAM-backed storage removes most fsync waits, but only Linux provides it at a
    fixed path, and it also consumes memory, so it needs ``shm_min_bytes`` free.
    """
    if override:
        return Path(override)
    if platform.startswith("linux"):
        available = free_bytes(shm)
        if available is not None and available >= shm_min_bytes:
            return shm
    return fallback


def worker_count(requested: str, *, cpus: int, available_memory: int | None, free_disk: int) -> int:
    """Return the xdist worker count; ``auto`` fits CPU, memory, and disk budgets."""
    if requested != "auto":
        try:
            count = int(requested)
        except ValueError:
            raise ValueError("--workers must be auto or a positive integer") from None
        if count < 1:
            raise ValueError("--workers must be auto or a positive integer")
        return count
    limits = [cpus, (free_disk - DISK_RESERVE_BYTES) // WORKER_DISK_BYTES]
    if available_memory is not None:
        limits.append(available_memory // WORKER_MEMORY_BYTES)
    return max(1, min(limits))


# Cleanup.


def remove_tree(path: Path, *, attempts: int = 3) -> list[str]:
    """Delete ``path``; return what could not be removed after retries.

    Windows keeps files open briefly after their process exits, so removal is
    retried, clearing read-only attributes on the way.
    """

    def _writable_retry(function: Callable[..., object], target: str, _: object) -> None:
        with contextlib.suppress(OSError):
            os.chmod(target, 0o700)
            function(target)

    for attempt in range(attempts):
        if not path.exists():
            return []
        try:
            if sys.version_info >= (3, 12):
                shutil.rmtree(path, onexc=_writable_retry)
            else:
                shutil.rmtree(path, onerror=_writable_retry)
        except OSError:
            if attempt + 1 < attempts:
                time.sleep(0.5 * (attempt + 1))
    return [str(path)] if path.exists() else []


def _record(directory: Path) -> dict[str, object] | None:
    try:
        value = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _remove_run(directory: Path, record: dict[str, object] | None) -> list[str]:
    leftovers: list[str] = []
    extra = record.get("extra_dirs", []) if record else []
    for value in extra if isinstance(extra, list) else []:
        leftovers.extend(remove_tree(Path(str(value))))
    leftovers.extend(remove_tree(directory))
    return leftovers


def sweep_stale_runs(
    parent: Path,
    *,
    alive: Callable[[int], bool],
    host: str,
    keep_failed: int,
    now: datetime,
) -> list[Path]:
    """Remove crashed, incompletely deleted, and surplus failed runs under ``parent``."""
    removed: list[Path] = []
    failed: list[tuple[str, Path, dict[str, object]]] = []
    try:
        candidates = sorted(parent.glob(RUN_PREFIX + "*"))
    except OSError:
        return removed
    for directory in candidates:
        if not directory.is_dir():
            continue
        record = _record(directory)
        if record is None:
            # Left by the earlier Makefile wrapper, or a run that never wrote its record.
            try:
                age = now - datetime.fromtimestamp(directory.stat().st_mtime, UTC)
            except OSError:
                continue
            if age > LEGACY_AGE and not _remove_run(directory, None):
                removed.append(directory)
            continue
        if record.get("host") != host:
            continue
        state = record.get("state")
        pid = record.get("pid")
        stale = state == "incomplete" or (
            state == "running" and isinstance(pid, int) and not alive(pid)
        )
        if stale:
            if not _remove_run(directory, record):
                removed.append(directory)
        elif state == "failed":
            failed.append((str(record.get("created", "")), directory, record))
    failed.sort(reverse=True)
    for _, directory, record in failed[max(keep_failed, 0) :]:
        if not _remove_run(directory, record):
            removed.append(directory)
    return removed


# The run.


class _Interrupted(Exception):
    pass


def _write_record(directory: Path, record: dict[str, object]) -> None:
    staging = directory / "run.json.partial"
    staging.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    staging.replace(directory / "run.json")


def _start(command: list[str], environment: dict[str, str]) -> subprocess.Popen[bytes]:
    if sys.platform == "win32":
        return subprocess.Popen(
            command, env=environment, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
    # A separate process group lets the runner stop xdist workers and anything a
    # test left running before it deletes their files.
    return subprocess.Popen(command, env=environment, start_new_session=True)


def _signal_group(process: subprocess.Popen[bytes], signum: int) -> None:
    if sys.platform == "win32":
        if signum == signal.SIGINT:
            with contextlib.suppress(OSError):
                process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(process.pid)],
                capture_output=True,
                check=False,
            )
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signum)


def _stop(process: subprocess.Popen[bytes]) -> None:
    """Interrupt pytest, then terminate and kill its process tree if it lingers."""
    for signum, grace in ((signal.SIGINT, STOP_GRACE_SECONDS), (signal.SIGTERM, 10)):
        _signal_group(process, signum)
        try:
            process.wait(grace)
            break
        except subprocess.TimeoutExpired:
            continue
    else:
        _signal_group(process, getattr(signal, "SIGKILL", signal.SIGTERM))
        process.wait(10)


def _reap_group(process: subprocess.Popen[bytes]) -> None:
    """Kill whatever is left in pytest's process group so it cannot hold files."""
    if sys.platform != "win32":
        _signal_group(process, signal.SIGKILL)


def _parse(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    arguments = list(argv)
    pytest_arguments: list[str] = []
    if "--" in arguments:
        index = arguments.index("--")
        arguments, pytest_arguments = arguments[:index], arguments[index + 1 :]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--tmp-parent",
        default=os.environ.get("CAPICOVE_TEST_TMP_PARENT") or None,
        help="Directory to hold the run directory (default: /dev/shm on Linux when "
        "large enough, otherwise the system temporary directory).",
    )
    parser.add_argument(
        "--workers",
        default=os.environ.get("CAPICOVE_TEST_WORKERS", "auto"),
        help="xdist workers: auto (fit CPU, memory, and disk) or a positive integer.",
    )
    parser.add_argument(
        "--keep-failed",
        type=int,
        default=1,
        help="Failed or interrupted run directories to keep (default: 1).",
    )
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=MIN_FREE_GIB,
        help=f"Refuse to start below this much free space (default: {MIN_FREE_GIB:g}).",
    )
    return parser.parse_args(arguments), pytest_arguments


def main(argv: Sequence[str] | None = None) -> int:
    """Run pytest in a private, budgeted, cleaned-up temporary directory."""
    options, pytest_arguments = _parse(sys.argv[1:] if argv is None else argv)
    disk_temp = system_temp_dir()
    parent = choose_temp_parent(
        override=options.tmp_parent,
        platform=sys.platform,
        shm=SHM,
        fallback=disk_temp,
        free_bytes=free_bytes,
        shm_min_bytes=SHM_MIN_BYTES,
    )
    host = socket.gethostname()
    now = datetime.now(UTC)
    for location in dict.fromkeys((parent, SHM, disk_temp)):
        if location.is_dir():
            for removed in sweep_stale_runs(
                location, alive=process_alive, host=host, keep_failed=options.keep_failed, now=now
            ):
                _say(f"removed stale run {removed}")
    available = free_bytes(parent)
    if available is None or available < options.min_free_gib * GIB:
        shown = "unknown" if available is None else f"{available / GIB:.1f} GiB"
        _say(
            f"{parent} has {shown} free; at least {options.min_free_gib:g} GiB is required. "
            "Free space or choose another directory with --tmp-parent "
            "(make test PYTEST_TMP_PARENT=DIR)."
        )
        return 2
    try:
        workers = worker_count(
            options.workers,
            cpus=os.cpu_count() or 1,
            available_memory=available_memory(),
            free_disk=available,
        )
    except ValueError as error:
        _say(str(error))
        return 2

    run = Path(tempfile.mkdtemp(prefix=RUN_PREFIX, dir=parent))
    record: dict[str, object] = {
        "pid": os.getpid(),
        "host": host,
        "state": "running",
        "created": now.isoformat(),
        "extra_dirs": [],
    }
    _write_record(run, record)
    base, scratch = run / "base", run / "tmp"
    scratch.mkdir()
    # Shared wheel environments hardlink from uv's cache only on its filesystem.
    wheels = run / "wheels"
    cache = uv_cache_dir()
    if cache is not None and not _same_device(run, cache) and _same_device(disk_temp, cache):
        wheels = Path(tempfile.mkdtemp(prefix=WHEEL_PREFIX, dir=disk_temp))
        record["extra_dirs"] = [str(wheels)]
        _write_record(run, record)
    wheels.mkdir(exist_ok=True)

    environment = dict(os.environ)
    for name in ("TMPDIR", "TEMP", "TMP"):
        environment[name] = str(scratch)
    environment["CAPICOVE_TEST_WHEEL_ROOT"] = str(wheels)
    command = [
        *(sys.executable, "-m", "pytest", "--basetemp", str(base)),
        *("-n", str(workers) if workers > 1 else "0"),
        *pytest_arguments,
    ]
    _say(f"run directory {run} ({available / GIB:.1f} GiB free), {workers} worker(s)")

    def _on_terminate(signum: int, frame: FrameType | None) -> None:
        raise _Interrupted

    previous = signal.signal(signal.SIGTERM, _on_terminate)
    process = _start(command, environment)
    interrupted = False
    try:
        code = process.wait()
    except (KeyboardInterrupt, _Interrupted):
        interrupted = True
        _stop(process)
        code = process.returncode if process.returncode not in (None, 0) else 130
    finally:
        signal.signal(signal.SIGTERM, previous)
        _reap_group(process)

    if code == 0 and not interrupted:
        leftovers = _remove_run(run, record)
        if leftovers:
            record["state"] = "incomplete"
            with contextlib.suppress(OSError):
                _write_record(run, record)
            _say(
                "tests passed, but these paths could not be deleted (the next run retries): "
                + ", ".join(leftovers)
            )
        return 0
    record["state"] = "failed"
    _write_record(run, record)
    # Apply the failed-run retention now, so at most ``--keep-failed`` remain.
    sweep_stale_runs(
        parent,
        alive=process_alive,
        host=host,
        keep_failed=options.keep_failed,
        now=datetime.now(UTC),
    )
    reason = "interrupted" if interrupted else f"exit status {code}"
    _say(f"pytest {reason}; kept failure evidence in {run}")
    return code


if __name__ == "__main__":
    sys.exit(main())
