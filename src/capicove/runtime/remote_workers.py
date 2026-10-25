"""Isolated execution of remote operations in per-domain Linux sandboxes (R3.04).

Each call runs exactly one registered operation in a fresh process tree with its
own PID, mount, network, IPC and UTS namespaces, as the unprivileged identity of
its execution domain, inside a minimal read-only root whose only writable path is
the domain scratch directory. There is no network interface except a down
loopback. The supervisor alone selects the operation registry and import roots;
payloads are data and cannot name code.

The response travels over a private ``SOCK_SEQPACKET`` pair. Identity is taken
from kernel-attached ``SCM_CREDENTIALS``, never from the payload. The envelope is
bound to the request by a per-call fence and HMAC key, size-limited, and rejected
unless all three checks agree. On deadline, the namespace's process group is
killed; the kernel then terminates every remaining descendant in the PID
namespace. The supervisor requires Linux, root, ``unshare`` and ``chroot``
support; it fails closed with ``isolation_unavailable`` otherwise.
"""

from __future__ import annotations

import hmac
import importlib
import json
import os
import re
import secrets
import shutil
import signal
import socket
import struct
import subprocess
import sys
import sysconfig
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from capicove.runtime._remote.worker_entry import MAX_MESSAGE_BYTES, canonical, sign


DEFAULT_REGISTRY = "capicove.runtime._remote.worker_operations"
_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
_MODULE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
_MAX_IDENTITY = 2**31 - 2
_SO_PASSCRED = 16
_SCM_CREDENTIALS = 2
_SO_SNDBUFFORCE = 32
_SO_RCVBUFFORCE = 33
_BUFFER = 4 * MAX_MESSAGE_BYTES
_BOOT = (
    "import sys;sys.path.insert(0,sys.argv[1]);"
    "from capicove.runtime._remote.worker_sandbox import main;raise SystemExit(main(sys.argv[2]))"
)
_LIMITS = {
    "RLIMIT_CORE": 0,
    "RLIMIT_NOFILE": 64,
    "RLIMIT_FSIZE": 64 * 1024 * 1024,
    "RLIMIT_AS": 2 * 1024 * 1024 * 1024,
}
_FAILURE_CODES = frozenset({"unknown_operation", "output_too_large"})


class WorkerError(RuntimeError):
    """A typed isolated-execution failure; ``code`` is stable and public."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


@dataclass(frozen=True)
class WorkerDomain:
    """One execution domain: a dedicated unprivileged identity and scratch tree."""

    name: str
    uid: int
    gid: int
    root: Path
    scratch: Path

    @classmethod
    def provision(cls, name: str, root: Path, *, uid: int, gid: int) -> WorkerDomain:
        """Create or verify the domain scratch (owner-only) under a supervisor root."""
        if not _NAME.fullmatch(name):
            raise ValueError("domain name must be 1-63 lowercase letters, digits or hyphens")
        for value in (uid, gid):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("domain identities must be integers")
            if not 0 < value <= _MAX_IDENTITY:
                raise ValueError("domain identities must be unprivileged and non-root")
        root = Path(root)
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        _require_supervisor_directory(root)
        (root / ".mnt").mkdir(mode=0o700, exist_ok=True)
        _require_supervisor_directory(root / ".mnt")
        scratch = root / name
        scratch.mkdir(mode=0o700, exist_ok=True)
        status = scratch.lstat()
        if not scratch.is_dir() or scratch.is_symlink():
            raise ValueError("domain scratch must be a real directory")
        if (status.st_uid, status.st_gid) not in {(uid, gid), (os.geteuid(), os.getegid())}:
            raise ValueError("domain scratch is owned by another identity")
        os.chown(scratch, uid, gid, follow_symlinks=False)
        scratch.chmod(0o700)
        return cls(name=name, uid=uid, gid=gid, root=root, scratch=scratch)


@dataclass(frozen=True)
class CgroupLimits:
    """Per-call cgroup v2 limits under a delegated, supervisor-owned cgroup."""

    root: Path
    pids_max: int = 64
    memory_max: int = 1024 * 1024 * 1024


@dataclass(frozen=True)
class WorkerResult:
    """A verified response: kernel peer identity plus the operation output."""

    operation: str
    output: Mapping[str, Any]
    peer_uid: int
    peer_gid: int
    seconds: float


def _require_supervisor_directory(path: Path) -> None:
    status = path.lstat()
    if path.is_symlink() or not path.is_dir() or status.st_uid != os.geteuid():
        raise ValueError("domain roots must be real directories owned by the supervisor")
    if status.st_mode & 0o022:
        raise ValueError("domain roots must not be group- or world-writable")


def _registered(registry: str) -> frozenset[str]:
    if not _MODULE.fullmatch(registry):
        raise WorkerError("unknown_operation", "registry must be a module name")
    operations = getattr(importlib.import_module(registry), "OPERATIONS", None)
    if not isinstance(operations, Mapping):
        raise WorkerError("unknown_operation", "registry has no OPERATIONS mapping")
    return frozenset(operations)


def _package_parent() -> Path:
    import capicove

    return Path(capicove.__file__).resolve().parent.parent


def _interpreter() -> tuple[Path, str]:
    prefix = Path(sys.base_prefix).resolve()
    candidates = [
        Path(sys.executable).resolve(),
        prefix / "bin" / f"python{sys.version_info.major}.{sys.version_info.minor}",
    ]
    for executable in candidates:
        try:
            return prefix, str(executable.relative_to(prefix))
        except ValueError:
            continue
    raise WorkerError("isolation_unavailable", "interpreter is outside its base prefix")


def _seconds_left(deadline: datetime) -> float:
    if deadline.tzinfo is None:
        raise ValueError("deadline must be timezone-aware")
    return (deadline - datetime.now(UTC)).total_seconds()


def _receive(channel: socket.socket, deadline: datetime) -> tuple[bytes, int, int, bool]:
    remaining = _seconds_left(deadline)
    if remaining <= 0:
        raise WorkerError("deadline_exceeded")
    channel.settimeout(remaining)
    try:
        data, ancillary, flags, _ = channel.recvmsg(
            MAX_MESSAGE_BYTES + 1, socket.CMSG_SPACE(struct.calcsize("3i"))
        )
    except TimeoutError as error:
        raise WorkerError("deadline_exceeded") from error
    except OSError as error:
        raise WorkerError("worker_failed", "response channel failed") from error
    uid = gid = -1
    for level, kind, value in ancillary:
        if level == socket.SOL_SOCKET and kind == _SCM_CREDENTIALS:
            _, uid, gid = struct.unpack("3i", value[: struct.calcsize("3i")])
    truncated = bool(flags & socket.MSG_TRUNC) or len(data) > MAX_MESSAGE_BYTES
    return data, uid, gid, truncated


def _verify(
    data: bytes, *, operation: str, fence: str, key: bytes
) -> tuple[str, Mapping[str, Any]]:
    try:
        envelope = json.loads(data)
    except (UnicodeDecodeError, ValueError) as error:
        raise WorkerError("invalid_envelope", "response is not JSON") from error
    if not isinstance(envelope, dict) or set(envelope) != {
        "fence",
        "operation",
        "status",
        "output",
        "mac",
    }:
        raise WorkerError("invalid_envelope", "response fields are not exact")
    mac = envelope.pop("mac")
    if not isinstance(mac, str) or not hmac.compare_digest(mac, sign(key, envelope)):
        raise WorkerError("invalid_envelope", "response MAC does not verify")
    if envelope["fence"] != fence or envelope["operation"] != operation:
        raise WorkerError("invalid_envelope", "response is not bound to this request")
    status, output = envelope["status"], envelope["output"]
    if status not in {"succeeded", "failed"} or not isinstance(output, dict):
        raise WorkerError("invalid_envelope", "response status or output is malformed")
    return status, output


def _open_cgroup(limits: CgroupLimits, fence: str) -> Path:
    try:
        controllers = set((limits.root / "cgroup.subtree_control").read_text().split())
    except OSError as error:
        raise WorkerError("isolation_unavailable", "cgroup root is not readable") from error
    if not {"pids", "memory"} <= controllers:
        raise WorkerError("isolation_unavailable", "cgroup root lacks pids/memory control")
    leaf = limits.root / f"capicove-op-{fence}"
    leaf.mkdir(mode=0o700)
    (leaf / "pids.max").write_text(str(limits.pids_max))
    (leaf / "memory.max").write_text(str(limits.memory_max))
    if (leaf / "memory.swap.max").exists():
        (leaf / "memory.swap.max").write_text("0")
    return leaf


def _kill(process: subprocess.Popen[bytes], leaf: Path | None) -> None:
    if leaf is not None:
        (leaf / "cgroup.kill").write_text("1")
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()
    if leaf is not None:
        for _attempt in range(200):
            try:
                leaf.rmdir()
                return
            except OSError:
                time.sleep(0.01)
        raise WorkerError("worker_failed", "operation cgroup did not drain")


def run_isolated(
    domain: WorkerDomain,
    operation: str,
    payload: Mapping[str, Any],
    *,
    deadline: datetime,
    registry: str = DEFAULT_REGISTRY,
    import_roots: Sequence[Path] = (),
    cgroup: CgroupLimits | None = None,
) -> WorkerResult:
    """Run one registered operation in a fresh sandbox of ``domain``.

    Args:
        domain: The provisioned execution domain.
        operation: A name present in the registry's ``OPERATIONS`` mapping.
        payload: JSON data handed to the operation.
        deadline: Absolute timezone-aware deadline for the whole call.
        registry: Supervisor-selected operation registry module.
        import_roots: Extra read-only import roots mounted for the registry.
        cgroup: Optional delegated cgroup v2 root; each call then runs in its own
            leaf with ``pids.max``/``memory.max`` and is torn down by ``cgroup.kill``.

    Returns:
        The verified result with kernel-attested peer identity.

    Raises:
        WorkerError: ``unknown_operation``, ``deadline_exceeded``,
            ``invalid_envelope``, ``output_too_large``, ``identity_mismatch``,
            ``operation_failed``, ``worker_failed`` or ``isolation_unavailable``.
    """
    if not isinstance(operation, str) or operation not in _registered(registry):
        raise WorkerError("unknown_operation")
    request = canonical(dict(payload))
    if len(request) > MAX_MESSAGE_BYTES // 2:
        raise WorkerError("invalid_input", "payload exceeds the request bound")
    unshare = shutil.which("unshare")
    if not sys.platform.startswith("linux") or os.geteuid() != 0 or unshare is None:
        raise WorkerError("isolation_unavailable", "requires Linux, root and unshare")
    prefix, executable = _interpreter()
    package = _package_parent()
    roots = [package, *(Path(root).resolve() for root in import_roots)]
    purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
    if purelib.is_dir():
        roots.append(purelib)
    key = secrets.token_bytes(32)
    fence = secrets.token_hex(16)
    leaf = _open_cgroup(cgroup, fence) if cgroup is not None else None
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    started = time.monotonic()
    try:
        parent.setsockopt(socket.SOL_SOCKET, _SO_PASSCRED, 1)
        parent.setsockopt(socket.SOL_SOCKET, _SO_RCVBUFFORCE, _BUFFER)
        parent.setsockopt(socket.SOL_SOCKET, _SO_SNDBUFFORCE, _BUFFER)
        child.setsockopt(socket.SOL_SOCKET, _SO_SNDBUFFORCE, _BUFFER)
        child.setsockopt(socket.SOL_SOCKET, _SO_RCVBUFFORCE, _BUFFER)
        spec = {
            "mount_point": str(domain.root / ".mnt"),
            "scratch": str(domain.scratch),
            "uid": domain.uid,
            "gid": domain.gid,
            "prefix": str(prefix),
            "executable": executable,
            "import_roots": [str(root) for root in roots],
            "registry": registry,
            "channel": child.fileno(),
            "limits": _LIMITS,
            "cgroup": str(leaf) if leaf is not None else None,
        }
        command = [
            unshare,
            "--fork",
            "--pid",
            "--mount",
            "--net",
            "--ipc",
            "--uts",
            "--kill-child",
            "--",
            str(prefix / executable),
            "-I",
            "-B",
            "-c",
            _BOOT,
            str(package),
            json.dumps(spec),
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            pass_fds=(child.fileno(),),
            env={"PATH": "/usr/bin:/bin"},
            start_new_session=True,
        )
    except OSError as error:
        parent.close()
        child.close()
        if leaf is not None:
            leaf.rmdir()
        raise WorkerError("isolation_unavailable", "sandbox could not start") from error
    child.close()
    try:
        parent.sendall(
            canonical(
                {"operation": operation, "payload": dict(payload), "fence": fence, "key": key.hex()}
            )
        )
        data, uid, gid, truncated = _receive(parent, deadline)
        if truncated:
            raise WorkerError("output_too_large")
        if not data:
            raise WorkerError("worker_failed", "worker exited without a response")
        if (uid, gid) != (domain.uid, domain.gid):
            raise WorkerError("identity_mismatch", "peer credentials do not match the domain")
        status, output = _verify(data, operation=operation, fence=fence, key=key)
    finally:
        parent.close()
        _kill(process, leaf)
    if status == "failed":
        failure = str(output.get("error"))
        raise WorkerError(failure if failure in _FAILURE_CODES else "operation_failed", failure)
    return WorkerResult(
        operation=operation,
        output=output,
        peer_uid=uid,
        peer_gid=gid,
        seconds=time.monotonic() - started,
    )
