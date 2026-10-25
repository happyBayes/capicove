"""Root stage inside fresh namespaces: build a minimal root, drop privileges, exec.

Runs as PID 1 of a new PID namespace (``unshare --fork --pid --mount --net --ipc
--uts --kill-child``) and is the only code that executes with privileges inside
the namespaces. It mounts a private tmpfs root that contains read-only system
libraries, the interpreter prefix and explicitly listed import roots, plus the
domain scratch directory at ``/work``. It then irrevocably drops to the domain
identity (no groups, empty capability bounding set, ``no_new_privs``) and execs
the worker entry. Linux only; uses ``ctypes`` because Python 3.11 lacks mount APIs.
"""

from __future__ import annotations

import ctypes
import json
import os
import resource
import sys
from pathlib import Path


MS_RDONLY = 1
MS_NOSUID = 2
MS_NODEV = 4
MS_NOEXEC = 8
MS_REMOUNT = 32
MS_BIND = 4096
MS_REC = 16384
MS_PRIVATE = 1 << 18
PR_CAPBSET_DROP = 24
PR_SET_NO_NEW_PRIVS = 38
PR_CAP_AMBIENT = 47
PR_CAP_AMBIENT_CLEAR_ALL = 4
SYSTEM_PATHS = ("usr", "bin", "lib", "lib64", "lib32", "libx32", "sbin")
DEVICES = ("null", "zero", "urandom", "random")
# Loader inputs only; account, credential and host configuration stay invisible.
SYSTEM_ETC = ("ld.so.cache", "alternatives")
WORKER_BOOT = (
    "import sys;n=int(sys.argv[3]);sys.path[:0]=sys.argv[4:4+n];del sys.argv[3:];"
    "import runpy;runpy.run_module('capicove.runtime._remote.worker_entry',"
    "run_name='__main__',alter_sys=True)"
)

_libc = ctypes.CDLL(None, use_errno=True)


def _check(result: int, action: str) -> None:
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, f"{action}: {os.strerror(error)}")


def _mount(source: str | None, target: str, fstype: str | None, flags: int, data: str = "") -> None:
    _check(
        _libc.mount(
            source.encode() if source is not None else None,
            target.encode(),
            fstype.encode() if fstype is not None else None,
            ctypes.c_ulong(flags),
            data.encode() if data else None,
        ),
        f"mount {target}",
    )


def _bind(source: str, target: Path, *, writable: bool = False, device: bool = False) -> None:
    _mount(source, str(target), None, MS_BIND | MS_REC)
    flags = MS_REMOUNT | MS_BIND | MS_NOSUID
    if not writable:
        flags |= MS_RDONLY
    if not device:
        flags |= MS_NODEV
    _mount(None, str(target), None, flags)


def _build_root(spec: dict[str, object], new_root: Path) -> None:
    _mount(None, "/", None, MS_REC | MS_PRIVATE)
    _mount("tmpfs", str(new_root), "tmpfs", MS_NOSUID | MS_NODEV, "mode=0755,size=16m")
    for name in SYSTEM_PATHS:
        host = Path("/") / name
        inside = new_root / name
        if host.is_symlink():
            inside.symlink_to(os.readlink(host))
        elif host.is_dir():
            inside.mkdir()
            _bind(str(host), inside)
    etc = new_root / "etc"
    etc.mkdir()
    for name in SYSTEM_ETC:
        host = Path("/etc") / name
        if host.is_dir() and not host.is_symlink():
            (etc / name).mkdir()
        elif host.is_file():
            (etc / name).touch()
        else:
            continue
        _bind(str(host), etc / name)
    opt = new_root / "opt"
    opt.mkdir()
    for index, source in enumerate([spec["prefix"], *spec["import_roots"]]):  # type: ignore[misc]
        target = opt / ("python" if index == 0 else f"import-{index - 1}")
        target.mkdir()
        _bind(str(source), target)
    (new_root / "work").mkdir()
    _bind(str(spec["scratch"]), new_root / "work", writable=True)
    (new_root / "proc").mkdir()
    _mount("proc", str(new_root / "proc"), "proc", MS_NOSUID | MS_NODEV | MS_NOEXEC)
    dev = new_root / "dev"
    dev.mkdir()
    for name in DEVICES:
        (dev / name).touch()
        _bind(f"/dev/{name}", dev / name, writable=True, device=True)
    (new_root / "tmp").mkdir()
    _mount("tmpfs", str(new_root / "tmp"), "tmpfs", MS_NOSUID | MS_NODEV, "mode=1777,size=16m")
    _mount(None, str(new_root), None, MS_REMOUNT | MS_RDONLY | MS_NOSUID | MS_NODEV)


def _drop(uid: int, gid: int, limits: dict[str, int]) -> None:
    last = int(Path("/proc/sys/kernel/cap_last_cap").read_text())
    for capability in range(last + 1):
        _check(_libc.prctl(PR_CAPBSET_DROP, capability, 0, 0, 0), "drop bounding capability")
    _check(_libc.prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL, 0, 0, 0), "clear ambient")
    for name, value in limits.items():
        resource.setrlimit(getattr(resource, name), (value, value))
    os.setgroups([])
    os.setresgid(gid, gid, gid)
    os.setresuid(uid, uid, uid)
    _check(_libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), "set no_new_privs")
    if os.getresuid() != (uid, uid, uid) or os.getgroups():
        raise OSError("privilege drop did not take effect")


def main(raw: str) -> int:
    """Build the root from the JSON spec, drop to the domain identity and exec."""
    spec = json.loads(raw)
    if spec.get("cgroup"):
        # Join the per-call leaf before any descendant exists.
        Path(spec["cgroup"], "cgroup.procs").write_text("0", encoding="ascii")
    new_root = Path(spec["mount_point"])
    _build_root(spec, new_root)
    os.chroot(new_root)
    os.chdir("/work")
    _drop(int(spec["uid"]), int(spec["gid"]), dict(spec["limits"]))
    inside = [f"/opt/import-{index}" for index in range(len(spec["import_roots"]))]
    channel = int(spec["channel"])
    os.set_inheritable(channel, True)
    os.closerange(3, channel)
    os.closerange(channel + 1, 4096)
    executable = "/opt/python/" + str(spec["executable"])
    argv = [executable, "-I", "-B", "-c", WORKER_BOOT, str(channel), str(spec["registry"])]
    argv += [str(len(inside)), *inside]
    environment = {"PATH": "/usr/bin:/bin", "HOME": "/work", "LANG": "C.UTF-8", "TMPDIR": "/tmp"}
    os.execve(executable, argv, environment)


if __name__ == "__main__":  # pragma: no cover - entered through the supervisor bootstrap
    raise SystemExit(main(sys.argv[1]))
