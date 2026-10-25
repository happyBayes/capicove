"""Process lifecycle for `capicove http serve` (RFC-0020, ADR-0025).

Order of startup: dependencies, configuration and catalog, workspace lock,
worker, listener, token, readiness record. Shutdown stops admission, stops the
worker's process group, removes this server's token, and releases the lock.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import secrets
import select
import signal
import socket
import stat
import subprocess  # nosec B404
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import IO, TYPE_CHECKING

from capicove.api.workspace import (
    ToolInvocationFailed,
    WorkerTimeout,
    WorkerUnavailable,
    create_workspace_app,
)
from capicove.mcp.bindings import InvalidToolArguments


if TYPE_CHECKING:
    pass

LOOPBACK = "127.0.0.1"
_STARTUP_SECONDS = 60.0
_TERMINATE_GRACE_SECONDS = 5.0
_GRACEFUL_SHUTDOWN_SECONDS = 3


class ServeError(RuntimeError):
    """Report a stable startup failure without token or request content."""


class WorkerProcess:
    """Own one worker process and every process it starts, across restarts.

    The worker receives the configuration bytes frozen at server startup, not a
    path, so a restart can never pick up a replaced configuration. With child
    reaping enabled, processes orphaned by a dead worker reparent to the server
    and are stopped before the next start.
    """

    def __init__(
        self,
        config_payload: bytes,
        *,
        workspace_identity: tuple[int, int] | None = None,
        command: list[str] | None = None,
    ) -> None:
        self._startup = json.dumps(
            {
                "config": base64.b64encode(config_payload).decode("ascii"),
                "workspace_identity": list(workspace_identity) if workspace_identity else None,
            }
        ).encode()
        self._command = command or [sys.executable, "-m", "capicove.api._worker"]
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._closing = threading.Event()

    def start(self) -> None:
        """Start the worker and wait until it reports a composed catalog."""
        _stop_orphans(exclude=None)
        process = subprocess.Popen(  # nosec B603 - own interpreter worker, no shell
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            start_new_session=True,
        )
        self._process = process
        try:
            assert process.stdin is not None
            process.stdin.write(self._startup + b"\n")
            process.stdin.flush()
            message = self._read(process, _STARTUP_SECONDS)
        except (OSError, WorkerTimeout, WorkerUnavailable):
            message = None
        if message != {"ready": True}:
            self.terminate()
            raise ServeError("the selected Tools could not start in the worker")

    def invoke(
        self, tool_id: str, arguments: Mapping[str, object], *, timeout: float
    ) -> dict[str, object]:
        """Send one request; a missed deadline stops the worker and its children."""
        with self._lock:
            if self._closing.is_set():
                raise WorkerUnavailable
            process = self._process
            if process is None or process.poll() is not None:
                if process is not None:
                    self._stop(process)
                try:
                    self.start()
                except ServeError:
                    raise WorkerUnavailable from None
                process = self._process
            assert process is not None and process.stdin is not None
            try:
                line = json.dumps({"tool_id": tool_id, "arguments": dict(arguments)})
                process.stdin.write(line.encode() + b"\n")
                process.stdin.flush()
                message = self._read(process, timeout + 5.0)
            except WorkerTimeout:
                self._stop(process)
                raise
            except (OSError, WorkerUnavailable):
                self._stop(process)
                raise WorkerUnavailable from None
        if not isinstance(message, dict):
            raise WorkerUnavailable
        if message.get("error") == "invalid_arguments":
            raise InvalidToolArguments("Invalid Tool arguments")
        result = message.get("result")
        if not isinstance(result, dict):
            raise ToolInvocationFailed
        return result

    def terminate(self) -> None:
        """Stop admission, the worker, and every process it started."""
        self._closing.set()
        process = self._process
        if process is not None:
            self._stop(process)
        else:
            _stop_orphans(exclude=None)

    @staticmethod
    def _read(process: subprocess.Popen[bytes], timeout: float) -> object:
        stream: IO[bytes] | None = process.stdout
        assert stream is not None
        readable, _, _ = select.select([stream], [], [], timeout)
        if not readable:
            raise WorkerTimeout
        line = stream.readline()
        if not line:
            raise WorkerUnavailable
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            raise WorkerUnavailable from None

    @staticmethod
    def _stop(process: subprocess.Popen[bytes]) -> None:
        # Media adapters start FFmpeg or Whisper in their own sessions, so a group
        # signal alone would miss them. Snapshot the live tree, and afterwards stop
        # anything that was orphaned to this (subreaper) process.
        targets = [process.pid, *_descendants(process.pid)] if process.poll() is None else []
        for signal_number in (signal.SIGTERM, signal.SIGKILL):
            if not targets:
                break
            _signal_all(process.pid, targets, signal_number)
            try:
                process.wait(_TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                continue
            if not any(_alive(pid) for pid in targets[1:]):
                break
        _signal_all(process.pid, targets, signal.SIGKILL)
        try:
            process.wait(_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        _stop_orphans(exclude=None)


def _is_linux() -> bool:
    """Return whether this process is running on Linux."""
    return sys.platform.startswith("linux")


def enable_child_reaping() -> None:
    """Make this process a Linux child subreaper, so orphans reparent to it."""
    import ctypes
    import ctypes.util

    library = ctypes.util.find_library("c")
    if not _is_linux() or library is None:
        raise ServeError("HTTP serve requires Linux child-subreaper support in this release")
    libc = ctypes.CDLL(library, use_errno=True)
    pr_set_child_subreaper = 36
    if libc.prctl(pr_set_child_subreaper, 1, 0, 0, 0) != 0:
        raise ServeError("HTTP serve could not enable child reaping")


def _process_table() -> dict[int, list[int]]:
    """Map each parent PID to its children from Linux procfs, without spawning a process."""
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as stream:
                fields = stream.read().rsplit(b")", 1)[1].split()
        except OSError:
            continue
        # After the command name: state, ppid, ...
        children.setdefault(int(fields[1]), []).append(int(entry))
    return children


def _descendants(root: int) -> list[int]:
    """List every descendant process of ``root``."""
    table = _process_table()
    found: list[int] = []
    pending = [root]
    while pending:
        for child in table.get(pending.pop(), []):
            if child not in found:
                found.append(child)
                pending.append(child)
    return found


def _stop_orphans(*, exclude: int | None) -> None:
    """Stop and reap processes reparented to this subreaper by a dead worker."""
    table = _process_table()
    orphans = [pid for pid in table.get(os.getpid(), []) if pid != exclude]
    targets = [*orphans, *(pid for orphan in orphans for pid in _descendants(orphan))]
    for signal_number in (signal.SIGTERM, signal.SIGKILL):
        for pid in targets:
            try:
                os.kill(pid, signal_number)
            except ProcessLookupError:
                continue
        deadline = time.monotonic() + (0.5 if signal_number == signal.SIGTERM else 2.0)
        while time.monotonic() < deadline and any(_alive(pid) for pid in targets):
            time.sleep(0.02)
    # Collect every orphan's exit status so none is left behind as a zombie.
    for pid in orphans:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                if os.waitpid(pid, os.WNOHANG)[0] != 0:
                    break
            except ChildProcessError:
                break
            time.sleep(0.02)


def _signal_all(group: int, pids: list[int], signal_number: int) -> None:
    try:
        os.killpg(group, signal_number)
    except ProcessLookupError:
        pass
    for pid in pids:
        try:
            os.kill(pid, signal_number)
        except ProcessLookupError:
            continue


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", "rb") as stream:
            state = stream.read().rsplit(b")", 1)[1].split()[0]
    except OSError:
        return False
    return state != b"Z"


def serve(config_path: Path, *, port: int, token_file: Path | None) -> int:
    """Run the server in the foreground until SIGTERM or SIGINT."""
    import importlib.util

    if any(importlib.util.find_spec(name) is None for name in ("starlette", "uvicorn")):
        raise ServeError("install capicove[api] to serve HTTP")
    if os.name != "posix" or not hasattr(os, "killpg"):
        raise ServeError("HTTP serve requires POSIX process groups in this release")
    from capicove.mcp.bindings import catalog_payload
    from capicove.runtime.mcp_config import (
        WorkspaceHostConfiguration,
        parse_mcp_host_config,
        read_mcp_host_config_bytes,
    )
    from capicove.runtime.mcp_host import compose_mcp_host

    # Freeze the authority once: workers receive these bytes, never the path.
    payload = read_mcp_host_config_bytes(config_path)
    config = parse_mcp_host_config(payload)
    if not isinstance(config, WorkspaceHostConfiguration):
        raise ServeError("HTTP serve requires a version-2 workspace config")
    if not 0 <= port <= 65535:
        raise ServeError("--port must be from 0 through 65535")
    root = config.workspace_root
    _require_private_root(root)
    target = _token_target(root, token_file)
    with compose_mcp_host(config, catalog_only=True) as bindings:
        catalog = catalog_payload(config, bindings)
        timeouts = {item.spec.tool_id: float(item.spec.timeout_seconds) for item in bindings}
    enable_child_reaping()
    lock = _acquire_lock(root)
    root_metadata = os.stat(root, follow_symlinks=False)
    worker = WorkerProcess(payload, workspace_identity=(root_metadata.st_dev, root_metadata.st_ino))
    token: str | None = None
    listener: socket.socket | None = None
    stop_requested = threading.Event()
    handled = (signal.SIGINT, signal.SIGTERM)
    # Own both signals from here until cleanup ends, so shutdown always runs below.
    original_handlers = {
        number: signal.signal(number, lambda *_: stop_requested.set()) for number in handled
    }
    try:
        worker.start()
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind((LOOPBACK, port))
        listener.listen(64)
        token = secrets.token_hex(32)
        _write_token(root, target, token)
        app = create_workspace_app(catalog=catalog, token=token, invoker=worker, timeouts=timeouts)
        server = _server(app, worker)
        for number in handled:
            signal.signal(number, server.handle_exit)
        if stop_requested.is_set():
            server.should_exit = True
        print(
            json.dumps(
                {
                    "event": "ready",
                    "address": LOOPBACK,
                    "port": listener.getsockname()[1],
                    "token_file": str(target),
                }
            ),
            file=sys.stderr,
            flush=True,
        )
        server.run(sockets=[listener])
    finally:
        for number in handled:
            signal.signal(number, lambda *_: None)
        worker.terminate()
        if listener is not None:
            listener.close()
        if token is not None:
            _remove_own_token(root, target, token)
        os.close(lock)
        for number, handler in original_handlers.items():
            signal.signal(number, handler)
    return 0


def _server(app: object, worker: WorkerProcess):  # type: ignore[no-untyped-def]
    import uvicorn

    class _Server(uvicorn.Server):
        @contextlib.contextmanager
        def capture_signals(self) -> Iterator[None]:
            # serve() owns SIGTERM/SIGINT from before readiness until cleanup ends;
            # uvicorn's capture would re-raise the signal and skip that cleanup.
            yield

        def handle_exit(self, sig: int, frame: object) -> None:
            # Stop admission and in-flight work first, so draining connections is fast.
            app.state.closing.set()  # type: ignore[attr-defined]
            threading.Thread(target=worker.terminate, daemon=True).start()
            super().handle_exit(sig, frame)  # type: ignore[arg-type]

    return _Server(
        uvicorn.Config(
            app,  # type: ignore[arg-type]
            log_level="warning",
            access_log=False,
            lifespan="off",
            timeout_graceful_shutdown=_GRACEFUL_SHUTDOWN_SECONDS,
        )
    )


def _require_private_root(root: Path) -> None:
    try:
        metadata = root.lstat()
    except OSError:
        raise ServeError("workspace_root must be an existing directory") from None
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o022
    ):
        raise ServeError("workspace_root must be an owner-controlled directory without symlinks")


def _token_target(root: Path, token_file: Path | None) -> Path:
    target = (root / "http.token") if token_file is None else token_file.absolute()
    if ".." in target.parts or not target.is_relative_to(root) or target == root:
        raise ServeError("--token-file must be inside workspace_root")
    return target


def _open_parent(root: Path, target: Path) -> int:
    """Open the token's parent through no-follow handles pinned from the workspace root."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        descriptor = os.open(root.anchor, flags)
        for component in target.parent.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except OSError:
        raise ServeError("the token file directory must exist without symlinks") from None
    return descriptor


def _acquire_lock(root: Path) -> int:
    import fcntl

    directory = _open_parent(root, root / ".http.lock")
    try:
        # Non-blocking: a FIFO planted at the lock path must fail, not hang startup.
        descriptor = os.open(
            ".http.lock",
            os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
            0o600,
            dir_fd=directory,
        )
    except OSError:
        raise ServeError("the server lock could not be opened") from None
    finally:
        os.close(directory)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ServeError("the server lock must be a regular file")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise ServeError("an HTTP server is already running for this workspace") from None
    return descriptor


def _write_token(root: Path, target: Path, token: str) -> None:
    directory = _open_parent(root, target)
    temporary = f".{target.name}.{os.getpid()}.tmp"
    try:
        try:
            existing = os.stat(target.name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and (
            not stat.S_ISREG(existing.st_mode)
            or existing.st_uid != os.geteuid()
            or existing.st_mode & 0o077
        ):
            raise ServeError("--token-file exists and is not an owner-private regular file")
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(token + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target.name, src_dir_fd=directory, dst_dir_fd=directory)
        os.fsync(directory)
    except OSError:
        try:
            os.unlink(temporary, dir_fd=directory)
        except OSError:
            pass
        raise ServeError("the token file could not be written") from None
    finally:
        os.close(directory)


def _remove_own_token(root: Path, target: Path, token: str) -> None:
    try:
        directory = _open_parent(root, target)
    except ServeError:
        return
    try:
        descriptor = os.open(
            target.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
        with os.fdopen(descriptor, encoding="utf-8") as stream:
            # Never read from a FIFO or device that replaced the token file.
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                return
            current = stream.read(256).strip()
        if current == token:
            os.unlink(target.name, dir_fd=directory)
    except OSError:
        pass
    finally:
        os.close(directory)


__all__ = ["LOOPBACK", "ServeError", "WorkerProcess", "serve"]
