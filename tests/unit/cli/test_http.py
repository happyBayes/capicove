"""RFC-0020 `capicove http serve`: startup, token and lock lifecycle, and shutdown."""

from __future__ import annotations

import json
import os
import select
import signal
import stat
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.unit.runtime.test_local_retrieval import (
    Workspace,
    build,
    import_source,
    search,
)
from tests.unit.runtime.test_local_retrieval import (
    workspace as workspace,  # re-exported pytest fixture
)


pytestmark = [
    pytest.mark.skipif(os.name != "posix", reason="HTTP serve is POSIX-only (ADR-0025)"),
]


class Server:
    def __init__(self, process: subprocess.Popen[str], ready: dict[str, object]) -> None:
        self.process = process
        self.ready = ready
        self.base = f"http://{ready['address']}:{ready['port']}"
        self.token_file = Path(str(ready["token_file"]))

    @property
    def token(self) -> str:
        return self.token_file.read_text(encoding="utf-8").strip()

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}


_PR_SET_CHILD_SUBREAPER = 36
_PR_GET_CHILD_SUBREAPER = 37


def _process_table() -> dict[int, tuple[int, str]]:
    """Map PID to (parent PID, state) from procfs; empty where procfs is absent."""
    table: dict[int, tuple[int, str]] = {}
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            stat_line = (entry / "stat").read_text()
        except OSError:
            continue
        # The command name may contain spaces or parentheses; fields follow the last ')'.
        fields = stat_line[stat_line.rindex(")") + 2 :].split()
        table[int(entry.name)] = (int(fields[1]), fields[0])
    return table


def descendants(pid: int) -> set[int]:
    """Return every live or zombie descendant of ``pid``, across sessions."""
    table = _process_table()
    found: set[int] = set()
    frontier = [pid]
    while frontier:
        parent = frontier.pop()
        for child, (ppid, _) in table.items():
            if ppid == parent and child not in found:
                found.add(child)
                frontier.append(child)
    return found


def _kill(pids: set[int]) -> None:
    """SIGKILL ``pids``, reap those this process adopted, and wait until they are gone."""
    for pid in pids:
        with suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        for pid in pids:
            with suppress(ChildProcessError):
                os.waitpid(pid, os.WNOHANG)
        if not any(alive(pid) for pid in pids):
            return
        time.sleep(0.02)


def _is_linux() -> bool:
    """Return whether this process is running on Linux."""
    return sys.platform.startswith("linux")


def _set_subreaper(enabled: bool) -> bool:
    """Set the Linux child-subreaper flag; return its previous value."""
    if not _is_linux():
        return False
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    libc.prctl(_PR_GET_CHILD_SUBREAPER, ctypes.byref(previous), 0, 0, 0)
    libc.prctl(_PR_SET_CHILD_SUBREAPER, int(enabled), 0, 0, 0)
    return bool(previous.value)


@contextmanager
def reaping_descendants() -> Iterator[None]:
    """Adopt orphans while active; afterwards kill and reap every new descendant.

    The server starts its worker in a separate session, so killing the server's
    process group misses it, and once the server dies the worker is reparented.
    As a child subreaper, this process adopts such orphans and can stop them.
    """
    me = os.getpid()
    before = descendants(me)
    previous = _set_subreaper(True)
    try:
        yield
    finally:
        for _ in range(20):
            remaining = descendants(me) - before
            if not remaining:
                break
            _kill(remaining)
        _set_subreaper(previous)


@pytest.fixture(autouse=True)
def reap_server_descendants() -> Iterator[None]:
    """No server, worker, or media child outlives the test that started it."""
    with reaping_descendants():
        yield


def _stop_tree(process: subprocess.Popen[str], *, grace: float, terminate: bool = False) -> None:
    """Optionally SIGTERM the server, wait, then kill it and all descendants.

    The tree is recorded first: once the server exits, its separate-session
    children are reparented and no longer appear below it.
    """
    tree = descendants(process.pid)
    if terminate and process.poll() is None:
        process.send_signal(signal.SIGTERM)
    try:
        process.wait(grace)
    except subprocess.TimeoutExpired:
        pass
    tree |= descendants(process.pid)
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)
    _kill(tree)
    process.wait(5)


def start(
    config: Path,
    *extra: str,
    timeout: float = 60,
    executable: str | None = None,
    cwd: Path | None = None,
) -> tuple[subprocess.Popen[str], dict[str, object] | None, str]:
    command = [executable] if executable else [sys.executable, "-m", "capicove"]
    process = subprocess.Popen(
        [*command, "http", "serve", "--config", str(config), *extra],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert process.stderr is not None
    descriptor = process.stderr.fileno()
    os.set_blocking(descriptor, False)
    deadline = time.monotonic() + timeout
    seen = b""  # Everything read so far, for failure messages.
    buffer = b""  # The trailing partial line.
    closed = False
    try:
        while not closed and (remaining := deadline - time.monotonic()) > 0:
            readable, _, _ = select.select([descriptor], [], [], min(remaining, 0.2))
            if not readable:
                if process.poll() is not None:
                    break
                continue
            try:
                chunk = os.read(descriptor, 65536)
            except BlockingIOError:
                continue
            closed = not chunk
            seen += chunk
            buffer += chunk
            *lines, buffer = buffer.split(b"\n")
            for line in lines:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict) and record.get("event") == "ready":
                    os.set_blocking(descriptor, True)
                    return process, record, seen.decode(errors="replace")
    except BaseException:
        _stop_tree(process, grace=0)
        raise
    _stop_tree(process, grace=10)
    with suppress(OSError):
        seen += os.read(descriptor, 65536)
    return process, None, seen.decode(errors="replace")


@contextmanager
def serving(config: Path, *extra: str, executable: str | None = None) -> Iterator[Server]:
    process, ready, output = start(config, *extra, executable=executable)
    assert ready is not None, output
    server = Server(process, ready)
    try:
        yield server
    finally:
        _stop_tree(process, grace=15, terminate=True)


def prepared(space: Workspace, capsys: pytest.CaptureFixture[str]) -> Workspace:
    import_source(capsys, space, "paris.md")
    build(capsys, space)
    return space


def test_missing_api_extra_exits_before_any_write(workspace: Workspace) -> None:
    script = (
        "import sys; sys.modules['uvicorn'] = None; "
        "from capicove.cli.app import main; "
        f"sys.exit(main(['http', 'serve', '--config', {str(workspace.config)!r}]))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )

    assert completed.returncode == 2
    assert "capicove[api]" in completed.stderr
    assert not (workspace.root / "http.token").exists()
    assert not (workspace.root / ".http.lock").exists()


@pytest.mark.parametrize("where", ["outside", "symlink", "shared"])
def test_unsafe_token_files_are_refused_before_listening(
    workspace: Workspace, tmp_path: Path, where: str
) -> None:
    if where == "outside":
        target = tmp_path / "outside.token"
    elif where == "symlink":
        target = workspace.root / "linked.token"
        target.symlink_to(tmp_path / "victim.token")
    else:
        target = workspace.root / "shared.token"
        target.write_text("foreign\n")
        target.chmod(0o644)

    process, ready, output = start(workspace.config, "--token-file", str(target))

    assert ready is None and process.returncode == 2, output
    assert not (tmp_path / "victim.token").exists()
    if where == "shared":
        assert target.read_text() == "foreign\n"


def test_ready_record_token_and_authenticated_catalog(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    prepared(workspace, capsys)
    from capicove.cli.app import main

    assert main(["mcp", "catalog", "--config", str(workspace.config)]) == 0
    expected_catalog = json.loads(capsys.readouterr().out)

    with serving(workspace.config) as server:
        assert server.ready["address"] == "127.0.0.1"
        assert int(str(server.ready["port"])) > 0
        assert server.token_file == workspace.root / "http.token"
        mode = stat.S_IMODE(server.token_file.stat().st_mode)
        assert mode == 0o600
        token = server.token
        assert len(token) == 64
        assert httpx.get(f"{server.base}/v1/health").json() == {"status": "ok"}
        assert httpx.get(f"{server.base}/v1/tools").status_code == 401
        catalog = httpx.get(f"{server.base}/v1/tools", headers=server.headers())
        assert catalog.json() == expected_catalog
        process = server.process
    assert process.stdout is not None and token not in process.stdout.read()


def test_http_search_matches_cli_search(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    prepared(workspace, capsys)
    _, cli = search(capsys, workspace, "Paris")

    with serving(workspace.config) as server:
        direct = httpx.post(
            f"{server.base}/v1/tools/capicove.search",
            headers=server.headers(),
            json={"query": "Paris"},
        ).json()
        alias = httpx.post(
            f"{server.base}/v1/search", headers=server.headers(), json={"query": "Paris"}
        ).json()
        evidence_id = direct["evidence_refs"][0]
        opened = httpx.post(
            f"{server.base}/v1/evidence",
            headers=server.headers(),
            json={"evidence_id": evidence_id},
        ).json()
        invalid = httpx.post(
            f"{server.base}/v1/search", headers=server.headers(), json={"unexpected": 1}
        )

    def stable(payload: dict[str, Any]) -> object:
        result = dict(payload["result"])
        result.pop("query_id")
        return payload["status"], payload["tool_id"], payload["evidence_refs"], result

    assert stable(direct) == stable(alias) == stable(cli)
    assert opened["status"] == "succeeded"
    assert "Alice moved to Paris" in opened["result"]["metadata"]["text"]
    assert invalid.status_code == 400


def test_second_server_is_rejected_without_rotating_the_token(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    prepared(workspace, capsys)
    with serving(workspace.config) as first:
        token = first.token
        other = workspace.root / "other.token"
        process, ready, output = start(workspace.config, "--token-file", str(other))
        assert ready is None and process.returncode == 2, output
        assert "already running" in output
        assert first.token == token
        assert not other.exists()
        assert httpx.get(f"{first.base}/v1/tools", headers=first.headers()).status_code == 200


def test_sigterm_cleans_up_and_restart_rotates_the_token(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    prepared(workspace, capsys)
    with serving(workspace.config) as server:
        first_token = server.token
        worker_pids = children(server.process.pid)
        server.process.send_signal(signal.SIGTERM)
        started = time.monotonic()
        server.process.wait(15)
        assert time.monotonic() - started < 10
        assert server.process.returncode == 0
    assert not (workspace.root / "http.token").exists()
    assert all(not alive(pid) for pid in worker_pids)

    with serving(workspace.config) as restarted:
        assert restarted.token != first_token


def test_restart_after_sigkill_replaces_stale_token(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    prepared(workspace, capsys)
    process, ready, output = start(workspace.config)
    assert ready is not None, output
    stale = (workspace.root / "http.token").read_text()
    os.killpg(process.pid, signal.SIGKILL)
    process.wait(10)

    with serving(workspace.config) as restarted:
        assert restarted.token != stale.strip()
        response = httpx.get(f"{restarted.base}/v1/tools", headers=restarted.headers())
        assert response.status_code == 200


def test_network_enabled_admin_config_still_binds_loopback(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    prepared(workspace, capsys)
    workspace.write_config(
        profile="admin", network={"enabled": True, "source_hosts": ["example.com"]}
    )

    with serving(workspace.config) as server:
        assert server.ready["address"] == "127.0.0.1"


def children(pid: int) -> list[int]:
    try:
        output = subprocess.run(
            ["ps", "-o", "pid=", "--ppid", str(pid)], capture_output=True, text=True, check=False
        ).stdout
    except FileNotFoundError:
        return []
    return [int(value) for value in output.split()]


def alive(pid: int) -> bool:
    """Return whether ``pid`` is running; a zombie awaiting its reaper is not."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    state = _process_table().get(pid)
    return state is None or state[1] != "Z"


@pytest.mark.slow
@pytest.mark.skipif(
    __import__("shutil").which("ffmpeg") is None, reason="needs local FFmpeg for media work"
)
def test_media_work_keeps_health_responsive_and_sigterm_reaps_children(tmp_path: Path) -> None:
    import shutil
    import threading

    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    assert ffmpeg is not None and ffprobe is not None
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    clips = tmp_path / "clips"
    clips.mkdir()
    subprocess.run(
        [
            *(
                ffmpeg,
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=40:size=640x480:rate=25",
            ),
            *("-c:v", "libx264", "-g", "25", "-pix_fmt", "yuv420p", str(clips / "long.mp4")),
        ],
        check=True,
        timeout=120,
    )
    config = root / "host.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "server_name": "capicove-local",
                "workspace_id": "workspace",
                "actor_id": "owner",
                "environment_id": "local",
                "workspace_root": str(root),
                "profile": "write",
                "permissions": ["artifact:write", "subprocess:execute", "video:read"],
                "enabled_tools": ["capicove.video.split"],
                "trace_store": None,
                "capabilities": {
                    "video": {
                        "schema_version": 1,
                        "read_roots": [str(clips)],
                        "staging_root": str(root / "staging"),
                        "output_root": str(root / "output"),
                        "backend": "ffmpeg",
                        "ffmpeg": ffmpeg,
                        "ffprobe": ffprobe,
                        "max_input_bytes": 1024 * 1024 * 1024,
                    }
                },
                "network": {"enabled": False, "source_hosts": []},
            }
        )
    )
    config.chmod(0o600)
    process, ready, output = start(config)
    assert ready is not None, output
    server = Server(process, ready)
    responses: list[httpx.Response] = []

    def long_split() -> None:
        responses.append(
            httpx.post(
                f"{server.base}/v1/tools/capicove.video.split",
                headers=server.headers(),
                json={
                    "source": {"type": "path", "path": str(clips / "long.mp4")},
                    "strategy": "duration",
                    "config": {
                        "split_duration": 0.5,
                        "overlap_duration": 0.0,
                        "min_last_split_duration": 0.0,
                    },
                    "output_format": "path",
                    "backend": "ffmpeg",
                    "write_mode": "accurate",
                },
                timeout=120,
            )
        )

    worker = threading.Thread(target=long_split)
    worker.start()
    try:
        from capicove.api.workspace_server import _descendants

        deadline = time.monotonic() + 60
        media: list[int] = []
        while time.monotonic() < deadline:
            # Server -> worker -> FFmpeg; FFmpeg runs in its own session.
            media = [pid for pid in _descendants(process.pid) if pid not in children(process.pid)]
            if media:
                break
            time.sleep(0.05)
        assert media, "the split never started an FFmpeg child"
        tree = _descendants(process.pid)
        started = time.monotonic()
        assert httpx.get(f"{server.base}/v1/health", timeout=5).json() == {"status": "ok"}
        assert time.monotonic() - started < 1.0
        busy_response = httpx.post(
            f"{server.base}/v1/search", headers=server.headers(), json={}, timeout=5
        )
        assert busy_response.status_code == 404  # search is not selected on this host
        second = httpx.post(
            f"{server.base}/v1/tools/capicove.video.split",
            headers=server.headers(),
            json={},
            timeout=5,
        )
        assert second.status_code == 429

        process.send_signal(signal.SIGTERM)
        stopping = time.monotonic()
        assert process.wait(15) == 0
        assert time.monotonic() - stopping < 10
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        worker.join(30)
    time.sleep(0.2)
    assert [pid for pid in tree if alive(pid)] == []
    assert responses and responses[0].status_code in {503, 504}
    assert not (root / "http.token").exists()


# Review findings (2026-10-06)


def _worker_pid(server_pid: int) -> int:
    (pid,) = children(server_pid)
    return pid


def test_missing_starlette_exits_with_install_guidance(workspace: Workspace) -> None:
    script = (
        "import sys; sys.modules['starlette'] = None; "
        "from capicove.cli.app import main; "
        f"sys.exit(main(['http', 'serve', '--config', {str(workspace.config)!r}]))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )

    assert completed.returncode == 2, completed.stderr
    assert "capicove[api]" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_restarted_worker_keeps_the_startup_configuration(
    workspace: Workspace, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    prepared(workspace, capsys)
    other_notes = tmp_path / "other-notes"
    other_notes.mkdir()
    (other_notes / "secret.md").write_text("Zanzibar is the other workspace.\n")
    other_root = tmp_path / "other"
    other_root.mkdir(mode=0o700)
    other = Workspace(other_root, other_notes, other_root / "index", other_root / "host.json")
    other.write_config()
    import_source(capsys, other, "secret.md")
    build(capsys, other)

    with serving(workspace.config) as server:
        workspace.config.write_text(other.config.read_text())
        os.kill(_worker_pid(server.process.pid), signal.SIGKILL)
        time.sleep(0.5)
        found = httpx.post(
            f"{server.base}/v1/search", headers=server.headers(), json={"query": "Zanzibar"}
        )
        paris = httpx.post(
            f"{server.base}/v1/search", headers=server.headers(), json={"query": "Paris"}
        )

    assert found.status_code in {200, 503}
    if found.status_code == 200:
        assert found.json()["result"]["hits"] == []
    assert paris.status_code == 200 and paris.json()["result"]["hits"]


def test_start_helper_kills_a_server_that_never_becomes_ready(tmp_path: Path) -> None:
    """A hung server must not outlive its test (one survived for days)."""
    hung = tmp_path / "hung-server"
    hung.write_text("#!/bin/sh\nexec sleep 300\n")
    hung.chmod(0o700)

    process, ready, _ = start(tmp_path / "unused.json", timeout=0.5, executable=str(hung))

    assert ready is None
    assert process.poll() is not None


def _fake_server(tmp_path: Path, body: str) -> str:
    """Write an executable that ignores its arguments and runs ``body``."""
    script = tmp_path / "fake-server"
    script.write_text(f"#!{sys.executable}\n" + textwrap.dedent(body))
    script.chmod(0o700)
    return str(script)


def _detached_child(record: Path) -> str:
    """Python source that starts a sleeper in its own session and records its PID."""
    return f"""
        import subprocess, sys
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(300)"], start_new_session=True
        )
        open({str(record)!r}, "w").write(str(child.pid))
    """


def _wait_for_pid(record: Path) -> int:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if record.exists() and record.read_text():
            return int(record.read_text())
        time.sleep(0.05)
    raise AssertionError("the fake server never started its child")


def test_start_helper_times_out_on_a_partial_line(tmp_path: Path) -> None:
    """A line without a newline must not block the readiness deadline."""
    partial = _fake_server(
        tmp_path,
        """
        import sys, time
        sys.stderr.write('{"event": "rea')
        sys.stderr.flush()
        time.sleep(300)
        """,
    )
    began = time.monotonic()

    process, ready, output = start(tmp_path / "unused.json", timeout=1, executable=partial)

    assert ready is None
    assert time.monotonic() - began < 15
    assert process.poll() is not None
    assert '{"event": "rea' in output


def test_start_helper_cleans_up_when_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = tmp_path / "child.pid"
    server = _fake_server(
        tmp_path,
        _detached_child(record)
        + """
        import time
        time.sleep(300)
        """,
    )
    started: list[subprocess.Popen[str]] = []
    original_popen = subprocess.Popen

    def recording_popen(*args: Any, **kwargs: Any) -> Any:
        process = original_popen(*args, **kwargs)
        started.append(process)
        return process

    def interrupt(*_: object) -> object:
        _wait_for_pid(record)
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "Popen", recording_popen)
    monkeypatch.setattr(select, "select", interrupt)

    with pytest.raises(KeyboardInterrupt):
        start(tmp_path / "unused.json", executable=server)

    assert started and started[0].poll() is not None
    assert not alive(int(record.read_text()))


def test_start_helper_kills_children_in_separate_sessions(tmp_path: Path) -> None:
    record = tmp_path / "child.pid"
    server = _fake_server(
        tmp_path,
        _detached_child(record)
        + """
        import time
        time.sleep(300)
        """,
    )

    _, ready, _ = start(tmp_path / "unused.json", timeout=3, executable=server)

    assert ready is None
    assert not alive(_wait_for_pid(record))


def _ready_server(tmp_path: Path, record: Path) -> str:
    token = tmp_path / "token"
    token.write_text("token")
    return _fake_server(
        tmp_path,
        _detached_child(record)
        + f"""
        import json, sys, time
        ready = {{"event": "ready", "address": "127.0.0.1", "port": 9, "token_file": {str(token)!r}}}
        print(json.dumps(ready), file=sys.stderr, flush=True)
        time.sleep(300)
        """,
    )


def test_serving_helper_kills_children_in_separate_sessions(tmp_path: Path) -> None:
    record = tmp_path / "child.pid"
    server = _ready_server(tmp_path, record)

    with pytest.raises(RuntimeError, match="test body failed"):
        with serving(tmp_path / "unused.json", executable=server) as running:
            child = _wait_for_pid(record)
            assert alive(child)
            raise RuntimeError("test body failed")

    assert running.process.poll() is not None
    assert not alive(child)


def test_reaper_kills_children_orphaned_by_a_killed_server(tmp_path: Path) -> None:
    """SIGKILL of the server must not leave its separate-session worker behind."""
    record = tmp_path / "child.pid"
    server = _ready_server(tmp_path, record)

    with reaping_descendants():
        process, ready, output = start(tmp_path / "unused.json", executable=server)
        assert ready is not None, output
        child = _wait_for_pid(record)
        os.kill(process.pid, signal.SIGKILL)
        process.wait(10)
        assert alive(child)

    assert not alive(child)


def test_fifo_lock_fails_startup_instead_of_blocking(workspace: Workspace) -> None:
    os.mkfifo(workspace.root / ".http.lock", 0o600)

    process, ready, output = start(workspace.config, timeout=20)

    assert ready is None and process.returncode == 2, output


def test_fifo_token_cannot_block_shutdown(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    prepared(workspace, capsys)
    process, ready, output = start(workspace.config)
    assert ready is not None, output
    token = Path(str(ready["token_file"]))
    token.unlink()
    os.mkfifo(token, 0o600)

    process.send_signal(signal.SIGTERM)
    try:
        process.wait(12)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(5)
    assert process.returncode == 0


def test_detached_children_are_reaped_after_the_worker_exits(tmp_path: Path) -> None:
    """A worker that starts a pipe-less child in its own session, then exits."""
    record = tmp_path / "orphan.pid"
    fake_worker = tmp_path / "fake_worker.py"
    fake_worker.write_text(
        textwrap.dedent(
            f"""
            import subprocess, sys
            child = subprocess.Popen(
                ["sleep", "60"],
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            open({str(record)!r}, "w").write(str(child.pid))
            print('{{"ready": true}}', flush=True)
            sys.exit(0)
            """
        )
    )
    harness = textwrap.dedent(
        f"""
        import os, sys, time
        from pathlib import Path
        from capicove.api.workspace_server import WorkerProcess, enable_child_reaping
        enable_child_reaping()
        worker = WorkerProcess(b"{{}}", command=[sys.executable, {str(fake_worker)!r}])
        worker.start()
        while not Path({str(record)!r}).exists():
            time.sleep(0.01)
        time.sleep(0.3)  # The worker has exited; its child lives on in its own session.
        worker.terminate()
        orphan = int(Path({str(record)!r}).read_text())
        time.sleep(0.2)
        try:
            os.kill(orphan, 0)
            print("alive", orphan)
        except ProcessLookupError:
            print("reaped", orphan)
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", harness], capture_output=True, text=True, timeout=60
    )

    assert completed.stdout.startswith("reaped"), completed.stdout + completed.stderr


def _wait_for(probe: Any, seconds: float) -> list[int]:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        found = probe()
        if found:
            return list(found)
        time.sleep(0.05)
    raise AssertionError("condition was not reached")


def _long_split(clips: Path, segment_seconds: float = 0.5) -> dict[str, object]:
    return {
        "source": {"type": "path", "path": str(clips / "long.mp4")},
        "strategy": "duration",
        "config": {
            "split_duration": segment_seconds,
            "overlap_duration": 0.0,
            "min_last_split_duration": 0.0,
        },
        "output_format": "path",
        "backend": "ffmpeg",
        "write_mode": "accurate",
    }


def _long_video_server(
    tmp_path: Path,
) -> tuple[subprocess.Popen[str], dict[str, object], Path, Path]:
    import shutil

    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    assert ffmpeg is not None and ffprobe is not None
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    clips = tmp_path / "clips"
    clips.mkdir()
    subprocess.run(
        [
            *(
                ffmpeg,
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=40:size=640x480:rate=25",
            ),
            *("-c:v", "libx264", "-g", "25", "-pix_fmt", "yuv420p", str(clips / "long.mp4")),
        ],
        check=True,
        timeout=120,
    )
    config = root / "host.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "server_name": "capicove-local",
                "workspace_id": "workspace",
                "actor_id": "owner",
                "environment_id": "local",
                "workspace_root": str(root),
                "profile": "write",
                "permissions": ["artifact:write", "subprocess:execute", "video:read"],
                "enabled_tools": ["capicove.video.split"],
                "trace_store": None,
                "capabilities": {
                    "video": {
                        "schema_version": 1,
                        "read_roots": [str(clips)],
                        "staging_root": str(root / "staging"),
                        "output_root": str(root / "output"),
                        "backend": "ffmpeg",
                        "ffmpeg": ffmpeg,
                        "ffprobe": ffprobe,
                        "max_input_bytes": 1024 * 1024 * 1024,
                    }
                },
                "network": {"enabled": False, "source_hosts": []},
            }
        )
    )
    config.chmod(0o600)
    process, ready, output = start(config)
    assert ready is not None, output
    return process, ready, config, clips
