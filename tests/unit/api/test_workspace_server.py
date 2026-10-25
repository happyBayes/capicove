"""RFC-0020 server lifecycle pieces exercised in-process: worker client, token, lock."""

from __future__ import annotations

import base64
import io
import json
import os
import stat
import sys
import textwrap
from pathlib import Path

import pytest

from capicove.api import _worker
from capicove.api.workspace import ToolInvocationFailed, WorkerTimeout, WorkerUnavailable
from capicove.api.workspace_server import (
    ServeError,
    WorkerProcess,
    _acquire_lock,
    _remove_own_token,
    _require_private_root,
    _token_target,
    _write_token,
)
from capicove.mcp.bindings import InvalidToolArguments
from tests.unit.runtime.test_local_retrieval import (
    Workspace,
    build,
    import_source,
)
from tests.unit.runtime.test_local_retrieval import (
    workspace as workspace,  # re-exported pytest fixture
)


pytestmark = pytest.mark.skipif(os.name != "posix", reason="HTTP serve is POSIX-only")


def fake_worker(tmp_path: Path, body: str) -> list[str]:
    """Return a worker command that answers the protocol with ``body`` per request."""
    script = tmp_path / "worker.py"
    script.write_text(
        textwrap.dedent(
            """
            import json, sys, time
            sys.stdin.readline()
            print(json.dumps({"ready": True}), flush=True)
            for line in sys.stdin:
            """
        )
        + textwrap.indent(textwrap.dedent(body), "    ")
    )
    return [sys.executable, str(script)]


# Worker client


def test_worker_answers_through_the_frozen_configuration(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    import_source(capsys, workspace, "paris.md")
    build(capsys, workspace)
    worker = WorkerProcess(workspace.config.read_bytes())
    worker.start()
    try:
        workspace.config.write_text("{}")  # A later edit must not reach the worker.
        result = worker.invoke("capicove.search", {"query": "Paris"}, timeout=30)
        with pytest.raises(InvalidToolArguments):
            worker.invoke("capicove.search", {"unexpected": True}, timeout=30)
    finally:
        worker.terminate()

    assert result["status"] == "succeeded"
    assert result["result"]["hits"]  # type: ignore[index]
    with pytest.raises(WorkerUnavailable):
        worker.invoke("capicove.search", {"query": "Paris"}, timeout=30)


def test_worker_rejects_a_configuration_that_cannot_compose(tmp_path: Path) -> None:
    worker = WorkerProcess(b'{"schema_version": 2}')

    with pytest.raises(ServeError):
        worker.start()


def test_worker_rejects_a_replaced_workspace_root(workspace: Workspace) -> None:
    worker = WorkerProcess(workspace.config.read_bytes(), workspace_identity=(0, 0))

    with pytest.raises(ServeError):
        worker.start()


def test_a_missed_deadline_stops_and_replaces_the_worker(tmp_path: Path) -> None:
    command = fake_worker(
        tmp_path,
        """
        request = json.loads(line)
        if request["tool_id"] == "slow":
            time.sleep(60)
        print(json.dumps({"result": {"status": "succeeded"}}), flush=True)
        """,
    )
    worker = WorkerProcess(b"{}", command=command)
    worker.start()
    try:
        with pytest.raises(WorkerTimeout):
            worker.invoke("slow", {}, timeout=-4.5)
        assert worker.invoke("fast", {}, timeout=10) == {"status": "succeeded"}
    finally:
        worker.terminate()


@pytest.mark.parametrize(
    ("answer", "error"),
    [
        ('print("not json", flush=True)', WorkerUnavailable),
        ('print(json.dumps({"error": "internal"}), flush=True)', ToolInvocationFailed),
        ("sys.exit(0)", WorkerUnavailable),
    ],
)
def test_worker_protocol_failures_map_to_stable_errors(
    tmp_path: Path, answer: str, error: type[Exception]
) -> None:
    worker = WorkerProcess(b"{}", command=fake_worker(tmp_path, answer))
    worker.start()
    try:
        with pytest.raises(error):
            worker.invoke("any", {}, timeout=10)
    finally:
        worker.terminate()


# Worker process main


def test_worker_main_serves_requests_and_reports_errors(
    workspace: Workspace, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import_source(capsys, workspace, "paris.md")
    build(capsys, workspace)
    capsys.readouterr()
    metadata = os.stat(workspace.root)
    startup = {
        "config": base64.b64encode(workspace.config.read_bytes()).decode(),
        "workspace_identity": [metadata.st_dev, metadata.st_ino],
    }
    requests = [
        {"tool_id": "capicove.search", "arguments": {"query": "Paris"}},
        {"tool_id": "capicove.search", "arguments": {"bad": 1}},
        {"tool_id": "capicove.unknown", "arguments": {}},
    ]
    lines = [json.dumps(startup), *(json.dumps(request) for request in requests)]
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(lines) + "\n"))

    assert _worker.main([]) == 0

    replies = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert replies[0] == {"ready": True}
    assert replies[1]["result"]["status"] == "succeeded"
    assert replies[2] == {"error": "invalid_arguments"}
    assert replies[3] == {"error": "internal"}


def test_worker_main_refuses_a_foreign_workspace_identity(
    workspace: Workspace, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    startup = {
        "config": base64.b64encode(workspace.config.read_bytes()).decode(),
        "workspace_identity": [0, 0],
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(startup) + "\n"))

    assert _worker.main([]) == 2
    assert json.loads(capsys.readouterr().out) == {"failed": "startup"}


# Token, lock, and root checks


def private_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    return root


def test_token_is_written_private_and_removed_only_when_still_ours(tmp_path: Path) -> None:
    root = private_root(tmp_path)
    target = _token_target(root, None)

    _write_token(root, target, "first")
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    _write_token(root, target, "second")
    _remove_own_token(root, target, "first")
    assert target.read_text().strip() == "second"
    _remove_own_token(root, target, "second")
    assert not target.exists()


@pytest.mark.parametrize("kind", ["symlink", "shared", "fifo"])
def test_unsafe_existing_token_files_are_refused(tmp_path: Path, kind: str) -> None:
    root = private_root(tmp_path)
    target = root / "http.token"
    if kind == "symlink":
        target.symlink_to(tmp_path / "elsewhere")
    elif kind == "shared":
        target.write_text("foreign")
        target.chmod(0o644)
    else:
        os.mkfifo(target, 0o600)

    with pytest.raises(ServeError):
        _write_token(root, target, "token")
    _remove_own_token(root, target, "token")  # Must return without blocking or deleting.
    assert target.exists() or target.is_symlink()


def test_token_target_must_stay_inside_the_workspace(tmp_path: Path) -> None:
    root = private_root(tmp_path)

    assert _token_target(root, root / "custom.token") == root / "custom.token"
    for outside in (tmp_path / "outside.token", root, root / ".." / "x"):
        with pytest.raises(ServeError):
            _token_target(root, outside)


def test_lock_admits_one_server_and_refuses_a_fifo(tmp_path: Path) -> None:
    root = private_root(tmp_path)
    first = _acquire_lock(root)
    try:
        with pytest.raises(ServeError, match="already running"):
            _acquire_lock(root)
    finally:
        os.close(first)
    (tmp_path / "fifo").mkdir()
    fifo_root = private_root(tmp_path / "fifo")
    os.mkfifo(fifo_root / ".http.lock", 0o600)
    with pytest.raises(ServeError):
        _acquire_lock(fifo_root)


def test_workspace_root_must_be_owner_controlled(tmp_path: Path) -> None:
    root = private_root(tmp_path)
    _require_private_root(root)
    root.chmod(0o777)
    with pytest.raises(ServeError):
        _require_private_root(root)
    with pytest.raises(ServeError):
        _require_private_root(tmp_path / "missing")
