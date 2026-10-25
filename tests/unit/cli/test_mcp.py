"""CLI startup errors stay off the MCP protocol stream."""

import importlib
import io
import json
import os
import sqlite3
import subprocess
import sys
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from capicove.cli.app import main
from capicove.contracts import ToolHandler
from capicove.mcp import stdio
from capicove.mcp.bindings import McpToolBinding
from capicove.mcp.stdio import protocol_stdout
from capicove.runtime.mcp_config import McpHostConfiguration, WorkspaceHostConfiguration
from tests.unit.runtime.test_mcp_config import config_body, workspace_config_body


def test_embedded_protocol_stream_is_preserved(capsys: pytest.CaptureFixture[str]) -> None:
    stream = io.StringIO()
    with protocol_stdout(stream) as protocol:
        print("diagnostic")
        print("protocol", file=protocol)
    assert not stream.closed
    assert stream.getvalue() == "protocol\n"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "diagnostic\n"


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor isolation")
def test_protocol_stdout_isolates_native_fd_writes_and_restores_fd_one(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """A native writer must not corrupt the MCP protocol stream."""
    with os.fdopen(1, "w", encoding="utf-8", closefd=False) as standard:
        with protocol_stdout(standard) as protocol:
            os.write(1, b"native diagnostic\n")
            print("protocol response", file=protocol, flush=True)
    os.write(1, b"restored output\n")

    captured = capfd.readouterr()
    assert captured.out == "protocol response\nrestored output\n"
    assert captured.err == "native diagnostic\n"


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor isolation")
def test_protocol_stdout_keeps_body_error_when_handle_restore_fails(
    capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A restore error must not hide the Tool failure or strand fd 1."""
    calls = 0
    original = stdio._sync_windows_stdout_handle

    def fail_on_restore(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("fixture handle restoration failure")
        original(descriptor)

    monkeypatch.setattr(stdio, "_sync_windows_stdout_handle", fail_on_restore)
    with os.fdopen(1, "w", encoding="utf-8", closefd=False) as standard:
        with pytest.raises(ValueError, match="fixture body failure") as error:
            with protocol_stdout(standard):
                raise ValueError("fixture body failure")
    os.write(1, b"restored after failure\n")

    assert calls == 2
    assert any("handle restoration failure" in note for note in error.value.__notes__)
    assert capfd.readouterr().out == "restored after failure\n"


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor isolation")
def test_protocol_stdout_restores_fd_one_after_redirection_failure(
    capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed native-handle redirect must not strand the protocol fd."""
    calls = 0

    def fail_first(_descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("fixture redirect failure")

    monkeypatch.setattr(stdio, "_sync_windows_stdout_handle", fail_first)
    with os.fdopen(1, "w", encoding="utf-8", closefd=False) as standard:
        with pytest.raises(OSError, match="fixture redirect failure"):
            with protocol_stdout(standard):
                pytest.fail("body must not run after redirect failure")
    os.write(1, b"restored after redirect failure\n")

    assert calls == 2
    assert capfd.readouterr().out == "restored after redirect failure\n"


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor isolation")
def test_protocol_stdout_reports_restore_failure_without_masking_fd_recovery(
    capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed native-handle restore remains visible after CRT fd recovery."""
    calls = 0

    def fail_second(_descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("fixture native restore failure")

    monkeypatch.setattr(stdio, "_sync_windows_stdout_handle", fail_second)
    with os.fdopen(1, "w", encoding="utf-8", closefd=False) as standard:
        with pytest.raises(OSError, match="fixture native restore failure"):
            with protocol_stdout(standard):
                pass
    os.write(1, b"restored despite native failure\n")

    assert calls == 2
    assert capfd.readouterr().out == "restored despite native failure\n"


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor isolation")
def test_protocol_stdout_requires_separate_stderr_descriptor(
    capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """MCP startup must reject fd 1 when diagnostics cannot be isolated."""
    with os.fdopen(1, "w", encoding="utf-8", closefd=False) as standard:
        monkeypatch.setattr(sys, "stderr", io.StringIO())
        with pytest.raises(RuntimeError, match="separate stderr descriptor"):
            with protocol_stdout(standard):
                pytest.fail("protocol must not start without isolated stderr")
    os.write(1, b"stdout still available\n")

    assert capfd.readouterr().out == "stdout still available\n"


def test_run_stdio_drains_server_with_dedicated_protocol_stream(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The synchronous host owns the SDK lifecycle through server completion."""
    protocol = io.StringIO()
    reader = object()
    writer = object()
    observed: list[tuple[object, object, object]] = []

    @asynccontextmanager
    async def fake_stdio_server(*, stdout: object) -> AsyncIterator[tuple[object, object]]:
        assert stdout is not protocol
        yield reader, writer

    class FakeServer:
        def create_initialization_options(self) -> str:
            return "options"

        async def run(self, actual_reader: object, actual_writer: object, options: object) -> None:
            print("server diagnostic")
            observed.append((actual_reader, actual_writer, options))

    original_import = importlib.import_module
    monkeypatch.setattr(
        stdio,
        "importlib",
        SimpleNamespace(
            import_module=lambda name: (
                SimpleNamespace(stdio_server=fake_stdio_server)
                if name == "mcp.server.stdio"
                else original_import(name)
            )
        ),
    )

    stdio.run_stdio(FakeServer(), stdout=protocol)  # type: ignore[arg-type]

    assert observed == [(reader, writer, "options")]
    assert protocol.getvalue() == ""
    assert capsys.readouterr().err == "server diagnostic\n"


def test_missing_config_stops_before_protocol_redirection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_protocol() -> None:
        pytest.fail("missing config must not redirect protocol stdout")

    monkeypatch.setattr(stdio, "protocol_stdout", fail_protocol)
    assert main(["mcp", "catalog", "--config", str(tmp_path / "host.json")]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    expected = (
        "MCP Windows path could not be securely opened"
        if os.name == "nt"
        else "MCP config must be readable regular UTF-8 JSON"
    )
    assert expected in captured.err


def test_catalog_lists_only_selected_tools(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "host.json"
    path.write_text(json.dumps(config_body(tmp_path)))
    assert main(["mcp", "catalog", "--config", str(path)]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert [item["tool_id"] for item in payload["tools"]] == ["capicove.image.filter.shape"]
    assert payload["tools"][0]["annotations"]["readOnlyHint"] is True
    assert captured.err == ""


def test_workspace_catalog_lists_selected_image_tool(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "host-v2.json"
    path.write_text(json.dumps(workspace_config_body(tmp_path)))

    assert main(["mcp", "catalog", "--config", str(path)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 2
    assert payload["capabilities"] == ["image"]
    assert [item["tool_id"] for item in payload["tools"]] == ["capicove.image.filter.shape"]


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, KeyError])
@pytest.mark.parametrize("command", ["catalog", "serve"])
def test_mcp_command_does_not_disclose_selected_host_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error_type: type[Exception],
    command: str,
) -> None:
    """A selected provider can fail with private source content in its exception."""
    import capicove.runtime.mcp_host as mcp_host

    path = tmp_path / "host-v2.json"
    path.write_text(json.dumps(workspace_config_body(tmp_path)))

    def fail_selected_host(*_args: object, **_kwargs: object) -> None:
        raise error_type("private-token /outside/workspace source body")

    monkeypatch.setattr(mcp_host, "compose_mcp_host", fail_selected_host)

    assert main(["mcp", command, "--config", str(path)]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "MCP startup failed" in captured.err
    assert "private-token" not in captured.err
    assert "/outside/workspace" not in captured.err
    assert "source body" not in captured.err


def test_catalog_does_not_publish_partial_result_when_host_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import capicove.runtime.mcp_host as mcp_host

    path = tmp_path / "host-v2.json"
    path.write_text(json.dumps(workspace_config_body(tmp_path)))
    original_compose = mcp_host.compose_mcp_host

    @contextmanager
    def fail_after_catalog(
        config: McpHostConfiguration | WorkspaceHostConfiguration, *, catalog_only: bool = False
    ) -> Iterator[tuple[McpToolBinding, ...]]:
        with original_compose(config, catalog_only=catalog_only) as bindings:
            yield bindings
        raise RuntimeError("private-token in cleanup")

    monkeypatch.setattr(mcp_host, "compose_mcp_host", fail_after_catalog)

    assert main(["mcp", "catalog", "--config", str(path)]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "MCP startup failed" in captured.err
    assert "private-token" not in captured.err


@pytest.mark.skipif(os.name == "nt", reason="Memory fails closed on Windows")
def test_workspace_memory_runtime_closes_after_stdio_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from capicove.knowledge import SQLiteMemoryRepository
    from capicove.mcp import stdio
    from capicove.runtime import memory_runtime
    from capicove.tools.memory_tools import MemoryToolHandler

    body = workspace_config_body(tmp_path)
    body["capabilities"] = {
        "image": {"schema_version": 1, "read_roots": [str(tmp_path)]},
        "memory": {
            "schema_version": 1,
            "root": str(tmp_path / "memory"),
            "scopes": ["notes"],
        },
    }
    body["permissions"] = ["image:read", "memory:read"]
    body["enabled_tools"] = ["capicove.image.filter.shape", "capicove.memory.recall"]
    path = tmp_path / "host-v2.json"
    path.write_text(json.dumps(body))
    connections: list[sqlite3.Connection] = []
    original_close = memory_runtime.close_memory_runtime

    def complete_stdio(server: object, *, stdout: object) -> None:
        del server, stdout
        assert (tmp_path / "memory" / "memory.sqlite3").is_file()

    def record_close(handlers: Mapping[str, ToolHandler]) -> None:
        owner = handlers["doctor"]
        assert isinstance(owner, MemoryToolHandler)
        repository = owner.service.knowledge.repository
        assert isinstance(repository, SQLiteMemoryRepository)
        connections.append(repository._db)
        original_close(handlers)

    monkeypatch.setattr(stdio, "run_stdio", complete_stdio)
    monkeypatch.setattr(memory_runtime, "close_memory_runtime", record_close)

    assert main(["mcp", "serve", "--config", str(path)]) == 0
    assert capsys.readouterr().out == ""
    assert len(connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")


@pytest.mark.parametrize("change", ["capability", "tool", "permission", "root", "extra", "trace"])
def test_unavailable_host_fails_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], change: str
) -> None:
    body = config_body(tmp_path)
    if change == "capability":
        body["capability"] = "memory"
    elif change == "tool":
        body["enabled_tools"] = ["capicove.image.ocr"]
    elif change == "permission":
        body["permissions"] = []
    elif change == "root":
        body["capability_config"] = {"read_roots": [str(tmp_path / "absent")]}
    elif change == "trace":
        body["trace_store"] = str(tmp_path / "absent" / "private.db")
    else:
        body["capability_config"] = {"read_roots": [str(tmp_path)], "provider": "secret-value"}
    path = tmp_path / "host.json"
    path.write_text(json.dumps(body))
    assert main(["mcp", "serve", "--config", str(path)]) != 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err
    assert "secret-value" not in captured.err
    assert not (tmp_path / "private.db").exists()


def test_public_imports_and_help_do_not_load_optional_sdks() -> None:
    script = """
import sys
import capicove
import capicove.mcp
import capicove.runtime
from capicove.cli.app import _build_parser
_build_parser().parse_args(["mcp", "catalog", "--config", "/tmp/unused"])
assert not {
    "mcp", "torch", "transformers", "ray", "networkx", "pyarrow",
    "capicove.runtime.reverse_image", "capicove.runtime.mcp_config",
    "capicove.runtime.mcp_host",
}.intersection(sys.modules)
assert set(capicove.runtime.__all__) <= set(dir(capicove.runtime))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr


def test_catalog_accepts_relative_config_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "host-v2.json").write_text(json.dumps(workspace_config_body(tmp_path)))
    monkeypatch.chdir(tmp_path)

    assert main(["mcp", "catalog", "--config", "host-v2.json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert [item["tool_id"] for item in payload["tools"]] == ["capicove.image.filter.shape"]


def test_serve_explains_why_config_path_must_be_absolute(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "host-v2.json").write_text(json.dumps(workspace_config_body(tmp_path)))
    monkeypatch.chdir(tmp_path)

    assert main(["mcp", "serve", "--config", "host-v2.json"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "mcp serve requires an absolute --config path" in captured.err
    assert str(tmp_path / "host-v2.json") in captured.err
