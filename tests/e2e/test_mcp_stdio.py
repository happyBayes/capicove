"""Real stdio subprocesses exercise configured Tools and filesystem rejection."""

import asyncio
import json
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from PIL import Image

from tests.unit.runtime.test_mcp_config import config_body


pytestmark = pytest.mark.e2e


def test_protocol_descriptor_is_private_and_restored_after_failure() -> None:
    script = """
import os
from capicove.mcp.stdio import protocol_stdout
try:
    with protocol_stdout() as stream:
        assert not os.get_inheritable(stream.fileno())
        print('protocol', file=stream, flush=True)
        os.write(1, b'diagnostic\\n')
        raise ValueError('fixture')
except ValueError:
    pass
os.write(1, b'restored\\n')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "protocol\nrestored\n"
    assert result.stderr == "diagnostic\n"


def test_protocol_stdout_restores_descriptor_if_redirection_fails() -> None:
    script = """
import os
from capicove.mcp import stdio

calls = 0
original = stdio._sync_windows_stdout_handle
def fail_once(descriptor):
    global calls
    calls += 1
    if calls == 1:
        raise RuntimeError('fixture redirection failure')
    original(descriptor)
stdio._sync_windows_stdout_handle = fail_once
try:
    with stdio.protocol_stdout():
        raise AssertionError('body must not run after redirection failure')
except RuntimeError as error:
    assert str(error) == 'fixture redirection failure'
os.write(1, b'restored-after-failure\\n')
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert result.stdout == b"restored-after-failure\n"


def test_protocol_stdout_preserves_body_failure_if_handle_restore_fails() -> None:
    script = """
import os
from capicove.mcp import stdio

calls = 0
original = stdio._sync_windows_stdout_handle
def fail_on_restore(descriptor):
    global calls
    calls += 1
    if calls == 2:
        raise OSError('fixture handle restoration failure')
    original(descriptor)
stdio._sync_windows_stdout_handle = fail_on_restore
try:
    with stdio.protocol_stdout():
        raise ValueError('fixture body failure')
except ValueError as error:
    assert str(error) == 'fixture body failure'
    assert any('fixture handle restoration failure' in note for note in error.__notes__)
else:
    raise AssertionError('body failure was masked')
os.write(1, b'restored-after-body-failure\\n')
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert result.stdout == b"restored-after-body-failure\n"


def test_protocol_stdout_reports_handle_restore_failure_without_body_error() -> None:
    script = """
import os
from capicove.mcp import stdio

calls = 0
original = stdio._sync_windows_stdout_handle
def fail_on_restore(descriptor):
    global calls
    calls += 1
    if calls == 2:
        raise OSError('fixture handle restoration failure')
    original(descriptor)
stdio._sync_windows_stdout_handle = fail_on_restore
try:
    with stdio.protocol_stdout():
        pass
except OSError as error:
    assert str(error) == 'fixture handle restoration failure'
else:
    raise AssertionError('handle restoration failure was hidden')
os.write(1, b'restored-after-handle-failure\\n')
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert result.stdout == b"restored-after-handle-failure\n"


def test_protocol_stdout_reports_restore_failure_inside_unrelated_except() -> None:
    script = """
import os
from capicove.mcp import stdio

calls = 0
original = stdio._sync_windows_stdout_handle
def fail_on_restore(descriptor):
    global calls
    calls += 1
    if calls == 2:
        raise OSError('fixture handle restoration failure')
    original(descriptor)
stdio._sync_windows_stdout_handle = fail_on_restore
try:
    raise ValueError('unrelated outer failure')
except ValueError:
    try:
        with stdio.protocol_stdout():
            pass
    except OSError as error:
        assert str(error) == 'fixture handle restoration failure'
    else:
        raise AssertionError('handle restoration failure was hidden')
os.write(1, b'restored-inside-outer-except\\n')
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert result.stdout == b"restored-inside-outer-except\n"


def test_protocol_stdout_reports_cleanup_failure_inside_unrelated_except() -> None:
    script = """
import os
from capicove.mcp import stdio

saved = None
original_dup = stdio.os.dup
original_close = stdio.os.close
def record_dup(descriptor):
    global saved
    result = original_dup(descriptor)
    if descriptor == 1 and saved is None:
        saved = result
    return result
def fail_saved_close(descriptor):
    original_close(descriptor)
    if descriptor == saved:
        raise OSError('fixture descriptor cleanup failure')
stdio.os.dup = record_dup
stdio.os.close = fail_saved_close
try:
    raise ValueError('unrelated outer failure')
except ValueError:
    try:
        with stdio.protocol_stdout():
            pass
    except OSError as error:
        assert str(error) == 'fixture descriptor cleanup failure'
    else:
        raise AssertionError('descriptor cleanup failure was hidden')
os.write(1, b'restored-after-cleanup-failure\\n')
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert result.stdout == b"restored-after-cleanup-failure\n"


def test_windows_stdout_capture_normalizes_crt_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    from capicove.mcp import stdio

    def fail_crt_handle(_descriptor: int) -> int:
        raise OSError("fixture invalid descriptor")

    monkeypatch.setattr(stdio, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(
        stdio,
        "importlib",
        SimpleNamespace(import_module=lambda _name: SimpleNamespace(get_osfhandle=fail_crt_handle)),
    )
    with pytest.raises(
        RuntimeError, match="Windows standard output redirection is unavailable"
    ) as error:
        stdio._windows_stdout_handles()
    assert isinstance(error.value.__cause__, OSError)


@pytest.mark.skipif(os.name != "nt", reason="requires Win32 standard handles")
def test_windows_protocol_isolates_crt_native_and_child_stdout() -> None:
    script = """
import ctypes
import os
import subprocess
import sys
from capicove.mcp.stdio import protocol_stdout

kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
kernel32.GetStdHandle.argtypes = (ctypes.c_uint32,)
kernel32.GetStdHandle.restype = ctypes.c_void_p
kernel32.SetStdHandle.argtypes = (ctypes.c_uint32, ctypes.c_void_p)
kernel32.SetStdHandle.restype = ctypes.c_int
kernel32.WriteFile.argtypes = (
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p,
)
kernel32.WriteFile.restype = ctypes.c_int
def native_write(value):
    written = ctypes.c_uint32()
    handle = kernel32.GetStdHandle(0xFFFFFFF5)
    assert kernel32.WriteFile(handle, value, len(value), ctypes.byref(written), None)
    assert written.value == len(value)

with protocol_stdout() as protocol:
    assert not os.get_inheritable(protocol.fileno())
    print('protocol', file=protocol, flush=True)
    os.write(1, b'crt-diagnostic\\n')
    native_write(b'win32-diagnostic\\n')
    subprocess.run([sys.executable, '-c', "print('child-diagnostic')"], check=True)
os.write(1, b'crt-restored\\n')
native_write(b'win32-restored\\n')
assert kernel32.SetStdHandle(0xFFFFFFF5, kernel32.GetStdHandle(0xFFFFFFF4))
with protocol_stdout() as protocol:
    print('protocol-again', file=protocol, flush=True)
native_write(b'original-native-stderr\\n')
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert result.stdout == b"protocol\ncrt-restored\nwin32-restored\nprotocol-again\n"
    assert b"crt-diagnostic" in result.stderr
    assert b"win32-diagnostic" in result.stderr
    assert b"child-diagnostic" in result.stderr
    assert b"original-native-stderr" in result.stderr


@pytest.mark.slow
def test_cli_catalog_isolates_provider_startup_native_output(tmp_path: Path) -> None:
    config = tmp_path / "host.json"
    config.write_text(json.dumps(config_body(tmp_path)))
    script = """
import os, sys
from capicove.runtime import image_mcp
from capicove.cli.app import main
original = image_mcp.compose_image_mcp
def noisy(*args):
    os.write(1, b'startup-diagnostic\\n')
    return original(*args)
image_mcp.compose_image_mcp = noisy
raise SystemExit(main(['mcp', 'catalog', '--config', sys.argv[1]]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(config)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["tools"][0]["tool_id"] == "capicove.image.filter.shape"
    assert "startup-diagnostic" in result.stderr


async def exercise_stdio(python: str, config: Path, inside: Path, outside: Path) -> None:
    parameters = StdioServerParameters(
        command=python, args=["-m", "capicove", "mcp", "serve", "--config", str(config)]
    )
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(
            reader, writer, read_timeout_seconds=timedelta(seconds=60)
        ) as session:
            await session.initialize()
            listing = await session.list_tools()
            assert [tool.name for tool in listing.tools] == ["capicove_image_filter_shape"]
            good = await session.call_tool(
                listing.tools[0].name, {"sources": [{"path": str(inside)}]}
            )
            assert not good.isError, good
            assert good.structuredContent is not None
            assert good.structuredContent["status"] == "succeeded"
            assert good.structuredContent["result"]["keep"] is True
            bad = await session.call_tool(
                listing.tools[0].name, {"sources": [{"path": str(outside)}]}
            )
            assert bad.structuredContent is not None
            assert bad.structuredContent["error"]["code"] == "permission_denied"
            unknown = await session.call_tool("capicove_memory_erase_scope", {})
            assert unknown.isError
            final = await session.call_tool(listing.tools[0].name, {"sources": []})
            assert not final.isError


@pytest.mark.slow
def test_stdio_roundtrip_and_authority(tmp_path: Path) -> None:
    root = tmp_path / "inputs"
    root.mkdir()
    inside, outside = root / "sample.png", tmp_path / "private.png"
    Image.new("RGB", (2, 3), "red").save(inside)
    Image.new("RGB", (4, 5), "blue").save(outside)
    config = tmp_path / "host.json"
    config.write_text(json.dumps(config_body(root)))
    asyncio.run(exercise_stdio(sys.executable, config, inside, outside))


@pytest.mark.slow
def test_stdio_eof_drains_worker_and_keeps_stdout_protocol_only() -> None:
    script = """
import os, subprocess, sys, time
from capicove.contracts import ToolSpec, ToolResult, ToolStatus
from capicove.mcp.bindings import McpToolBinding, default_context
from capicove.mcp.server import create_bound_mcp_server
from capicove.mcp.stdio import run_stdio
spec = ToolSpec("capicove.test.wait", 1, "Wait for bounded fixture work.",
                {"type":"object"}, {"type":"object"})
def handler(context, arguments):
    print("worker-started", flush=True)
    time.sleep(0.2)
    os.write(1, b"native-diagnostic\\n")
    subprocess.run([sys.executable, "-c", "print('child-diagnostic')"], check=True)
    print("worker-finished", flush=True)
    return ToolResult(context.request_id, context.trace_id, spec.tool_id, 1,
                      ToolStatus.SUCCEEDED, result={})
run_stdio(create_bound_mcp_server(
    (McpToolBinding("capicove_test_wait", spec, handler, default_context),), server_name="test"))
"""

    async def run() -> None:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            script,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert (
            process.stdin is not None and process.stdout is not None and process.stderr is not None
        )
        try:
            process.stdin.write(
                (
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "initialize",
                            "params": {
                                "protocolVersion": "2025-06-18",
                                "capabilities": {},
                                "clientInfo": {"name": "test", "version": "1"},
                            },
                        }
                    )
                    + "\n"
                ).encode()
            )
            await process.stdin.drain()
            initialized = json.loads(await asyncio.wait_for(process.stdout.readline(), 15))
            assert initialized["id"] == 1 and "result" in initialized
            messages = [
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "capicove_test_wait", "arguments": {}},
                },
            ]
            process.stdin.write(
                "".join(json.dumps(message) + "\n" for message in messages).encode()
            )
            await process.stdin.drain()
            while True:
                line = await asyncio.wait_for(process.stderr.readline(), 5)
                assert line, "worker must start before EOF"
                if b"worker-started" in line:
                    break
            process.stdin.close()
            stdout, stderr = await asyncio.wait_for(process.communicate(), 10)
            assert process.returncode == 0, stderr.decode()
            assert b"worker-finished" in stderr
            assert b"native-diagnostic" in stderr
            assert b"child-diagnostic" in stderr
            assert b"worker-" not in stdout
            for line in stdout.splitlines():
                assert json.loads(line)["jsonrpc"] == "2.0"
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    asyncio.run(run())
