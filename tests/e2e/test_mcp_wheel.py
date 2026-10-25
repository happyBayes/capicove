"""Verify the actual installed command, lazy imports, missing extra, and stdio."""

import asyncio
import json
import os
import subprocess
from datetime import timedelta
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from PIL import Image

from tests.e2e.test_mcp_stdio import exercise_stdio
from tests.e2e.wheel_cache import WheelCache
from tests.unit.runtime.test_mcp_config import config_body


pytestmark = pytest.mark.e2e


async def exercise_selected_memory_stdio(python: str, config: Path) -> None:
    """Check selected Memory authority in the installed MCP process."""
    parameters = StdioServerParameters(
        command=python, args=["-m", "capicove", "mcp", "serve", "--config", str(config)]
    )
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(
            reader, writer, read_timeout_seconds=timedelta(seconds=60)
        ) as session:
            await session.initialize()
            listing = await session.list_tools()
            assert [tool.name for tool in listing.tools] == [
                "capicove_image_filter_shape",
                "capicove_memory_recall",
            ]
            good = await session.call_tool("capicove_memory_recall", {"scope": "notes"})
            assert good.isError is False
            assert good.structuredContent is not None
            assert good.structuredContent["status"] == "succeeded"
            denied = await session.call_tool("capicove_memory_recall", {"scope": "foreign"})
            assert denied.structuredContent is not None
            assert denied.structuredContent["error"]["code"] == "permission_denied"


@pytest.mark.slow
def test_installed_mcp_wheel(tmp_path: Path, wheel_cache: WheelCache) -> None:
    # This journey changes its environment (base install, then the mcp extra), so it
    # owns a private one; the wheel and locked constraints come from the session.
    wheel = wheel_cache.wheel()
    private = wheel_cache.private_environment(tmp_path / "venv")
    python, executable = private.python, private.executable
    constraints = wheel_cache.constraints(("mcp",))
    installed = subprocess.run(
        ["uv", "pip", "install", "--constraints", str(constraints), "--python", python, str(wheel)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert installed.returncode == 0, f"{installed.args}\n{installed.stdout}\n{installed.stderr}"
    for args in (["--version"], ["mcp", "--help"], ["mcp", "serve", "--help"]):
        result = subprocess.run([executable, *args], capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr
    probe = subprocess.run(
        [
            python,
            "-c",
            "import capicove, capicove.mcp, capicove.runtime, sys; assert 'mcp' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert probe.returncode == 0, probe.stderr
    root = tmp_path / "inputs"
    root.mkdir()
    inside, outside = root / "sample.png", tmp_path / "private.png"
    Image.new("RGB", (2, 3), "red").save(inside)
    Image.new("RGB", (4, 5), "blue").save(outside)
    config = tmp_path / "host.json"
    config.write_text(json.dumps(config_body(root)))
    missing = subprocess.run(
        [executable, "mcp", "serve", "--config", str(config)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert missing.returncode == 2
    assert missing.stdout == ""
    assert "capicove[mcp]" in missing.stderr
    installed = subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--constraints",
            str(constraints),
            "--python",
            python,
            f"{wheel}[mcp]",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert installed.returncode == 0, f"{installed.args}\n{installed.stdout}\n{installed.stderr}"
    catalog = subprocess.run(
        [executable, "mcp", "catalog", "--config", str(config)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert catalog.returncode == 0, catalog.stderr
    assert len(json.loads(catalog.stdout)["tools"]) == 1
    asyncio.run(exercise_stdio(python, config, inside, outside))

    workspace = tmp_path / "new-workspace"
    initialized = subprocess.run(
        [executable, "setup", "init", str(workspace)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert initialized.returncode == 0, initialized.stderr
    version_two = workspace / "host.json"
    readiness = subprocess.run(
        [executable, "setup", "doctor", "--config", str(version_two)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert readiness.returncode == 0, readiness.stderr
    assert json.loads(readiness.stdout)["ready"] is True
    version_two_catalog = subprocess.run(
        [executable, "mcp", "catalog", "--config", str(version_two)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert version_two_catalog.returncode == 0, version_two_catalog.stderr
    assert json.loads(version_two_catalog.stdout)["schema_version"] == 2
    starter_image = workspace / "inputs" / "sample.png"
    Image.new("RGB", (2, 3), "red").save(starter_image)
    asyncio.run(exercise_stdio(python, version_two, starter_image, outside))

    if os.name != "nt":
        selected = json.loads(version_two.read_text())
        memory_root = workspace / "memory"
        selected["capabilities"]["memory"] = {
            "schema_version": 1,
            "root": str(memory_root),
            "scopes": ["notes"],
        }
        selected["permissions"].append("memory:read")
        selected["enabled_tools"].append("capicove.memory.recall")
        selected_config = workspace / "selected-host.json"
        selected_config.write_text(json.dumps(selected))
        selected_readiness = subprocess.run(
            [executable, "setup", "doctor", "--config", str(selected_config)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert selected_readiness.returncode == 0, selected_readiness.stderr
        assert json.loads(selected_readiness.stdout)["tools"] == [
            {"tool_id": "capicove.image.filter.shape", "status": "ready"},
            {"tool_id": "capicove.memory.recall", "status": "ready"},
        ]
        assert not memory_root.exists()
        selected_catalog = subprocess.run(
            [executable, "mcp", "catalog", "--config", str(selected_config)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert selected_catalog.returncode == 0, selected_catalog.stderr
        assert [tool["tool_id"] for tool in json.loads(selected_catalog.stdout)["tools"]] == [
            "capicove.image.filter.shape",
            "capicove.memory.recall",
        ]
        assert not memory_root.exists()
        asyncio.run(exercise_selected_memory_stdio(python, selected_config))
