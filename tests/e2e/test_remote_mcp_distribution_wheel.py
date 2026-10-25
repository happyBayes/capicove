"""R6.01: the built wheel on Linux x86-64 / Python 3.11 with and without remote extras."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.e2e.wheel_cache import WheelCache
from tests.integration.mcp.remote.test_launch import _setup


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux distribution check"),
]
LIGHTWEIGHT = (
    "import json, sys; import capicove, capicove.runtime.remote_launch, capicove.tools.remote_control;"
    "heavy = sorted(m for m in ('mcp', 'jwt', 'httpx', 'uvicorn', 'starlette', 'torch', 'numpy')"
    " if m in sys.modules); print(json.dumps(heavy))"
)


def _check(executable: str, config: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [executable, "mcp", "remote-check", "--config", str(config)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
        env={**os.environ, "PYTHONPATH": ""},
    )


@pytest.mark.slow
def test_base_wheel_stays_offline_and_lightweight_and_refuses_readiness(
    tmp_path: Path, wheel_cache: WheelCache
) -> None:
    installed = wheel_cache.install(())
    imported = subprocess.run(
        [installed.python, "-c", LIGHTWEIGHT],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
        env={**os.environ, "PYTHONPATH": ""},
    )
    assert json.loads(imported.stdout) == []
    result = _check(installed.executable, _setup(tmp_path))
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert "ready" not in result.stdout


@pytest.mark.slow
def test_remote_extra_wheel_reports_sanitized_readiness(
    tmp_path: Path, wheel_cache: WheelCache
) -> None:
    installed = wheel_cache.install(("remote-mcp",))
    config = _setup(tmp_path)
    result = _check(installed.executable, config)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "ready" and report["listener"] == "loopback"
    assert str(tmp_path) not in result.stdout
