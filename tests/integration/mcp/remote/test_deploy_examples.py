"""R6.01 deployment examples parse, verify and stay disabled-by-default safe."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[4]
DEPLOY = ROOT / "deploy" / "remote-mcp"


def test_example_launch_decodes_and_keeps_secrets_out() -> None:
    from capicove.runtime.remote_launch import decode_launch

    text = (DEPLOY / "remote-launch.example.json").read_text()
    launch = decode_launch(text)
    assert launch.listen_host == "127.0.0.1"
    assert launch.revocation_journal is not None and launch.control_store is not None
    assert not launch.revocation_journal.is_relative_to(launch.control_store)
    for word in ('secret"', "password", "token", "BEGIN "):
        assert word not in text


@pytest.mark.skipif(shutil.which("systemd-analyze") is None, reason="systemd is not installed")
def test_systemd_unit_verifies_with_a_real_executable(tmp_path: Path) -> None:
    unit = (DEPLOY / "capicove-mcp.service").read_text()
    assert "NoNewPrivileges=yes" in unit and "CapabilityBoundingSet=\n" in unit
    assert "ProtectSystem=strict" in unit and "/var/run/docker.sock" not in unit
    executable = Path(sys.executable).with_name("capicove")
    assert executable.exists()
    local = tmp_path / "capicove-mcp.service"
    local.write_text(unit.replace("/opt/capicove/venv/bin/capicove", str(executable)))
    result = subprocess.run(
        ["systemd-analyze", "verify", str(local)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
