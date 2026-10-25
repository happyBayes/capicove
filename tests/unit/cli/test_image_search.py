"""Executable image-search CLI boundaries."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("command", ["search", "serve"])
def test_image_search_entry_points_are_executable_and_lazy(command: str) -> None:
    code = (
        "import sys; from capicove.cli.app import main; "
        f"\ntry: main(['image', '{command}', '--help'])"
        "\nexcept SystemExit as error:"
        "\n assert error.code == 0"
        "\nassert 'onnxruntime' not in sys.modules"
        "\nassert 'mcp' not in sys.modules"
        "\nassert 'cv2' not in sys.modules"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "--config" in completed.stdout


@pytest.mark.slow
@pytest.mark.parametrize("command", ["search", "serve"])
def test_invalid_startup_is_sanitized(tmp_path: Path, command: str) -> None:
    private = tmp_path / "private-customer-name.json"
    private.write_text('{"secret": "not-a-real-secret"}')  # pragma: allowlist secret
    args = ["image", command, "--config", str(private)]
    if command == "search":
        args.extend(["--arguments", str(private)])
    completed = subprocess.run(
        [sys.executable, "-m", "capicove", *args], capture_output=True, text=True, timeout=10
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "startup or transport failed" in completed.stderr
    assert "private-customer" not in completed.stderr
    assert "not-a-real-secret" not in completed.stderr


@pytest.mark.parametrize("command", ["search", "serve"])
def test_image_search_rejects_trace_store(command: str) -> None:
    args = ["--trace-store", "/nonexistent/trace", "image", command, "--config", "/missing.json"]
    if command == "search":
        args.extend(["--arguments", "/missing-query.json"])
    completed = subprocess.run(
        [sys.executable, "-m", "capicove", *args], capture_output=True, text=True, timeout=10
    )
    assert completed.returncode == 2
    assert "do not yet support --trace-store" in completed.stderr
