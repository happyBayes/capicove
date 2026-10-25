"""The reusable VLM launcher verifies local assets before invoking vLLM."""

import hashlib
import runpy
import tomllib
from pathlib import Path
from typing import Any

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/vlm/deploy.py"


def receipt(root: Path) -> dict[str, Any]:
    (root / "config.json").write_text("{}")
    return {
        "schema_version": 1,
        "repository": "Qwen/Qwen3.5-9B",
        "revision": "a" * 40,
        "license": "apache-2.0",
        "format": "safetensors",
        "conversion": "none",
        "files": {"config.json": hashlib.sha256(b"{}").hexdigest()},
    }


def test_asset_change_and_extra_files_fail_before_start(tmp_path: Path) -> None:
    module = runpy.run_path(str(SCRIPT))
    manifest = receipt(tmp_path)
    module["verify_assets"](tmp_path, manifest)
    (tmp_path / "config.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="hash"):
        module["verify_assets"](tmp_path, manifest)
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "unreviewed.py").write_text("")
    with pytest.raises(ValueError, match="inventory"):
        module["verify_assets"](tmp_path, manifest)


def test_manifest_paths_cannot_escape_model_root(tmp_path: Path) -> None:
    module = runpy.run_path(str(SCRIPT))
    manifest = receipt(tmp_path)
    manifest["files"] = {"../outside": "b" * 64}
    with pytest.raises(ValueError):
        module["verify_assets"](tmp_path, manifest)


def test_qwen_command_has_tools_and_uses_no_credential_argument(tmp_path: Path) -> None:
    module = runpy.run_path(str(SCRIPT))
    config = tomllib.loads((SCRIPT.parent / "qwen35-9b.toml").read_text())
    command = module["serve_command"](tmp_path, config)
    assert "--enable-auto-tool-choice" in command
    assert "qwen3_xml" in command
    assert "--api-key" not in command
    assert "--trust-remote-code" not in command
