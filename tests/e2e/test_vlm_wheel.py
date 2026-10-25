"""Installed VLM commands and HTTP client need no GPU SDK or source checkout."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from tests.e2e.wheel_cache import WheelCache
from tests.unit.model.test_openai_vlm import upstream as upstream
from tests.unit.model.test_vlm import configuration


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.parametrize("dependencies", [False, True])
def test_installed_vlm_cli_calls_generic_endpoint(
    tmp_path: Path,
    wheel_cache: WheelCache,
    dependencies: bool,
    upstream: tuple[str, dict[str, object]],
) -> None:
    installed = wheel_cache.install(extras=("vlm",), dependencies=dependencies)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    raw = configuration()
    raw["profiles"]["vision"]["endpoint"] = upstream[0]
    config = tmp_path / "models.json"
    config.write_text(json.dumps(raw))
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps({"messages": [{"role": "user", "content": "Describe the picture."}]})
    )
    for args in (
        ["--help"],
        ["list", "--config", str(config)],
        ["serve", "--help"],
        [
            "generate",
            "--config",
            str(config),
            "--binding",
            "assistant.vision",
            "--input",
            str(request),
        ],
    ):
        result = subprocess.run(
            [installed.executable, "models", "vlm", *args],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if not dependencies and args[0] == "generate":
            assert result.returncode == 2
            continue
        assert result.returncode == 0, result.stderr
    if dependencies:
        assert json.loads(result.stdout)["choices"][0]["message"]["content"] == "A red square."
    result = subprocess.run(
        [
            installed.python,
            "-c",
            "from capicove.cli.app import _build_parser; import sys; _build_parser().parse_args(['models','vlm','serve','--config','/tmp/models.json']); assert not {'torch','transformers','fastapi','uvicorn','PIL'} & sys.modules.keys()",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.e2e
@pytest.mark.slow
def test_source_distribution_ships_reusable_vlm_deployment(tmp_path: Path) -> None:
    import tarfile

    subprocess.run(
        ["uv", "build", "--sdist", "--out-dir", str(tmp_path)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    (archive,) = tmp_path.glob("capicove-*.tar.gz")
    with tarfile.open(archive) as source:
        paths = {"/".join(member.name.split("/")[1:]) for member in source}
    assert {
        "scripts/vlm/deploy.py",
        "scripts/vlm/client.py",
        "scripts/vlm/models.toml",
        "scripts/vlm/qwen35-9b.toml",
        "scripts/vlm/README.md",
    } <= paths
    assert not any(path.endswith(".safetensors") or "/tmp/" in path for path in paths)
