"""The base wheel exposes model serving without importing optional runtimes."""

import os
import subprocess
from pathlib import Path

import pytest

from tests.e2e.wheel_cache import WheelCache


@pytest.mark.e2e
@pytest.mark.slow
def test_installed_wheel_exposes_bge_serving_help_without_model_extras(
    tmp_path: Path, wheel_cache: WheelCache
) -> None:
    installed = wheel_cache.install()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [installed.executable, "models", "serve", "bge-m3", "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--model-manifest-sha256" in result.stdout
    probe = subprocess.run(
        [
            installed.python,
            "-c",
            "import sys; import capicove.api, capicove.cli.app; "
            "assert not {'torch','fastapi','uvicorn','sentence_transformers'} & sys.modules.keys()",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert probe.returncode == 0, probe.stderr
