"""A base wheel exposes reranker commands without eager optional imports."""

import os
import subprocess
from pathlib import Path

import pytest

from tests.e2e.wheel_cache import WheelCache


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.parametrize("dependencies", [False, True])
def test_installed_wheel_exposes_reranker_without_sdk_imports(
    tmp_path: Path, wheel_cache: WheelCache, dependencies: bool
) -> None:
    installed = wheel_cache.install(dependencies=dependencies)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    for args in (
        ["models", "serve", "bge-reranker-v2-m3", "--help"],
        ["models", "rerank", "--help"],
    ):
        result = subprocess.run(
            [installed.executable, *args],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
    probe = subprocess.run(
        [
            installed.python,
            "-c",
            "from capicove.cli.app import _build_parser; import sys; "
            "_build_parser().parse_args(['models','rerank','--model-directory','/tmp/model','--input','-']); "
            "assert not {'torch','transformers','fastapi','uvicorn','huggingface_hub'} & sys.modules.keys()",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert probe.returncode == 0, probe.stderr
