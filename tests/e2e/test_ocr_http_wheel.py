"""The installed wheel exposes portable OCR serving without model imports."""

import os
import subprocess
from pathlib import Path

import pytest

from tests.e2e.wheel_cache import WheelCache


@pytest.mark.e2e
@pytest.mark.slow
def test_installed_wheel_exposes_ocr_serving_without_model_extras(
    tmp_path: Path, wheel_cache: WheelCache
) -> None:
    installed = wheel_cache.install()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [installed.executable, "models", "serve", "paddleocr", "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--max-pixels" in result.stdout
    probe = subprocess.run(
        [
            installed.python,
            "-c",
            "import sys; import capicove.cli.app; "
            "assert not {'paddleocr','paddle','fastapi','uvicorn','torch'} & sys.modules.keys()",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert probe.returncode == 0, probe.stderr
