"""Verify the image-model lock and downloader ship in the base wheel."""

from __future__ import annotations

import os
import subprocess
import zipfile
from pathlib import Path

import pytest

from tests.e2e.wheel_cache import WheelCache


pytestmark = pytest.mark.e2e


@pytest.mark.slow
def test_wheel_exposes_locked_image_model_dry_run(tmp_path: Path, wheel_cache: WheelCache) -> None:
    wheel = wheel_cache.wheel()
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    assert "capicove/model/image_models.toml" in names
    assert "capicove/model/cli.py" in names
    assert "capicove/model/factory.py" in names
    assert "capicove/model/bge_m3.lock.toml" in names
    assert not any(name.startswith("capicove/model_management/") for name in names)

    environment = wheel_cache.install(dependencies=False).environment
    executable = environment / (
        "Scripts/capicove-download-image-models.exe"
        if os.name == "nt"
        else "bin/capicove-download-image-models"
    )

    result = subprocess.run(
        [str(executable), "--model", "clip-vit-b32", "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "openai/clip-vit-base-patch32" in result.stdout
    assert "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268" in result.stdout  # pragma: allowlist secret
    assert result.stderr == ""
    smoke = subprocess.run(
        [
            str(environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")),
            "-c",
            "from capicove.model import ModelFactory, ModelConfiguration; "
            "import capicove.model.receipt; import importlib.util; import sys; "
            "assert importlib.util.find_spec('capicove.model_management') is None; "
            "assert 'torch' not in sys.modules",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert smoke.returncode == 0, smoke.stderr
