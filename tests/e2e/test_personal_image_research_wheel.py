"""Verify the research module ships without becoming a public retrieval API."""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tests.e2e.wheel_cache import WheelCache


pytestmark = pytest.mark.e2e


@pytest.mark.slow
def test_wheel_contains_importable_internal_personal_image_research(
    tmp_path: Path,
    wheel_cache: WheelCache,
) -> None:
    wheel = wheel_cache.wheel()

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())

    assert "capicove/retrieval/personal_image_research.py" in names
    assert "capicove/retrieval/_adapters/opencv_phash.py" in names
    assert not any(name.startswith(("tests/", "reference/")) for name in names)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(wheel)
    imported = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import capicove.retrieval as public_retrieval; "
                "import capicove.retrieval.personal_image_research as research; "
                "assert '.whl/capicove/retrieval/personal_image_research.py' "
                "in research.__file__.replace('\\\\', '/'); "
                "assert not hasattr(public_retrieval, 'PersonalImageResearchIndex'); "
                "assert not hasattr(public_retrieval, 'ResearchSearchIntent')"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert imported.returncode == 0, imported.stderr
