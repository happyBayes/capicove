"""The repository test configuration bounds temporary storage to active tests."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


PYPROJECT = Path(__file__).resolve().parents[3] / "pyproject.toml"


@pytest.mark.slow
def test_passing_tests_release_tmp_path_while_failures_keep_it(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "test_sample.py").write_text(
        textwrap.dedent(
            """
            def test_passes(tmp_path):
                (tmp_path / "artifact").write_text("passed")

            def test_fails(tmp_path):
                (tmp_path / "artifact").write_text("failed")
                assert False
            """
        )
    )
    base = tmp_path / "base"

    completed = subprocess.run(
        [
            *(sys.executable, "-m", "pytest", "-c", str(PYPROJECT), "--rootdir", str(suite)),
            *("-p", "no:cacheprovider", "-p", "no:xdist", "--basetemp", str(base), "-q"),
            str(suite / "test_sample.py"),
        ],
        cwd=suite,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert not (base / "test_passes0").exists()
    assert (base / "test_fails0" / "artifact").read_text() == "failed"
