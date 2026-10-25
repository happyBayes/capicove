"""Contract tests for execution backend plugin metadata."""

from __future__ import annotations

import subprocess
import sys


def test_plugin_package_import_does_not_load_execution_sdks() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import capicove.plugins; import sys; "
                "assert 'pyspark' not in sys.modules; "
                "assert 'ray' not in sys.modules; "
                "assert 'elasticsearch' not in sys.modules; "
                "assert 'paddleocr' not in sys.modules; "
                "assert 'paddle' not in sys.modules; "
                "assert 'paddlex' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
