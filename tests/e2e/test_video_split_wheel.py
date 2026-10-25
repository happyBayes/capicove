import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from tests.e2e.wheel_cache import WheelCache


pytestmark = pytest.mark.e2e


@pytest.mark.slow
def test_base_wheel_exposes_video_tool_without_optional_backends(
    tmp_path: Path, wheel_cache: WheelCache
) -> None:
    wheel = wheel_cache.wheel()
    installed = wheel_cache.install()
    python = Path(installed.python)
    executable = Path(installed.executable)

    imported = subprocess.run(
        [
            str(python),
            "-c",
            "import capicove, capicove.processing.video, capicove.tools.video_split",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    help_result = subprocess.run(
        [str(executable), "video", "split", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    unavailable = subprocess.run(
        [
            str(executable),
            "video",
            "split",
            "--input",
            "-",
            "--strategy",
            "duration",
            "--backend",
            "av",
            "--output",
            "json",
        ],
        input=b"video",
        check=False,
        capture_output=True,
    )

    assert imported.returncode == 0
    assert help_result.returncode == 0
    assert "Split a video" in help_result.stdout
    assert unavailable.returncode == 1
    payload = json.loads(unavailable.stdout)
    assert payload["error"]["code"] == "unavailable"
    assert "Traceback" not in unavailable.stderr.decode()

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    assert "capicove/processing/video/splitter.py" in names
    assert "capicove/tools/video_split.py" in names
    assert "capicove/mcp/video_split.py" in names
    assert not any("video_spliter.py" in name for name in names)
    assert not any(name.startswith(("tests/", "reference/")) for name in names)
