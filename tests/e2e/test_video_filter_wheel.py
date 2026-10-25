import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tests.e2e.wheel_cache import WheelCache


pytestmark = pytest.mark.e2e


def test_unrelated_transports_do_not_load_image_retrieval_dependencies() -> None:
    imported = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import builtins; "
                "original_import = builtins.__import__; "
                "blocked = {'PIL', 'cv2', 'numpy'}; "
                "\ndef guarded_import(name, *args, **kwargs):"
                "\n    if name.partition('.')[0] in blocked:"
                "\n        raise AssertionError(f'unexpected optional import: {name}')"
                "\n    return original_import(name, *args, **kwargs)"
                "\nbuiltins.__import__ = guarded_import; "
                "\nimport capicove.tools.video_filter; "
                "import capicove.mcp.video_filter"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert imported.returncode == 0, imported.stderr


@pytest.mark.slow
def test_base_wheel_exposes_filter_without_optional_providers(
    tmp_path: Path, wheel_cache: WheelCache
) -> None:
    wheel = wheel_cache.wheel()

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode("utf-8")

    assert "capicove/processing/video/filtering/video_filter_contracts.py" in names
    assert "capicove/processing/video/filtering/video_filter_pipeline.py" in names
    assert "capicove/tools/video_filter.py" in names
    assert "capicove/mcp/video_filter.py" in names
    assert "capicove/cli/commands/video_filter.py" in names
    assert "capicove/retrieval/search.py" in names
    assert "capicove/runtime/reverse_image.py" in names
    assert "capicove/tools/search.py" in names
    assert "capicove/mcp/search.py" in names
    for extra in (
        "retrieval-cpu",
        "video-filter",
        "video-filter-ocr",
        "video-filter-models",
        "video-filter-tags",
        "video-all",
    ):
        assert f"Provides-Extra: {extra}" in metadata
    assert "Requires-Dist: onnxruntime<2,>=1.20; extra == 'retrieval-cpu'" in metadata
    assert not any(name.startswith(("tests/", "reference/")) for name in names)

    installed = wheel_cache.install(dependencies=False)
    python = Path(installed.python)
    executable = Path(installed.executable)

    imported = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import capicove.processing.video.filtering; "
                "import capicove.tools.video_filter; import capicove.mcp.video_filter"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    help_result = subprocess.run(
        [str(executable), "video", "filter", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    unavailable = subprocess.run(
        [
            str(python),
            "-c",
            (
                "from capicove.processing.video.filtering import LocalModelReference; "
                "from capicove.processing.video.filtering._adapters."
                "huggingface_frame_scorers import HuggingFaceLocalModelResolver; "
                "from capicove.processing.video.filtering.video_filter_ports import "
                "BackendUnavailableFilterError; "
                "\ntry: HuggingFaceLocalModelResolver().resolve("
                "LocalModelReference('missing-local-model'), model_kind='classification')"
                "\nexcept BackendUnavailableFilterError: raise SystemExit(0)"
                "\nraise SystemExit(1)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert imported.returncode == 0
    assert help_result.returncode == 0
    assert "criteria locally" in help_result.stdout
    assert unavailable.returncode == 0
    assert "Traceback" not in unavailable.stderr
