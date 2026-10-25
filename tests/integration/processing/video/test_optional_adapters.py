import builtins
import hashlib
import importlib
import importlib.metadata
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from packaging.requirements import Requirement

from capicove.processing.video import (
    AcceptedVideoSource,
    SceneDetectorName,
    SceneSplitConfig,
    TimeRange,
)
from capicove.processing.video.ports import BackendUnavailableError, PendingOutput


pytestmark = pytest.mark.integration


@pytest.mark.parametrize("extra", ["video-scene", "video", "video-all"])
def test_video_scene_extra_installs_a_headless_opencv_runtime(extra: str) -> None:
    requirements = [Requirement(value) for value in importlib.metadata.requires("capicove") or []]
    selected = {
        requirement.name
        for requirement in requirements
        if requirement.marker is None or requirement.marker.evaluate({"extra": extra})
    }

    assert {"opencv-python-headless", "scenedetect-headless"} <= selected
    assert {"opencv-python", "scenedetect"}.isdisjoint(selected)


def test_public_video_import_does_not_import_optional_media_packages() -> None:
    sys.modules.pop("av", None)
    sys.modules.pop("scenedetect", None)

    importlib.import_module("capicove")
    importlib.import_module("capicove.processing.video")
    importlib.import_module("capicove.tools")

    assert "av" not in sys.modules
    assert "scenedetect" not in sys.modules


@pytest.mark.parametrize(
    ("module_name", "class_name", "method_name"),
    [
        ("capicove.processing.video._adapters.pyav", "PyAvVideoInspector", "inspect"),
        ("capicove.processing.video._adapters.scenedetect", "PySceneDetectSceneDetector", "detect"),
    ],
)
def test_optional_adapter_reports_actionable_missing_dependency(
    module_name: str,
    class_name: str,
    method_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module(module_name)
    adapter = getattr(module, class_name)()
    source = AcceptedVideoSource(
        source_id="source-1",
        sha256="a" * 64,
        mime_type="video/mp4",
        data=b"video",
    )
    real_import = builtins.__import__

    def blocked_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        if name.split(".", maxsplit=1)[0] in {"av", "scenedetect"}:
            raise ImportError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(BackendUnavailableError, match=r"capicove\[video-"):
        if method_name == "inspect":
            adapter.inspect(source, datetime.now(UTC) + timedelta(minutes=1))
        else:
            adapter.detect(
                source,
                SceneSplitConfig(SceneDetectorName.CONTENT),
                datetime.now(UTC) + timedelta(minutes=1),
            )


def test_pyav_inspects_and_writes_when_installed(tmp_path: Path) -> None:
    pytest.importorskip("av", reason="install capicove[video-av]")
    dataset = Path("tests/datasets/muscian_sound.mp4")
    if not dataset.exists():
        pytest.skip("local ignored test video is not present")
    from capicove.processing.video._adapters.pyav import PyAvClipWriter, PyAvVideoInspector

    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    source = AcceptedVideoSource(
        source_id=f"sha256:{digest}",
        sha256=digest,
        mime_type="video/mp4",
        path=dataset.resolve(),
    )
    deadline = datetime.now(UTC) + timedelta(minutes=1)
    destination = PendingOutput("b" * 64, tmp_path / "segment.mp4.partial")

    metadata = PyAvVideoInspector().inspect(source, deadline)
    written = PyAvClipWriter().write(source, TimeRange(0.0, 1.0), destination, deadline)
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            str(written.path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream_types = {item["codec_type"] for item in json.loads(completed.stdout)["streams"]}

    assert metadata.duration_seconds > 0
    assert written.path.stat().st_size > 0
    assert stream_types == {"video", "audio"}


def test_scene_detect_returns_normalized_ranges_when_installed() -> None:
    pytest.importorskip("scenedetect", reason="install capicove[video-scene]")
    dataset = Path("tests/datasets/muscian_sound.mp4")
    if not dataset.exists():
        pytest.skip("local ignored test video is not present")
    from capicove.processing.video._adapters.scenedetect import PySceneDetectSceneDetector

    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    ranges = PySceneDetectSceneDetector().detect(
        AcceptedVideoSource(
            source_id=f"sha256:{digest}",
            sha256=digest,
            mime_type="video/mp4",
            path=dataset.resolve(),
        ),
        SceneSplitConfig(SceneDetectorName.CONTENT, {"threshold": 27.0}),
        datetime.now(UTC) + timedelta(minutes=1),
    )

    assert all(item.end_seconds > item.start_seconds for item in ranges)
