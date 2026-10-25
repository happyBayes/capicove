from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest import MonkeyPatch

from capicove.processing.video import (
    AcceptedVideoSource,
    SceneDetectorName,
    SceneSplitConfig,
)
from capicove.processing.video._adapters.scenedetect import PySceneDetectSceneDetector
from capicove.processing.video.ports import BackendUnavailableError


class FakeTimecode:
    def __init__(self, seconds: float) -> None:
        self._seconds = seconds

    @property
    def seconds(self) -> float:
        return self._seconds


def test_scene_dependency_error_identifies_a_missing_opencv_runtime(
    monkeypatch: MonkeyPatch,
) -> None:
    def fail_import(_: str) -> object:
        raise ModuleNotFoundError("OpenCV could not be found")

    monkeypatch.setattr(
        "capicove.processing.video._adapters.scenedetect.importlib.import_module",
        fail_import,
    )

    with pytest.raises(BackendUnavailableError, match="OpenCV runtime"):
        PySceneDetectSceneDetector().detect(
            AcceptedVideoSource(
                source_id="source-1",
                sha256="a" * 64,
                mime_type="video/mp4",
                data=b"video",
            ),
            SceneSplitConfig(SceneDetectorName.CONTENT),
            datetime.now(UTC) + timedelta(minutes=1),
        )


def test_scene_detection_includes_content_before_the_first_cut(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    source_path = tmp_path / "video.mp4"
    source_path.write_bytes(b"video")
    observed: dict[str, bool] = {}

    class FakeContentDetector:
        def __init__(self, **options: object) -> None:
            del options

    def detect(
        path: str,
        detector: object,
        *,
        start_in_scene: bool,
    ) -> list[tuple[FakeTimecode, FakeTimecode]]:
        del path, detector
        observed["start_in_scene"] = start_in_scene
        return [(FakeTimecode(0.0), FakeTimecode(3.0))]

    fake_module = SimpleNamespace(
        ContentDetector=FakeContentDetector,
        AdaptiveDetector=object,
        ThresholdDetector=object,
        detect=detect,
    )
    monkeypatch.setattr(
        "capicove.processing.video._adapters.scenedetect._load_scenedetect",
        lambda: fake_module,
    )
    monkeypatch.setattr(
        "capicove.processing.video._adapters.scenedetect.importlib.metadata.version",
        lambda _: "0.7.1",
    )

    ranges = PySceneDetectSceneDetector().detect(
        AcceptedVideoSource(
            source_id="source-1",
            sha256="a" * 64,
            mime_type="video/mp4",
            path=source_path,
        ),
        SceneSplitConfig(SceneDetectorName.CONTENT),
        datetime.now(UTC) + timedelta(minutes=1),
    )

    assert observed == {"start_in_scene": True}
    assert [(item.start_seconds, item.end_seconds) for item in ranges] == [(0.0, 3.0)]
