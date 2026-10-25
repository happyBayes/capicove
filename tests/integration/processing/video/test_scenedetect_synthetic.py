"""Real scene detection compatibility without private video fixtures."""

import hashlib
import warnings
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
import pytest

from capicove.processing.video import AcceptedVideoSource, SceneDetectorName, SceneSplitConfig
from capicove.processing.video._adapters.scenedetect import PySceneDetectSceneDetector
from capicove.processing.video.ports import SceneDetectionError


pytestmark = pytest.mark.integration


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    path = tmp_path / "three-scenes.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 30, (64, 64))
    if not writer.isOpened():
        raise RuntimeError("Synthetic MJPG writer is unavailable")
    try:
        for level in (0, 255, 0):
            frame = np.full((64, 64, 3), level, dtype=np.uint8)
            for _ in range(30):
                writer.write(frame)
    finally:
        writer.release()
    return path


@pytest.mark.parametrize("detector", tuple(SceneDetectorName))
@pytest.mark.parametrize("inline", [False, True])
def test_scene_07_preserves_range_bounds_for_path_and_bytes(
    synthetic_video: Path, detector: SceneDetectorName, inline: bool
) -> None:
    pytest.importorskip("scenedetect", reason="install capicove[video-scene]")
    payload = synthetic_video.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    source = AcceptedVideoSource(
        source_id=f"sha256:{digest}",
        sha256=digest,
        mime_type="video/avi",
        path=None if inline else synthetic_video,
        data=payload if inline else None,
    )
    adapter = PySceneDetectSceneDetector()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        ranges = adapter.detect(
            source,
            SceneSplitConfig(detector, {"min_scene_len": 5}),
            datetime.now(UTC) + timedelta(seconds=30),
        )
    assert ranges
    assert ranges[0].start_seconds == pytest.approx(0)
    assert ranges[-1].end_seconds == pytest.approx(3, abs=1 / 30)
    assert all(r.end_seconds > r.start_seconds for r in ranges)
    assert all(left.end_seconds == right.start_seconds for left, right in pairwise(ranges))
    if detector in (SceneDetectorName.CONTENT, SceneDetectorName.ADAPTIVE):
        assert [r.start_seconds for r in ranges] == pytest.approx([0, 1, 2], abs=1 / 30)
    assert adapter.identity[0] == "scenedetect"
    assert adapter.identity[1] is not None
    assert adapter.identity[1].startswith("0.7.")


def test_scene_07_invalid_media_is_a_canonical_detection_error() -> None:
    pytest.importorskip("scenedetect", reason="install capicove[video-scene]")
    with pytest.raises(SceneDetectionError):
        PySceneDetectSceneDetector().detect(
            AcceptedVideoSource(
                source_id="invalid-media",
                sha256="a" * 64,
                mime_type="video/mp4",
                data=b"not a video",
            ),
            SceneSplitConfig(SceneDetectorName.CONTENT),
            datetime.now(UTC) + timedelta(seconds=30),
        )
