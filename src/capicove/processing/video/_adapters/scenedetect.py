"""Lazy PySceneDetect scene range adapter."""

from __future__ import annotations

import importlib
import importlib.metadata
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

from capicove.processing.video.models import (
    AcceptedVideoSource,
    SceneDetectorName,
    SceneSplitConfig,
    TimeRange,
)
from capicove.processing.video.ports import BackendUnavailableError, SceneDetectionError


def _load_scenedetect() -> ModuleType:
    try:
        return importlib.import_module("scenedetect")
    except ImportError as error:
        if "opencv" in str(error).lower() or "cv2" in str(error).lower():
            raise BackendUnavailableError(
                "the OpenCV runtime is unavailable; reinstall capicove[video-scene]"
            ) from error
        raise BackendUnavailableError(
            "install capicove[video-scene] to use PySceneDetect"
        ) from error


class PySceneDetectSceneDetector:
    """Detect scene boundaries with PySceneDetect loaded on first use."""

    def __init__(self) -> None:
        self._version: str | None = None

    @property
    def identity(self) -> tuple[str, str | None]:
        """Return the detector backend identity after availability resolution."""
        return "scenedetect", self._version

    def detect(
        self,
        source: AcceptedVideoSource,
        config: SceneSplitConfig,
        deadline: datetime,
    ) -> tuple[TimeRange, ...]:
        """Return normalized detector ranges without writing media outputs."""
        module = _load_scenedetect()
        try:
            self._version = importlib.metadata.version("scenedetect-headless")
        except importlib.metadata.PackageNotFoundError:
            self._version = None
        detector_class = {
            SceneDetectorName.CONTENT: module.ContentDetector,
            SceneDetectorName.ADAPTIVE: module.AdaptiveDetector,
            SceneDetectorName.THRESHOLD: module.ThresholdDetector,
        }[config.detector]
        detector: Any = detector_class(**dict(config.options))
        try:
            with _materialized_path(source) as source_path:
                scenes: Any = module.detect(
                    str(source_path),
                    detector,
                    start_in_scene=True,
                )
            return tuple(
                TimeRange(float(start.seconds), float(end.seconds))
                for start, end in scenes
                if float(end.seconds) > float(start.seconds)
            )
        except BackendUnavailableError:
            raise
        except Exception as error:
            raise SceneDetectionError("PySceneDetect failed to detect scenes") from error


@contextmanager
def _materialized_path(source: AcceptedVideoSource) -> Iterator[Path]:
    if source.path is not None:
        yield source.path
        return
    if source.data is None:
        raise SceneDetectionError("video source is not materialized")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temporary:
            temporary.write(source.data)
            temporary_path = Path(temporary.name)
        yield temporary_path
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
