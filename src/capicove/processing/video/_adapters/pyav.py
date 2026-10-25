"""Lazy PyAV inspection and clip writing adapters."""

from __future__ import annotations

import importlib
import io
from datetime import datetime
from types import ModuleType
from typing import Any

from capicove.processing.video._adapters.ffmpeg import FfmpegClipWriter
from capicove.processing.video.models import (
    AcceptedVideoSource,
    TimeRange,
    VideoWriteMode,
)
from capicove.processing.video.ports import (
    BackendUnavailableError,
    PendingOutput,
    VideoBackendBundle,
    VideoInspectionError,
    VideoMetadata,
    WrittenClip,
)


def _load_av() -> ModuleType:
    try:
        return importlib.import_module("av")
    except ImportError as error:
        raise BackendUnavailableError("install capicove[video-av] to use PyAV") from error


def _source_value(source: AcceptedVideoSource) -> str | io.BytesIO:
    if source.path is not None:
        return str(source.path)
    if source.data is not None:
        return io.BytesIO(source.data)
    raise VideoInspectionError("video source is not materialized")


class PyAvVideoInspector:
    """Inspect normalized metadata with PyAV loaded only on first use."""

    def inspect(self, source: AcceptedVideoSource, deadline: datetime) -> VideoMetadata:
        """Return duration and packet keyframe timestamps."""
        av = _load_av()
        container: Any = None
        try:
            container = av.open(_source_value(source))
            stream = next((item for item in container.streams if item.type == "video"), None)
            if stream is None:
                raise VideoInspectionError("PyAV found no video stream")
            time_base = stream.time_base
            if stream.duration is not None and time_base is not None:
                duration = float(stream.duration * time_base)
            elif container.duration is not None:
                duration = float(container.duration / av.time_base)
            else:
                raise VideoInspectionError("PyAV found no video duration")
            timestamps = tuple(
                float(packet.pts * time_base)
                for packet in container.demux(stream)
                if packet.is_keyframe and packet.pts is not None and time_base is not None
            )
            return VideoMetadata(duration, timestamps, "av", str(av.__version__))
        except BackendUnavailableError:
            raise
        except VideoInspectionError:
            raise
        except Exception as error:
            raise VideoInspectionError("PyAV failed to inspect the video") from error
        finally:
            if container is not None:
                container.close()


class PyAvClipWriter:
    """Write PyAV-inspected ranges with the safe shared FFmpeg writer."""

    def __init__(
        self,
        write_mode: VideoWriteMode = VideoWriteMode.ACCURATE,
        *,
        ffmpeg_executable: str = "ffmpeg",
    ) -> None:
        # The writer still shells out to FFmpeg; a host passes its pinned executable.
        self._writer = FfmpegClipWriter(ffmpeg_executable, write_mode=write_mode)

    def write(
        self,
        source: AcceptedVideoSource,
        time_range: TimeRange,
        destination: PendingOutput,
        deadline: datetime,
    ) -> WrittenClip:
        """Write an accurate or copy-mode clip while preserving audio."""
        _load_av()
        return self._writer.write(source, time_range, destination, deadline)


def create_pyav_backend(
    write_mode: VideoWriteMode = VideoWriteMode.ACCURATE,
    *,
    ffmpeg_executable: str = "ffmpeg",
) -> VideoBackendBundle:
    """Create a PyAV backend bundle or raise an actionable availability error."""
    av = _load_av()
    return VideoBackendBundle(
        name="av",
        version=str(av.__version__),
        inspector=PyAvVideoInspector(),
        writer=PyAvClipWriter(write_mode, ffmpeg_executable=ffmpeg_executable),
    )
