"""Lazy PyAV adapter for bounded chronological frame sampling."""

from __future__ import annotations

import importlib
import importlib.metadata
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from capicove.processing.video.filtering.video_filter_contracts import (
    FrameSamplingConfig,
    SamplingMethod,
    VideoFilterWarning,
    VideoFilterWarningCode,
)
from capicove.processing.video.filtering.video_filter_ports import (
    BackendUnavailableFilterError,
    DecodeFailedFilterError,
    FrameSampleBatch,
    SampledFrame,
    VideoFilterCapabilityError,
    VideoFilterTimeoutError,
)
from capicove.processing.video.models import AcceptedVideoSource


FrameDecoder = Callable[
    [AcceptedVideoSource, datetime],
    tuple[SampledFrame, ...],
]


class PyavVideoFrameSampler:
    """Decode once and select bounded frames without public PyAV values."""

    def __init__(self, *, decoder: FrameDecoder | None = None) -> None:
        self._decoder = decoder
        try:
            version = importlib.metadata.version("av")
        except importlib.metadata.PackageNotFoundError:
            version = None
        self._identity = ("pyav", version)

    @property
    def identity(self) -> tuple[str, str | None]:
        """Return the PyAV adapter identity."""
        return self._identity

    def sample(
        self,
        source: AcceptedVideoSource,
        config: FrameSamplingConfig,
        deadline: datetime,
    ) -> FrameSampleBatch:
        """Return deterministic bounded uniform frames or keyframes."""
        if datetime.now(UTC) >= deadline:
            raise VideoFilterTimeoutError("The frame sampling deadline expired.")
        if self._decoder is None:
            return self._decode_selected(source, config, deadline)
        frames = self._decoder(source, deadline)
        return _select_decoded_frames(frames, config)

    def _decode_selected(
        self,
        source: AcceptedVideoSource,
        config: FrameSamplingConfig,
        deadline: datetime,
    ) -> FrameSampleBatch:
        """Decode pixels only for the bounded set of selected frames."""
        try:
            av = importlib.import_module("av")
        except ModuleNotFoundError as error:
            raise BackendUnavailableFilterError(
                "PyAV is unavailable; install capicove[video-filter]."
            ) from error

        temporary_path: Path | None = None
        if source.path is not None:
            input_path = source.path
        elif source.data is not None:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as stream:
                stream.write(source.data)
                temporary_path = Path(stream.name)
            input_path = temporary_path
        else:
            raise DecodeFailedFilterError("Accepted video source is not materialized.")

        try:
            if config.method is SamplingMethod.ALL_KEYFRAMES:
                keyframe_indices = self._scan_indices(av, input_path, deadline, keyframes_only=True)
                if not keyframe_indices:
                    raise DecodeFailedFilterError("The video contains no decodable keyframes.")
                requested = len(keyframe_indices)
                positions = _uniform_indices(requested, min(requested, config.max_frames))
                target_indices = tuple(keyframe_indices[index] for index in positions)
                warnings = _sample_limit_warning(requested, config.max_frames, label="Keyframe")
            else:
                frame_count = self._declared_frame_count(av, input_path)
                if frame_count <= 0:
                    frame_count = len(
                        self._scan_indices(av, input_path, deadline, keyframes_only=False)
                    )
                if frame_count <= 0:
                    raise DecodeFailedFilterError("The video contains no decodable frames.")
                target_count = min(config.frame_count, config.max_frames, frame_count)
                target_indices = _uniform_indices(frame_count, target_count)
                warnings = _sample_limit_warning(
                    config.frame_count, config.max_frames, label="Uniform"
                )

            frames = self._decode_indices(av, input_path, target_indices, deadline)
            if not frames:
                raise DecodeFailedFilterError("The video contains no decodable frames.")
            return FrameSampleBatch(frames, warnings)
        except VideoFilterCapabilityError:
            raise
        except Exception as error:
            raise DecodeFailedFilterError(
                f"PyAV could not decode the accepted video: {type(error).__name__}."
            ) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _declared_frame_count(av: Any, input_path: Path) -> int:
        container = None
        try:
            container = av.open(str(input_path), mode="r")
            if not container.streams.video:
                raise DecodeFailedFilterError("The media contains no video stream.")
            return int(container.streams.video[0].frames or 0)
        finally:
            if container is not None:
                container.close()

    @staticmethod
    def _scan_indices(
        av: Any,
        input_path: Path,
        deadline: datetime,
        *,
        keyframes_only: bool,
    ) -> tuple[int, ...]:
        container = None
        try:
            container = av.open(str(input_path), mode="r")
            if not container.streams.video:
                raise DecodeFailedFilterError("The media contains no video stream.")
            video_stream = container.streams.video[0]
            indices: list[int] = []
            for index, frame in enumerate(container.decode(video_stream)):
                if datetime.now(UTC) >= deadline:
                    raise VideoFilterTimeoutError("The frame sampling deadline expired.")
                if not keyframes_only or frame.key_frame:
                    indices.append(index)
            return tuple(indices)
        finally:
            if container is not None:
                container.close()

    @staticmethod
    def _decode_indices(
        av: Any,
        input_path: Path,
        target_indices: tuple[int, ...],
        deadline: datetime,
    ) -> tuple[SampledFrame, ...]:
        if not target_indices:
            return ()
        target_set = frozenset(target_indices)
        last_target = target_indices[-1]
        container = None
        try:
            container = av.open(str(input_path), mode="r")
            if not container.streams.video:
                raise DecodeFailedFilterError("The media contains no video stream.")
            video_stream = container.streams.video[0]
            decoded: list[SampledFrame] = []
            for index, frame in enumerate(container.decode(video_stream)):
                if datetime.now(UTC) >= deadline:
                    raise VideoFilterTimeoutError("The frame sampling deadline expired.")
                if index not in target_set:
                    if index >= last_target:
                        break
                    continue
                timestamp = frame.time
                if timestamp is None and frame.pts is not None:
                    timestamp = float(frame.pts * video_stream.time_base)
                if timestamp is None:
                    timestamp = float(index)
                decoded.append(
                    SampledFrame(
                        index=index,
                        timestamp_seconds=float(timestamp),
                        width=int(frame.width),
                        height=int(frame.height),
                        payload=frame.to_ndarray(format="bgr24"),
                        key_frame=bool(frame.key_frame),
                    )
                )
                if index >= last_target:
                    break
            return tuple(decoded)
        finally:
            if container is not None:
                container.close()


def _select_decoded_frames(
    frames: tuple[SampledFrame, ...],
    config: FrameSamplingConfig,
) -> FrameSampleBatch:
    if not frames:
        raise DecodeFailedFilterError("The video contains no decodable frames.")
    chronological = tuple(sorted(frames, key=lambda frame: (frame.timestamp_seconds, frame.index)))
    if config.method is SamplingMethod.ALL_KEYFRAMES:
        candidates = tuple(frame for frame in chronological if frame.key_frame)
        if not candidates:
            raise DecodeFailedFilterError("The video contains no decodable keyframes.")
        requested = len(candidates)
        target_count = min(requested, config.max_frames)
        selected = _select_uniform(candidates, target_count)
        warnings = _sample_limit_warning(requested, config.max_frames, label="Keyframe")
    else:
        target_count = min(config.frame_count, config.max_frames, len(chronological))
        selected = _select_uniform(chronological, target_count)
        warnings = _sample_limit_warning(config.frame_count, config.max_frames, label="Uniform")
    return FrameSampleBatch(selected, warnings)


def _sample_limit_warning(
    requested: int,
    maximum: int,
    *,
    label: str,
) -> tuple[VideoFilterWarning, ...]:
    if requested <= maximum:
        return ()
    return (
        VideoFilterWarning(
            VideoFilterWarningCode.SAMPLE_LIMIT_REACHED,
            f"{label} samples were limited from {requested} to {maximum}.",
        ),
    )


def _uniform_indices(length: int, count: int) -> tuple[int, ...]:
    if count <= 0 or length <= 0:
        return ()
    if count >= length:
        return tuple(range(length))
    if count == 1:
        return ((length - 1) // 2,)
    return tuple(round(index * (length - 1) / (count - 1)) for index in range(count))


def _select_uniform(
    frames: tuple[SampledFrame, ...],
    count: int,
) -> tuple[SampledFrame, ...]:
    return tuple(frames[index] for index in _uniform_indices(len(frames), count))
