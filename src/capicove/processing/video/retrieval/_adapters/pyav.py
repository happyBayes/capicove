"""One sequential bounded local decode; PyAV is imported only on use."""

import importlib
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO

from capicove.contracts.retrieval import ProcessingFingerprint
from capicove.processing.video.models import AcceptedVideoSource
from capicove.processing.video.ports import BackendUnavailableError
from capicove.processing.video.retrieval.models import (
    FrameObservation,
    SceneMetric,
    VideoDecodeObservation,
)
from capicove.processing.video.retrieval.pipeline import check_deadline
from capicove.processing.video.retrieval.ports import PreviewStore


class PyAVVideoDecodePass:
    """Persist sampled normalized previews without retaining decoded video frames."""

    def __init__(
        self,
        preview_store: PreviewStore,
        *,
        sample_interval_ms: int = 500,
        maximum_preview_edge: int = 384,
        max_decoded_frames: int = 1_000_000,
        max_observations: int = 100_000,
        max_pixels: int = 16_777_216,
        video_stream_index: int = 0,
    ) -> None:
        if (
            min(
                sample_interval_ms,
                maximum_preview_edge,
                max_decoded_frames,
                max_observations,
                max_pixels,
            )
            <= 0
            or video_stream_index < 0
        ):
            raise ValueError("decode bounds must be positive")
        self._store = preview_store
        self._settings = dict(
            sample_interval_ms=sample_interval_ms,
            maximum_preview_edge=maximum_preview_edge,
            max_decoded_frames=max_decoded_frames,
            max_observations=max_observations,
            max_pixels=max_pixels,
            video_stream_index=video_stream_index,
        )

    @property
    def fingerprint(self) -> ProcessingFingerprint:
        """Bind decoder settings, color, resize, quality and preview encoding."""
        versions = {}
        for package in ("av", "Pillow"):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = "unavailable"
        return ProcessingFingerprint.from_manifest(
            {
                "decoder": "pyav-rgb-preview-v1",
                "package_versions": versions,
                "settings": self._settings,
                "preview": "png-rgb-lanczos-v1",
                "quality": "contrast-v1",
                "black_threshold": 3,
            }
        )

    def decode(self, source: AcceptedVideoSource, *, deadline: datetime) -> VideoDecodeObservation:
        """Decode a selected local stream once with enforced frame/pixel limits."""
        check_deadline(deadline)
        try:
            av = importlib.import_module("av")
            image_module = importlib.import_module("PIL.Image")
            image_stat = importlib.import_module("PIL.ImageStat")
        except ImportError as error:
            raise BackendUnavailableError("local video dependencies unavailable") from error
        source_input = str(source.path) if source.path is not None else BytesIO(source.data or b"")
        frames: list[FrameObservation] = []
        metrics = []
        previous_rgb = None
        next_sample = 0
        end_ms = 0
        with av.open(source_input) as container:
            streams = container.streams.video
            index = self._settings["video_stream_index"]
            if index >= len(streams):
                raise ValueError("configured video stream unavailable")
            stream = streams[index]
            if (
                stream.codec_context.width * stream.codec_context.height
                > self._settings["max_pixels"]
            ):
                raise ValueError("decoded pixel capacity exceeded")
            step_ms = max(1, round(1000 / float(stream.average_rate))) if stream.average_rate else 1
            origin = stream.start_time or 0
            for count, frame in enumerate(container.decode(stream)):
                check_deadline(deadline)
                if count >= self._settings["max_decoded_frames"]:
                    raise ValueError("decoded frame capacity exceeded")
                if frame.width * frame.height > self._settings["max_pixels"]:
                    raise ValueError("decoded pixel capacity exceeded")
                if frame.pts is None or frame.time_base is None:
                    raise ValueError("missing frame presentation timestamp")
                timestamp_ms = round(float((frame.pts - origin) * frame.time_base) * 1000)
                if timestamp_ms < 0:
                    continue
                end_ms = max(end_ms, timestamp_ms + step_ms)
                if timestamp_ms < next_sample:
                    continue
                if len(frames) >= self._settings["max_observations"]:
                    raise ValueError("observation capacity exceeded")
                picture = frame.to_image().convert("RGB")
                edge = self._settings["maximum_preview_edge"]
                picture.thumbnail((edge, edge), image_module.Resampling.LANCZOS)
                stats = image_stat.Stat(picture)
                rgb = tuple(float(v) for v in stats.mean)
                quality = max(0.01, min(1.0, sum(stats.stddev) / (3 * 128) + 0.1))
                metric = (
                    0.0
                    if previous_rgb is None
                    else sum(abs(a - b) for a, b in zip(rgb, previous_rgb, strict=True)) / (3 * 255)
                )
                previous_rgb = rgb
                preview = BytesIO()
                picture.save(preview, format="PNG", optimize=False)
                reference = self._store.put(preview.getvalue())
                frames.append(
                    FrameObservation(
                        timestamp_ms,
                        quality,
                        reference,
                        black_frame=max(rgb) < 3,
                        corrupt=bool(frame.is_corrupt),
                    )
                )
                metrics.append(SceneMetric(timestamp_ms, metric))
                next_sample = timestamp_ms + self._settings["sample_interval_ms"]
            if stream.duration is not None and stream.time_base is not None:
                end_ms = max(end_ms, round(float(stream.duration * stream.time_base) * 1000))
        check_deadline(deadline)
        return VideoDecodeObservation(end_ms, tuple(frames), tuple(metrics))
