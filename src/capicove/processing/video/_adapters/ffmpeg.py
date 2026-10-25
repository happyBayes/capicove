"""ffprobe inspection and ffmpeg clip writing adapters."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess  # nosec B404
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from capicove.processing.video.models import (
    AcceptedVideoSource,
    TimeRange,
    VideoBackend,
    VideoWriteMode,
)
from capicove.processing.video.ports import (
    BackendUnavailableError,
    ClipWriteError,
    PendingOutput,
    VideoBackendBundle,
    VideoInspectionError,
    VideoMetadata,
    WrittenClip,
)


# Media commands use fixed argv construction, deadlines, and no shell.


_MAX_DIAGNOSTIC_CHARS = 8192


class FfprobeVideoInspector:
    """Inspect normalized video metadata through ffprobe JSON output."""

    def __init__(self, executable: str = "ffprobe", version: str | None = None) -> None:
        self._executable = executable
        self._version = version

    def inspect(self, source: AcceptedVideoSource, deadline: datetime) -> VideoMetadata:
        """Return duration and chronological keyframe timestamps."""
        with _materialized_source(source) as source_path:
            completed = _run(
                [
                    self._executable,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "format=duration:frame=key_frame,best_effort_timestamp_time",
                    "-of",
                    "json",
                    str(source_path),
                ],
                deadline,
                VideoInspectionError,
            )
        try:
            payload = json.loads(completed.stdout)
            duration = float(payload["format"]["duration"])
            frames = payload.get("frames", [])
            timestamps = tuple(
                float(frame["best_effort_timestamp_time"])
                for frame in frames
                if int(frame.get("key_frame", 0)) == 1 and "best_effort_timestamp_time" in frame
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise VideoInspectionError("ffprobe returned invalid video metadata") from error
        if not math.isfinite(duration) or duration <= 0:
            raise VideoInspectionError("ffprobe returned an invalid duration")
        return VideoMetadata(duration, timestamps, "ffmpeg", self._version)


class FfmpegClipWriter:
    """Write one requested range through a fixed ffmpeg argument template."""

    def __init__(
        self,
        executable: str = "ffmpeg",
        write_mode: VideoWriteMode = VideoWriteMode.ACCURATE,
    ) -> None:
        self._executable = executable
        self._write_mode = write_mode

    def write(
        self,
        source: AcceptedVideoSource,
        time_range: TimeRange,
        destination: PendingOutput,
        deadline: datetime,
    ) -> WrittenClip:
        """Write and hash a clip without exposing arbitrary caller flags."""
        with _materialized_source(source) as source_path:
            arguments = [
                self._executable,
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-ss",
                f"{time_range.start_seconds:.9f}",
                "-i",
                str(source_path),
                "-t",
                f"{time_range.end_seconds - time_range.start_seconds:.9f}",
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
            ]
            if self._write_mode is VideoWriteMode.COPY:
                arguments.extend(["-c", "copy", "-avoid_negative_ts", "make_zero"])
            else:
                arguments.extend(
                    [
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "18",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "192k",
                        "-movflags",
                        "+faststart",
                    ]
                )
            arguments.extend(["-f", "mp4", str(destination.path)])
            _run(
                arguments,
                deadline,
                ClipWriteError,
            )
        if not destination.path.is_file() or destination.path.stat().st_size == 0:
            raise ClipWriteError("ffmpeg did not create a non-empty output")
        return WrittenClip(
            path=destination.path,
            sha256=_hash_file(destination.path),
            mime_type="video/mp4",
        )


BackendFactory = Callable[[VideoWriteMode], VideoBackendBundle]


class DefaultVideoBackendResolver:
    """Resolve explicit or automatic backends without import-time probing."""

    def __init__(
        self,
        av_factory: BackendFactory | None = None,
        *,
        executables: tuple[str, str] | None = None,
        path_lookup: bool = True,
    ) -> None:
        if executables is not None and not all(Path(item).is_absolute() for item in executables):
            raise ValueError("configured ffmpeg and ffprobe executables must be absolute")
        self._av_factory = av_factory
        self._executables = executables
        self._path_lookup = path_lookup

    def resolve(
        self,
        requested: VideoBackend,
        write_mode: VideoWriteMode = VideoWriteMode.ACCURATE,
    ) -> VideoBackendBundle:
        """Resolve ffmpeg first for auto, then an injected optional PyAV backend."""
        if requested in (VideoBackend.AUTO, VideoBackend.FFMPEG):
            ffmpeg_path: str | None
            ffprobe_path: str | None
            if self._executables is not None:
                ffmpeg_path, ffprobe_path = self._executables
            elif self._path_lookup:
                ffmpeg_path, ffprobe_path = shutil.which("ffmpeg"), shutil.which("ffprobe")
            else:
                ffmpeg_path = ffprobe_path = None
            if ffmpeg_path is not None and ffprobe_path is not None:
                version = _read_version(ffmpeg_path)
                return VideoBackendBundle(
                    name="ffmpeg",
                    version=version,
                    inspector=FfprobeVideoInspector(ffprobe_path, version),
                    writer=FfmpegClipWriter(ffmpeg_path, write_mode),
                )
            if requested is VideoBackend.FFMPEG:
                raise BackendUnavailableError("ffmpeg and ffprobe are unavailable")
        if self._av_factory is not None:
            return self._av_factory(write_mode)
        backend = requested.value if requested is not VideoBackend.AUTO else "ffmpeg or av"
        raise BackendUnavailableError(f"{backend} backend is unavailable")


@contextmanager
def _materialized_source(source: AcceptedVideoSource) -> Iterator[Path]:
    if source.path is not None:
        yield source.path
        return
    if source.data is None:
        raise VideoInspectionError("video source is not materialized")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temporary:
            temporary.write(source.data)
            temporary_path = Path(temporary.name)
        yield temporary_path
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _run(
    arguments: Sequence[str],
    deadline: datetime,
    error_type: type[VideoInspectionError] | type[ClipWriteError],
) -> subprocess.CompletedProcess[str]:
    remaining = (deadline - datetime.now(UTC)).total_seconds()
    if remaining <= 0:
        raise error_type("media command deadline expired")
    try:
        # This private adapter runs only fixed ffmpeg and ffprobe argument templates.
        completed = subprocess.run(  # nosec B603
            list(arguments),
            check=False,
            capture_output=True,
            text=True,
            timeout=remaining,
            start_new_session=True,
            shell=False,
        )
    except FileNotFoundError as error:
        raise error_type("media executable is unavailable") from error
    except subprocess.TimeoutExpired as error:
        raise error_type("media command timed out") from error
    if completed.returncode != 0:
        diagnostic = completed.stderr[-_MAX_DIAGNOSTIC_CHARS:].strip()
        raise error_type(f"media command failed: {diagnostic or 'no diagnostic'}")
    return completed


def _read_version(executable: str) -> str | None:
    try:
        completed = _run(
            [executable, "-version"],
            datetime.now(UTC) + timedelta(seconds=5),
            VideoInspectionError,
        )
    except VideoInspectionError:
        return None
    first_line = completed.stdout.splitlines()[0] if completed.stdout else ""
    parts = first_line.split()
    return parts[2] if len(parts) >= 3 else None


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
