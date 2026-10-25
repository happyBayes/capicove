"""ffprobe adapter for normalized video filter metadata."""

from __future__ import annotations

import json
import subprocess  # nosec B404
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from capicove.processing.video.filtering.video_filter_contracts import VideoFilterMetadata
from capicove.processing.video.filtering.video_filter_ports import (
    BackendUnavailableFilterError,
    UnsupportedMediaFilterError,
    VideoFilterTimeoutError,
)
from capicove.processing.video.models import AcceptedVideoSource


# ffprobe uses a validated executable (bare name or absolute path) and fixed arguments.


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class FfprobeVideoMetadataProbe:
    """Probe media with trusted ffprobe arguments and bounded diagnostics."""

    def __init__(
        self,
        ffprobe_executable: str = "ffprobe",
        *,
        runner: CommandRunner = subprocess.run,
    ) -> None:
        # A bare name is resolved through PATH by the legacy CLI; a host passes an
        # absolute, operator-configured executable instead. Relative paths are refused.
        if not ffprobe_executable or (
            "/" in ffprobe_executable and not ffprobe_executable.startswith("/")
        ):
            raise ValueError("ffprobe_executable must be a bare name or an absolute path")
        self._executable = ffprobe_executable
        self._runner = runner
        self._version: str | None = None

    @property
    def identity(self) -> tuple[str, str | None]:
        """Return the backend identity observed by the latest probe."""
        return "ffprobe", self._version

    def probe(
        self,
        source: AcceptedVideoSource,
        deadline: datetime,
    ) -> VideoFilterMetadata:
        """Return normalized metadata for one accepted source."""
        remaining = (deadline - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            raise VideoFilterTimeoutError("The metadata probe deadline expired.")

        temporary_path: Path | None = None
        if source.path is not None:
            input_path = source.path
        elif source.data is not None:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as stream:
                stream.write(source.data)
                temporary_path = Path(stream.name)
            input_path = temporary_path
        else:
            raise UnsupportedMediaFilterError("Accepted video source is not materialized.")

        command = [
            self._executable,
            "-v",
            "error",
            "-show_program_version",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(input_path),
        ]
        try:
            try:
                completed = self._runner(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=remaining,
                    check=False,
                )
            except FileNotFoundError as error:
                raise BackendUnavailableFilterError(
                    f"{self._executable} is not installed or executable."
                ) from error
            except subprocess.TimeoutExpired as error:
                raise VideoFilterTimeoutError(
                    "ffprobe exceeded the invocation deadline."
                ) from error
            if completed.returncode != 0:
                diagnostic = completed.stderr.strip()[:8192]
                raise UnsupportedMediaFilterError(
                    f"ffprobe could not inspect the video: {diagnostic or 'unknown error'}"
                )
            try:
                parsed = json.loads(completed.stdout)
            except (json.JSONDecodeError, TypeError) as error:
                raise UnsupportedMediaFilterError(
                    "ffprobe returned invalid JSON metadata."
                ) from error
            if not isinstance(parsed, dict):
                raise UnsupportedMediaFilterError("ffprobe returned an invalid metadata object.")
            return self._normalize(parsed)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _normalize(self, payload: Mapping[str, Any]) -> VideoFilterMetadata:
        streams = payload.get("streams", [])
        if not isinstance(streams, list):
            raise UnsupportedMediaFilterError("ffprobe streams metadata is invalid.")
        video_stream = next(
            (
                stream
                for stream in streams
                if isinstance(stream, dict) and stream.get("codec_type") == "video"
            ),
            None,
        )
        if video_stream is None:
            raise UnsupportedMediaFilterError("The media contains no video stream.")

        width = _positive_int(video_stream.get("width"), "video width")
        height = _positive_int(video_stream.get("height"), "video height")
        rotation = _rotation(video_stream)
        if abs(rotation) % 180 == 90:
            display_width, display_height = height, width
        else:
            display_width, display_height = width, height

        format_data = payload.get("format")
        format_mapping = format_data if isinstance(format_data, dict) else {}
        duration_value = format_mapping.get("duration", video_stream.get("duration"))
        duration = _non_negative_float(duration_value, "video duration")

        program_version = payload.get("program_version")
        if isinstance(program_version, dict):
            version_value = program_version.get("version")
            self._version = str(version_value) if version_value is not None else None

        return VideoFilterMetadata(
            duration_seconds=duration,
            encoded_width=width,
            encoded_height=height,
            display_width=display_width,
            display_height=display_height,
            rotation_degrees=rotation,
            sample_aspect_ratio=_optional_string(video_stream.get("sample_aspect_ratio")),
            display_aspect_ratio=_optional_string(video_stream.get("display_aspect_ratio")),
            backend="ffprobe",
            backend_version=self._version,
        )


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, bytes, bytearray, int, float)):
        raise UnsupportedMediaFilterError(f"{name} is missing or invalid.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise UnsupportedMediaFilterError(f"{name} is missing or invalid.") from error
    if parsed <= 0:
        raise UnsupportedMediaFilterError(f"{name} must be positive.")
    return parsed


def _non_negative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, bytes, bytearray, int, float)):
        raise UnsupportedMediaFilterError(f"{name} is missing or invalid.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise UnsupportedMediaFilterError(f"{name} is missing or invalid.") from error
    if parsed < 0:
        raise UnsupportedMediaFilterError(f"{name} must be non-negative.")
    return parsed


def _rotation(stream: Mapping[str, Any]) -> int:
    tags = stream.get("tags")
    if isinstance(tags, dict) and "rotate" in tags:
        try:
            return int(float(tags["rotate"]))
        except (TypeError, ValueError):
            pass
    side_data = stream.get("side_data_list")
    if isinstance(side_data, list):
        for entry in side_data:
            if isinstance(entry, dict) and "rotation" in entry:
                try:
                    return int(float(entry["rotation"]))
                except (TypeError, ValueError):
                    continue
    return 0


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    parsed = str(value).strip()
    return parsed or None
