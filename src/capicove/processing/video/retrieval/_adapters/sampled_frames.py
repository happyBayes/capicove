"""Fixed FFmpeg/ffprobe commands that sample PNG keyframes from a preserved video.

Inputs are verified, content-addressed evidence blobs inside a private store. Every
command pins the demuxer, allows only the ``file`` protocol, discards metadata and
bounds its output and time; no caller text reaches an argument except the store
path and the integer timestamp.
"""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


VIDEO_DEMUXERS = {".mp4": "mov", ".mov": "mov", ".mkv": "matroska"}
MAX_PROBE_BYTES = 64 * 1024
MAX_FRAME_BYTES = 16 * 1024 * 1024


class FrameSamplingError(RuntimeError):
    """A bounded, sanitized video decoding failure."""


@dataclass(frozen=True, slots=True)
class VideoProbe:
    """Stream counts and duration of one preserved container."""

    duration_ms: int
    video_streams: int
    audio_streams: int


def _run(arguments: list[str], timeout: float, limit: int) -> bytes:
    deadline = time.monotonic() + timeout
    try:
        process = subprocess.Popen(  # nosec B603 - fixed argument list, no shell
            arguments,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        raise FrameSamplingError("video decoder is unavailable") from None
    assert process.stdout is not None
    output = bytearray()
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise FrameSamplingError("video decoding timed out")
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    break
                output.extend(chunk)
                if len(output) > limit:
                    raise FrameSamplingError("video decoder output exceeds its bound")
        if process.wait(max(0.0, deadline - time.monotonic())) != 0:
            raise FrameSamplingError("video could not be decoded")
    except subprocess.TimeoutExpired:
        raise FrameSamplingError("video decoding timed out") from None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
    return bytes(output)


class FfmpegFrameSampler:
    """Probe and sample frames with explicitly configured executables."""

    def __init__(self, ffmpeg: str, ffprobe: str, *, timeout_seconds: float = 60.0) -> None:
        self._ffmpeg = ffmpeg
        self._ffprobe = ffprobe
        self._timeout = timeout_seconds

    def probe(self, blob: Path, demuxer: str) -> VideoProbe:
        """Return duration and stream counts of a preserved container."""
        output = _run(
            [
                self._ffprobe,
                "-v",
                "error",
                "-protocol_whitelist",
                "file",
                "-f",
                demuxer,
                "-show_entries",
                "format=duration:stream=codec_type",
                "-of",
                "json",
                str(blob),
            ],
            self._timeout,
            MAX_PROBE_BYTES,
        )
        try:
            payload = json.loads(output)
            kinds = [stream["codec_type"] for stream in payload.get("streams", [])]
            duration = float(payload["format"]["duration"])
        except (ValueError, KeyError, TypeError):
            raise FrameSamplingError("video stream inventory is invalid") from None
        if not 0 < duration < 7 * 24 * 3600:
            raise FrameSamplingError("video duration is invalid")
        return VideoProbe(
            duration_ms=int(duration * 1000),
            video_streams=kinds.count("video"),
            audio_streams=kinds.count("audio"),
        )

    def frame(self, blob: Path, demuxer: str, timestamp_ms: int, max_side: int) -> bytes:
        """Return one PNG frame at ``timestamp_ms`` from the first video stream."""
        if type(timestamp_ms) is not int or timestamp_ms < 0:
            raise FrameSamplingError("invalid frame timestamp")
        seconds = f"{timestamp_ms // 1000}.{timestamp_ms % 1000:03d}"
        payload = _run(
            [
                self._ffmpeg,
                "-v",
                "error",
                "-nostdin",
                "-protocol_whitelist",
                "file",
                "-f",
                demuxer,
                "-ss",
                seconds,
                "-i",
                str(blob),
                "-map",
                "0:v:0",
                "-an",
                "-sn",
                "-dn",
                "-frames:v",
                "1",
                "-vf",
                f"scale=w='min({max_side},iw)':h='min({max_side},ih)':force_original_aspect_ratio=decrease",
                "-map_metadata",
                "-1",
                "-c:v",
                "png",
                "-f",
                "image2pipe",
                "pipe:1",
            ],
            self._timeout,
            MAX_FRAME_BYTES,
        )
        if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise FrameSamplingError("video frame could not be decoded")
        return payload
