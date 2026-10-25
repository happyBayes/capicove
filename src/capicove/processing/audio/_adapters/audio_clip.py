"""Fixed, bounded FFprobe/FFmpeg commands for descriptor-bound clipping."""

from __future__ import annotations

import json
import math
import os
import selectors
import struct
import subprocess  # nosec B404
import sys
import tempfile
from datetime import datetime
from decimal import Decimal, DecimalException
from fractions import Fraction
from pathlib import Path
from typing import Any

from capicove.processing.audio.clip_models import (
    AudioClipGeometry,
    AudioClipMetadata,
    AudioClippingError,
    remaining_seconds,
)
from capicove.processing.audio.models import AudioFailureCode


def _run(arguments: list[str], deadline: datetime, descriptors: tuple[int, ...] = ()) -> bytes:
    """Capture at most 64 KiB and always reap a timed-out or failed child."""
    timeout = remaining_seconds(deadline)
    try:
        process = subprocess.Popen(  # nosec B603
            arguments,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            pass_fds=descriptors,
            start_new_session=True,
            shell=False,
        )
    except OSError:
        raise AudioClippingError(
            AudioFailureCode.BACKEND_UNAVAILABLE, "audio backend is unavailable"
        ) from None
    output = bytearray()
    assert process.stdout is not None
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                timeout = remaining_seconds(deadline)
                if not selector.select(timeout):
                    raise AudioClippingError(AudioFailureCode.TIMEOUT, "audio backend timed out")
                chunk = os.read(process.stdout.fileno(), min(8192, 65537 - len(output)))
                if not chunk:
                    break
                output.extend(chunk)
                if len(output) > 65536:
                    raise AudioClippingError(
                        AudioFailureCode.INVALID_AUDIO, "backend output exceeds the limit"
                    )
        try:
            returncode = process.wait(timeout=remaining_seconds(deadline))
        except subprocess.TimeoutExpired:
            raise AudioClippingError(AudioFailureCode.TIMEOUT, "audio backend timed out") from None
        if returncode:
            raise AudioClippingError(
                AudioFailureCode.INVALID_AUDIO, "audio backend rejected the media"
            )
        return bytes(output)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
        process.stdout.close()


class FfmpegAudioClipBackend:
    """Use seekable fd protocols and sample counts, with no caller-supplied flags."""

    def __init__(self, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> None:
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.ffprobe_version = "unavailable"
        self.fd_protocols = {"ffprobe_input": False, "ffmpeg_input": False, "ffmpeg_output": False}

    def preflight(self, deadline: datetime) -> str:
        """Verify that both binaries actually consume inherited, seekable descriptors."""
        if sys.platform not in ("linux", "darwin"):
            raise AudioClippingError(
                AudioFailureCode.BACKEND_UNAVAILABLE,
                "unsupported clipping platform",
                reason="unsupported_os",
            )
        for name, reason in (
            ("O_NOFOLLOW", "nofollow_unavailable"),
            ("O_DIRECTORY", "directory_fd_unavailable"),
            ("pread", "filesystem_profile_unavailable"),
            ("fstatvfs", "filesystem_profile_unavailable"),
        ):
            if not hasattr(os, name):
                raise AudioClippingError(
                    AudioFailureCode.BACKEND_UNAVAILABLE,
                    "secure filesystem APIs are unavailable",
                    reason=reason,
                )
        phase = "ffmpeg_unavailable"
        try:
            version = (
                _run([self.ffmpeg, "-version"], deadline)
                .splitlines()[0]
                .decode("utf-8", "replace")[:128]
            )
            phase = "ffprobe_unavailable"
            self.ffprobe_version = (
                _run([self.ffprobe, "-version"], deadline)
                .splitlines()[0]
                .decode("utf-8", "replace")[:128]
            )
            phase = "filesystem_profile_unavailable"
            with tempfile.TemporaryDirectory(prefix="capicove-clip-probe-") as directory:
                from capicove.processing.audio._adapters.clip_files import (
                    admit_directory,
                    open_directory,
                    verify_wav,
                )

                with open_directory(Path(directory)) as parent:
                    admit_directory(parent)
                payload = b"\0" * 1600
                header = (
                    b"RIFF"
                    + struct.pack("<I", 36 + len(payload))
                    + b"WAVEfmt "
                    + struct.pack("<IHHIIHH", 16, 1, 1, 8000, 16000, 2, 16)
                    + b"data"
                    + struct.pack("<I", len(payload))
                )
                source = Path(directory) / "probe.wav"
                source.write_bytes(header + payload)
                with (
                    source.open("rb") as input_stream,
                    (Path(directory) / "output.wav").open("w+b") as output,
                ):
                    phase = "ffprobe_fd_input_unavailable"
                    metadata = self.inspect(input_stream.fileno(), "wav", deadline)
                    self.fd_protocols["ffprobe_input"] = True
                    phase = "ffmpeg_fd_input_unavailable"
                    os.lseek(input_stream.fileno(), 0, os.SEEK_SET)
                    _run(
                        [
                            self.ffmpeg,
                            "-v",
                            "error",
                            "-nostdin",
                            "-protocol_whitelist",
                            "fd",
                            "-f",
                            "wav",
                            "-fd",
                            str(input_stream.fileno()),
                            "-i",
                            "fd:",
                            "-map",
                            "0:a:0",
                            "-f",
                            "null",
                            "-fd",
                            str(output.fileno()),
                            "fd:",
                        ],
                        deadline,
                        (input_stream.fileno(), output.fileno()),
                    )
                    self.fd_protocols["ffmpeg_input"] = True
                    phase = "ffmpeg_fd_output_unavailable"
                    geometry = AudioClipGeometry(0, 800, 8000, 8000, 1, 800)
                    self.write(input_stream.fileno(), output.fileno(), "wav", geometry, deadline)
                    verify_wav(output.fileno(), geometry, deadline)
                    self.fd_protocols["ffmpeg_output"] = True
                    if metadata.sample_rate_hz != 8000:
                        raise ValueError("invalid protocol probe")
            return version
        except AudioClippingError as error:
            if error.code is AudioFailureCode.TIMEOUT:
                raise
            raise AudioClippingError(
                AudioFailureCode.BACKEND_UNAVAILABLE,
                "seekable descriptor protocols are unavailable",
                reason=phase,
            ) from None
        except (OSError, ValueError, IndexError):
            raise AudioClippingError(
                AudioFailureCode.BACKEND_UNAVAILABLE,
                "secure runtime profile is unavailable",
                reason=phase,
            ) from None

    def inspect(self, descriptor: int, demuxer: str, deadline: datetime) -> AudioClipMetadata:
        """Read actual audio-stream metadata, never substitute container duration."""
        os.lseek(descriptor, 0, os.SEEK_SET)
        output = _run(
            [
                self.ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index,codec_name,sample_rate,channels,bits_per_sample,bits_per_raw_sample,bit_rate,duration,duration_ts,time_base:stream_tags=DURATION:format=format_name",
                "-of",
                "json",
                "-protocol_whitelist",
                "fd",
                "-f",
                demuxer,
                "-fd",
                str(descriptor),
                "-i",
                "fd:",
            ],
            deadline,
            (descriptor,),
        )
        try:
            root = json.loads(output)
            streams = root["streams"]
            if len(streams) != 1:
                code = (
                    AudioFailureCode.NO_AUDIO_STREAM
                    if not streams
                    else AudioFailureCode.AUDIO_STREAM_SELECTION_REQUIRED
                )
                raise AudioClippingError(code, "clipping requires exactly one audio stream")
            stream = streams[0]
            duration_exact = _duration(stream)
            duration = float(duration_exact)
            rate, channels = int(stream["sample_rate"]), int(stream["channels"])
            if rate <= 0 or channels <= 0 or not math.isfinite(duration) or duration <= 0:
                raise ValueError("invalid stream geometry")
            return AudioClipMetadata(
                duration,
                rate,
                channels,
                str(stream["codec_name"]),
                str(root["format"]["format_name"]),
                _positive_optional(stream.get("bits_per_raw_sample"))
                or _positive_optional(stream.get("bits_per_sample")),
                _positive_optional(stream.get("bit_rate")),
                0,
                int(stream["index"]),
                duration_exact,
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
            ZeroDivisionError,
            AttributeError,
            DecimalException,
        ) as error:
            if isinstance(error, AudioClippingError):
                raise
            raise AudioClippingError(
                AudioFailureCode.INVALID_AUDIO, "selected audio metadata is missing or invalid"
            ) from None

    def write(
        self,
        source: int,
        output: int,
        demuxer: str,
        geometry: AudioClipGeometry,
        deadline: datetime,
    ) -> None:
        """Trim decoded source samples before conversion, then write a real WAV header."""
        os.lseek(source, 0, os.SEEK_SET)
        os.lseek(output, 0, os.SEEK_SET)
        filters = f"atrim=start_sample={geometry.start_frame}:end_sample={geometry.end_frame},asetpts=N/SR/TB"
        if geometry.sample_rate_hz != geometry.source_rate_hz:
            filters += f",aresample={geometry.sample_rate_hz}"
        _run(
            [
                self.ffmpeg,
                "-v",
                "error",
                "-xerror",
                "-nostdin",
                "-protocol_whitelist",
                "fd",
                "-f",
                demuxer,
                "-fd",
                str(source),
                "-i",
                "fd:",
                "-map",
                "0:a:0",
                "-af",
                filters,
                "-ac",
                str(geometry.channels),
                "-ar",
                str(geometry.sample_rate_hz),
                "-c:a",
                "pcm_s16le",
                "-map_metadata",
                "-1",
                "-f",
                "wav",
                "-fd",
                str(output),
                "fd:",
            ],
            deadline,
            (source, output),
        )


def _positive_optional(value: object) -> int | None:
    if value in (None, "N/A", "0", 0):
        return None
    parsed = int(str(value))
    if parsed < 0:
        raise ValueError("negative metadata")
    return parsed


def _duration(stream: dict[str, Any]) -> Fraction:
    if not isinstance(stream, dict):
        raise ValueError("audio stream must be an object")
    if stream.get("duration_ts") is not None and stream.get("time_base"):
        numerator, denominator = str(stream["time_base"]).split("/")
        if int(numerator) <= 0 or int(denominator) <= 0:
            raise ValueError("invalid audio time base")
        return Fraction(int(stream["duration_ts"]) * int(numerator), int(denominator))
    value = stream.get("duration")
    if value not in (None, "N/A"):
        return Fraction(Decimal(str(value)))
    # Matroska records per-stream duration in tags rather than stream.duration.
    tagged = stream.get("tags", {}).get("DURATION")
    if isinstance(tagged, str):
        hours, minutes, seconds = tagged.split(":")
        return Fraction(int(hours) * 3600 + int(minutes) * 60) + Fraction(Decimal(seconds))
    raise ValueError("audio stream duration is unavailable")
