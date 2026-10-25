"""FFmpeg-backed full-stream audio validation."""

from __future__ import annotations

# Required only by this private adapter for a fixed FFmpeg command.
import subprocess  # nosec B404
from datetime import UTC, datetime
from pathlib import Path

from capicove.processing.audio.models import AudioFailureCode
from capicove.processing.audio.ports import AudioValidationError


# Demuxer per accepted extension. Forcing it, with only the local file protocol,
# stops playlist or concat demuxers from following references to other files.
AUDIO_DEMUXERS = {
    ".aac": "aac",
    ".flac": "flac",
    ".m4a": "mov",
    ".mp3": "mp3",
    ".ogg": "ogg",
    ".opus": "ogg",
    ".wav": "wav",
    ".webm": "matroska",
}
# Soundtracks of video containers; admitted only when a caller explicitly asks.
VIDEO_CONTAINER_DEMUXERS = {".mp4": "mov", ".mov": "mov", ".mkv": "matroska"}


def _demuxer_from_signature(source: Path) -> str | None:
    try:
        with source.open("rb") as stream:
            head = stream.read(16)
    except OSError:
        return None
    if head.startswith(b"RIFF") and head[8:12] == b"WAVE":
        return "wav"
    if head.startswith(b"fLaC"):
        return "flac"
    if head.startswith(b"OggS"):
        return "ogg"
    if head[4:8] == b"ftyp":
        return "mov"
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        return "matroska"
    if head.startswith(b"ADIF") or (len(head) >= 2 and head[0] == 0xFF and head[1] & 0xF6 == 0xF0):
        return "aac"
    if head.startswith(b"ID3") or (len(head) >= 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0):
        return "mp3"
    return None


def restricted_input_arguments(source: Path) -> list[str]:
    """Return FFmpeg input options that pin the demuxer and allow only local files.

    The demuxer comes from the extension, or for extension-less evidence blobs from
    the container signature.
    """
    demuxer = AUDIO_DEMUXERS.get(source.suffix.lower()) or (
        None if source.suffix else _demuxer_from_signature(source)
    )
    if demuxer is None:
        raise AudioValidationError(
            AudioFailureCode.UNSUPPORTED_FORMAT, "unsupported audio extension"
        )
    return ["-protocol_whitelist", "file", "-f", demuxer]


class FfmpegAudioValidator:
    """Require FFmpeg to decode one selected audio stream from start to finish."""

    def __init__(
        self,
        executable: str = "ffmpeg",
        timeout_seconds: float = 300.0,
        *,
        restrict_inputs: bool = False,
    ) -> None:
        self._executable = executable
        self._timeout_seconds = timeout_seconds
        self._restrict_inputs = restrict_inputs

    def validate(
        self,
        source: Path,
        *,
        audio_stream_index: int,
        deadline: datetime | None = None,
    ) -> None:
        """Reject missing decoders, timeouts, absent streams, and decode failures."""
        timeout = self._timeout_seconds
        if deadline is not None:
            if deadline.tzinfo is None or deadline.utcoffset() is None:
                raise ValueError("deadline must be timezone-aware")
            remaining = (deadline - datetime.now(UTC)).total_seconds()
            if remaining <= 0:
                raise AudioValidationError(
                    AudioFailureCode.TIMEOUT,
                    "audio stream validation timed out",
                )
            timeout = min(timeout, remaining)
        try:
            # Arguments use a fixed template and are passed directly without a shell.
            completed = subprocess.run(  # nosec B603
                [
                    self._executable,
                    "-v",
                    "error",
                    "-xerror",
                    "-nostdin",
                    *(restricted_input_arguments(source) if self._restrict_inputs else []),
                    "-i",
                    str(source),
                    "-map",
                    f"0:a:{audio_stream_index}",
                    "-f",
                    "null",
                    "-",
                ],
                check=False,
                capture_output=True,
                timeout=timeout,
                start_new_session=True,
                shell=False,
            )
        except OSError as error:
            raise AudioValidationError(
                AudioFailureCode.DECODER_UNAVAILABLE,
                "audio decoder is unavailable",
            ) from error
        except subprocess.TimeoutExpired as error:
            raise AudioValidationError(
                AudioFailureCode.TIMEOUT,
                "audio stream validation timed out",
            ) from error
        if completed.returncode != 0:
            raise AudioValidationError(
                AudioFailureCode.SELECTED_AUDIO_STREAM_UNDECODABLE,
                "audio stream is not decodable",
            )
