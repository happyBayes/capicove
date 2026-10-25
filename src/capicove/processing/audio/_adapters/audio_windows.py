"""Fixed FFmpeg command decoding one selected audio stream to raw mono PCM16."""

from __future__ import annotations

import os
import sys
from datetime import datetime

from capicove.processing.audio._adapters.audio_clip import _run
from capicove.processing.audio.clip_models import AudioClippingError
from capicove.processing.audio.models import AudioFailureCode


class FfmpegAudioWindowBackend:
    """Decode caller-owned descriptors; never open caller-controlled paths."""

    def __init__(self, ffmpeg: str = "ffmpeg") -> None:
        self.ffmpeg = ffmpeg

    def preflight(self, deadline: datetime) -> str:
        """Return the first FFmpeg version line, bounded for receipts."""
        if sys.platform not in ("linux", "darwin"):
            raise AudioClippingError(
                AudioFailureCode.BACKEND_UNAVAILABLE, "unsupported decoding platform"
            )
        first = _run([self.ffmpeg, "-version"], deadline).splitlines()[0]
        words = first.decode("utf-8", "replace").split()
        if len(words) < 3 or words[0] != "ffmpeg" or words[1] != "version":
            raise AudioClippingError(
                AudioFailureCode.BACKEND_UNAVAILABLE, "unrecognized FFmpeg version output"
            )
        return f"ffmpeg {words[2][:96]}"

    def decode(
        self,
        source: int,
        output: int,
        demuxer: str,
        audio_stream_index: int,
        sample_rate_hz: int,
        deadline: datetime,
    ) -> None:
        """Write raw little-endian mono PCM16 of exactly one selected stream."""
        os.lseek(source, 0, os.SEEK_SET)
        os.lseek(output, 0, os.SEEK_SET)
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
                f"0:a:{audio_stream_index}",
                "-vn",
                "-sn",
                "-dn",
                "-ac",
                "1",
                "-ar",
                str(sample_rate_hz),
                "-c:a",
                "pcm_s16le",
                "-map_metadata",
                "-1",
                "-f",
                "s16le",
                "-fd",
                str(output),
                "fd:",
            ],
            deadline,
            (source, output),
        )
