"""Backend-independent request, geometry, and ports for audio clipping."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Protocol

from capicove.processing.audio.models import AudioFailureCode


MAX_PCM_BYTES = 512 * 1024**2
DISK_RESERVE_BYTES = 16 * 1024**2


class AudioClippingError(ValueError):
    """Report a stable clipping failure without private paths or diagnostics."""

    def __init__(self, code: AudioFailureCode, message: str, *, reason: str | None = None) -> None:
        super().__init__(f"{code.value}: {message}")
        self.code = code
        self.reason = reason


def remaining_seconds(deadline: datetime) -> float:
    """Require an aware, unexpired invocation deadline."""
    if deadline.tzinfo is None or deadline.utcoffset() is None:
        raise ValueError("deadline must be timezone-aware")
    remaining = (deadline - datetime.now(UTC)).total_seconds()
    if remaining <= 0:
        raise AudioClippingError(AudioFailureCode.TIMEOUT, "audio clipping timed out")
    return min(remaining, 300.0)


@dataclass(frozen=True, slots=True)
class AudioClipRequest:
    """Select one half-open local audio range and a PCM16 WAV destination."""

    source: Path
    destination: Path
    start_seconds: float
    end_seconds: float
    profile: str = "pcm16-wav"
    sample_rate_hz: int | None = None
    channels: int | None = None

    def __post_init__(self) -> None:
        for path in (self.source, self.destination):
            if not isinstance(path, Path) or ".." in path.parts or "\0" in str(path):
                raise ValueError("audio paths must be normalized local paths")
        if self.destination.suffix != ".wav":
            raise ValueError("destination must use the lowercase .wav extension")
        for value in (self.start_seconds, self.end_seconds):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("range endpoints must be finite numbers")
        if self.start_seconds < 0 or self.end_seconds <= self.start_seconds:
            raise ValueError("range must be non-negative and non-empty")
        if self.profile != "pcm16-wav":
            raise ValueError("only the pcm16-wav profile is supported")
        if self.sample_rate_hz is not None and (
            type(self.sample_rate_hz) is not int or not 8000 <= self.sample_rate_hz <= 96000
        ):
            raise ValueError("sample rate must be an integer from 8000 through 96000")
        if self.channels is not None and (
            type(self.channels) is not int or self.channels not in (1, 2)
        ):
            raise ValueError("channels must be one or two")


@dataclass(frozen=True, slots=True)
class AudioClipMetadata:
    """Describe the actual selected stream without confusing codec and container."""

    duration_seconds: float
    sample_rate_hz: int
    channels: int
    codec: str
    container: str
    bit_depth: int | None
    bit_rate_bps: int | None
    audio_stream_index: int
    container_stream_index: int
    duration_exact: Fraction | None = None


@dataclass(frozen=True, slots=True)
class AudioClipGeometry:
    """Resolve exact source frames and bounded output geometry."""

    start_frame: int
    end_frame: int
    source_rate_hz: int
    sample_rate_hz: int
    channels: int
    expected_frames: int

    @classmethod
    def resolve(cls, request: AudioClipRequest, source: AudioClipMetadata) -> AudioClipGeometry:
        """Quantize boundaries using decimal arithmetic and half-up frame counts."""
        rate = request.sample_rate_hz or source.sample_rate_hz
        channels = request.channels or source.channels
        if not 8000 <= rate <= 96000 or channels not in (1, 2):
            raise AudioClippingError(
                AudioFailureCode.UNSUPPORTED_AUDIO_PROFILE,
                "specify a supported rate and channel count",
            )
        upper_bound = source.duration_exact or Fraction(Decimal(str(source.duration_seconds)))
        if Fraction(Decimal(str(request.end_seconds))) > upper_bound:
            raise AudioClippingError(
                AudioFailureCode.INVALID_RANGE, "range exceeds the selected audio stream"
            )
        with localcontext() as context:
            context.prec = 400
            start = int(
                (Decimal(str(request.start_seconds)) * source.sample_rate_hz).to_integral_value(
                    rounding=ROUND_CEILING
                )
            )
            end = int(
                (Decimal(str(request.end_seconds)) * source.sample_rate_hz).to_integral_value(
                    rounding=ROUND_CEILING
                )
            )
        frames = end - start
        if frames <= 0 or (
            rate != source.sample_rate_hz and frames * 1000 < source.sample_rate_hz * 20
        ):
            raise AudioClippingError(
                AudioFailureCode.INVALID_RANGE, "range is empty or too short for resampling"
            )
        quotient, remainder = divmod(frames * rate, source.sample_rate_hz)
        expected = quotient + int(2 * remainder >= source.sample_rate_hz)
        if expected * channels * 2 > MAX_PCM_BYTES:
            raise AudioClippingError(
                AudioFailureCode.INVALID_RANGE, "PCM output exceeds the size limit"
            )
        return cls(start, end, source.sample_rate_hz, rate, channels, expected)


class AudioClipBackend(Protocol):
    """Probe and transform only caller-owned open descriptors."""

    def preflight(self, deadline: datetime) -> str:
        """Check descriptor protocol behavior and return the backend version."""
        ...

    def inspect(self, descriptor: int, demuxer: str, deadline: datetime) -> AudioClipMetadata:
        """Require one audio stream with trustworthy duration and geometry."""
        ...

    def write(
        self,
        source: int,
        output: int,
        demuxer: str,
        geometry: AudioClipGeometry,
        deadline: datetime,
    ) -> None:
        """Decode, sample-trim, and write a seekable PCM16 WAV."""
        ...
