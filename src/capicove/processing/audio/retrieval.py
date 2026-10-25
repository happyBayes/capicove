"""Bounded, sample-exact PCM window preparation for native audio retrieval (RFC-0032).

The audio capability owns stream selection, decoding and window geometry. Retrieval
consumes :class:`PreparedAudio` and its receipts; it never invokes FFmpeg itself or
re-extracts PCM from rounded display times. No speech VAD or ASR is involved:
environmental sounds and music are windowed exactly like speech.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import secrets
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Protocol

from capicove.contracts.retrieval import ProcessingFingerprint, TemporalRange
from capicove.processing.audio.clip_models import AudioClippingError, remaining_seconds
from capicove.processing.audio.models import (
    AudioEvidence,
    AudioFailureCode,
    AudioStreamSelection,
)
from capicove.processing.audio.processor import AudioImportError, AudioProcessor


AUDIO_WINDOW_VERSION = "capicove-audio-window-v1"
RECEIPT_SCHEMA = "capicove.audio.window-receipt.v1"
MAX_PREVIEW_DURATION_SECONDS = 4 * 3600
MAX_PREVIEW_WINDOWS = 10_000
_BYTES_PER_FRAME = 2
_READ_CHUNK = 1024 * 1024
_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "source_sha256",
        "audio_stream_index",
        "selection_method",
        "start_frame",
        "end_frame",
        "sample_rate_hz",
        "channels",
        "decoder",
        "pcm_sha256",
        "preprocessing_sha256",
    }
)


class AudioWindowError(ValueError):
    """Report a stable, path-free window preparation failure."""

    def __init__(self, code: AudioFailureCode, message: str) -> None:
        super().__init__(f"{code.value}: {message}")
        self.code = code


def _positive_int(name: str, value: object, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer from 1 to {maximum}")
    return value


@dataclass(frozen=True, slots=True)
class AudioWindowPolicy:
    """Fix every parameter that changes prepared PCM or window geometry."""

    sample_rate_hz: int = 16_000
    window_seconds: int = 30
    hop_seconds: int = 25
    max_duration_seconds: int = MAX_PREVIEW_DURATION_SECONDS
    max_windows: int = MAX_PREVIEW_WINDOWS

    def __post_init__(self) -> None:
        _positive_int("sample_rate_hz", self.sample_rate_hz, 96_000)
        if self.sample_rate_hz < 8_000:
            raise ValueError("sample_rate_hz must be at least 8000")
        _positive_int("window_seconds", self.window_seconds, 600)
        _positive_int("hop_seconds", self.hop_seconds, self.window_seconds)
        _positive_int(
            "max_duration_seconds", self.max_duration_seconds, MAX_PREVIEW_DURATION_SECONDS
        )
        _positive_int("max_windows", self.max_windows, MAX_PREVIEW_WINDOWS)

    @property
    def window_frames(self) -> int:
        """Return the window length in output frames."""
        return self.window_seconds * self.sample_rate_hz

    @property
    def hop_frames(self) -> int:
        """Return the window advance in output frames."""
        return self.hop_seconds * self.sample_rate_hz

    @property
    def fingerprint(self) -> ProcessingFingerprint:
        """Hash the complete preprocessing identity, including the downmix rule."""
        return ProcessingFingerprint.from_manifest(
            {
                "version": AUDIO_WINDOW_VERSION,
                "sample_rate_hz": self.sample_rate_hz,
                "channels": 1,
                "sample_format": "pcm_s16le",
                "downmix": "ffmpeg-ac-1",
                "window_frames": self.window_frames,
                "hop_frames": self.hop_frames,
                "short_final_window": "preserve",
            }
        )


@dataclass(frozen=True, slots=True)
class AudioWindow:
    """Describe one half-open frame interval and its derived display range."""

    index: int
    start_frame: int
    end_frame: int
    sample_rate_hz: int

    @property
    def time_range(self) -> TemporalRange:
        """Floor the start and ceil the end; never use it to re-extract PCM."""
        start = self.start_frame * 1000 // self.sample_rate_hz
        end = -(-self.end_frame * 1000 // self.sample_rate_hz)
        return TemporalRange(start, end)


def plan_windows(total_frames: int, policy: AudioWindowPolicy) -> tuple[AudioWindow, ...]:
    """Emit ``[start, min(start + W, N))`` from zero, stopping once a window reaches N."""
    if isinstance(total_frames, bool) or not isinstance(total_frames, int) or total_frames < 1:
        raise ValueError("audio must contain at least one frame")
    if total_frames > policy.max_duration_seconds * policy.sample_rate_hz:
        raise ValueError("audio duration exceeds the configured limit")
    count = 1
    if total_frames > policy.window_frames:
        count += math.ceil((total_frames - policy.window_frames) / policy.hop_frames)
    if count > policy.max_windows:
        raise ValueError("audio window count exceeds the configured limit")
    windows: list[AudioWindow] = []
    start = 0
    while True:
        end = min(start + policy.window_frames, total_frames)
        windows.append(AudioWindow(len(windows), start, end, policy.sample_rate_hz))
        if end == total_frames:
            return tuple(windows)
        start += policy.hop_frames


@dataclass(frozen=True, slots=True)
class AudioWindowReceipt:
    """Bind one window's PCM to its source, stream, samples, decoder and policy."""

    source_sha256: str
    audio_stream_index: int
    selection_method: str
    start_frame: int
    end_frame: int
    sample_rate_hz: int
    channels: int
    decoder: str
    pcm_sha256: str
    preprocessing_sha256: str
    schema: str = field(default=RECEIPT_SCHEMA)

    def __post_init__(self) -> None:
        if self.schema != RECEIPT_SCHEMA:
            raise ValueError("unsupported audio window receipt schema")
        for name in ("source_sha256", "pcm_sha256", "preprocessing_sha256"):
            ProcessingFingerprint(getattr(self, name))
        for name in ("audio_stream_index", "start_frame"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            isinstance(self.end_frame, bool)
            or not isinstance(self.end_frame, int)
            or self.end_frame <= self.start_frame
        ):
            raise ValueError("end_frame must exceed start_frame")
        _positive_int("sample_rate_hz", self.sample_rate_hz, 96_000)
        if self.channels != 1 or isinstance(self.channels, bool):
            raise ValueError("audio window receipts describe mono PCM")
        if self.selection_method not in {"explicit_index", "sole_stream_implicit"}:
            raise ValueError("invalid stream selection method")
        if not isinstance(self.decoder, str) or not 1 <= len(self.decoder) <= 128:
            raise ValueError("decoder identity must be bounded text")

    @property
    def time_range(self) -> TemporalRange:
        """Return the display range derived from exact sample offsets."""
        return AudioWindow(0, self.start_frame, self.end_frame, self.sample_rate_hz).time_range

    def to_payload(self) -> dict[str, object]:
        """Return the closed JSON projection persisted with each searchable unit."""
        return {name: getattr(self, name) for name in sorted(_RECEIPT_FIELDS)}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> AudioWindowReceipt:
        """Decode exactly the closed field set; unknown fields fail closed."""
        if not isinstance(payload, Mapping) or set(payload) != _RECEIPT_FIELDS:
            raise ValueError("audio window receipt fields are invalid")
        try:
            return cls(**{name: payload[name] for name in _RECEIPT_FIELDS})  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError("audio window receipt fields are invalid") from error

    @property
    def digest(self) -> str:
        """Return the canonical receipt digest bound by evidence identifiers."""
        body = json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(body.encode()).hexdigest()


class AudioWindowBackend(Protocol):
    """Decode one selected stream from a caller-owned descriptor to mono PCM16."""

    def preflight(self, deadline: datetime) -> str:
        """Return the bounded decoder identity."""
        ...

    def decode(
        self,
        source: int,
        output: int,
        demuxer: str,
        audio_stream_index: int,
        sample_rate_hz: int,
        deadline: datetime,
    ) -> None:
        """Write raw PCM16LE mono frames to ``output``."""
        ...


class PreparedAudio:
    """Own one private staged PCM file; window reads verify their recorded digest."""

    def __init__(
        self,
        *,
        evidence: AudioEvidence,
        selection: AudioStreamSelection,
        policy: AudioWindowPolicy,
        decoder: str,
        descriptor: int,
        staging: int,
        name: str,
        total_frames: int,
        pcm_sha256: str,
        windows: tuple[AudioWindow, ...],
        receipts: tuple[AudioWindowReceipt, ...],
    ) -> None:
        self.evidence = evidence
        self.selection = selection
        self.policy = policy
        self.decoder = decoder
        self.total_frames = total_frames
        self.pcm_sha256 = pcm_sha256
        self.windows = windows
        self.receipts = receipts
        self._descriptor: int | None = descriptor
        self._staging = staging
        self._name = name

    def window_pcm(self, index: int) -> bytes:
        """Return one window's exact PCM bytes after re-verifying its digest."""
        if self._descriptor is None:
            raise AudioWindowError(AudioFailureCode.STORE_FAILED, "prepared audio is closed")
        window, receipt = self.windows[index], self.receipts[index]
        length = (window.end_frame - window.start_frame) * _BYTES_PER_FRAME
        payload = os.pread(self._descriptor, length, window.start_frame * _BYTES_PER_FRAME)
        if len(payload) != length or hashlib.sha256(payload).hexdigest() != receipt.pcm_sha256:
            raise AudioWindowError(AudioFailureCode.STORE_CORRUPTED, "prepared PCM changed")
        return payload

    def close(self) -> None:
        """Remove the staged PCM once; the descriptor is closed only after unlinking."""
        if self._descriptor is None:
            return
        descriptor, self._descriptor = self._descriptor, None
        try:
            try:
                os.unlink(self._name, dir_fd=self._staging)
            except FileNotFoundError:
                pass
        finally:
            os.close(descriptor)
            os.close(self._staging)

    def __enter__(self) -> PreparedAudio:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class AudioWindowPreparer:
    """Preserve evidence, decode the selected stream once and window it exactly."""

    def __init__(
        self,
        processor: AudioProcessor,
        backend: AudioWindowBackend,
        policy: AudioWindowPolicy | None = None,
    ) -> None:
        self._processor = processor
        self._backend = backend
        self.policy = policy or AudioWindowPolicy()

    def prepare(
        self,
        source: Path,
        store_root: Path,
        staging_root: Path,
        *,
        stream_index: int | None = None,
        deadline: datetime,
    ) -> PreparedAudio:
        """Return prepared windows; every failure removes the staged PCM."""
        from capicove.processing.audio._adapters.ffmpeg import AUDIO_DEMUXERS

        remaining_seconds(deadline)
        demuxer = AUDIO_DEMUXERS.get(source.suffix.lower())
        if demuxer is None:
            raise AudioWindowError(
                AudioFailureCode.UNSUPPORTED_FORMAT, "unsupported audio extension"
            )
        try:
            decoder = self._backend.preflight(deadline)
            imported = self._processor.import_file_with_selection(
                source,
                store_root,
                **({} if stream_index is None else {"audio_stream_index": stream_index}),
                deadline=deadline,
            )
        except AudioImportError as error:
            raise AudioWindowError(error.code, "source evidence could not be imported") from None
        except AudioClippingError as error:
            raise AudioWindowError(error.code, "audio decoder is unavailable") from None
        return self._decode(
            imported.evidence, imported.stream_selection, demuxer, decoder, staging_root, deadline
        )

    def prepare_preserved(
        self,
        evidence: AudioEvidence,
        selection: AudioStreamSelection,
        suffix: str,
        staging_root: Path,
        *,
        deadline: datetime,
        container: str = "audio",
    ) -> PreparedAudio:
        """Decode bytes preserved at import time; the original path is never reopened.

        ``container="video"`` explicitly admits a video container's soundtrack
        (``.mp4``, ``.mov``, ``.mkv``); audio-only callers keep the audio allowlist.
        """
        from capicove.processing.audio._adapters.ffmpeg import (
            AUDIO_DEMUXERS,
            VIDEO_CONTAINER_DEMUXERS,
        )

        remaining_seconds(deadline)
        if container not in {"audio", "video"}:
            raise AudioWindowError(AudioFailureCode.UNSUPPORTED_FORMAT, "unsupported container")
        demuxers = VIDEO_CONTAINER_DEMUXERS if container == "video" else AUDIO_DEMUXERS
        demuxer = demuxers.get(suffix.lower())
        if demuxer is None:
            raise AudioWindowError(
                AudioFailureCode.UNSUPPORTED_FORMAT, "unsupported audio extension"
            )
        try:
            decoder = self._backend.preflight(deadline)
        except AudioClippingError as error:
            raise AudioWindowError(error.code, "audio decoder is unavailable") from None
        return self._decode(evidence, selection, demuxer, decoder, staging_root, deadline)

    def _decode(
        self,
        evidence: AudioEvidence,
        selection: AudioStreamSelection,
        demuxer: str,
        decoder: str,
        staging_root: Path,
        deadline: datetime,
    ) -> PreparedAudio:
        staging = _open_private_directory(staging_root)
        name = f".pcm-{secrets.token_hex(16)}"
        try:
            output = os.open(
                name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=staging,
            )
        except BaseException:
            os.close(staging)
            raise
        try:
            source_descriptor = _open_verified_evidence(evidence, deadline)
            try:
                self._backend.decode(
                    source_descriptor,
                    output,
                    demuxer,
                    selection.selected_audio_stream_index,
                    self.policy.sample_rate_hz,
                    deadline,
                )
                if _hash_descriptor(source_descriptor, deadline) != evidence.sha256:
                    raise AudioWindowError(AudioFailureCode.SOURCE_CHANGED, "source changed")
            finally:
                os.close(source_descriptor)
            size = os.fstat(output).st_size
            if size == 0 or size % _BYTES_PER_FRAME:
                raise AudioWindowError(
                    AudioFailureCode.INVALID_AUDIO, "decoded PCM is empty or partial"
                )
            try:
                windows = plan_windows(size // _BYTES_PER_FRAME, self.policy)
            except ValueError as error:
                raise AudioWindowError(AudioFailureCode.INVALID_AUDIO, str(error)) from None
            pcm_sha256 = _hash_descriptor(output, deadline)
            receipts = tuple(
                AudioWindowReceipt(
                    source_sha256=evidence.sha256,
                    audio_stream_index=selection.selected_audio_stream_index,
                    selection_method=selection.selection_method,
                    start_frame=window.start_frame,
                    end_frame=window.end_frame,
                    sample_rate_hz=self.policy.sample_rate_hz,
                    channels=1,
                    decoder=decoder,
                    pcm_sha256=_hash_range(output, window, deadline),
                    preprocessing_sha256=self.policy.fingerprint.sha256,
                )
                for window in windows
            )
        except BaseException as error:
            try:
                os.unlink(name, dir_fd=staging)
            finally:
                os.close(output)
                os.close(staging)
            if isinstance(error, AudioClippingError):
                raise AudioWindowError(error.code, "audio decoding failed") from None
            if isinstance(error, OSError) and error.errno == errno.ENOSPC:
                raise AudioWindowError(
                    AudioFailureCode.STORE_FAILED, "staging disk is full"
                ) from None
            raise
        return PreparedAudio(
            evidence=evidence,
            selection=selection,
            policy=self.policy,
            decoder=decoder,
            descriptor=output,
            staging=staging,
            name=name,
            total_frames=size // _BYTES_PER_FRAME,
            pcm_sha256=pcm_sha256,
            windows=windows,
            receipts=receipts,
        )


def _open_private_directory(path: Path) -> int:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    identity = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(identity.st_mode)
        or identity.st_uid != os.geteuid()
        or identity.st_mode & 0o077
    ):
        os.close(descriptor)
        raise AudioWindowError(AudioFailureCode.STORE_FAILED, "staging root must be private")
    return descriptor


def _open_verified_evidence(evidence: AudioEvidence, deadline: datetime) -> int:
    descriptor = os.open(evidence.stored_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        identity = os.fstat(descriptor)
        if (
            not stat.S_ISREG(identity.st_mode)
            or identity.st_size != evidence.size_bytes
            or _hash_descriptor(descriptor, deadline) != evidence.sha256
        ):
            raise AudioWindowError(AudioFailureCode.STORE_CORRUPTED, "source evidence is invalid")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _hash_descriptor(descriptor: int, deadline: datetime) -> str:
    digest = hashlib.sha256()
    offset = 0
    while chunk := os.pread(descriptor, _READ_CHUNK, offset):
        remaining_seconds(deadline)
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _hash_range(descriptor: int, window: AudioWindow, deadline: datetime) -> str:
    digest = hashlib.sha256()
    offset = window.start_frame * _BYTES_PER_FRAME
    end = window.end_frame * _BYTES_PER_FRAME
    while offset < end:
        remaining_seconds(deadline)
        chunk = os.pread(descriptor, min(_READ_CHUNK, end - offset), offset)
        if not chunk:
            raise AudioWindowError(AudioFailureCode.STORE_CORRUPTED, "prepared PCM is truncated")
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


__all__ = [
    "AUDIO_WINDOW_VERSION",
    "MAX_PREVIEW_DURATION_SECONDS",
    "MAX_PREVIEW_WINDOWS",
    "AudioWindow",
    "AudioWindowBackend",
    "AudioWindowError",
    "AudioWindowPolicy",
    "AudioWindowPreparer",
    "AudioWindowReceipt",
    "PreparedAudio",
    "plan_windows",
]
