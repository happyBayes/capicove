"""Ports consumed by the audio processing capability."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Protocol

from capicove.processing.audio.models import (
    AudioFailureCode,
    AudioStreamInventory,
    AudioTranscript,
)


class AudioValidationError(ValueError):
    """Report audio that a selected decoder cannot fully validate."""

    def __init__(self, code: AudioFailureCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class AudioPathPolicyError(PermissionError):
    """Report that an audio source is outside explicitly authorized roots."""


class AudioValidator(Protocol):
    """Validate that an audio source can be decoded from start to finish."""

    def validate(
        self,
        source: Path,
        *,
        audio_stream_index: int,
        deadline: datetime | None = None,
    ) -> None:
        """Raise ``AudioValidationError`` when the source is not decodable."""


class AudioStreamInspector(Protocol):
    """Inventory audio-stream ordinals without choosing one by metadata."""

    def inspect(
        self,
        source: Path,
        *,
        deadline: datetime | None = None,
    ) -> AudioStreamInventory:
        """Return a bounded inventory or raise ``AudioValidationError``."""


class AudioTranscriber(Protocol):
    """Transcribe validated local audio through one explicit backend."""

    def transcribe(
        self,
        source: Path,
        *,
        audio_stream_index: int,
        language: str,
        deadline: datetime,
    ) -> AudioTranscript:
        """Return ordered transcript segments or raise a stable audio error."""
