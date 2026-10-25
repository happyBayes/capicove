"""Dependency-free uploaded-audio inference port."""

from datetime import datetime
from pathlib import Path
from typing import Protocol

from capicove.processing.audio.models import AudioTranscript


class AudioTranscriptionProvider(Protocol):
    """Transcribe an authorized local file with a bounded processing deadline."""

    model_id: str

    def transcribe(self, source: Path, *, language: str, deadline: datetime) -> AudioTranscript: ...

    def close(self) -> None: ...
