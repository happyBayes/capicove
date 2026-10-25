"""Application service for evidence-preserving local audio transcription."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from capicove.processing.audio.models import (
    AudioFailureCode,
    AudioTranscript,
    AudioTranscriptionResult,
    AudioWarningCode,
)
from capicove.processing.audio.ports import AudioTranscriber
from capicove.processing.audio.processor import (
    _AUDIO_STREAM_INDEX_OMITTED,
    AudioProcessor,
    _validate_audio_stream_index,
)


_SUPPORTED_LANGUAGES = frozenset({"auto", "zh", "en", "zh-CN", "en-US", "en-GB"})


class AudioTranscriptionError(RuntimeError):
    """Report a stable local-ASR failure without exposing private content."""

    def __init__(self, code: AudioFailureCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class AudioTranscriptionService:
    """Preserve original evidence before deriving a local transcript."""

    def __init__(self, processor: AudioProcessor, transcriber: AudioTranscriber) -> None:
        self._processor = processor
        self._transcriber = transcriber

    def transcribe_file(
        self,
        source_path: Path,
        store_root: Path,
        *,
        language: str,
        deadline: datetime,
        audio_stream_index: object = _AUDIO_STREAM_INDEX_OMITTED,
    ) -> AudioTranscriptionResult:
        """Import one local file and derive an auditable timestamped transcript."""
        _validate_audio_stream_index(audio_stream_index)
        _require_future_deadline(deadline)
        if language not in _SUPPORTED_LANGUAGES:
            raise AudioTranscriptionError(
                AudioFailureCode.UNSUPPORTED_LANGUAGE,
                "requested language is unsupported",
            )
        imported = self._processor.import_file_with_selection(
            source_path,
            store_root,
            audio_stream_index=audio_stream_index,
            deadline=deadline,
        )
        evidence = imported.evidence
        transcript = self._transcriber.transcribe(
            evidence.stored_path,
            audio_stream_index=imported.stream_selection.selected_audio_stream_index,
            language=language,
            deadline=deadline,
        )
        _require_future_deadline(deadline)
        if _hash_file(evidence.stored_path, deadline) != evidence.sha256:
            raise AudioTranscriptionError(
                AudioFailureCode.STORE_CORRUPTED,
                "immutable audio evidence changed during transcription",
            )
        if (
            not transcript.segments
            and AudioWarningCode.NO_SPEECH_DETECTED not in transcript.warnings
        ):
            transcript = AudioTranscript(
                text="",
                language=transcript.language,
                segments=(),
                warnings=(*transcript.warnings, AudioWarningCode.NO_SPEECH_DETECTED),
                lineage=transcript.lineage,
                input_duration_seconds=transcript.input_duration_seconds,
            )
        if (
            imported.stream_selection.selection_method == "explicit_index"
            and imported.stream_selection.preflight_audio_stream_count > 1
            and AudioWarningCode.UNSELECTED_AUDIO_STREAMS_IGNORED not in transcript.warnings
        ):
            transcript = AudioTranscript(
                text=transcript.text,
                language=transcript.language,
                segments=transcript.segments,
                warnings=(
                    *transcript.warnings,
                    AudioWarningCode.UNSELECTED_AUDIO_STREAMS_IGNORED,
                ),
                lineage=transcript.lineage,
                input_duration_seconds=transcript.input_duration_seconds,
            )
        return AudioTranscriptionResult(evidence, transcript, imported.stream_selection)


def _require_future_deadline(deadline: datetime) -> None:
    if deadline.tzinfo is None or deadline.utcoffset() is None:
        raise ValueError("deadline must be timezone-aware")
    if deadline <= datetime.now(UTC):
        raise AudioTranscriptionError(
            AudioFailureCode.TIMEOUT,
            "audio transcription deadline has expired",
        )


def _hash_file(path: Path, deadline: datetime) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while True:
                _require_future_deadline(deadline)
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as error:
        raise AudioTranscriptionError(
            AudioFailureCode.STORE_CORRUPTED,
            "immutable audio evidence changed during transcription",
        ) from error
    return digest.hexdigest()
