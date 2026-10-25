from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from capicove.processing.audio import (
    AudioBackendLineage,
    AudioFailureCode,
    AudioImportError,
    AudioProcessor,
    AudioStreamInventory,
    AudioStreamSelection,
    AudioTranscript,
    AudioTranscriptionError,
    AudioTranscriptionSegment,
    AudioTranscriptionService,
    AudioWarningCode,
)


_WAV = b"RIFF\x04\x00\x00\x00WAVE"


class _AcceptingValidator:
    def validate(
        self,
        source: Path,
        *,
        audio_stream_index: int,
        deadline: datetime | None = None,
    ) -> None:
        pass


class _Inspector:
    def __init__(self, count: int = 1) -> None:
        self.count = count

    def inspect(
        self,
        source: Path,
        *,
        deadline: datetime | None = None,
    ) -> AudioStreamInventory:
        return AudioStreamInventory(self.count)


class _StubTranscriber:
    def __init__(
        self,
        segments: tuple[AudioTranscriptionSegment, ...],
        *,
        language: str = "zh",
    ) -> None:
        self._segments = segments
        self._language = language
        self.sources: list[Path] = []
        self.stream_indexes: list[int] = []
        self.mutate_evidence = False

    def transcribe(
        self,
        source: Path,
        *,
        audio_stream_index: int,
        language: str,
        deadline: datetime,
    ) -> AudioTranscript:
        self.sources.append(source)
        self.stream_indexes.append(audio_stream_index)
        if self.mutate_evidence:
            source.write_bytes(b"changed")
        return AudioTranscript(
            text="".join(segment.text for segment in self._segments),
            language=self._language,
            segments=self._segments,
            warnings=(),
            lineage=AudioBackendLineage(
                backend="fake-asr",
                backend_version="1.0",
                model="fake-multilingual",
                model_sha256="a" * 64,
                device="cpu",
                precision="fp32",
            ),
        )


def _service(
    transcriber: _StubTranscriber,
    *,
    stream_count: int = 1,
) -> AudioTranscriptionService:
    return AudioTranscriptionService(
        AudioProcessor(_Inspector(stream_count), _AcceptingValidator()),
        transcriber,
    )


def test_transcription_segment_requires_finite_positive_ordered_time() -> None:
    with pytest.raises(ValueError, match="start_seconds"):
        AudioTranscriptionSegment(0, float("nan"), 1.0, "hello")
    with pytest.raises(ValueError, match="end_seconds"):
        AudioTranscriptionSegment(0, 1.0, 1.0, "hello")
    with pytest.raises(ValueError, match="text"):
        AudioTranscriptionSegment(0, 0.0, 1.0, "   ")


def test_transcript_rejects_overlapping_or_mismatched_segments() -> None:
    first = AudioTranscriptionSegment(0, 0.0, 1.0, "hello")
    overlapping = AudioTranscriptionSegment(1, 0.5, 1.5, "world")
    lineage = AudioBackendLineage("fake", "1", "model", "a" * 64, "cpu", "fp32")

    with pytest.raises(ValueError, match="overlap"):
        AudioTranscript("helloworld", "en", (first, overlapping), (), lineage)
    with pytest.raises(ValueError, match="concatenated"):
        AudioTranscript("different", "en", (first,), (), lineage)


def test_service_preserves_original_and_transcribes_verified_evidence(tmp_path: Path) -> None:
    source = tmp_path / "meeting.wav"
    source.write_bytes(_WAV)
    segments = (
        AudioTranscriptionSegment(0, 0.0, 1.2, "第一项。"),
        AudioTranscriptionSegment(1, 1.2, 2.4, "Second item."),
    )
    transcriber = _StubTranscriber(segments)

    result = _service(transcriber).transcribe_file(
        source,
        tmp_path / "store",
        language="auto",
        deadline=datetime.now(UTC) + timedelta(minutes=1),
    )

    assert source.read_bytes() == _WAV
    assert result.evidence.sha256 == hashlib.sha256(_WAV).hexdigest()
    assert transcriber.sources == [result.evidence.stored_path]
    assert transcriber.stream_indexes == [0]
    assert result.transcript.text == "第一项。Second item."
    assert result.transcript.segments == segments
    assert result.stream_selection == AudioStreamSelection(
        selected_audio_stream_index=0,
        selection_method="sole_stream_implicit",
        preflight_audio_stream_count=1,
    )


def test_service_returns_explicit_silence_warning_without_fabricated_text(
    tmp_path: Path,
) -> None:
    source = tmp_path / "silence.wav"
    source.write_bytes(_WAV)
    transcriber = _StubTranscriber(())

    result = _service(transcriber).transcribe_file(
        source,
        tmp_path / "store",
        language="auto",
        deadline=datetime.now(UTC) + timedelta(minutes=1),
    )

    assert result.transcript.text == ""
    assert result.transcript.segments == ()
    assert result.transcript.warnings == (AudioWarningCode.NO_SPEECH_DETECTED,)


def test_service_rejects_expired_deadline_before_processing(tmp_path: Path) -> None:
    source = tmp_path / "meeting.wav"
    source.write_bytes(_WAV)

    with pytest.raises(AudioTranscriptionError, match="deadline") as captured:
        _service(_StubTranscriber(())).transcribe_file(
            source,
            tmp_path / "store",
            language="auto",
            deadline=datetime.now(UTC) - timedelta(seconds=1),
        )

    assert captured.value.code is AudioFailureCode.TIMEOUT
    assert not (tmp_path / "store").exists()


def test_service_rejects_evidence_modified_during_transcription(tmp_path: Path) -> None:
    source = tmp_path / "meeting.wav"
    source.write_bytes(_WAV)
    transcriber = _StubTranscriber(())
    transcriber.mutate_evidence = True

    with pytest.raises(AudioTranscriptionError, match="evidence changed") as captured:
        _service(transcriber).transcribe_file(
            source,
            tmp_path / "store",
            language="auto",
            deadline=datetime.now(UTC) + timedelta(minutes=1),
        )

    assert captured.value.code is AudioFailureCode.STORE_CORRUPTED
    assert source.read_bytes() == _WAV


def test_service_enforces_deadline_after_transcriber_returns(tmp_path: Path) -> None:
    source = tmp_path / "meeting.wav"
    source.write_bytes(_WAV)
    transcriber = _StubTranscriber(())
    expired = AudioTranscriptionError(
        AudioFailureCode.TIMEOUT,
        "audio transcription deadline has expired",
    )

    with (
        patch(
            "capicove.processing.audio.transcription._require_future_deadline",
            side_effect=[None, expired],
        ),
        pytest.raises(AudioTranscriptionError, match="deadline") as captured,
    ):
        _service(transcriber).transcribe_file(
            source,
            tmp_path / "store",
            language="auto",
            deadline=datetime.now(UTC) + timedelta(minutes=1),
        )

    assert captured.value.code is AudioFailureCode.TIMEOUT


@pytest.mark.parametrize("language", ["", "zh_CN", "zh-XX", "en-XX", "english", "*"])
def test_service_rejects_unsupported_language_before_processing(
    tmp_path: Path,
    language: str,
) -> None:
    source = tmp_path / "meeting.wav"
    source.write_bytes(_WAV)

    with pytest.raises(AudioTranscriptionError) as captured:
        _service(_StubTranscriber(())).transcribe_file(
            source,
            tmp_path / "store",
            language=language,
            deadline=datetime.now(UTC) + timedelta(minutes=1),
        )

    assert captured.value.code is AudioFailureCode.UNSUPPORTED_LANGUAGE
    assert not (tmp_path / "store").exists()


def test_service_passes_explicit_stream_and_adds_ignored_warning_once(
    tmp_path: Path,
) -> None:
    source = tmp_path / "meeting.wav"
    source.write_bytes(_WAV)
    transcriber = _StubTranscriber(())

    result = _service(transcriber, stream_count=2).transcribe_file(
        source,
        tmp_path / "store",
        language="auto",
        deadline=datetime.now(UTC) + timedelta(minutes=1),
        audio_stream_index=1,
    )

    assert transcriber.stream_indexes == [1]
    assert result.stream_selection == AudioStreamSelection(
        selected_audio_stream_index=1,
        selection_method="explicit_index",
        preflight_audio_stream_count=2,
    )
    assert result.transcript.warnings == (
        AudioWarningCode.NO_SPEECH_DETECTED,
        AudioWarningCode.UNSELECTED_AUDIO_STREAMS_IGNORED,
    )


@pytest.mark.parametrize("selector", [None, True, 1.0, "1", -1])
def test_service_invalid_selector_does_not_write_or_invoke_asr(
    tmp_path: Path,
    selector: object,
) -> None:
    source = tmp_path / "meeting.wav"
    source.write_bytes(_WAV)
    store = tmp_path / "store"
    transcriber = _StubTranscriber(())

    with pytest.raises(AudioImportError) as captured:
        _service(transcriber, stream_count=2).transcribe_file(
            source,
            store,
            language="auto",
            deadline=datetime.now(UTC) + timedelta(minutes=1),
            audio_stream_index=selector,
        )

    assert captured.value.code is AudioFailureCode.INVALID_AUDIO_STREAM_INDEX
    assert transcriber.sources == []
    assert not store.exists()


def test_service_validates_selector_before_deadline_and_source_access(
    tmp_path: Path,
) -> None:
    with pytest.raises(AudioImportError) as captured:
        _service(_StubTranscriber(())).transcribe_file(
            tmp_path / "missing.wav",
            tmp_path / "store",
            language="auto",
            deadline=datetime.now(UTC) - timedelta(seconds=1),
            audio_stream_index=None,
        )

    assert captured.value.code is AudioFailureCode.INVALID_AUDIO_STREAM_INDEX
    assert not (tmp_path / "store").exists()


def test_service_multistream_omission_rejects_before_asr_or_storage(
    tmp_path: Path,
) -> None:
    source = tmp_path / "meeting.wav"
    source.write_bytes(_WAV)
    store = tmp_path / "store"
    transcriber = _StubTranscriber(())

    with pytest.raises(AudioImportError) as captured:
        _service(transcriber, stream_count=2).transcribe_file(
            source,
            store,
            language="auto",
            deadline=datetime.now(UTC) + timedelta(minutes=1),
        )

    assert captured.value.code is AudioFailureCode.AUDIO_STREAM_SELECTION_REQUIRED
    assert transcriber.sources == []
    assert not store.exists()
