"""Local ASR adaptation preserves the public service's operational failures."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from capicove.contracts.retrieval import ProcessingFingerprint
from capicove.processing.audio.models import AudioFailureCode
from capicove.processing.audio.transcription import (
    AudioTranscriptionError,
    AudioTranscriptionService,
)
from capicove.processing.video.models import AcceptedVideoSource
from capicove.processing.video.ports import BackendUnavailableError
from capicove.processing.video.retrieval._adapters.providers import LocalASRProcessor


@pytest.mark.parametrize("index", [None, 1])
def test_asr_passes_selected_stream_without_inventing_a_default(
    tmp_path: Path, index: int | None
) -> None:
    service = MagicMock(spec=AudioTranscriptionService)
    service.transcribe_file.return_value.transcript.segments = ("speech",)
    adapter = LocalASRProcessor(
        service,
        tmp_path,
        provider_fingerprint=ProcessingFingerprint("a" * 64),
        language="en",
        audio_stream_index=index,
    )
    source = AcceptedVideoSource("source", "a" * 64, "video/mp4", path=tmp_path / "video.mp4")
    deadline = datetime.now(UTC) + timedelta(seconds=1)
    assert (
        adapter.transcribe(source, deadline=deadline)
        is service.transcribe_file.return_value.transcript
    )
    kwargs = service.transcribe_file.call_args.kwargs
    assert kwargs["deadline"] == deadline
    assert kwargs["language"] == "en"
    if index is None:
        assert "audio_stream_index" not in kwargs
    else:
        assert kwargs["audio_stream_index"] == index


def test_asr_no_speech_no_audio_and_operational_failure(tmp_path: Path) -> None:
    service = MagicMock(spec=AudioTranscriptionService)
    service.transcribe_file.return_value.transcript.segments = ()
    adapter = LocalASRProcessor(
        service, tmp_path, provider_fingerprint=ProcessingFingerprint("a" * 64)
    )
    source = AcceptedVideoSource("source", "a" * 64, "video/mp4", path=tmp_path / "video.mp4")
    deadline = datetime.now(UTC) + timedelta(seconds=1)
    assert adapter.transcribe(source, deadline=deadline) is None
    service.transcribe_file.side_effect = AudioTranscriptionError(
        AudioFailureCode.NO_AUDIO_STREAM, "no audio"
    )
    assert adapter.transcribe(source, deadline=deadline) is None
    service.transcribe_file.side_effect = AudioTranscriptionError(
        AudioFailureCode.MODEL_UNAVAILABLE, "model unavailable"
    )
    with pytest.raises(BackendUnavailableError):
        adapter.transcribe(source, deadline=deadline)
    service.transcribe_file.side_effect = AudioTranscriptionError(
        AudioFailureCode.MODEL_LOAD_FAILED, "corrupt model"
    )
    with pytest.raises(AudioTranscriptionError) as caught:
        adapter.transcribe(source, deadline=deadline)
    assert caught.value.code == AudioFailureCode.MODEL_LOAD_FAILED
    with pytest.raises(ValueError, match="materialized"):
        adapter.transcribe(
            AcceptedVideoSource("source", "a" * 64, "video/mp4", data=b"bytes"), deadline=deadline
        )
