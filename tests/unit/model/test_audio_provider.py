"""ASR policy fails closed before native inference."""

from pathlib import Path

import pytest

from capicove.model.configuration import ModelCapability, ModelProfile
from capicove.model.providers.audio import validate_audio


def test_audio_rejects_unbounded_duration(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        validate_audio(
            ModelProfile(
                "speech",
                "whisper-cpp",
                ModelCapability.AUDIO_TRANSCRIPTION,
                tmp_path,
                "a" * 64,
                options={"max_duration_seconds": 0},
            )
        )


def test_whisper_duration_limit_is_explicit(tmp_path: Path) -> None:
    from capicove.processing.audio._adapters.whisper_cpp import WhisperCppTranscriber

    with pytest.raises(ValueError):
        WhisperCppTranscriber(tmp_path / "m", vad_model_path=tmp_path / "v", max_duration_seconds=0)


def test_decode_limit_rejects_before_whisper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import wave
    from datetime import UTC, datetime, timedelta
    from typing import Any

    from capicove.processing.audio._adapters.whisper_cpp import WhisperCppTranscriber
    from capicove.processing.audio.transcription import AudioTranscriptionError

    model, vad = tmp_path / "m", tmp_path / "v"
    model.write_bytes(b"model")
    vad.write_bytes(b"vad")
    transcriber = WhisperCppTranscriber(model, vad_model_path=vad, max_duration_seconds=1)

    def normalize(source: Path, output: Path, *args: Any) -> None:
        with wave.open(str(output), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(b"\0" * 64000)

    monkeypatch.setattr(transcriber, "_normalize", normalize)
    with pytest.raises(AudioTranscriptionError, match="duration limit"):
        transcriber.transcribe(
            tmp_path / "input", language="en", deadline=datetime.now(UTC) + timedelta(seconds=10)
        )


def test_asset_replacement_rejected_before_decode(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta

    from capicove.processing.audio._adapters.whisper_cpp import WhisperCppTranscriber
    from capicove.processing.audio.transcription import AudioTranscriptionError

    model, vad = tmp_path / "m", tmp_path / "v"
    model.write_bytes(b"changed")
    vad.write_bytes(b"vad")
    provider = WhisperCppTranscriber(model, vad_model_path=vad, expected_model_sha256="a" * 64)
    with pytest.raises(AudioTranscriptionError, match="digest mismatch"):
        provider.transcribe(
            tmp_path / "missing-audio",
            language="en",
            deadline=datetime.now(UTC) + timedelta(seconds=10),
        )
