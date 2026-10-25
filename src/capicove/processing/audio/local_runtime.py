"""Public composition for already-authorized, explicitly provisioned local ASR."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from capicove.processing.audio.models import AudioTranscript


def transcribe_local(
    path: Path,
    *,
    model: Path,
    vad_model: Path,
    whisper_cli: str,
    whisper_vad_cli: str,
    ffmpeg: str,
    ffprobe: str,
    backend_version: str,
    deadline: datetime,
) -> AudioTranscript:
    """Transcribe one authorized staged file; callers verify all supplied assets."""
    from capicove.processing.audio._adapters.ffprobe import FfprobeAudioStreamInspector
    from capicove.processing.audio._adapters.whisper_cpp import WhisperCppTranscriber

    inventory = FfprobeAudioStreamInspector(ffprobe, restrict_inputs=True).inspect(
        path, deadline=deadline
    )
    if inventory.audio_stream_count != 1:
        raise ValueError("knowledge_requires_one_audio_stream")

    return WhisperCppTranscriber(
        model,
        vad_model_path=vad_model,
        executable=whisper_cli,
        vad_executable=whisper_vad_cli,
        ffmpeg_executable=ffmpeg,
        backend_version=backend_version,
        restrict_inputs=True,
    ).transcribe(path, language="auto", deadline=deadline)


class LocalAudioProvider:
    """Compose single-stream inspection and bounded Whisper transcription."""

    def __init__(
        self,
        *,
        model: Path,
        vad_model: Path,
        whisper_cli: str,
        whisper_vad_cli: str,
        ffmpeg: str,
        ffprobe: str,
        backend_version: str,
        max_duration_seconds: int = 1800,
        expected_model_sha256: str | None = None,
        expected_vad_sha256: str | None = None,
    ) -> None:
        from capicove.processing.audio._adapters.ffprobe import FfprobeAudioStreamInspector
        from capicove.processing.audio._adapters.whisper_cpp import WhisperCppTranscriber

        self.model_id = model.name
        self._inspector = FfprobeAudioStreamInspector(ffprobe, restrict_inputs=True)
        self._transcriber = WhisperCppTranscriber(
            model,
            vad_model_path=vad_model,
            executable=whisper_cli,
            vad_executable=whisper_vad_cli,
            ffmpeg_executable=ffmpeg,
            backend_version=backend_version,
            restrict_inputs=True,
            max_duration_seconds=max_duration_seconds,
            expected_model_sha256=expected_model_sha256,
            expected_vad_sha256=expected_vad_sha256,
        )

    def transcribe(self, source: Path, *, language: str, deadline: datetime) -> AudioTranscript:
        """Require one stream and enforce limits on decoded samples, not metadata."""
        if self._inspector.inspect(source, deadline=deadline).audio_stream_count != 1:
            raise ValueError("exactly_one_audio_stream_required")
        return self._transcriber.transcribe(source, language=language, deadline=deadline)

    def close(self) -> None:
        """Whisper subprocesses are owned and reaped by each bounded invocation."""
