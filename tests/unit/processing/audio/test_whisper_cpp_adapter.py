from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from capicove.processing.audio import (
    AudioFailureCode,
    AudioTranscriptionError,
    AudioWarningCode,
)
from capicove.processing.audio._adapters.whisper_cpp import (
    WhisperCppTranscriber,
    _parse_vad_segments,
)


def _adapter(
    tmp_path: Path,
    *,
    segment_language_detection: bool = False,
) -> WhisperCppTranscriber:
    model = tmp_path / "ggml-tiny.bin"
    model.write_bytes(b"model")
    vad_model = tmp_path / "ggml-silero-v6.2.0.bin"
    vad_model.write_bytes(b"vad-model")
    return WhisperCppTranscriber(
        model_path=model,
        vad_model_path=vad_model,
        executable="trusted-whisper-cli",
        vad_executable="trusted-whisper-vad-speech-segments",
        ffmpeg_executable="trusted-ffmpeg",
        backend_version="1.7.6",
        timeout_seconds=30.0,
        segment_language_detection=segment_language_detection,
    )


def _write_result(arguments: list[str], payload: dict[str, object]) -> None:
    output_prefix = Path(arguments[arguments.index("--output-file") + 1])
    output_prefix.with_suffix(".json").write_text(json.dumps(payload), encoding="utf-8")


def _write_wav(path: Path, *, frames: int = 16000) -> None:
    import wave

    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x00\x00" * frames)


def test_adapter_uses_fixed_offline_cpu_commands_and_parses_ordered_segments(
    tmp_path: Path,
) -> None:
    source = tmp_path / "evidence-blob"
    source.write_bytes(b"audio")
    calls: list[list[str]] = []

    def run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(arguments)
        if arguments[0] == "trusted-ffmpeg":
            _write_wav(Path(arguments[-1]), frames=48000)
        else:
            _write_result(
                arguments,
                {
                    "result": {"language": "zh"},
                    "transcription": [
                        {"offsets": {"from": 0, "to": 1200}, "text": " 你好"},
                        {"offsets": {"from": 1200, "to": 2600}, "text": " world"},
                    ],
                },
            )
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    with patch("subprocess.run", side_effect=run):
        transcript = _adapter(tmp_path).transcribe(
            source,
            audio_stream_index=1,
            language="auto",
            deadline=datetime.now(UTC) + timedelta(minutes=1),
        )

    assert calls[0][0:8] == [
        "trusted-ffmpeg",
        "-v",
        "error",
        "-xerror",
        "-nostdin",
        "-i",
        str(source),
        "-map",
    ]
    assert calls[0][8] == "0:a:1"
    assert calls[1][0] == "trusted-whisper-cli"
    assert "--no-gpu" in calls[1]
    assert "--output-json" in calls[1]
    assert "--no-prints" in calls[1]
    assert calls[1][calls[1].index("--vad-model") + 1].endswith("ggml-silero-v6.2.0.bin")
    assert "--vad" in calls[1]
    assert transcript.text == "你好 world"
    assert [segment.start_seconds for segment in transcript.segments] == [0.0, 1.2]
    assert transcript.lineage.model_sha256
    assert transcript.lineage.vad_model == "ggml-silero-v6.2.0.bin"
    assert transcript.lineage.vad_model_sha256
    assert transcript.lineage.device == "cpu"


def test_adapter_maps_locale_language_to_backend_language(tmp_path: Path) -> None:
    source = tmp_path / "evidence-blob"
    source.write_bytes(b"audio")
    calls: list[list[str]] = []

    def run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(arguments)
        if arguments[0] == "trusted-ffmpeg":
            _write_wav(Path(arguments[-1]))
        else:
            _write_result(
                arguments,
                {
                    "result": {"language": "zh"},
                    "transcription": [{"offsets": {"from": 0, "to": 1000}, "text": " 你好"}],
                },
            )
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    with patch("subprocess.run", side_effect=run):
        _adapter(tmp_path).transcribe(
            source,
            language="zh-CN",
            deadline=datetime.now(UTC) + timedelta(minutes=1),
        )

    whisper_call = calls[1]
    assert whisper_call[whisper_call.index("--language") + 1] == "zh"


def test_adapter_rejects_missing_model_without_running_subprocess(tmp_path: Path) -> None:
    with (
        patch("subprocess.run") as run,
        pytest.raises(AudioTranscriptionError, match="model is unavailable") as captured,
    ):
        WhisperCppTranscriber(
            model_path=tmp_path / "missing.bin",
            vad_model_path=tmp_path / "missing-vad.bin",
        ).transcribe(
            tmp_path / "audio",
            language="auto",
            deadline=datetime.now(UTC) + timedelta(minutes=1),
        )

    assert captured.value.code is AudioFailureCode.MODEL_UNAVAILABLE
    run.assert_not_called()


def test_adapter_rejects_expired_deadline_before_model_resolution(tmp_path: Path) -> None:
    with (
        patch("pathlib.Path.resolve") as resolve,
        patch("subprocess.run") as run,
        pytest.raises(AudioTranscriptionError, match="deadline") as captured,
    ):
        WhisperCppTranscriber(
            model_path=tmp_path / "missing.bin",
            vad_model_path=tmp_path / "missing-vad.bin",
        ).transcribe(
            tmp_path / "audio",
            language="auto",
            deadline=datetime.now(UTC) - timedelta(seconds=1),
        )

    assert captured.value.code is AudioFailureCode.TIMEOUT
    resolve.assert_not_called()
    run.assert_not_called()


def test_adapter_rejects_missing_vad_model_without_running_subprocess(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    with (
        patch("subprocess.run") as run,
        pytest.raises(AudioTranscriptionError, match="VAD model is unavailable") as captured,
    ):
        WhisperCppTranscriber(
            model_path=model,
            vad_model_path=tmp_path / "missing-vad.bin",
        ).transcribe(
            tmp_path / "audio",
            language="auto",
            deadline=datetime.now(UTC) + timedelta(minutes=1),
        )

    assert captured.value.code is AudioFailureCode.MODEL_UNAVAILABLE
    run.assert_not_called()


def test_adapter_maps_model_hash_race_to_stable_unavailable_error(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    source = tmp_path / "audio"
    source.write_bytes(b"audio")

    with (
        patch(
            "capicove.processing.audio._adapters.whisper_cpp._hash_file",
            side_effect=OSError("private model path"),
        ),
        patch("subprocess.run") as run,
        pytest.raises(AudioTranscriptionError, match="model is unavailable") as captured,
    ):
        adapter.transcribe(
            source,
            language="auto",
            deadline=datetime.now(UTC) + timedelta(minutes=1),
        )

    assert captured.value.code is AudioFailureCode.MODEL_UNAVAILABLE
    assert "private model path" not in str(captured.value)
    run.assert_not_called()


def test_adapter_rechecks_deadline_after_model_hashing(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    source = tmp_path / "audio"
    source.write_bytes(b"audio")

    with (
        patch(
            "capicove.processing.audio._adapters.whisper_cpp._remaining_timeout",
            side_effect=[30.0, AudioTranscriptionError(AudioFailureCode.TIMEOUT, "expired")],
        ) as remaining,
        patch("subprocess.run") as run,
        pytest.raises(AudioTranscriptionError) as captured,
    ):
        adapter.transcribe(
            source,
            language="auto",
            deadline=datetime.now(UTC) + timedelta(minutes=1),
        )

    assert captured.value.code is AudioFailureCode.TIMEOUT
    assert remaining.call_count == 2
    run.assert_not_called()


def test_adapter_rejects_model_changed_during_transcription(tmp_path: Path) -> None:
    model = tmp_path / "ggml-tiny.bin"
    model.write_bytes(b"model")
    adapter = _adapter(tmp_path)
    source = tmp_path / "audio"
    source.write_bytes(b"audio")

    def run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if arguments[0] == "trusted-ffmpeg":
            _write_wav(Path(arguments[-1]))
        else:
            model.write_bytes(b"changed-model")
            _write_result(
                arguments,
                {
                    "result": {"language": "en"},
                    "transcription": [{"offsets": {"from": 0, "to": 1000}, "text": "hello"}],
                },
            )
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    with (
        patch("subprocess.run", side_effect=run),
        pytest.raises(AudioTranscriptionError, match="model changed") as captured,
    ):
        adapter.transcribe(
            source,
            language="en",
            deadline=datetime.now(UTC) + timedelta(minutes=1),
        )

    assert captured.value.code is AudioFailureCode.MODEL_LOAD_FAILED


def test_adapter_maps_missing_executable_timeout_and_invalid_json(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    source = tmp_path / "audio"
    source.write_bytes(b"audio")
    deadline = datetime.now(UTC) + timedelta(minutes=1)

    with (
        patch("subprocess.run", side_effect=FileNotFoundError),
        pytest.raises(AudioTranscriptionError) as missing,
    ):
        adapter.transcribe(source, language="auto", deadline=deadline)
    assert missing.value.code is AudioFailureCode.DECODER_UNAVAILABLE

    subprocess_calls = 0

    def missing_output(
        arguments: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        nonlocal subprocess_calls
        subprocess_calls += 1
        if subprocess_calls == 1:
            _write_wav(Path(arguments[-1]))
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    with (
        patch("subprocess.run", side_effect=missing_output),
        pytest.raises(AudioTranscriptionError) as missing_result,
    ):
        adapter.transcribe(source, language="auto", deadline=deadline)
    assert missing_result.value.code is AudioFailureCode.MODEL_LOAD_FAILED

    with (
        patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(["ffmpeg"], timeout=1),
        ),
        pytest.raises(AudioTranscriptionError) as timeout,
    ):
        adapter.transcribe(source, language="auto", deadline=deadline)
    assert timeout.value.code is AudioFailureCode.TIMEOUT

    call_count = 0

    def invalid_json(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            _write_wav(Path(arguments[-1]))
        else:
            output_prefix = Path(arguments[arguments.index("--output-file") + 1])
            output_prefix.with_suffix(".json").write_text("not-json", encoding="utf-8")
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    with (
        patch("subprocess.run", side_effect=invalid_json),
        pytest.raises(AudioTranscriptionError) as invalid,
    ):
        adapter.transcribe(source, language="auto", deadline=deadline)
    assert invalid.value.code is AudioFailureCode.MODEL_LOAD_FAILED


def test_adapter_rejects_backend_timestamps_that_overlap(tmp_path: Path) -> None:
    source = tmp_path / "audio"
    source.write_bytes(b"audio")
    call_count = 0

    def run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            _write_wav(Path(arguments[-1]), frames=32000)
        else:
            _write_result(
                arguments,
                {
                    "result": {"language": "en"},
                    "transcription": [
                        {"offsets": {"from": 0, "to": 1000}, "text": "one"},
                        {"offsets": {"from": 500, "to": 1500}, "text": "two"},
                    ],
                },
            )
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    with (
        patch("subprocess.run", side_effect=run),
        pytest.raises(AudioTranscriptionError) as captured,
    ):
        _adapter(tmp_path).transcribe(
            source,
            language="auto",
            deadline=datetime.now(UTC) + timedelta(minutes=1),
        )

    assert captured.value.code is AudioFailureCode.MODEL_LOAD_FAILED


def test_adapter_skips_blank_segments_and_bounds_invalid_json_shape(tmp_path: Path) -> None:
    source = tmp_path / "audio"
    source.write_bytes(b"audio")
    adapter = _adapter(tmp_path)
    call_count = 0

    def blank_segment(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            _write_wav(Path(arguments[-1]))
        else:
            _write_result(
                arguments,
                {
                    "result": {"language": "en"},
                    "transcription": [{"offsets": {"from": 0, "to": 1000}, "text": "   "}],
                },
            )
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    with patch("subprocess.run", side_effect=blank_segment):
        transcript = adapter.transcribe(
            source,
            language="auto",
            deadline=datetime.now(UTC) + timedelta(minutes=1),
        )
    assert transcript.text == ""
    assert transcript.segments == ()

    call_count = 0

    def wrong_shape(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            _write_wav(Path(arguments[-1]))
        else:
            _write_result(arguments, {"unexpected": "private output"})
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    with (
        patch("subprocess.run", side_effect=wrong_shape),
        pytest.raises(AudioTranscriptionError, match="invalid result") as captured,
    ):
        adapter.transcribe(
            source,
            language="auto",
            deadline=datetime.now(UTC) + timedelta(minutes=1),
        )
    assert captured.value.code is AudioFailureCode.MODEL_LOAD_FAILED


def test_adapter_repairs_encoding_clamps_end_and_reports_warnings(tmp_path: Path) -> None:
    source = tmp_path / "audio"
    source.write_bytes(b"audio")
    call_count = 0

    def run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            _write_wav(Path(arguments[-1]), frames=16000)
        else:
            output_prefix = Path(arguments[arguments.index("--output-file") + 1])
            output_prefix.with_suffix(".json").write_bytes(
                b'{"result":{"language":"en"},"transcription":['
                b'{"offsets":{"from":0,"to":1500},"text":"hello\xff"}]}'
            )
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    with patch("subprocess.run", side_effect=run):
        transcript = _adapter(tmp_path).transcribe(
            source,
            language="auto",
            deadline=datetime.now(UTC) + timedelta(minutes=1),
        )

    assert transcript.segments[0].end_seconds == 1.0
    assert transcript.text == "hello�"
    assert transcript.warnings == (
        AudioWarningCode.TRANSCRIPT_ENCODING_REPAIRED,
        AudioWarningCode.TIMESTAMP_CLAMPED,
    )


def test_adapter_silently_clamps_backend_rounding_within_fifty_milliseconds(
    tmp_path: Path,
) -> None:
    source = tmp_path / "audio"
    source.write_bytes(b"audio")
    call_count = 0

    def run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            _write_wav(Path(arguments[-1]), frames=16000)
        else:
            _write_result(
                arguments,
                {
                    "result": {"language": "en"},
                    "transcription": [{"offsets": {"from": 0, "to": 1030}, "text": "hello"}],
                },
            )
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    with patch("subprocess.run", side_effect=run):
        transcript = _adapter(tmp_path).transcribe(
            source,
            language="auto",
            deadline=datetime.now(UTC) + timedelta(minutes=1),
        )

    assert transcript.segments[0].end_seconds == 1.0
    assert transcript.warnings == ()


def test_adapter_maps_blank_audio_marker_to_no_speech(tmp_path: Path) -> None:
    source = tmp_path / "audio"
    source.write_bytes(b"audio")
    call_count = 0

    def run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            _write_wav(Path(arguments[-1]))
        else:
            _write_result(
                arguments,
                {
                    "result": {"language": "en"},
                    "transcription": [
                        {"offsets": {"from": 0, "to": 1000}, "text": "[BLANK_AUDIO]"}
                    ],
                },
            )
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    with patch("subprocess.run", side_effect=run):
        transcript = _adapter(tmp_path).transcribe(
            source,
            language="auto",
            deadline=datetime.now(UTC) + timedelta(minutes=1),
        )

    assert transcript.text == ""
    assert transcript.segments == ()
    assert transcript.warnings == (AudioWarningCode.NO_SPEECH_DETECTED,)


def test_adapter_warns_on_highly_repetitive_potential_hallucination(tmp_path: Path) -> None:
    source = tmp_path / "audio"
    source.write_bytes(b"audio")
    call_count = 0

    def run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            _write_wav(Path(arguments[-1]), frames=480000)
        else:
            _write_result(
                arguments,
                {
                    "result": {"language": "zh"},
                    "transcription": [
                        {
                            "offsets": {"from": 0, "to": 30000},
                            "text": "炒炉炒炉炒炉炒炉炒炉炒炉炒炉炒炉炒炉炒炉炒炉炒炉",
                        }
                    ],
                },
            )
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    with patch("subprocess.run", side_effect=run):
        transcript = _adapter(tmp_path).transcribe(
            source,
            language="auto",
            deadline=datetime.now(UTC) + timedelta(minutes=1),
        )

    assert transcript.text
    assert transcript.warnings == (AudioWarningCode.POTENTIAL_HALLUCINATION,)


def test_adapter_transcribes_vad_segments_independently_and_restores_timeline(
    tmp_path: Path,
) -> None:
    source = tmp_path / "code-switch.wav"
    source.write_bytes(b"audio")
    calls: list[list[str]] = []

    def run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(arguments)
        if arguments[0] == "trusted-ffmpeg":
            _write_wav(Path(arguments[-1]), frames=320000)
            return subprocess.CompletedProcess(arguments, 0, b"", b"")
        if arguments[0] == "trusted-whisper-vad-speech-segments":
            return subprocess.CompletedProcess(
                arguments,
                0,
                (
                    b"Detected 2 speech segments:\n"
                    b"Speech segment 0: start = 3.00, end = 1046.00\n"
                    b"Speech segment 1: start = 1072.00, end = 1504.00\n"
                ),
                b"",
            )
        output_prefix = Path(arguments[arguments.index("--output-file") + 1])
        if output_prefix.name.endswith("000"):
            _write_result(
                arguments,
                {
                    "result": {"language": "zh"},
                    "transcription": [{"offsets": {"from": 0, "to": 6880}, "text": " 中文内容"}],
                },
            )
        else:
            _write_result(
                arguments,
                {
                    "result": {"language": "en"},
                    "transcription": [
                        {"offsets": {"from": 0, "to": 4280}, "text": " English content"}
                    ],
                },
            )
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    with patch("subprocess.run", side_effect=run):
        transcript = _adapter(tmp_path, segment_language_detection=True).transcribe(
            source,
            language="auto",
            deadline=datetime.now(UTC) + timedelta(minutes=1),
        )

    assert transcript.language == "multilingual"
    assert transcript.text == "中文内容 English content"
    assert [(segment.start_seconds, segment.end_seconds) for segment in transcript.segments] == [
        (0.03, 6.91),
        (10.72, 15.0),
    ]
    whisper_calls = [call for call in calls if call[0] == "trusted-whisper-cli"]
    assert len(whisper_calls) == 2


def test_adapter_does_not_duplicate_existing_separator_across_vad_segments(
    tmp_path: Path,
) -> None:
    source = tmp_path / "english.wav"
    source.write_bytes(b"audio")

    def run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if arguments[0] == "trusted-ffmpeg":
            _write_wav(Path(arguments[-1]), frames=64000)
        elif arguments[0] == "trusted-whisper-vad-speech-segments":
            return subprocess.CompletedProcess(
                arguments,
                0,
                (
                    b"Detected 2 speech segments:\n"
                    b"Speech segment 0: start = 0.00, end = 180.00\n"
                    b"Speech segment 1: start = 200.00, end = 380.00\n"
                ),
                b"",
            )
        else:
            output_prefix = Path(arguments[arguments.index("--output-file") + 1])
            transcription = (
                [{"offsets": {"from": 0, "to": 1000}, "text": "first"}]
                if output_prefix.name.endswith("000")
                else [
                    {"offsets": {"from": 0, "to": 800}, "text": " second"},
                    {"offsets": {"from": 800, "to": 1600}, "text": " item"},
                ]
            )
            _write_result(
                arguments,
                {
                    "result": {"language": "en"},
                    "transcription": transcription,
                },
            )
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    with patch("subprocess.run", side_effect=run):
        transcript = _adapter(tmp_path, segment_language_detection=True).transcribe(
            source,
            language="auto",
            deadline=datetime.now(UTC) + timedelta(minutes=1),
        )

    assert transcript.text == "first second item"


def test_vad_segment_parser_accepts_silence_and_rejects_invalid_timeline() -> None:
    assert _parse_vad_segments(b"Detected 0 speech segments:\n") == ()
    with pytest.raises(ValueError, match="ordered"):
        _parse_vad_segments(
            b"Detected 2 speech segments:\n"
            b"Speech segment 0: start = 200.00, end = 300.00\n"
            b"Speech segment 1: start = 100.00, end = 400.00\n"
        )
    with pytest.raises(ValueError, match="valid segment summary"):
        _parse_vad_segments(b"private backend output")
    with pytest.raises(ValueError, match="count"):
        _parse_vad_segments(
            b"Detected 2 speech segments:\nSpeech segment 0: start = 0.00, end = 100.00\n"
        )


def test_vad_segment_beyond_audio_is_clamped_to_original_timeline(tmp_path: Path) -> None:
    source = tmp_path / "audio.wav"
    source.write_bytes(b"audio")

    def run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if arguments[0] == "trusted-ffmpeg":
            _write_wav(Path(arguments[-1]), frames=16000)
        elif arguments[0] == "trusted-whisper-vad-speech-segments":
            return subprocess.CompletedProcess(
                arguments,
                0,
                (b"Detected 1 speech segments:\nSpeech segment 0: start = 0.00, end = 150.00\n"),
                b"",
            )
        else:
            _write_result(
                arguments,
                {
                    "result": {"language": "en"},
                    "transcription": [{"offsets": {"from": 0, "to": 1500}, "text": "hello"}],
                },
            )
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    with patch("subprocess.run", side_effect=run):
        transcript = _adapter(tmp_path, segment_language_detection=True).transcribe(
            source,
            language="auto",
            deadline=datetime.now(UTC) + timedelta(minutes=1),
        )

    assert transcript.segments[0].end_seconds == 1.0
    assert transcript.warnings == (AudioWarningCode.TIMESTAMP_CLAMPED,)


def test_short_spurious_language_segment_does_not_mark_transcript_multilingual(
    tmp_path: Path,
) -> None:
    source = tmp_path / "mandarin.wav"
    source.write_bytes(b"audio")
    calls = 0

    def run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        nonlocal calls
        calls += 1
        if arguments[0] == "trusted-ffmpeg":
            _write_wav(Path(arguments[-1]), frames=160000)
        elif arguments[0] == "trusted-whisper-vad-speech-segments":
            return subprocess.CompletedProcess(
                arguments,
                0,
                (
                    b"Detected 2 speech segments:\n"
                    b"Speech segment 0: start = 0.00, end = 800.00\n"
                    b"Speech segment 1: start = 820.00, end = 880.00\n"
                ),
                b"",
            )
        else:
            language = "zh" if calls == 3 else "ja"
            duration = 8000 if language == "zh" else 600
            _write_result(
                arguments,
                {
                    "result": {"language": language},
                    "transcription": [{"offsets": {"from": 0, "to": duration}, "text": " 中文"}],
                },
            )
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    with patch("subprocess.run", side_effect=run):
        transcript = _adapter(tmp_path, segment_language_detection=True).transcribe(
            source,
            language="auto",
            deadline=datetime.now(UTC) + timedelta(minutes=1),
        )

    assert transcript.language == "zh"
