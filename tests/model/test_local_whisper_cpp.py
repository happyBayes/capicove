"""Opt-in native ASR acceptance using operator-provisioned, hash-pinned files.

CAPICOVE_WHISPER_QUALIFICATION selects a local JSON file. Its ``files`` mapping
contains model, vad_model, speech, whisper_cli, whisper_vad_cli, ffmpeg and ffprobe;
each entry has an absolute path and sha256. No test downloads assets.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import wave
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from capicove.processing.audio import AudioFailureCode, AudioTranscriptionError, AudioWarningCode
from capicove.processing.audio._adapters.whisper_cpp import WhisperCppTranscriber


pytestmark = pytest.mark.model_small


@pytest.fixture(scope="module")
def configuration() -> dict[str, Any]:
    selected = os.environ.get("CAPICOVE_WHISPER_QUALIFICATION")
    if not selected:
        pytest.skip("Set CAPICOVE_WHISPER_QUALIFICATION to qualify native local ASR")
    config: dict[str, Any] = json.loads(Path(selected).read_text(encoding="utf-8"))
    for name in (
        "model",
        "vad_model",
        "speech",
        "whisper_cli",
        "whisper_vad_cli",
        "ffmpeg",
        "ffprobe",
    ):
        record = config["files"][name]
        path = Path(record["path"])
        assert path.is_absolute() and path.is_file(), name
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == record["sha256"], name
    return config


def _path(configuration: dict[str, Any], name: str) -> str:
    return str(configuration["files"][name]["path"])


def _adapter(configuration: dict[str, Any]) -> WhisperCppTranscriber:
    return WhisperCppTranscriber(
        Path(_path(configuration, "model")),
        vad_model_path=Path(_path(configuration, "vad_model")),
        executable=_path(configuration, "whisper_cli"),
        vad_executable=_path(configuration, "whisper_vad_cli"),
        ffmpeg_executable=_path(configuration, "ffmpeg"),
        backend_version=configuration["backend_version"],
    )


@pytest.mark.parametrize("language", ["en", "auto"])
def test_real_speech_timestamps_and_lineage(configuration: dict[str, Any], language: str) -> None:
    transcript = _adapter(configuration).transcribe(
        Path(_path(configuration, "speech")),
        language=language,
        deadline=datetime.now(UTC) + timedelta(seconds=180),
    )
    assert transcript.language == "en"
    assert transcript.segments
    assert transcript.input_duration_seconds is not None
    for keyword in configuration["speech_keywords"]:
        assert keyword.lower() in transcript.text.lower()
    previous_end = 0.0
    for segment in transcript.segments:
        assert previous_end <= segment.start_seconds < segment.end_seconds
        assert segment.end_seconds <= transcript.input_duration_seconds
        previous_end = segment.end_seconds
    assert transcript.lineage.model_sha256 == configuration["files"]["model"]["sha256"]
    assert transcript.lineage.vad_model_sha256 == configuration["files"]["vad_model"]["sha256"]


@pytest.fixture
def silence(tmp_path: Path) -> Path:
    path = tmp_path / "silence.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 16000 * 3)
    return path


@pytest.mark.parametrize("language", ["en", "auto"])
def test_real_vad_rejects_silence(
    configuration: dict[str, Any], silence: Path, language: str
) -> None:
    transcript = _adapter(configuration).transcribe(
        silence, language=language, deadline=datetime.now(UTC) + timedelta(seconds=60)
    )
    assert not transcript.text and not transcript.segments
    if language == "auto":
        assert transcript.language == "und"
    assert AudioWarningCode.NO_SPEECH_DETECTED in transcript.warnings


@pytest.mark.parametrize("asset", ["model", "vad_model"])
def test_real_runtime_rejects_corrupt_weights(
    configuration: dict[str, Any], tmp_path: Path, asset: str
) -> None:
    corrupted = tmp_path / f"truncated-{asset}.bin"
    with Path(_path(configuration, asset)).open("rb") as stream:
        corrupted.write_bytes(stream.read(256))
    invalid = {
        **configuration,
        "files": {**configuration["files"], asset: {"path": str(corrupted)}},
    }
    with pytest.raises(AudioTranscriptionError) as error:
        _adapter(invalid).transcribe(
            Path(_path(configuration, "speech")),
            language="auto",
            deadline=datetime.now(UTC) + timedelta(seconds=60),
        )
    assert error.value.code is AudioFailureCode.MODEL_LOAD_FAILED
    assert str(corrupted) not in str(error.value)


def test_real_cli_stream_selection_and_ambiguity(
    configuration: dict[str, Any], silence: Path, tmp_path: Path
) -> None:
    source = tmp_path / "two-streams.m4a"
    subprocess.run(
        [
            _path(configuration, "ffmpeg"),
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(silence),
            "-i",
            _path(configuration, "speech"),
            "-map",
            "0:a:0",
            "-map",
            "1:a:0",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    command = [sys.executable, "-m", "capicove", "audio", "transcribe"]
    for flag, name in (
        ("--model", "model"),
        ("--vad-model", "vad_model"),
        ("--whisper-cli", "whisper_cli"),
        ("--whisper-vad-cli", "whisper_vad_cli"),
        ("--ffmpeg", "ffmpeg"),
        ("--ffprobe", "ffprobe"),
    ):
        command.extend((flag, _path(configuration, name)))
    command.extend(
        (
            "--input",
            str(source),
            "--store",
            str(tmp_path / "evidence"),
            "--language",
            "auto",
            "--output",
            "json",
        )
    )
    ambiguous = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert ambiguous.returncode != 0
    failure = json.loads(ambiguous.stdout)["result"]["failure"]
    assert failure["code"] == AudioFailureCode.AUDIO_STREAM_SELECTION_REQUIRED.value
    out_of_range = subprocess.run(
        [*command, "--audio-stream-index", "2"], capture_output=True, text=True, timeout=60
    )
    assert out_of_range.returncode != 0
    assert json.loads(out_of_range.stdout)["result"]["failure"]["code"] == (
        AudioFailureCode.AUDIO_STREAM_INDEX_OUT_OF_RANGE.value
    )
    silent_stream = subprocess.run(
        [*command, "--audio-stream-index", "0"], capture_output=True, text=True, timeout=60
    )
    assert silent_stream.returncode == 0, silent_stream.stdout + silent_stream.stderr
    silent_payload = json.loads(silent_stream.stdout)
    assert silent_payload["status"] == "succeeded"
    assert silent_payload["result"]["text"] == ""
    assert AudioWarningCode.NO_SPEECH_DETECTED.value in silent_payload["result"]["warnings"]
    assert (
        silent_payload["result"]["lineage"]["audio_stream_selection"]["selected_audio_stream_index"]
        == 0
    )
    selected = subprocess.run(
        [*command, "--audio-stream-index", "1"], capture_output=True, text=True, timeout=240
    )
    assert selected.returncode == 0, selected.stdout + selected.stderr
    payload = json.loads(selected.stdout)
    assert payload["status"] == "succeeded"
    for keyword in configuration["speech_keywords"]:
        assert keyword.lower() in payload["result"]["text"].lower()
    selection = payload["result"]["lineage"]["audio_stream_selection"]
    assert selection["selected_audio_stream_index"] == 1
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
