from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from pytest import CaptureFixture

from capicove.cli.app import _build_parser
from capicove.cli.commands.audio_transcribe import run_capabilities, run_transcribe
from capicove.contracts import (
    InvocationContext,
    ToolError,
    ToolErrorCode,
    ToolResult,
    ToolStatus,
)


class _Handler:
    def __init__(self, result: ToolResult) -> None:
        self.result = result
        self.arguments: Mapping[str, object] | None = None

    def __call__(
        self,
        context: InvocationContext,
        arguments: Mapping[str, object],
        /,
    ) -> ToolResult:
        self.arguments = arguments
        return self.result


def _result() -> ToolResult:
    return ToolResult(
        request_id="request-1",
        trace_id="trace-1",
        tool_id="capicove.audio.transcribe",
        tool_version=2,
        status=ToolStatus.SUCCEEDED,
        result={
            "status": "succeeded",
            "text": "hello",
            "language": "en",
            "segments": [
                {
                    "index": 0,
                    "start_seconds": 0.0,
                    "end_seconds": 1.0,
                    "text": "hello",
                }
            ],
            "warnings": ["UNSELECTED_AUDIO_STREAMS_IGNORED"],
            "evidence": {
                "evidence_id": "sha256:" + "b" * 64,
                "sha256": "b" * 64,
                "size_bytes": 10,
                "media_type": "audio/wav",
                "source_name": "meeting.wav",
            },
            "lineage": {
                "backend": "whisper.cpp",
                "backend_version": "1.7.6",
                "model": "tiny",
                "model_sha256": "a" * 64,
                "vad_model": "silero-v6.2.0",
                "vad_model_sha256": "c" * 64,
                "device": "cpu",
                "precision": "q5_1",
                "audio_stream_selection": {
                    "schema_version": 1,
                    "selected_audio_stream_index": 1,
                    "index_semantics": "zero_based_audio_stream_ordinal",
                    "selection_method": "explicit_index",
                    "preflight_audio_stream_count": 2,
                    "validation_scope": "selected_audio_stream_only",
                },
            },
        },
        evidence_refs=("sha256:" + "b" * 64,),
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )


def test_parser_accepts_audio_transcribe_configuration(tmp_path: Path) -> None:
    namespace = _build_parser().parse_args(
        [
            "audio",
            "transcribe",
            "--input",
            str(tmp_path / "meeting.wav"),
            "--model",
            str(tmp_path / "model.bin"),
            "--vad-model",
            str(tmp_path / "vad-model.bin"),
            "--whisper-vad-cli",
            "trusted-whisper-vad-speech-segments",
            "--ffprobe",
            "trusted-ffprobe",
            "--audio-stream-index",
            "1",
            "--store",
            str(tmp_path / "store"),
            "--language",
            "zh",
            "--output",
            "json",
        ]
    )

    assert namespace.command == "audio"
    assert namespace.audio_command == "transcribe"
    assert namespace.language == "zh"
    assert namespace.whisper_vad_cli == "trusted-whisper-vad-speech-segments"
    assert namespace.ffprobe == "trusted-ffprobe"
    assert namespace.audio_stream_index == 1


def test_parser_rejects_non_positive_or_unbounded_timeout(tmp_path: Path) -> None:
    base_arguments = [
        "audio",
        "transcribe",
        "--input",
        str(tmp_path / "meeting.wav"),
        "--model",
        str(tmp_path / "model.bin"),
        "--vad-model",
        str(tmp_path / "vad-model.bin"),
        "--timeout",
    ]
    for invalid_timeout in ("0", "-1", "301"):
        with pytest.raises(SystemExit):
            _build_parser().parse_args([*base_arguments, invalid_timeout])


def test_parser_rejects_negative_audio_stream_index(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            [
                "audio",
                "transcribe",
                "--input",
                str(tmp_path / "meeting.wav"),
                "--model",
                str(tmp_path / "model.bin"),
                "--vad-model",
                str(tmp_path / "vad-model.bin"),
                "--audio-stream-index",
                "-1",
            ]
        )


def test_transcribe_cli_maps_arguments_and_emits_canonical_json(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    source = tmp_path / "meeting.wav"
    namespace = _build_parser().parse_args(
        [
            "audio",
            "transcribe",
            "--input",
            str(source),
            "--model",
            str(tmp_path / "model.bin"),
            "--vad-model",
            str(tmp_path / "vad-model.bin"),
            "--store",
            str(tmp_path / "store"),
            "--language",
            "auto",
            "--output",
            "json",
        ]
    )
    handler = _Handler(_result())

    assert run_transcribe(namespace, handler=handler) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "succeeded"
    assert payload["result"]["text"] == "hello"
    assert handler.arguments == {
        "source": {"type": "path", "path": str(source)},
        "language": "auto",
    }


def test_transcribe_cli_serializes_explicit_selector_only_when_present(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    source = tmp_path / "meeting.wav"
    namespace = _build_parser().parse_args(
        [
            "audio",
            "transcribe",
            "--input",
            str(source),
            "--model",
            str(tmp_path / "model.bin"),
            "--vad-model",
            str(tmp_path / "vad-model.bin"),
            "--audio-stream-index",
            "1",
            "--output",
            "json",
        ]
    )
    handler = _Handler(_result())

    assert run_transcribe(namespace, handler=handler) == 0
    capsys.readouterr()

    assert handler.arguments == {
        "source": {"type": "path", "path": str(source)},
        "language": "auto",
        "audio_stream_index": 1,
    }


def test_transcribe_human_output_renders_structured_stream_rejection(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    namespace = _build_parser().parse_args(
        [
            "audio",
            "transcribe",
            "--input",
            str(tmp_path / "meeting.wav"),
            "--model",
            str(tmp_path / "model.bin"),
            "--vad-model",
            str(tmp_path / "vad-model.bin"),
        ]
    )
    rejected = ToolResult(
        request_id="request-1",
        trace_id="trace-1",
        tool_id="capicove.audio.transcribe",
        tool_version=2,
        status=ToolStatus.REJECTED,
        result={
            "status": "rejected",
            "failure": {
                "code": "AUDIO_STREAM_SELECTION_REQUIRED",
                "message": "multiple audio streams require audio_stream_index",
                "retryable": False,
                "preflight_audio_stream_count": 2,
            },
        },
        error=ToolError(
            ToolErrorCode.POLICY_REJECTED,
            "multiple audio streams require audio_stream_index",
            retryable=False,
        ),
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )

    assert run_transcribe(namespace, handler=_Handler(rejected)) == 1

    rendered = capsys.readouterr().out
    assert "AUDIO_STREAM_SELECTION_REQUIRED" in rendered
    assert "language: None" not in rendered


def test_capabilities_reports_missing_and_available_model(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    missing = _build_parser().parse_args(
        [
            "audio",
            "capabilities",
            "--model",
            str(tmp_path / "missing.bin"),
            "--vad-model",
            str(tmp_path / "missing-vad.bin"),
            "--ffmpeg",
            "trusted-ffmpeg",
            "--ffprobe",
            "trusted-ffprobe",
            "--output",
            "json",
        ]
    )

    assert run_capabilities(missing, executable_finder=lambda _: None) == 1
    unavailable = json.loads(capsys.readouterr().out)
    assert unavailable["available"] is False
    assert unavailable["model_sha256"] is None
    assert unavailable["vad"]["model_sha256"] is None

    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    vad_model = tmp_path / "vad-model.bin"
    vad_model.write_bytes(b"vad-model")
    available = _build_parser().parse_args(
        [
            "audio",
            "capabilities",
            "--model",
            str(model),
            "--vad-model",
            str(vad_model),
            "--ffmpeg",
            "trusted-ffmpeg",
            "--ffprobe",
            "trusted-ffprobe",
            "--output",
            "json",
        ]
    )
    assert (
        run_capabilities(
            available,
            executable_finder=lambda name: f"/usr/local/bin/{name}",
            version_reader=lambda _: "1.7.6",
        )
        == 0
    )
    discovered = json.loads(capsys.readouterr().out)
    assert discovered["available"] is True
    assert discovered["model_sha256"]
    assert discovered["vad"]["enabled"] is True
    assert discovered["vad"]["model_sha256"]
    assert discovered["backend_version"] == "1.7.6"
    assert discovered["vad"]["executable_available"] is True
    assert discovered["decoder"]["available"] is True
    assert discovered["decoder"]["probe_available"] is True


def test_capabilities_treats_model_hash_race_as_unavailable(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    vad_model = tmp_path / "vad-model.bin"
    vad_model.write_bytes(b"vad")
    arguments = _build_parser().parse_args(
        [
            "audio",
            "capabilities",
            "--model",
            str(model),
            "--vad-model",
            str(vad_model),
            "--output",
            "json",
        ]
    )

    with patch(
        "capicove.cli.commands.audio_transcribe._hash_file",
        side_effect=OSError("private model path"),
    ):
        assert run_capabilities(arguments, executable_finder=lambda name: name) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["available"] is False
    assert payload["model_sha256"] is None
