import json
from datetime import datetime
from pathlib import Path

import pytest
from pytest import CaptureFixture

from capicove.cli.app import _build_parser
from capicove.cli.commands.audio_transcribe import run_transcribe
from capicove.mcp.audio_transcribe import register_audio_transcribe_mcp
from capicove.processing.audio import (
    AudioBackendLineage,
    AudioEvidence,
    AudioFailureCode,
    AudioImportError,
    AudioStreamSelection,
    AudioTranscript,
    AudioTranscriptionResult,
    AudioTranscriptionSegment,
    AudioWarningCode,
)
from capicove.tools.audio_transcribe import AudioTranscribeToolHandler


pytestmark = pytest.mark.e2e
_OMITTED = object()


class _Authorizer:
    def authorize_read(self, policy_ref: str, path: Path) -> Path:
        return path.resolve()


class _TransportService:
    def __init__(self) -> None:
        self.selectors: list[object] = []

    def transcribe_file(
        self,
        source_path: Path,
        store_root: Path,
        *,
        language: str,
        deadline: datetime,
        audio_stream_index: object = _OMITTED,
    ) -> AudioTranscriptionResult:
        self.selectors.append(audio_stream_index)
        if audio_stream_index is _OMITTED:
            raise AudioImportError(
                AudioFailureCode.AUDIO_STREAM_SELECTION_REQUIRED,
                "multiple audio streams require audio_stream_index",
                preflight_audio_stream_count=2,
            )
        if audio_stream_index != 1:
            raise AudioImportError(
                AudioFailureCode.AUDIO_STREAM_INDEX_OUT_OF_RANGE,
                "audio_stream_index is outside the available audio stream range",
                preflight_audio_stream_count=2,
            )
        segment = AudioTranscriptionSegment(0, 0.0, 1.0, "你好 world")
        return AudioTranscriptionResult(
            AudioEvidence(
                "sha256:" + "b" * 64,
                "b" * 64,
                10,
                "audio/wav",
                "meeting.wav",
                Path("/store/blob"),
            ),
            AudioTranscript(
                "你好 world",
                "zh",
                (segment,),
                (AudioWarningCode.UNSELECTED_AUDIO_STREAMS_IGNORED,),
                AudioBackendLineage(
                    "whisper.cpp",
                    "1.7.6",
                    "tiny",
                    "a" * 64,
                    "cpu",
                    "q5_1",
                    "silero-v6.2.0",
                    "c" * 64,
                ),
                1.0,
            ),
            AudioStreamSelection(1, "explicit_index", 2),
        )


class _Server:
    def tool(self, **kwargs: object):  # type: ignore[no-untyped-def]
        def decorator(callback):  # type: ignore[no-untyped-def]
            return callback

        return decorator


def _semantic_payload(payload: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in payload.items() if key not in {"request_id", "trace_id"}}


@pytest.mark.parametrize(
    ("selector", "expected_status", "expected_code"),
    [
        (1, "succeeded", None),
        (_OMITTED, "rejected", "AUDIO_STREAM_SELECTION_REQUIRED"),
        (3, "failed", "AUDIO_STREAM_INDEX_OUT_OF_RANGE"),
    ],
)
def test_cli_and_mcp_preserve_selector_dependent_audio_outcomes(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    selector: object,
    expected_status: str,
    expected_code: str | None,
) -> None:
    service = _TransportService()
    handler = AudioTranscribeToolHandler(
        service,
        _Authorizer(),
        execution_policy_ref="test",
        store_root=Path("/store"),
    )
    cli_arguments = [
        "audio",
        "transcribe",
        "--input",
        str(tmp_path / "meeting.wav"),
        "--model",
        str(tmp_path / "model.bin"),
        "--vad-model",
        str(tmp_path / "vad-model.bin"),
        "--output",
        "json",
    ]
    mcp_arguments: dict[str, object] = {
        "source": {"type": "path", "path": str(tmp_path / "meeting.wav")},
        "language": "auto",
    }
    if selector is not _OMITTED:
        cli_arguments.extend(["--audio-stream-index", str(selector)])
        mcp_arguments["audio_stream_index"] = selector

    namespace = _build_parser().parse_args(cli_arguments)
    run_transcribe(namespace, handler=handler)
    cli = json.loads(capsys.readouterr().out)

    callback = register_audio_transcribe_mcp(_Server(), handler)
    mcp = dict(callback(mcp_arguments))

    assert _semantic_payload(cli) == _semantic_payload(mcp)
    assert cli["status"] == expected_status
    result = cli["result"]
    assert isinstance(result, dict)
    if expected_code is None:
        lineage = result["lineage"]
        assert isinstance(lineage, dict)
        selection = lineage["audio_stream_selection"]
        assert isinstance(selection, dict)
        assert selection["selected_audio_stream_index"] == 1
    else:
        failure = result["failure"]
        assert isinstance(failure, dict)
        assert failure["code"] == expected_code
    assert service.selectors == [selector, selector]
