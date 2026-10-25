"""CLI translation for local audio transcription and capability discovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess  # nosec B404
import sys
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

from capicove.contracts import InvocationContext, ToolHandler, ToolResult, ToolStatus
from capicove.processing.audio import AudioProcessor, AudioTranscriptionService
from capicove.processing.audio._adapters.ffmpeg import FfmpegAudioValidator
from capicove.processing.audio._adapters.ffprobe import FfprobeAudioStreamInspector
from capicove.processing.audio._adapters.filesystem import ExplicitAudioPathAuthorizer
from capicove.processing.audio._adapters.whisper_cpp import WhisperCppTranscriber
from capicove.tools.audio_transcribe import (
    AUDIO_TRANSCRIBE_SPEC,
    AudioTranscribeToolHandler,
    audio_transcribe_capabilities,
)
from capicove.traces import TraceRecorder


ExecutableFinder = Callable[[str], str | None]
VersionReader = Callable[[str], str]


def _bounded_timeout(value: str) -> float:
    timeout = float(value)
    if timeout <= 0 or timeout > AUDIO_TRANSCRIBE_SPEC.timeout_seconds:
        raise argparse.ArgumentTypeError(
            f"timeout must be greater than 0 and at most {AUDIO_TRANSCRIBE_SPEC.timeout_seconds:g}"
        )
    return timeout


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "audio stream index must be a non-negative integer"
        ) from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("audio stream index must be a non-negative integer")
    return parsed


def add_parsers(parent: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register `capicove audio transcribe` and capability discovery."""
    from capicove.cli.commands.audio_clip import add_parser as add_clip_parser

    add_clip_parser(parent)
    transcribe = parent.add_parser(
        "transcribe",
        help="Transcribe authorized local audio with an explicit local model.",
    )
    transcribe.add_argument("--input", required=True, help="Local audio file path.")
    transcribe.add_argument(
        "--model", help="Local whisper.cpp model path (required without --runtime-config)."
    )
    transcribe.add_argument(
        "--vad-model",
        help="Local Silero VAD model path (required without --runtime-config).",
    )
    transcribe.add_argument(
        "--store",
        help="Local immutable audio evidence store (default: .capicove/audio-evidence).",
    )
    transcribe.add_argument(
        "--language",
        choices=("auto", "zh", "en", "zh-CN", "en-US", "en-GB"),
        default="auto",
    )
    transcribe.add_argument("--whisper-cli", help="Default: whisper-cli.")
    transcribe.add_argument("--whisper-vad-cli", help="Default: whisper-vad-speech-segments.")
    transcribe.add_argument("--ffmpeg", help="Default: ffmpeg.")
    transcribe.add_argument("--ffprobe", help="Default: ffprobe.")
    transcribe.add_argument("--audio-stream-index", type=_non_negative_int)
    transcribe.add_argument("--timeout", type=_bounded_timeout, default=300.0)
    transcribe.add_argument("--output", choices=("human", "json"), default="human")
    transcribe.add_argument(
        "--runtime-config",
        type=Path,
        help="Use the version-2 workspace host's audio authority (RFC-0022).",
    )

    importer = parent.add_parser(
        "import", help="Preserve one audio file as evidence through the workspace host."
    )
    importer.add_argument("--runtime-config", type=Path, required=True)
    importer.add_argument("--input", required=True, help="Audio file below a read root.")
    importer.add_argument("--audio-stream-index", type=_non_negative_int)

    capabilities = parent.add_parser(
        "capabilities",
        help="Report the actual local audio transcription deployment.",
    )
    capabilities.add_argument("--component", choices=("transcribe", "clip"), default="transcribe")
    capabilities.add_argument("--model")
    capabilities.add_argument("--vad-model")
    capabilities.add_argument("--whisper-cli", default="whisper-cli")
    capabilities.add_argument(
        "--whisper-vad-cli",
        default="whisper-vad-speech-segments",
    )
    capabilities.add_argument("--ffmpeg", default="ffmpeg")
    capabilities.add_argument("--ffprobe", default="ffprobe")
    capabilities.add_argument("--output", choices=("human", "json"), default="human")


def run_transcribe(
    arguments: argparse.Namespace,
    *,
    handler: ToolHandler | None = None,
    recorder: TraceRecorder | None = None,
) -> int:
    """Compose local adapters, invoke the Tool, and render one canonical result."""
    if getattr(arguments, "runtime_config", None) is not None:
        return _run_configured_transcribe(arguments)
    if arguments.model is None or arguments.vad_model is None:
        print("--model and --vad-model are required without --runtime-config", file=sys.stderr)
        return 2
    for name, default in _LEGACY_DEFAULTS.items():
        if getattr(arguments, name) is None:
            setattr(arguments, name, default)
    source = Path(arguments.input).expanduser()
    if handler is None:
        resolved_source = source.resolve()
        store_root = Path(arguments.store).expanduser().resolve()
        transcriber = WhisperCppTranscriber(
            Path(arguments.model),
            vad_model_path=Path(arguments.vad_model),
            executable=arguments.whisper_cli,
            vad_executable=arguments.whisper_vad_cli,
            ffmpeg_executable=arguments.ffmpeg,
            backend_version=_read_version(arguments.whisper_cli),
            timeout_seconds=arguments.timeout,
        )
        handler = AudioTranscribeToolHandler(
            AudioTranscriptionService(
                AudioProcessor(
                    FfprobeAudioStreamInspector(arguments.ffprobe, arguments.timeout),
                    FfmpegAudioValidator(arguments.ffmpeg, arguments.timeout),
                ),
                transcriber,
            ),
            ExplicitAudioPathAuthorizer({"cli-local": (resolved_source.parent,)}),
            execution_policy_ref="cli-local",
            store_root=store_root,
        )
    tool_arguments: dict[str, object] = {
        "source": {"type": "path", "path": str(source)},
        "language": arguments.language,
    }
    if arguments.audio_stream_index is not None:
        tool_arguments["audio_stream_index"] = arguments.audio_stream_index
    context = InvocationContext(
        request_id=str(uuid.uuid4()),
        trace_id=str(uuid.uuid4()),
        workspace_id="cli-local",
        actor_id="cli-user",
        environment_id="cli-local",
        deadline=datetime.now(UTC) + timedelta(seconds=arguments.timeout),
        permissions=AUDIO_TRANSCRIBE_SPEC.required_permissions,
        privacy_class=AUDIO_TRANSCRIBE_SPEC.privacy_class,
    )
    from capicove.cli.tracing import invoke_cli_tool

    result = invoke_cli_tool(
        recorder=recorder,
        spec=AUDIO_TRANSCRIBE_SPEC,
        handler=handler,
        context=context,
        arguments=tool_arguments,
    )
    payload = encode_audio_result(result)
    if arguments.output == "json":
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        _render_human(payload)
    return 0 if result.status is ToolStatus.SUCCEEDED else 1


def run_capabilities(
    arguments: argparse.Namespace,
    *,
    executable_finder: ExecutableFinder = shutil.which,
    version_reader: VersionReader | None = None,
) -> int:
    """Inspect one explicit local deployment without downloading anything."""
    if getattr(arguments, "component", "transcribe") == "clip":
        from capicove.cli.commands.audio_clip import clip_capabilities

        report = clip_capabilities(arguments.ffmpeg, arguments.ffprobe)
        print(
            json.dumps(report, sort_keys=True)
            if arguments.output == "json"
            else f"available: {str(report['available']).lower()}"
        )
        return 0 if report["available"] else 1
    if arguments.model is None or arguments.vad_model is None:
        print(
            "--model and --vad-model are required for transcription capabilities", file=sys.stderr
        )
        return 2
    executable = executable_finder(arguments.whisper_cli)
    vad_executable = executable_finder(arguments.whisper_vad_cli)
    decoder = executable_finder(arguments.ffmpeg)
    probe = executable_finder(arguments.ffprobe)
    model = Path(arguments.model).expanduser()
    model_available = model.is_file()
    model_sha256 = _try_hash_file(model) if model_available else None
    model_available = model_available and model_sha256 is not None
    vad_model = Path(arguments.vad_model).expanduser()
    vad_model_available = vad_model.is_file()
    vad_model_sha256 = _try_hash_file(vad_model) if vad_model_available else None
    vad_model_available = vad_model_available and vad_model_sha256 is not None
    reader = version_reader or _read_version
    capabilities = audio_transcribe_capabilities(
        executable_available=executable is not None,
        vad_executable_available=vad_executable is not None,
        decoder_available=decoder is not None,
        probe_available=probe is not None,
        model_available=model_available,
        model_sha256=model_sha256,
        vad_model_available=vad_model_available,
        vad_model_sha256=vad_model_sha256,
        backend_version=reader(executable) if executable is not None else "unavailable",
    )
    if arguments.output == "json":
        print(json.dumps(capabilities, ensure_ascii=False, sort_keys=True))
    else:
        print(f"available: {str(capabilities['available']).lower()}")
        print(f"backend: {capabilities['backend']} {capabilities['backend_version']}")
    return 0 if capabilities["available"] else 1


def encode_audio_result(result: ToolResult) -> dict[str, object]:
    """Encode a canonical ToolResult for CLI and MCP parity."""
    payload: dict[str, object] = {
        "request_id": result.request_id,
        "trace_id": result.trace_id,
        "tool_id": result.tool_id,
        "tool_version": result.tool_version,
        "status": result.status.value,
        "result": dict(result.result) if result.result is not None else None,
        "evidence_refs": list(result.evidence_refs),
        "error": None,
    }
    if result.error is not None:
        payload["error"] = {
            "code": result.error.code.value,
            "message": result.error.message,
            "retryable": result.error.retryable,
        }
    return payload


def _render_human(payload: Mapping[str, object]) -> None:
    result = payload.get("result")
    if payload.get("status") != ToolStatus.SUCCEEDED.value:
        failure = result.get("failure") if isinstance(result, Mapping) else None
        if isinstance(failure, Mapping):
            print(f"audio transcription {payload.get('status')}: {failure.get('code')}")
            print(f"message: {failure.get('message')}")
        else:
            print(f"audio transcription failed: {payload.get('error')}")
        return
    if not isinstance(result, Mapping):
        print(f"audio transcription failed: {payload.get('error')}")
        return
    print(f"status: {payload['status']}")
    print(f"language: {result.get('language')}")
    print(f"text: {result.get('text')}")
    for warning in result.get("warnings", []):
        print(f"warning: {warning}")


_LEGACY_DEFAULTS = {
    "store": ".capicove/audio-evidence",
    "whisper_cli": "whisper-cli",
    "whisper_vad_cli": "whisper-vad-speech-segments",
    "ffmpeg": "ffmpeg",
    "ffprobe": "ffprobe",
}
_CONFIGURED_CONFLICTS = {
    "model": "--model",
    "vad_model": "--vad-model",
    "store": "--store",
    "whisper_cli": "--whisper-cli",
    "whisper_vad_cli": "--whisper-vad-cli",
    "ffmpeg": "--ffmpeg",
    "ffprobe": "--ffprobe",
}


def _run_configured_transcribe(arguments: argparse.Namespace) -> int:
    from capicove.cli.configured import invoke_configured

    conflicts = [
        flag for name, flag in _CONFIGURED_CONFLICTS.items() if getattr(arguments, name) is not None
    ]
    if conflicts:
        print(
            f"{', '.join(conflicts)} cannot be combined with --runtime-config; "
            "the host config owns models, executables, and the evidence store",
            file=sys.stderr,
        )
        return 2
    tool_arguments: dict[str, object] = {
        "source": {"type": "path", "path": str(Path(arguments.input).absolute())},
        "language": arguments.language,
    }
    if arguments.audio_stream_index is not None:
        tool_arguments["audio_stream_index"] = arguments.audio_stream_index
    return _render_configured(
        arguments, invoke_configured, AUDIO_TRANSCRIBE_SPEC.tool_id, tool_arguments
    )


def run_import(arguments: argparse.Namespace) -> int:
    """Run `capicove audio import` through the host-selected import Tool."""
    from capicove.cli.configured import invoke_configured

    tool_arguments: dict[str, object] = {
        "source": {"type": "path", "path": str(Path(arguments.input).absolute())}
    }
    if arguments.audio_stream_index is not None:
        tool_arguments["audio_stream_index"] = arguments.audio_stream_index
    arguments.output = "json"
    return _render_configured(arguments, invoke_configured, "capicove.audio.import", tool_arguments)


def _render_configured(
    arguments: argparse.Namespace,
    invoke: Callable[..., dict[str, object]],
    tool_id: str,
    tool_arguments: dict[str, object],
) -> int:
    try:
        payload = invoke(arguments.runtime_config, tool_id, tool_arguments)
    except (OSError, ValueError, ImportError) as error:
        print(f"Runtime startup failed: {error}", file=sys.stderr)
        return 2
    if arguments.output == "json":
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        result = payload.get("result")
        text = result.get("text") if isinstance(result, Mapping) else None
        print(text if isinstance(text, str) else json.dumps(payload.get("error")))
    return 0 if payload["status"] == "succeeded" else 1


def _read_version(executable: str) -> str:
    try:
        completed = subprocess.run(  # nosec B603
            [executable, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            start_new_session=True,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    output = (completed.stdout or completed.stderr).strip().splitlines()
    return output[0][:128] if output else "unknown"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _try_hash_file(path: Path) -> str | None:
    try:
        return _hash_file(path)
    except OSError:
        return None
