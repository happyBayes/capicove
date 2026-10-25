"""Compose audio import and transcription for the version-2 workspace host (RFC-0022).

Startup validates configuration, executable files, and pinned model hashes only.
Explicit calls stage the source (and, for transcription, the models) through the
shared handoff (ADR-0026) before any FFmpeg, FFprobe, or Whisper process runs.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess  # nosec B404
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

from capicove.contracts import (
    InvocationContext,
    ToolError,
    ToolErrorCode,
    ToolHandler,
    ToolResult,
    ToolSpec,
    ToolStatus,
)
from capicove.mcp.bindings import McpToolBinding
from capicove.runtime.executables import ExecutableChanged, PinnedExecutables
from capicove.runtime.mcp_config import WorkspaceHostConfiguration
from capicove.runtime.media_handoff import (
    AssetIntegrityError,
    MediaChanged,
    MediaDenied,
    MediaIntake,
    MediaInvalid,
    StagedFile,
    is_audio_container,
    restore_originals,
    verify_asset,
)


IMPORT = "capicove.audio.import"
TRANSCRIBE = "capicove.audio.transcribe"
CLIP = "capicove.audio.clip"
MAX_INPUT_BYTES = 64 * 1024**3
MAX_DURATION_SECONDS = 14_400
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FIELDS = frozenset(
    {
        "schema_version",
        "read_roots",
        "staging_root",
        "evidence_root",
        "ffmpeg",
        "ffprobe",
        "whisper_cli",
        "whisper_vad_cli",
        "model",
        "vad_model",
        "max_input_bytes",
        "max_duration_seconds",
    }
)
_TRANSCRIPTION_FIELDS = frozenset({"whisper_cli", "whisper_vad_cli", "model", "vad_model"})
_POLICY = "workspace-audio"


class AudioDependencyError(ImportError):
    """A selected Tool needs an executable that is not available."""


@dataclass(frozen=True, slots=True)
class PinnedAsset:
    """One configured model and its pinned SHA-256."""

    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class AudioSettings:
    """Validated startup authority for the audio capability."""

    read_roots: tuple[Path, ...]
    staging_root: Path
    evidence_root: Path
    ffmpeg: Path
    ffprobe: Path
    whisper_cli: Path | None
    whisper_vad_cli: Path | None
    model: PinnedAsset | None
    vad_model: PinnedAsset | None
    max_input_bytes: int
    max_duration_seconds: int
    output_root: Path | None = None


def audio_settings(config: WorkspaceHostConfiguration) -> AudioSettings:
    """Validate `capabilities.audio` and check the selected Tools' assets."""
    if os.name != "posix":
        raise ValueError("audio host Tools require POSIX no-follow file APIs in this release")
    raw = config.capabilities.get("audio")
    if raw is None:
        raise ValueError("capabilities.audio is not configured")
    transcription = TRANSCRIBE in config.enabled_tools
    required = _FIELDS if transcription else _FIELDS - _TRANSCRIPTION_FIELDS
    version = raw.get("schema_version")
    if type(version) is not int or version not in (1, 2):
        raise ValueError("capabilities.audio schema_version must be 1 or 2")
    allowed = _FIELDS | ({"output_root"} if version == 2 else set())
    if CLIP in config.enabled_tools:
        if version != 2:
            raise ValueError("audio clipping requires capabilities.audio schema_version 2")
        required = required | {"output_root"}
    if not required <= set(raw) <= allowed:
        raise ValueError("capabilities.audio requires exactly the documented fields")
    workspace = config.workspace_root
    roots = raw["read_roots"]
    if not isinstance(roots, tuple) or not 1 <= len(roots) <= 16 or len(set(roots)) != len(roots):
        raise ValueError("audio read_roots must contain 1 to 16 unique paths")
    read_roots = tuple(_existing_directory(value, "audio read_roots") for value in roots)
    staging = _private_child(raw["staging_root"], workspace, "audio staging_root")
    evidence = _private_child(raw["evidence_root"], workspace, "audio evidence_root")
    output = (
        _private_child(raw["output_root"], workspace, "audio output_root")
        if "output_root" in raw
        else None
    )
    for root in read_roots:
        for owned in (staging, evidence, *([output] if output else [])):
            if owned.is_relative_to(root) or root.is_relative_to(owned):
                raise ValueError(
                    "audio staging and evidence roots must be disjoint from read_roots"
                )
    if evidence.is_relative_to(staging) or staging.is_relative_to(evidence):
        raise ValueError("audio staging_root and evidence_root must be disjoint")
    if output and any(
        output.is_relative_to(owned) or owned.is_relative_to(output)
        for owned in (staging, evidence)
    ):
        raise ValueError("audio output_root must be disjoint from staging and evidence")
    for name, maximum in (
        ("max_input_bytes", MAX_INPUT_BYTES),
        ("max_duration_seconds", MAX_DURATION_SECONDS),
    ):
        value = raw[name]
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"audio {name} must be an integer from 1 through {maximum}")
    executables = {
        name: (_absolute(raw[name], f"audio {name}") if raw.get(name) is not None else None)
        for name in ("ffmpeg", "ffprobe", "whisper_cli", "whisper_vad_cli")
    }
    models = {
        name: _asset(raw[name], f"audio {name}") if raw.get(name) is not None else None
        for name in ("model", "vad_model")
    }
    if transcription and (None in executables.values() or None in models.values()):
        raise ValueError("audio transcription requires Whisper executables and pinned models")
    settings = AudioSettings(
        read_roots=read_roots,
        staging_root=staging,
        evidence_root=evidence,
        ffmpeg=executables["ffmpeg"] or Path("/"),
        ffprobe=executables["ffprobe"] or Path("/"),
        whisper_cli=executables["whisper_cli"],
        whisper_vad_cli=executables["whisper_vad_cli"],
        model=models["model"],
        vad_model=models["vad_model"],
        max_input_bytes=int(str(raw["max_input_bytes"])),
        max_duration_seconds=int(str(raw["max_duration_seconds"])),
        output_root=output,
    )
    if executables["ffmpeg"] is None or executables["ffprobe"] is None:
        raise ValueError("audio ffmpeg and ffprobe are required")
    for executable in (settings.ffmpeg, settings.ffprobe):
        _require_executable(executable)
    if transcription:
        assert settings.whisper_cli and settings.whisper_vad_cli
        assert settings.model and settings.vad_model
        _require_executable(settings.whisper_cli)
        _require_executable(settings.whisper_vad_cli)
        for asset in (settings.model, settings.vad_model):
            verify_asset(asset.path, asset.sha256)
    return settings


@contextmanager
def compose_audio_mcp(
    config: WorkspaceHostConfiguration,
    context_factory: Callable[[ToolSpec], InvocationContext],
    *,
    catalog_only: bool = False,
) -> Iterator[tuple[McpToolBinding, ...]]:
    """Validate audio authority and bind the selected Tools."""
    from capicove.tools.audio_clip import AUDIO_CLIP_SPEC
    from capicove.tools.audio_import import AUDIO_IMPORT_SPEC
    from capicove.tools.audio_transcribe import AUDIO_TRANSCRIBE_SPEC

    specs = {IMPORT: AUDIO_IMPORT_SPEC, TRANSCRIBE: AUDIO_TRANSCRIBE_SPEC, CLIP: AUDIO_CLIP_SPEC}
    if any(tool_id not in specs for tool_id in config.enabled_tools):
        raise ValueError("audio profile includes an unavailable Tool")
    for tool_id in config.enabled_tools:
        if not specs[tool_id].required_permissions.issubset(config.permissions):
            raise ValueError("configured permissions do not authorize every enabled Tool")
    settings = audio_settings(config)
    # Snapshot executable identities at startup; calls run only these exact files.
    base = {"ffmpeg": settings.ffmpeg, "ffprobe": settings.ffprobe}
    pins = {
        tool_id: PinnedExecutables(base)
        for tool_id in (IMPORT, CLIP)
        if tool_id in config.enabled_tools
    }
    if TRANSCRIBE in config.enabled_tools:
        assert settings.whisper_cli and settings.whisper_vad_cli
        pins[TRANSCRIBE] = PinnedExecutables(
            {
                **base,
                "whisper_cli": settings.whisper_cli,
                "whisper_vad_cli": settings.whisper_vad_cli,
            }
        )
    builders: dict[str, Callable[[], ToolHandler]] = {
        IMPORT: lambda: _import_handler(settings, pins[IMPORT]),
        TRANSCRIBE: lambda: _transcribe_handler(settings, pins[TRANSCRIBE]),
        CLIP: lambda: _clip_handler(settings, pins[CLIP]),
    }
    yield tuple(
        McpToolBinding(
            "capicove_audio_" + tool_id.rsplit(".", 1)[1],
            specs[tool_id],
            _catalog_only_handler if catalog_only else builders[tool_id](),
            context_factory,
        )
        for tool_id in sorted(config.enabled_tools)
    )


def _intake(settings: AudioSettings) -> MediaIntake:
    return MediaIntake(
        read_roots=settings.read_roots,
        staging_root=settings.staging_root,
        max_input_bytes=settings.max_input_bytes,
        accepts=is_audio_container,
    )


def _processor(executables: Mapping[str, str], timeout: float):  # type: ignore[no-untyped-def]
    from capicove.processing.audio import AudioProcessor
    from capicove.processing.audio._adapters.ffmpeg import FfmpegAudioValidator
    from capicove.processing.audio._adapters.ffprobe import FfprobeAudioStreamInspector

    return AudioProcessor(
        FfprobeAudioStreamInspector(executables["ffprobe"], timeout, restrict_inputs=True),
        FfmpegAudioValidator(executables["ffmpeg"], timeout, restrict_inputs=True),
    )


def _authorizer(settings: AudioSettings):  # type: ignore[no-untyped-def]
    from capicove.processing.audio._adapters.filesystem import ExplicitAudioPathAuthorizer

    return ExplicitAudioPathAuthorizer({_POLICY: (settings.staging_root,)})


def _import_handler(settings: AudioSettings, pins: PinnedExecutables) -> ToolHandler:
    from capicove.tools.audio_import import AUDIO_IMPORT_SPEC, AudioImportToolHandler

    intake = _intake(settings)

    def handler(context: InvocationContext, arguments: Mapping[str, object], /) -> ToolResult:
        path = _source_path(arguments)
        if path is None:
            return _failure(AUDIO_IMPORT_SPEC, context, ToolErrorCode.INVALID_INPUT)
        timeout = _remaining(context)

        def run(staged: tuple[StagedFile, ...], executables: Mapping[str, str]) -> ToolResult:
            settings.evidence_root.mkdir(mode=0o700, exist_ok=True)
            inner = AudioImportToolHandler(
                _processor(executables, timeout),
                _authorizer(settings),
                execution_policy_ref=_POLICY,
                store_root=settings.evidence_root,
            )
            return inner(context, _with_source(arguments, staged[0]))

        return _staged(AUDIO_IMPORT_SPEC, settings, intake, pins, context, path, arguments, (), run)

    return handler


def _transcribe_handler(settings: AudioSettings, pins: PinnedExecutables) -> ToolHandler:
    from capicove.processing.audio import AudioTranscriptionService
    from capicove.processing.audio._adapters.whisper_cpp import WhisperCppTranscriber
    from capicove.tools.audio_transcribe import AUDIO_TRANSCRIBE_SPEC, AudioTranscribeToolHandler

    assert settings.model and settings.vad_model
    assert settings.whisper_cli and settings.whisper_vad_cli
    intake = _intake(settings)
    assets = (
        (settings.model.path, settings.model.sha256),
        (settings.vad_model.path, settings.vad_model.sha256),
    )

    def handler(context: InvocationContext, arguments: Mapping[str, object], /) -> ToolResult:
        path = _source_path(arguments)
        if path is None:
            return _failure(AUDIO_TRANSCRIBE_SPEC, context, ToolErrorCode.INVALID_INPUT)
        timeout = _remaining(context)

        def run(staged: tuple[StagedFile, ...], executables: Mapping[str, str]) -> ToolResult:
            source, model, vad_model = staged
            settings.evidence_root.mkdir(mode=0o700, exist_ok=True)
            transcriber = WhisperCppTranscriber(
                model.staged,
                vad_model_path=vad_model.staged,
                executable=executables["whisper_cli"],
                vad_executable=executables["whisper_vad_cli"],
                ffmpeg_executable=executables["ffmpeg"],
                backend_version=_read_version(executables["whisper_cli"]),
                timeout_seconds=timeout,
                restrict_inputs=True,
            )
            inner = AudioTranscribeToolHandler(
                AudioTranscriptionService(_processor(executables, timeout), transcriber),
                _authorizer(settings),
                execution_policy_ref=_POLICY,
                store_root=settings.evidence_root,
            )
            return inner(context, _with_source(arguments, source))

        return _staged(
            AUDIO_TRANSCRIBE_SPEC, settings, intake, pins, context, path, arguments, assets, run
        )

    return handler


def _clip_handler(settings: AudioSettings, pins: PinnedExecutables) -> ToolHandler:
    from capicove.processing.audio import AudioClippingService
    from capicove.processing.audio._adapters.audio_clip import FfmpegAudioClipBackend
    from capicove.processing.audio.clip_models import AudioClippingError, remaining_seconds
    from capicove.runtime.media_handoff import open_private_directory
    from capicove.tools.audio_clip import AUDIO_CLIP_SPEC, AudioClipToolHandler, decode_clip_request

    assert settings.output_root is not None
    intake = _intake(settings)

    def handler(context: InvocationContext, arguments: Mapping[str, object], /) -> ToolResult:
        if AUDIO_CLIP_SPEC.required_permissions.difference(context.permissions):
            return _failure(AUDIO_CLIP_SPEC, context, ToolErrorCode.PERMISSION_DENIED)
        try:
            request = decode_clip_request(arguments)
        except (ValueError, TypeError, OverflowError):
            return _failure(AUDIO_CLIP_SPEC, context, ToolErrorCode.INVALID_INPUT)
        assert settings.output_root is not None
        if request.destination == settings.output_root or not request.destination.is_relative_to(
            settings.output_root
        ):
            return _failure(AUDIO_CLIP_SPEC, context, ToolErrorCode.PERMISSION_DENIED)

        def check_deadline() -> None:
            remaining_seconds(context.deadline)

        def run(staged: tuple[StagedFile, ...], executables: Mapping[str, str]) -> ToolResult:
            assert settings.output_root is not None
            # Create only the configured owner-private output root, during an explicit call.
            output_directory = open_private_directory(settings.output_root)
            os.close(output_directory)
            inner = AudioClipToolHandler(
                AudioClippingService(
                    _processor(executables, remaining_seconds(context.deadline)),
                    FfmpegAudioClipBackend(executables["ffmpeg"], executables["ffprobe"]),
                    max_duration_seconds=settings.max_duration_seconds,
                ),
                read_roots=(settings.staging_root,),
                output_root=settings.output_root,
                store_root=settings.evidence_root,
            )
            rewritten = dict(arguments)
            rewritten["source"] = {"type": "path", "path": staged[0].resolved}
            return inner(context, rewritten)

        try:
            check_deadline()
            intake.authorize(request.source)
            with (
                pins.open() as executables,
                intake.stage([request.source], check_deadline=check_deadline) as staged,
            ):
                check_deadline()
                result = run(staged, executables)
                evidence = (result.result or {}).get("evidence")
                if isinstance(evidence, Mapping) and evidence.get("sha256") != staged[0].sha256:
                    return _failure(AUDIO_CLIP_SPEC, context, ToolErrorCode.INTERNAL)
                return result
        except AudioClippingError:
            return _failure(AUDIO_CLIP_SPEC, context, ToolErrorCode.TIMEOUT)
        except MediaDenied:
            return _failure(AUDIO_CLIP_SPEC, context, ToolErrorCode.PERMISSION_DENIED)
        except (AssetIntegrityError, ExecutableChanged):
            return _failure(AUDIO_CLIP_SPEC, context, ToolErrorCode.UNAVAILABLE)
        except (MediaInvalid, MediaChanged):
            return _failure(AUDIO_CLIP_SPEC, context, ToolErrorCode.INVALID_INPUT)
        except OSError:
            return _failure(AUDIO_CLIP_SPEC, context, ToolErrorCode.INTERNAL)

    return handler


def _staged(
    spec: ToolSpec,
    settings: AudioSettings,
    intake: MediaIntake,
    pins: PinnedExecutables,
    context: InvocationContext,
    path: Path,
    arguments: Mapping[str, object],
    assets: tuple[tuple[Path, str], ...],
    run: Callable[[tuple[StagedFile, ...], Mapping[str, str]], ToolResult],
) -> ToolResult:
    if spec.required_permissions.difference(context.permissions):
        return _failure(spec, context, ToolErrorCode.PERMISSION_DENIED)
    index = arguments.get("audio_stream_index")
    try:
        intake.authorize(path)
        with pins.open() as executables, intake.stage([path], assets) as staged:
            duration = _duration_seconds(
                executables["ffprobe"], staged[0].staged, index if isinstance(index, int) else 0
            )
            if duration is not None and duration > settings.max_duration_seconds:
                raise MediaInvalid("audio exceeds max_duration_seconds")
            result = run(staged, executables)
            evidence = (result.result or {}).get("evidence")
            if isinstance(evidence, Mapping) and evidence.get("sha256") not in (
                None,
                staged[0].sha256,
            ):
                # The backend must have read the verified staged copy.
                return _failure(spec, context, ToolErrorCode.INVALID_INPUT)
            restored = restore_originals(
                dict(result.result) if result.result is not None else None, staged
            )
            error = result.error
            if error is not None:
                error = replace(error, message=str(restore_originals(error.message, staged)))
            return replace(
                result, result=restored if isinstance(restored, dict) else None, error=error
            )
    except MediaDenied:
        return _failure(spec, context, ToolErrorCode.PERMISSION_DENIED)
    except (AssetIntegrityError, ExecutableChanged):
        return _failure(spec, context, ToolErrorCode.UNAVAILABLE)
    except (MediaInvalid, MediaChanged):
        return _failure(spec, context, ToolErrorCode.INVALID_INPUT)


def _duration_seconds(ffprobe: str, path: Path, index: int) -> float | None:
    """Read the selected stream's duration with fixed, input-restricted FFprobe arguments."""
    from capicove.processing.audio._adapters.ffmpeg import restricted_input_arguments

    try:
        completed = subprocess.run(  # nosec B603 - pinned executable, fixed args
            [
                ffprobe,
                *("-v", "error", "-select_streams", f"a:{index}"),
                *("-show_entries", "stream=duration:format=duration", "-of", "json"),
                *restricted_input_arguments(path),
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            start_new_session=True,
        )
        payload = json.loads(completed.stdout or "{}")
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None
    for value in (
        *(stream.get("duration") for stream in payload.get("streams", [])),
        payload.get("format", {}).get("duration"),
    ):
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _read_version(executable: str) -> str:
    try:
        completed = subprocess.run(  # nosec B603 - absolute configured executable
            [executable, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            start_new_session=True,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    lines = (completed.stdout or completed.stderr).strip().splitlines()
    return lines[0][:128] if completed.returncode == 0 and lines else "unknown"


def _source_path(arguments: Mapping[str, object]) -> Path | None:
    source = arguments.get("source")
    if (
        not isinstance(source, Mapping)
        or set(source) != {"type", "path"}
        or source.get("type") != "path"
        or not isinstance(source.get("path"), str)
    ):
        return None
    return Path(str(source["path"]))


def _with_source(arguments: Mapping[str, object], staged: StagedFile) -> dict[str, object]:
    return {**arguments, "source": {"type": "path", "path": str(staged.staged)}}


def _remaining(context: InvocationContext) -> float:
    return max(1.0, (context.deadline - datetime.now(UTC)).total_seconds())


def _failure(spec: ToolSpec, context: InvocationContext, code: ToolErrorCode) -> ToolResult:
    messages = {
        ToolErrorCode.PERMISSION_DENIED: "The audio is outside the configured authority.",
        ToolErrorCode.INVALID_INPUT: "The audio request or media is not accepted by this host.",
        ToolErrorCode.UNAVAILABLE: "A configured model does not match its pinned SHA-256.",
        ToolErrorCode.TIMEOUT: "Audio clipping timed out.",
        ToolErrorCode.INTERNAL: "Audio storage or output verification failed.",
    }
    now = datetime.now(UTC)
    return ToolResult(
        request_id=context.request_id,
        trace_id=context.trace_id,
        tool_id=spec.tool_id,
        tool_version=spec.contract_version,
        status=ToolStatus.REJECTED
        if spec.tool_id == CLIP and code is ToolErrorCode.PERMISSION_DENIED
        else ToolStatus.FAILED,
        error=ToolError(code, messages[code], retryable=False),
        started_at=now,
        completed_at=now,
    )


def _catalog_only_handler(
    context: InvocationContext, arguments: Mapping[str, object], /
) -> NoReturn:
    raise RuntimeError("catalog-only audio bindings cannot be invoked")


def _existing_directory(value: object, field: str) -> Path:
    path = _absolute(value, field)
    if any(component.is_symlink() for component in (path, *path.parents)):
        raise ValueError(f"{field} must not contain symlinks")
    if not path.is_dir():
        raise ValueError(f"{field} must name existing directories")
    return path


def _private_child(value: object, workspace: Path, field: str) -> Path:
    path = _absolute(value, field)
    if path == workspace or not path.is_relative_to(workspace):
        raise ValueError(f"{field} must be strictly inside workspace_root")
    if any(component.is_symlink() for component in (path, *path.parents)):
        raise ValueError(f"{field} must not contain symlinks")
    if path.exists():
        metadata = path.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_mode & 0o077:
            raise ValueError(f"{field} must be an owner-private directory")
    return path


def _absolute(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.startswith("/") or "\0" in value:
        raise ValueError(f"{field} must be absolute paths")
    path = Path(value)
    if ".." in path.parts:
        raise ValueError(f"{field} must be normalized absolute paths")
    return path


def _asset(value: object, field: str) -> PinnedAsset:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{field} must be an object with path and sha256")
    digest = value["sha256"]
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{field} sha256 must be a lowercase hex digest")
    return PinnedAsset(_absolute(value["path"], f"{field} path"), digest)


def _require_executable(path: Path) -> None:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise AudioDependencyError(f"{path.name} is not an executable file")


__all__ = ["AudioDependencyError", "AudioSettings", "audio_settings", "compose_audio_mcp"]
