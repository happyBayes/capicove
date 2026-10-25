"""Canonical Tool contract and handler for local audio transcription."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from capicove.contracts import (
    InvocationContext,
    PrivacyClass,
    SideEffect,
    ToolError,
    ToolErrorCode,
    ToolResources,
    ToolResult,
    ToolSpec,
    ToolStatus,
)
from capicove.processing.audio import (
    AudioFailureCode,
    AudioImportError,
    AudioTranscriptionError,
    AudioTranscriptionResult,
)
from capicove.processing.audio.ports import AudioPathPolicyError
from capicove.tools.registry import ToolRegistry


AUDIO_TRANSCRIBE_INPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["source"],
    "properties": {
        "source": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "path"],
            "properties": {
                "type": {"const": "path"},
                "path": {"type": "string", "minLength": 1},
            },
        },
        "language": {"enum": ["auto", "zh", "en", "zh-CN", "en-US", "en-GB"]},
        "audio_stream_index": {"type": "integer", "minimum": 0},
    },
}

_AUDIO_STREAM_SELECTION_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "selected_audio_stream_index",
        "index_semantics",
        "selection_method",
        "preflight_audio_stream_count",
        "validation_scope",
    ],
    "properties": {
        "schema_version": {"const": 1},
        "selected_audio_stream_index": {"type": "integer", "minimum": 0, "maximum": 63},
        "index_semantics": {"const": "zero_based_audio_stream_ordinal"},
        "selection_method": {"enum": ["explicit_index", "sole_stream_implicit"]},
        "preflight_audio_stream_count": {"type": "integer", "minimum": 1, "maximum": 64},
        "validation_scope": {"const": "selected_audio_stream_only"},
    },
}

_AUDIO_TRANSCRIBE_SUCCESS_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "status",
        "text",
        "language",
        "segments",
        "warnings",
        "evidence",
        "lineage",
    ],
    "properties": {
        "status": {"const": "succeeded"},
        "text": {"type": "string"},
        "language": {"type": "string", "minLength": 1},
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["index", "start_seconds", "end_seconds", "text"],
                "properties": {
                    "index": {"type": "integer", "minimum": 0},
                    "start_seconds": {"type": "number", "minimum": 0},
                    "end_seconds": {"type": "number", "exclusiveMinimum": 0},
                    "text": {"type": "string", "minLength": 1},
                },
            },
        },
        "warnings": {
            "type": "array",
            "uniqueItems": True,
            "items": {
                "enum": [
                    "NO_SPEECH_DETECTED",
                    "POTENTIAL_HALLUCINATION",
                    "TIMESTAMP_CLAMPED",
                    "TRANSCRIPT_ENCODING_REPAIRED",
                    "UNSELECTED_AUDIO_STREAMS_IGNORED",
                ]
            },
        },
        "evidence": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "evidence_id",
                "sha256",
                "size_bytes",
                "media_type",
                "source_name",
            ],
            "properties": {
                "evidence_id": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
                "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "size_bytes": {"type": "integer", "minimum": 1},
                "media_type": {"type": "string", "minLength": 1},
                "source_name": {"type": "string", "minLength": 1},
            },
        },
        "lineage": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "backend",
                "backend_version",
                "model",
                "model_sha256",
                "vad_model",
                "vad_model_sha256",
                "device",
                "precision",
                "audio_stream_selection",
            ],
            "properties": {
                "backend": {"type": "string", "minLength": 1},
                "backend_version": {"type": "string", "minLength": 1},
                "model": {"type": "string", "minLength": 1},
                "model_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "vad_model": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
                "vad_model_sha256": {
                    "anyOf": [
                        {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                        {"type": "null"},
                    ]
                },
                "device": {"type": "string", "minLength": 1},
                "precision": {"type": "string", "minLength": 1},
                "audio_stream_selection": _AUDIO_STREAM_SELECTION_SCHEMA,
            },
        },
    },
}

_AUDIO_FAILURE_PROPERTIES: dict[str, object] = {
    "code": {"enum": [code.value for code in AudioFailureCode]},
    "message": {"type": "string", "minLength": 1, "maxLength": 256},
    "retryable": {"type": "boolean"},
    "preflight_audio_stream_count": {"type": "integer", "minimum": 0, "maximum": 64},
}

_AUDIO_TRANSCRIBE_REJECTED_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "failure"],
    "properties": {
        "status": {"const": "rejected"},
        "failure": {
            "type": "object",
            "additionalProperties": False,
            "required": ["code", "message", "retryable"],
            "properties": {
                **_AUDIO_FAILURE_PROPERTIES,
                "code": {"const": "AUDIO_STREAM_SELECTION_REQUIRED"},
            },
        },
    },
}

_AUDIO_TRANSCRIBE_FAILED_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "failure"],
    "properties": {
        "status": {"const": "failed"},
        "failure": {
            "type": "object",
            "additionalProperties": False,
            "required": ["code", "message", "retryable"],
            "properties": {
                **_AUDIO_FAILURE_PROPERTIES,
                "code": {
                    "enum": [
                        code.value
                        for code in AudioFailureCode
                        if code is not AudioFailureCode.AUDIO_STREAM_SELECTION_REQUIRED
                    ]
                },
            },
        },
    },
}

AUDIO_TRANSCRIBE_OUTPUT_SCHEMA: dict[str, object] = {
    "oneOf": [
        _AUDIO_TRANSCRIBE_SUCCESS_SCHEMA,
        _AUDIO_TRANSCRIBE_REJECTED_SCHEMA,
        _AUDIO_TRANSCRIBE_FAILED_SCHEMA,
    ]
}

AUDIO_TRANSCRIBE_SPEC = ToolSpec(
    tool_id="capicove.audio.transcribe",
    contract_version=2,
    description="Transcribe authorized local audio into ordered timestamped text evidence.",
    input_schema=AUDIO_TRANSCRIBE_INPUT_SCHEMA,
    output_schema=AUDIO_TRANSCRIBE_OUTPUT_SCHEMA,
    modalities=("audio", "text"),
    required_permissions=frozenset({"audio:read", "artifact:write", "subprocess:execute"}),
    privacy_class=PrivacyClass.CONFIDENTIAL,
    side_effect=SideEffect.WRITE,
    network_access=False,
    deterministic=False,
    idempotent=True,
    timeout_seconds=300.0,
    redacted_fields=("source.path",),
    resources=ToolResources(
        filesystem_read=True,
        filesystem_write=True,
        subprocesses=(
            "ffprobe",
            "ffmpeg",
            "whisper-cli",
            "whisper-vad-speech-segments",
        ),
    ),
)


class AudioTranscriptionPort(Protocol):
    """Narrow application operation consumed by the Tool handler."""

    def transcribe_file(
        self,
        source_path: Path,
        store_root: Path,
        *,
        language: str,
        deadline: datetime,
        audio_stream_index: object = ...,
    ) -> AudioTranscriptionResult: ...


class AudioPathAuthorizer(Protocol):
    """Authorize local audio paths against an explicit runtime policy."""

    def authorize_read(self, policy_ref: str, path: Path) -> Path: ...


class AudioTranscribeToolHandler:
    """Validate one request and map stable audio failures to Tool semantics."""

    def __init__(
        self,
        service: AudioTranscriptionPort,
        path_authorizer: AudioPathAuthorizer,
        *,
        execution_policy_ref: str,
        store_root: Path,
    ) -> None:
        if not execution_policy_ref.strip():
            raise ValueError("execution_policy_ref must not be blank")
        if not store_root.is_absolute():
            raise ValueError("store_root must be absolute")
        self._service = service
        self._path_authorizer = path_authorizer
        self._execution_policy_ref = execution_policy_ref
        self._store_root = store_root.resolve()

    def __call__(
        self,
        context: InvocationContext,
        arguments: Mapping[str, object],
        /,
    ) -> ToolResult:
        """Execute one authorized, offline audio transcription request."""
        started = datetime.now(UTC)
        missing = AUDIO_TRANSCRIBE_SPEC.required_permissions.difference(context.permissions)
        if missing:
            return _failure(
                context,
                started,
                ToolStatus.REJECTED,
                ToolErrorCode.PERMISSION_DENIED,
                f"Missing required permissions: {', '.join(sorted(missing))}.",
            )
        try:
            source, language, audio_stream_index = _parse_arguments(arguments)
            source = self._path_authorizer.authorize_read(self._execution_policy_ref, source)
            if audio_stream_index is _AUDIO_STREAM_INDEX_OMITTED:
                processed = self._service.transcribe_file(
                    source,
                    self._store_root,
                    language=language,
                    deadline=context.deadline,
                )
            else:
                processed = self._service.transcribe_file(
                    source,
                    self._store_root,
                    language=language,
                    deadline=context.deadline,
                    audio_stream_index=audio_stream_index,
                )
        except AudioPathPolicyError:
            return _failure(
                context,
                started,
                ToolStatus.REJECTED,
                ToolErrorCode.PERMISSION_DENIED,
                "Audio source is outside the authorized read scope.",
            )
        except AudioImportError as error:
            return _audio_failure(
                context,
                started,
                error.code,
                str(error),
                preflight_audio_stream_count=error.preflight_audio_stream_count,
            )
        except (ValueError, TypeError):
            return _failure(
                context,
                started,
                ToolStatus.FAILED,
                ToolErrorCode.INVALID_INPUT,
                "Audio transcription arguments are invalid.",
            )
        except AudioTranscriptionError as error:
            return _audio_failure(context, started, error.code, str(error))
        completed = datetime.now(UTC)
        total_seconds = (completed - started).total_seconds()
        metrics = {
            "segment_count": float(len(processed.transcript.segments)),
            "total_seconds": total_seconds,
        }
        if processed.transcript.input_duration_seconds is not None:
            metrics["input_duration_seconds"] = processed.transcript.input_duration_seconds
            metrics["end_to_end_real_time_factor"] = (
                total_seconds / processed.transcript.input_duration_seconds
            )
        return ToolResult(
            request_id=context.request_id,
            trace_id=context.trace_id,
            tool_id=AUDIO_TRANSCRIBE_SPEC.tool_id,
            tool_version=AUDIO_TRANSCRIBE_SPEC.contract_version,
            status=ToolStatus.SUCCEEDED,
            result=_serialize(processed),
            evidence_refs=(processed.evidence.evidence_id,),
            metrics=metrics,
            started_at=started,
            completed_at=completed,
        )


def register_audio_transcribe_tool(
    registry: ToolRegistry,
    service: AudioTranscriptionPort,
    path_authorizer: AudioPathAuthorizer,
    *,
    execution_policy_ref: str,
    store_root: Path,
) -> AudioTranscribeToolHandler:
    """Register the canonical audio-transcribe Tool handler."""
    handler = AudioTranscribeToolHandler(
        service,
        path_authorizer,
        execution_policy_ref=execution_policy_ref,
        store_root=store_root,
    )
    registry.register(AUDIO_TRANSCRIBE_SPEC, handler)
    return handler


def sanitize_audio_transcribe_arguments(arguments: Mapping[str, object]) -> dict[str, object]:
    """Remove the private source path before persisting trace arguments."""
    sanitized = dict(arguments)
    source = arguments.get("source")
    if isinstance(source, Mapping) and source.get("type") == "path":
        path_value = source.get("path")
        if isinstance(path_value, str):
            sanitized["source"] = {"type": "path", "name": Path(path_value).name}
    return sanitized


def audio_transcribe_capabilities(
    *,
    executable_available: bool,
    vad_executable_available: bool,
    decoder_available: bool,
    probe_available: bool,
    model_available: bool,
    model_sha256: str | None,
    vad_model_available: bool,
    vad_model_sha256: str | None,
    backend_version: str,
) -> dict[str, object]:
    """Describe the actual local ASR deployment without marketing overclaim."""
    available = (
        executable_available
        and vad_executable_available
        and decoder_available
        and probe_available
        and model_available
        and model_sha256 is not None
        and vad_model_available
        and vad_model_sha256 is not None
    )
    return {
        "component": AUDIO_TRANSCRIBE_SPEC.tool_id,
        "version": AUDIO_TRANSCRIBE_SPEC.contract_version,
        "available": available,
        "backend": "whisper.cpp",
        "backend_version": backend_version,
        "model_sha256": model_sha256,
        "vad": {
            "enabled": (
                vad_executable_available and vad_model_available and vad_model_sha256 is not None
            ),
            "executable_available": vad_executable_available,
            "model_sha256": vad_model_sha256,
            "streaming": False,
        },
        "decoder": {
            "available": decoder_available and probe_available,
            "backend": "ffmpeg",
            "probe_available": probe_available,
            "probe_backend": "ffprobe",
        },
        "audio_stream_selection": {
            "index_semantics": "zero_based_audio_stream_ordinal",
            "single_stream_omission": "sole_stream_implicit",
            "multi_stream_omission": "rejected",
            "validation_scope": "selected_audio_stream_only",
        },
        "network_access": False,
        "devices": ["cpu"],
        "streaming": False,
        "supported_languages": ["zh", "en", "auto"],
        "supported_formats": ["aac", "flac", "m4a", "mp3", "ogg", "opus", "wav", "webm"],
        "unimplemented_operations": [
            "audio.synthesize",
            "audio.detect_voice",
            "audio.diarize",
            "audio.denoise",
            "audio.get_job",
            "audio.cancel_job",
        ],
    }


_AUDIO_STREAM_INDEX_OMITTED = object()


def _parse_arguments(arguments: Mapping[str, object]) -> tuple[Path, str, object]:
    if (
        set(arguments).difference({"source", "language", "audio_stream_index"})
        or "source" not in arguments
    ):
        raise ValueError("arguments contain unknown or missing fields")
    source = arguments["source"]
    if not isinstance(source, Mapping) or set(source) != {"type", "path"}:
        raise ValueError("source representation is invalid")
    if source.get("type") != "path":
        raise ValueError("only authorized resource paths are supported")
    path_value = source.get("path")
    language = arguments.get("language", "auto")
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("source.path must not be blank")
    if not isinstance(language, str) or not language.strip():
        raise ValueError("language must not be blank")
    audio_stream_index = arguments.get(
        "audio_stream_index",
        _AUDIO_STREAM_INDEX_OMITTED,
    )
    if audio_stream_index is not _AUDIO_STREAM_INDEX_OMITTED and (
        not isinstance(audio_stream_index, int)
        or isinstance(audio_stream_index, bool)
        or audio_stream_index < 0
    ):
        raise AudioImportError(
            AudioFailureCode.INVALID_AUDIO_STREAM_INDEX,
            "audio_stream_index must be a non-negative integer",
        )
    return Path(path_value).expanduser(), language, audio_stream_index


def _serialize(processed: AudioTranscriptionResult) -> dict[str, object]:
    transcript = processed.transcript
    return {
        "status": "succeeded",
        "text": transcript.text,
        "language": transcript.language,
        "segments": [
            {
                "index": segment.index,
                "start_seconds": segment.start_seconds,
                "end_seconds": segment.end_seconds,
                "text": segment.text,
            }
            for segment in transcript.segments
        ],
        "warnings": [warning.value for warning in transcript.warnings],
        "evidence": {
            "evidence_id": processed.evidence.evidence_id,
            "sha256": processed.evidence.sha256,
            "size_bytes": processed.evidence.size_bytes,
            "media_type": processed.evidence.media_type,
            "source_name": processed.evidence.source_name,
        },
        "lineage": {
            "backend": transcript.lineage.backend,
            "backend_version": transcript.lineage.backend_version,
            "model": transcript.lineage.model,
            "model_sha256": transcript.lineage.model_sha256,
            "vad_model": transcript.lineage.vad_model,
            "vad_model_sha256": transcript.lineage.vad_model_sha256,
            "device": transcript.lineage.device,
            "precision": transcript.lineage.precision,
            "audio_stream_selection": {
                "schema_version": processed.stream_selection.schema_version,
                "selected_audio_stream_index": (
                    processed.stream_selection.selected_audio_stream_index
                ),
                "index_semantics": processed.stream_selection.index_semantics,
                "selection_method": processed.stream_selection.selection_method,
                "preflight_audio_stream_count": (
                    processed.stream_selection.preflight_audio_stream_count
                ),
                "validation_scope": processed.stream_selection.validation_scope,
            },
        },
    }


def _tool_error_code(code: AudioFailureCode) -> ToolErrorCode:
    if code in {
        AudioFailureCode.INVALID_AUDIO,
        AudioFailureCode.UNSUPPORTED_FORMAT,
        AudioFailureCode.UNSUPPORTED_LANGUAGE,
        AudioFailureCode.SOURCE_CHANGED,
        AudioFailureCode.INVALID_AUDIO_STREAM_INDEX,
        AudioFailureCode.NO_AUDIO_STREAM,
        AudioFailureCode.AUDIO_STREAM_INDEX_OUT_OF_RANGE,
        AudioFailureCode.SELECTED_AUDIO_STREAM_UNDECODABLE,
    }:
        return ToolErrorCode.INVALID_INPUT
    if code is AudioFailureCode.AUDIO_STREAM_SELECTION_REQUIRED:
        return ToolErrorCode.POLICY_REJECTED
    if code is AudioFailureCode.TIMEOUT:
        return ToolErrorCode.TIMEOUT
    if code in {
        AudioFailureCode.DECODER_UNAVAILABLE,
        AudioFailureCode.MODEL_UNAVAILABLE,
        AudioFailureCode.MODEL_LOAD_FAILED,
    }:
        return ToolErrorCode.UNAVAILABLE
    return ToolErrorCode.INTERNAL


def _audio_failure(
    context: InvocationContext,
    started: datetime,
    audio_code: AudioFailureCode,
    message: str,
    *,
    preflight_audio_stream_count: int | None = None,
    spec: ToolSpec | None = None,
) -> ToolResult:
    bounded_message = message if 0 < len(message) <= 256 else "audio operation failed"
    status = (
        ToolStatus.REJECTED
        if audio_code is AudioFailureCode.AUDIO_STREAM_SELECTION_REQUIRED
        else ToolStatus.FAILED
    )
    tool_code = _tool_error_code(audio_code)
    retryable = tool_code is ToolErrorCode.TIMEOUT
    failure: dict[str, object] = {
        "code": audio_code.value,
        "message": bounded_message,
        "retryable": retryable,
    }
    if (
        isinstance(preflight_audio_stream_count, int)
        and not isinstance(preflight_audio_stream_count, bool)
        and 0 <= preflight_audio_stream_count <= 64
    ):
        failure["preflight_audio_stream_count"] = preflight_audio_stream_count
    tool = spec or AUDIO_TRANSCRIBE_SPEC
    return ToolResult(
        request_id=context.request_id,
        trace_id=context.trace_id,
        tool_id=tool.tool_id,
        tool_version=tool.contract_version,
        status=status,
        result={"status": status.value, "failure": failure},
        error=ToolError(tool_code, bounded_message, retryable=retryable),
        started_at=started,
        completed_at=datetime.now(UTC),
    )


def audio_failure_result(
    spec: ToolSpec,
    context: InvocationContext,
    started: datetime,
    audio_code: AudioFailureCode,
    message: str,
    *,
    preflight_audio_stream_count: int | None = None,
) -> ToolResult:
    """Map one stable audio failure code for any audio Tool."""
    return _audio_failure(
        context,
        started,
        audio_code,
        message,
        preflight_audio_stream_count=preflight_audio_stream_count,
        spec=spec,
    )


def _failure(
    context: InvocationContext,
    started: datetime,
    status: ToolStatus,
    code: ToolErrorCode,
    message: str,
) -> ToolResult:
    return ToolResult(
        request_id=context.request_id,
        trace_id=context.trace_id,
        tool_id=AUDIO_TRANSCRIBE_SPEC.tool_id,
        tool_version=AUDIO_TRANSCRIBE_SPEC.contract_version,
        status=status,
        error=ToolError(code, message, retryable=code in {ToolErrorCode.TIMEOUT}),
        started_at=started,
        completed_at=datetime.now(UTC),
    )
