"""Canonical Tool contract and handler for video splitting."""

from __future__ import annotations

import base64
import binascii
import hashlib
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
from capicove.processing.video import (
    DurationSplitConfig,
    KeyframeSplitConfig,
    SceneDetectorName,
    SceneSplitConfig,
    VideoAsset,
    VideoBackend,
    VideoOutputFormat,
    VideoSegment,
    VideoSource,
    VideoSplitFailureCode,
    VideoSplitRequest,
    VideoSplitResult,
    VideoSplitStatus,
    VideoSplitStrategy,
    VideoSplitter,
    VideoSplitWarning,
    VideoWriteMode,
)
from capicove.processing.video._adapters.ffmpeg import DefaultVideoBackendResolver
from capicove.processing.video._adapters.filesystem import (
    ExplicitPathAuthorizer,
    FilesystemOutputStore,
)
from capicove.processing.video._adapters.pyav import create_pyav_backend
from capicove.processing.video._adapters.scenedetect import PySceneDetectSceneDetector
from capicove.processing.video.ports import PathPolicyError
from capicove.tools.registry import ToolRegistry


VIDEO_SPLIT_INPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["source", "strategy", "config", "output_format", "backend"],
    "properties": {
        "source": {
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["type", "path"],
                    "properties": {"type": {"const": "path"}, "path": {"type": "string"}},
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["type", "base64"],
                    "properties": {
                        "type": {"const": "bytes"},
                        "base64": {
                            "type": "string",
                            "contentEncoding": "base64",
                            "x-capicove-max-decoded-bytes": 4 * 1024 * 1024,
                        },
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["type", "ref", "sha256"],
                    "properties": {
                        "type": {"const": "artifact"},
                        "ref": {"type": "string"},
                        "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                    },
                },
            ]
        },
        "strategy": {"enum": ["duration", "keyframe", "scene"]},
        "config": {"type": "object"},
        "output_format": {"enum": ["path", "bytes"]},
        "output_directory": {"type": "string"},
        "backend": {"enum": ["auto", "ffmpeg", "av"]},
        "write_mode": {"enum": ["accurate", "copy"], "default": "accurate"},
    },
}

VIDEO_SPLIT_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "status",
        "original",
        "segments",
        "warnings",
        "degraded",
        "lineage",
    ],
    "properties": {
        "status": {"enum": ["succeeded", "failed"]},
        "original": {"type": "object"},
        "segments": {"type": "array", "items": {"type": "object"}},
        "warnings": {"type": "array", "items": {"type": "object"}},
        "degraded": {"type": "boolean"},
        "lineage": {"type": "object"},
        "failure": {"type": ["object", "null"]},
    },
}

VIDEO_SPLIT_SPEC = ToolSpec(
    tool_id="capicove.video.split",
    contract_version=1,
    description="Split a video by duration, keyframes, or detected scenes.",
    input_schema=VIDEO_SPLIT_INPUT_SCHEMA,
    output_schema=VIDEO_SPLIT_OUTPUT_SCHEMA,
    modalities=("video",),
    required_permissions=frozenset({"video:read", "artifact:write", "subprocess:execute"}),
    privacy_class=PrivacyClass.CONFIDENTIAL,
    side_effect=SideEffect.WRITE,
    network_access=False,
    deterministic=True,
    idempotent=True,
    timeout_seconds=300.0,
    redacted_fields=("source.base64",),
    resources=ToolResources(
        filesystem_read=True,
        filesystem_write=True,
        subprocesses=("ffmpeg", "ffprobe"),
    ),
)


class VideoSplitterPort(Protocol):
    """Narrow operation consumed by the Tool handler."""

    def split(self, request: VideoSplitRequest) -> VideoSplitResult: ...


class VideoSplitToolHandler:
    """Validate Tool arguments and map processing outcomes canonically."""

    def __init__(self, splitter: VideoSplitterPort, execution_policy_ref: str) -> None:
        if not execution_policy_ref.strip():
            raise ValueError("execution_policy_ref must not be blank")
        self._splitter = splitter
        self._execution_policy_ref = execution_policy_ref

    def __call__(
        self,
        context: InvocationContext,
        arguments: Mapping[str, object],
        /,
    ) -> ToolResult:
        """Execute a validated canonical Tool invocation."""
        started = datetime.now(UTC)
        missing = VIDEO_SPLIT_SPEC.required_permissions.difference(context.permissions)
        if missing:
            return ToolResult(
                request_id=context.request_id,
                trace_id=context.trace_id,
                tool_id=VIDEO_SPLIT_SPEC.tool_id,
                tool_version=VIDEO_SPLIT_SPEC.contract_version,
                status=ToolStatus.REJECTED,
                error=ToolError(
                    ToolErrorCode.PERMISSION_DENIED,
                    f"Missing required permissions: {', '.join(sorted(missing))}.",
                    retryable=False,
                ),
                started_at=started,
                completed_at=datetime.now(UTC),
            )
        try:
            request = _parse_request(context, arguments, self._execution_policy_ref)
            processing_result = self._splitter.split(request)
        except PathPolicyError:
            return ToolResult(
                request_id=context.request_id,
                trace_id=context.trace_id,
                tool_id=VIDEO_SPLIT_SPEC.tool_id,
                tool_version=VIDEO_SPLIT_SPEC.contract_version,
                status=ToolStatus.REJECTED,
                error=ToolError(
                    ToolErrorCode.PERMISSION_DENIED,
                    "The requested path is outside the execution policy.",
                    retryable=False,
                ),
                started_at=started,
                completed_at=datetime.now(UTC),
            )
        except (ValueError, TypeError, binascii.Error):
            return ToolResult(
                request_id=context.request_id,
                trace_id=context.trace_id,
                tool_id=VIDEO_SPLIT_SPEC.tool_id,
                tool_version=VIDEO_SPLIT_SPEC.contract_version,
                status=ToolStatus.FAILED,
                error=ToolError(
                    ToolErrorCode.INVALID_INPUT,
                    "Video split arguments are invalid.",
                    retryable=False,
                ),
                started_at=started,
                completed_at=datetime.now(UTC),
            )
        result_data = _serialize_result(processing_result)
        evidence_refs = (
            processing_result.original.source_id,
            *(segment.output_id for segment in processing_result.segments),
        )
        if processing_result.status is VideoSplitStatus.SUCCEEDED:
            return ToolResult(
                request_id=context.request_id,
                trace_id=context.trace_id,
                tool_id=VIDEO_SPLIT_SPEC.tool_id,
                tool_version=VIDEO_SPLIT_SPEC.contract_version,
                status=ToolStatus.SUCCEEDED,
                result=result_data,
                evidence_refs=evidence_refs,
                started_at=started,
                completed_at=datetime.now(UTC),
            )
        error_code = _tool_error_code(processing_result.failure_code)
        return ToolResult(
            request_id=context.request_id,
            trace_id=context.trace_id,
            tool_id=VIDEO_SPLIT_SPEC.tool_id,
            tool_version=VIDEO_SPLIT_SPEC.contract_version,
            status=ToolStatus.FAILED,
            result=result_data,
            evidence_refs=evidence_refs,
            error=ToolError(
                error_code,
                processing_result.failure_message or "Video splitting failed.",
                retryable=processing_result.retryable,
            ),
            started_at=started,
            completed_at=datetime.now(UTC),
        )


def register_video_split_tool(
    registry: ToolRegistry,
    splitter: VideoSplitterPort,
    *,
    execution_policy_ref: str,
) -> VideoSplitToolHandler:
    """Register one canonical video split Tool handler."""
    handler = VideoSplitToolHandler(splitter, execution_policy_ref)
    registry.register(VIDEO_SPLIT_SPEC, handler)
    return handler


def build_video_split_tool_handler(
    execution_policy_ref: str,
    read_roots: tuple[Path, ...],
    write_roots: tuple[Path, ...],
    output_root: Path,
) -> VideoSplitToolHandler:
    """Compose default local adapters without import-time availability probes."""
    authorizer = ExplicitPathAuthorizer({execution_policy_ref: (read_roots, write_roots)})
    splitter = VideoSplitter(
        backend_resolver=DefaultVideoBackendResolver(create_pyav_backend),
        output_store=FilesystemOutputStore(output_root),
        path_authorizer=authorizer,
        scene_detector=PySceneDetectSceneDetector(),
    )
    return VideoSplitToolHandler(splitter, execution_policy_ref)


def sanitize_video_split_arguments(arguments: Mapping[str, object]) -> dict[str, object]:
    """Remove inline media content before persisting ToolCall trace arguments."""
    sanitized = dict(arguments)
    source = _require_mapping(arguments.get("source"), "source")
    if source.get("type") == "bytes":
        encoded = _require_string(source.get("base64"), "source.base64")
        payload = base64.b64decode(encoded, validate=True)
        sanitized["source"] = {
            "type": "bytes",
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    else:
        sanitized["source"] = dict(source)
    return sanitized


def _parse_request(
    context: InvocationContext,
    arguments: Mapping[str, object],
    execution_policy_ref: str,
) -> VideoSplitRequest:
    allowed = {
        "source",
        "strategy",
        "config",
        "output_format",
        "output_directory",
        "backend",
        "write_mode",
    }
    if set(arguments) != allowed.intersection(arguments) or not {
        "source",
        "strategy",
        "config",
        "output_format",
        "backend",
    }.issubset(arguments):
        raise ValueError("arguments contain unknown or missing fields")
    strategy = VideoSplitStrategy(_require_string(arguments["strategy"], "strategy"))
    config_values = _require_mapping(arguments["config"], "config")
    config = _parse_config(strategy, config_values)
    output_format = VideoOutputFormat(_require_string(arguments["output_format"], "output_format"))
    output_directory_value = arguments.get("output_directory")
    output_directory = (
        Path(_require_string(output_directory_value, "output_directory"))
        if output_directory_value is not None
        else None
    )
    attempt_id = hashlib.sha256(f"{context.request_id}:1:attempt".encode()).hexdigest()
    operation_id = hashlib.sha256(
        f"{context.request_id}:capicove.video.split:1".encode()
    ).hexdigest()
    return VideoSplitRequest(
        request_id=context.request_id,
        trace_id=context.trace_id,
        attempt_id=attempt_id,
        operation_id=operation_id,
        execution_policy_ref=execution_policy_ref,
        source=_parse_source(_require_mapping(arguments["source"], "source")),
        strategy=strategy,
        config=config,
        output_format=output_format,
        output_directory=output_directory,
        backend=VideoBackend(_require_string(arguments["backend"], "backend")),
        deadline=context.deadline,
        write_mode=VideoWriteMode(
            _require_string(arguments.get("write_mode", "accurate"), "write_mode")
        ),
    )


def _parse_source(values: Mapping[str, object]) -> VideoSource:
    source_type = _require_string(values.get("type"), "source.type")
    if source_type == "path" and set(values) == {"type", "path"}:
        return VideoSource(path=Path(_require_string(values.get("path"), "source.path")))
    if source_type == "bytes" and set(values) == {"type", "base64"}:
        encoded = _require_string(values.get("base64"), "source.base64")
        return VideoSource(data=base64.b64decode(encoded, validate=True))
    if source_type == "artifact" and set(values) == {"type", "ref", "sha256"}:
        return VideoSource(
            artifact_ref=_require_string(values.get("ref"), "source.ref"),
            artifact_sha256=_require_string(values.get("sha256"), "source.sha256"),
        )
    raise ValueError("source representation is invalid")


def _parse_config(
    strategy: VideoSplitStrategy,
    values: Mapping[str, object],
) -> DurationSplitConfig | KeyframeSplitConfig | SceneSplitConfig:
    if strategy is VideoSplitStrategy.DURATION:
        allowed = {"split_duration", "overlap_duration", "min_last_split_duration"}
        if set(values).difference(allowed) or "split_duration" not in values:
            raise ValueError("duration config is invalid")
        return DurationSplitConfig(
            _require_number(values["split_duration"], "split_duration"),
            _require_number(values.get("overlap_duration", 0.0), "overlap_duration"),
            _require_number(
                values.get("min_last_split_duration", 0.0),
                "min_last_split_duration",
            ),
        )
    if strategy is VideoSplitStrategy.KEYFRAME:
        if set(values).difference({"min_segment_duration"}):
            raise ValueError("keyframe config is invalid")
        return KeyframeSplitConfig(
            _require_number(
                values.get("min_segment_duration", 10.0),
                "min_segment_duration",
            )
        )
    allowed = {"detector", "options"}
    if set(values).difference(allowed):
        raise ValueError("scene config is invalid")
    detector = SceneDetectorName(_require_string(values.get("detector", "content"), "detector"))
    options = _require_mapping(values.get("options", {}), "options")
    parsed_options: dict[str, str | int | float | bool] = {}
    for key, value in options.items():
        if not isinstance(key, str) or not isinstance(value, (str, int, float, bool)):
            raise ValueError("scene options contain an invalid value")
        parsed_options[key] = value
    return SceneSplitConfig(detector, parsed_options)


def _serialize_result(result: VideoSplitResult) -> dict[str, object]:
    return {
        "status": result.status.value,
        "original": _serialize_asset(result.original),
        "segments": [_serialize_segment(segment) for segment in result.segments],
        "warnings": [_serialize_warning(warning) for warning in result.warnings],
        "degraded": result.degraded,
        "lineage": {
            "request_id": result.request_id,
            "trace_id": result.trace_id,
            "attempt_id": result.attempt_id,
            "operation_id": result.operation_id,
            "strategy": result.strategy.value,
            "configuration": dict(result.normalized_config),
            "configuration_hash": result.configuration_hash,
            "backend": result.backend,
            "backend_version": result.backend_version,
        },
        "failure": (
            {
                "code": result.failure_code.value,
                "message": result.failure_message,
                "retryable": result.retryable,
            }
            if result.failure_code is not None
            else None
        ),
    }


def _serialize_asset(asset: VideoAsset) -> dict[str, object]:
    serialized: dict[str, object] = {
        "source_id": asset.source_id,
        "sha256": asset.sha256,
        "mime_type": asset.mime_type,
    }
    if asset.path is not None:
        serialized.update({"type": "path", "path": str(asset.path)})
    elif asset.artifact_ref is not None:
        serialized.update({"type": "artifact", "ref": asset.artifact_ref})
    elif asset.data is not None:
        serialized.update({"type": "bytes", "base64": base64.b64encode(asset.data).decode("ascii")})
    return serialized


def _serialize_segment(segment: VideoSegment) -> dict[str, object]:
    serialized: dict[str, object] = {
        "index": segment.index,
        "start_seconds": segment.time_range.start_seconds,
        "end_seconds": segment.time_range.end_seconds,
        "output_id": segment.output_id,
        "sha256": segment.sha256,
        "mime_type": segment.mime_type,
        "source_id": segment.source_id,
        "strategy": segment.strategy.value,
        "operation_id": segment.operation_id,
        "configuration_hash": segment.configuration_hash,
        "backend": segment.backend,
        "backend_version": segment.backend_version,
    }
    if segment.path is not None:
        serialized.update({"type": "path", "path": str(segment.path)})
    elif segment.data is not None:
        serialized.update(
            {"type": "bytes", "base64": base64.b64encode(segment.data).decode("ascii")}
        )
    return serialized


def _serialize_warning(warning: VideoSplitWarning) -> dict[str, object]:
    return {
        "code": warning.code.value,
        "message": warning.message,
        "segment_index": warning.segment_index,
        "start_seconds": (
            warning.time_range.start_seconds if warning.time_range is not None else None
        ),
        "end_seconds": (warning.time_range.end_seconds if warning.time_range is not None else None),
    }


def _tool_error_code(code: VideoSplitFailureCode | None) -> ToolErrorCode:
    return {
        VideoSplitFailureCode.INVALID_INPUT: ToolErrorCode.INVALID_INPUT,
        VideoSplitFailureCode.PERMISSION_DENIED: ToolErrorCode.PERMISSION_DENIED,
        VideoSplitFailureCode.BACKEND_UNAVAILABLE: ToolErrorCode.UNAVAILABLE,
        VideoSplitFailureCode.TIMEOUT: ToolErrorCode.TIMEOUT,
        VideoSplitFailureCode.INTERNAL: ToolErrorCode.INTERNAL,
        None: ToolErrorCode.INTERNAL,
    }[code]


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be an object")
    return value


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _require_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    return float(value)
