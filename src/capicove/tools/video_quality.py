"""Canonical Tool schema and composition for offline media quality inspection."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from capicove.contracts import (
    InvocationContext,
    PrivacyClass,
    SideEffect,
    ToolError,
    ToolErrorCode,
    ToolHandler,
    ToolResources,
    ToolResult,
    ToolSpec,
    ToolStatus,
)
from capicove.processing.video.quality import (
    CHECKS,
    QualityError,
    QualityLimits,
    QualityRequest,
    VideoQualityPipeline,
)
from capicove.processing.video.quality.video_quality_contracts import (
    DETECTOR_DEFAULTS,
    POLICY_FIELDS,
)
from capicove.processing.video.quality.video_quality_ports import QualityIntake
from capicove.tools.registry import ToolRegistry


def _object(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties) if required is None else required,
    }


def _nullable(kind: str) -> dict[str, Any]:
    return {"type": [kind, "null"]}


_POLICY_PROPERTIES: dict[str, Any] = {
    key: {"type": "boolean"}
    if key == "require_audio"
    else {"type": "array", "items": {"type": "string"}, "minItems": 1}
    if key.startswith("allowed_")
    else {"type": "integer", "minimum": 0}
    if any(s in key for s in ("width", "height", "channels", "sample_rate"))
    else {"type": "number"}
    for key in sorted(POLICY_FIELDS)
}
_OPTIONS_PROPERTIES: dict[str, Any] = {}
for _name, _value in DETECTOR_DEFAULTS.items():
    if isinstance(_value, bool):
        _OPTIONS_PROPERTIES[_name] = {"type": "boolean"}
    elif _name.endswith("min_seconds"):
        _OPTIONS_PROPERTIES[_name] = {"type": "number", "exclusiveMinimum": 0, "maximum": 7200}
    elif _name == "silence_threshold_dbfs":
        _OPTIONS_PROPERTIES[_name] = {"type": "number", "minimum": -120, "maximum": 0}
    else:
        _OPTIONS_PROPERTIES[_name] = {"type": "number", "minimum": 0, "maximum": 1}
for _name, _schema in _POLICY_PROPERTIES.items():
    if _name.endswith("fraction"):
        _schema.update(minimum=0, maximum=1)
    elif _schema.get("type") == "number" and "lufs" not in _name and "dbtp" not in _name:
        _schema["minimum"] = 0
VIDEO_QUALITY_INPUT_SCHEMA = _object(
    {
        "path": {"type": "string", "minLength": 1},
        "checks": {
            "type": "array",
            "items": {"enum": list(CHECKS)},
            "minItems": 1,
            "uniqueItems": True,
        },
        "video_stream_index": {"type": "integer", "minimum": 0},
        "audio_stream_index": {"type": "integer", "minimum": 0},
        "detector_options": _object(_OPTIONS_PROPERTIES, []),
        "policy": _object(_POLICY_PROPERTIES, []),
    },
    ["path"],
)
_METRIC = _object(
    {
        "value": {"type": ["number", "string", "boolean", "null"]},
        "unit": _nullable("string"),
        "reason": _nullable("string"),
    }
)
_EVENT = _object(
    {
        "type": {"enum": ["black", "freeze", "silence"]},
        "stream_index": {"type": "integer"},
        "channel": _nullable("integer"),
        "start_seconds": {"type": "number"},
        "end_seconds": {"type": "number"},
        "duration_seconds": {"type": "number", "minimum": 0},
        "closed_by_eof": {"type": "boolean"},
    }
)
_FINDING = _object(
    {
        "code": {"type": "string"},
        "severity": {"enum": ["info", "warning", "error"]},
        "message": {"type": "string"},
    }
)
_FAILURE = _object({"code": {"type": "string"}, "message": {"type": "string"}})
_CHECK = _object(
    {
        "check_id": {"enum": list(CHECKS)},
        "status": {"enum": ["completed", "not_applicable", "failed", "not_run"]},
        "verdict": {"enum": ["observed", "pass", "review", "fail", "unknown"]},
        "metrics": {"type": "object", "additionalProperties": _METRIC},
        "events": {"type": "array", "items": _EVENT},
        "findings": {"type": "array", "items": _FINDING},
        "unavailable_reason": _nullable("string"),
    }
)
_TRACK_COVERAGE = _object(
    {
        "start_seconds": _nullable("number"),
        "end_seconds": _nullable("number"),
        "decoded_duration_seconds": _nullable("number"),
        "eof": {"type": "boolean"},
        "frames": _nullable("integer"),
        "samples": _nullable("integer"),
    }
)
_TRACK = _object(
    {
        "index": {"type": "integer"},
        "codec_type": _nullable("string"),
        "codec_name": _nullable("string"),
        "disposition": {"type": ["object", "null"], "additionalProperties": {"type": "integer"}},
        "start_time": _nullable("string"),
        "duration": _nullable("string"),
    }
)
VIDEO_QUALITY_OUTPUT_SCHEMA = _object(
    {
        "schema_version": {"const": 1},
        "source": _object(
            {
                "path": {"type": "string"},
                "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "size_bytes": {"type": "integer"},
            }
        ),
        "selection": _object(
            {
                "video_index": {"type": "integer"},
                "audio_index": _nullable("integer"),
                "tracks": {"type": "array", "items": _TRACK},
            }
        ),
        "configuration": _object(
            {
                "checks": VIDEO_QUALITY_INPUT_SCHEMA["properties"]["checks"],
                "detector_options": _object(_OPTIONS_PROPERTIES),
                "policy": _object(_POLICY_PROPERTIES, []),
                "limits": _object(
                    {
                        key: {
                            "type": "integer" if type(value) is int else "number",
                            "exclusiveMinimum": 0,
                        }
                        for key, value in asdict(QualityLimits()).items()
                    }
                ),
                "sha256": {"type": "string"},
            }
        ),
        "coverage": _object(
            {
                "decoded": {"type": "boolean"},
                "origin_seconds": _nullable("number"),
                "passes": {
                    "type": "array",
                    "items": _object(
                        {
                            "backend": {"enum": ["ffprobe", "ffmpeg"]},
                            "purpose": {"type": "string"},
                            "stream_indexes": {"type": "array", "items": {"type": "integer"}},
                            "complete": {"type": "boolean"},
                        }
                    ),
                },
                "video": {"anyOf": [_TRACK_COVERAGE, {"type": "null"}]},
                "audio": {"anyOf": [_TRACK_COVERAGE, {"type": "null"}]},
            }
        ),
        "checks": {
            "type": "array",
            "items": _CHECK,
            "minItems": len(CHECKS),
            "maxItems": len(CHECKS),
        },
        "verdict": {"enum": ["observed", "pass", "review", "fail", "incomplete"]},
        "warnings": {"type": "array", "items": _FAILURE},
        "failures": {"type": "array", "items": _FAILURE},
        "lineage": _object(
            {
                "parser_version": {"const": 1},
                "ffprobe_version": _nullable("string"),
                "ffprobe_sha256": _nullable("string"),
                "ffmpeg_version": _nullable("string"),
                "ffmpeg_sha256": _nullable("string"),
                "tool_id": {"const": "capicove.video.quality"},
                "request_id": {"type": "string"},
                "trace_id": {"type": "string"},
                "operation_id": {"type": "string"},
                "policy_ref": {"type": "string"},
            }
        ),
    }
)
VIDEO_QUALITY_SPEC = ToolSpec(
    tool_id="capicove.video.quality",
    contract_version=1,
    description="Inspect local video media parameters, full decoding, temporal events and loudness.",
    input_schema=VIDEO_QUALITY_INPUT_SCHEMA,
    output_schema=VIDEO_QUALITY_OUTPUT_SCHEMA,
    modalities=("video", "audio"),
    required_permissions=frozenset({"video:read", "subprocess:execute"}),
    privacy_class=PrivacyClass.CONFIDENTIAL,
    side_effect=SideEffect.READ,
    network_access=False,
    deterministic=True,
    idempotent=True,
    timeout_seconds=300,
    resources=ToolResources(filesystem_read=True, subprocesses=("ffmpeg", "ffprobe")),
)


class VideoQualityToolHandler:
    """Enforce canonical validation and permissions before provider work."""

    def __init__(self, pipeline: VideoQualityPipeline, policy_ref: str) -> None:
        if not policy_ref.strip():
            raise ValueError("policy_ref must not be blank")
        self._pipeline, self._policy_ref = pipeline, policy_ref

    def __call__(
        self, context: InvocationContext, arguments: Mapping[str, object], /
    ) -> ToolResult:
        started = datetime.now(UTC)
        error: ToolError | None = None
        report: dict[str, Any] | None = None
        status = ToolStatus.SUCCEEDED
        try:
            if not VIDEO_QUALITY_SPEC.required_permissions.issubset(context.permissions):
                raise QualityError(
                    "permission_denied", "Video read and subprocess permissions are required."
                )
            request = QualityRequest.from_mapping(arguments)
            outcome = self._pipeline.inspect(
                request,
                min(
                    context.deadline,
                    started + timedelta(seconds=VIDEO_QUALITY_SPEC.timeout_seconds),
                ),
            )
            report = outcome.to_dict()
            report["lineage"].update(
                tool_id=VIDEO_QUALITY_SPEC.tool_id,
                request_id=context.request_id,
                trace_id=context.trace_id,
                policy_ref=self._policy_ref,
                operation_id=hashlib.sha256(
                    f"{context.request_id}:capicove.video.quality:1".encode()
                ).hexdigest(),
            )
            if outcome.failure_code:
                raise QualityError(
                    outcome.failure_code,
                    "Video quality inspection is incomplete; inspect failure details.",
                )
        except (QualityError, ValueError, OSError) as failure:
            code = failure.code if isinstance(failure, QualityError) else "invalid_input"
            status = (
                ToolStatus.REJECTED
                if code in {"invalid_input", "permission_denied"}
                else ToolStatus.FAILED
            )
            mapping = {
                "invalid_input": ToolErrorCode.INVALID_INPUT,
                "permission_denied": ToolErrorCode.PERMISSION_DENIED,
                "timeout": ToolErrorCode.TIMEOUT,
                "backend_unavailable": ToolErrorCode.UNAVAILABLE,
                "resource_limit": ToolErrorCode.POLICY_REJECTED,
            }
            error = ToolError(
                mapping.get(code, ToolErrorCode.INTERNAL),
                str(failure)
                if isinstance(failure, (QualityError, ValueError))
                else "Local video inspection failed.",
                False,
            )
        return ToolResult(
            context.request_id,
            context.trace_id,
            VIDEO_QUALITY_SPEC.tool_id,
            1,
            status,
            result=report,
            evidence_refs=(f"sha256:{report['source']['sha256']}",) if report else (),
            error=error,
            started_at=started,
            completed_at=datetime.now(UTC),
        )


def build_video_quality_tool_handler(
    policy_ref: str,
    read_roots: tuple[Path, ...],
    *,
    limits: QualityLimits | None = None,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    intake: QualityIntake | None = None,
) -> VideoQualityToolHandler:
    """Compose local adapters without running processes or loading optional SDKs."""
    from capicove.processing.video.quality._adapters.ffmpeg_quality import FfmpegQualityBackend
    from capicove.processing.video.quality._adapters.local_media import (
        LocalQualityIntake,
        VerifiedQualityIntake,
    )

    limits = limits or QualityLimits()
    return VideoQualityToolHandler(
        VideoQualityPipeline(
            VerifiedQualityIntake(intake, limits)
            if intake is not None
            else LocalQualityIntake(read_roots, limits),
            FfmpegQualityBackend(limits, ffmpeg=ffmpeg, ffprobe=ffprobe),
            limits,
        ),
        policy_ref,
    )


def register_video_quality_tool(registry: ToolRegistry, handler: ToolHandler) -> None:
    """Register the accepted canonical contract without invoking providers."""
    registry.register(VIDEO_QUALITY_SPEC, handler)
