"""Canonical Tool contract and handler for unified local video filtering."""

from __future__ import annotations

import base64
import binascii
import hashlib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

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
from capicove.processing.video import VideoSource
from capicove.processing.video._adapters.filesystem import ExplicitPathAuthorizer
from capicove.processing.video.filtering import (
    AestheticsCriterion,
    Aggregation,
    FrameSamplingConfig,
    FrameTextSimilarityCriterion,
    LocalModelReference,
    MetadataCriterion,
    MotionCriterion,
    NsfwCriterion,
    OcrAreaCriterion,
    Reduction,
    SamplingMethod,
    TagCriterion,
    VideoFilterFailureCode,
    VideoFilterItem,
    VideoFilterRequest,
    VideoFilterResult,
    VideoFilterStatus,
    WatermarkCriterion,
)
from capicove.processing.video.filtering._adapters.ffprobe_video_metadata import (
    FfprobeVideoMetadataProbe,
)
from capicove.processing.video.filtering._adapters.huggingface_frame_scorers import (
    HuggingFaceClassificationEvaluator,
    HuggingFaceClipSimilarityEvaluator,
)
from capicove.processing.video.filtering._adapters.opencv_farneback_motion import (
    OpenCvFarnebackMotionEvaluator,
)
from capicove.processing.video.filtering._adapters.paddleocr_text_area import (
    PaddleOcrTextAreaEvaluator,
)
from capicove.processing.video.filtering._adapters.pyav_frame_sampler import (
    PyavVideoFrameSampler,
)
from capicove.processing.video.filtering._adapters.ram_plus_video_tagger import (
    RamPlusLocalTagCriterionEvaluator,
)
from capicove.processing.video.filtering.video_filter_contracts import VideoFilterCriterion
from capicove.processing.video.filtering.video_filter_pipeline import VideoFilterPipeline
from capicove.tools.registry import ToolRegistry


if TYPE_CHECKING:
    from capicove.model import ModelFactory
    from capicove.processing.image.ocr_configuration import ImageOcrConfiguration


_SOURCE_SCHEMA: dict[str, object] = {
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
                "base64": {"type": "string", "contentEncoding": "base64"},
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
}

VIDEO_FILTER_INPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["sources", "criteria"],
    "properties": {
        "sources": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["source"],
                "properties": {
                    "item_id": {"type": "string"},
                    "source": _SOURCE_SCHEMA,
                    "text": {"type": "string", "maxLength": 8192},
                },
            },
        },
        "criteria": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["type", "criterion_id"],
                "properties": {
                    "type": {
                        "enum": [
                            "metadata",
                            "motion",
                            "ocr_area",
                            "aesthetics",
                            "nsfw",
                            "watermark",
                            "tags",
                            "frame_text_similarity",
                        ]
                    },
                    "criterion_id": {"type": "string"},
                    "videos_aggregation": {"enum": ["all", "any"], "default": "all"},
                },
            },
        },
        "aggregation": {"enum": ["all", "any"], "default": "all"},
    },
}

VIDEO_FILTER_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "status",
        "keep",
        "items",
        "criteria",
        "warnings",
        "failures",
        "aggregation",
        "lineage",
    ],
    "properties": {
        "status": {"enum": ["succeeded", "failed"]},
        "keep": {"type": ["boolean", "null"]},
        "items": {"type": "array", "items": {"type": "object"}},
        "criteria": {"type": "array", "items": {"type": "object"}},
        "warnings": {"type": "array", "items": {"type": "object"}},
        "failures": {"type": "array", "items": {"type": "object"}},
        "aggregation": {"enum": ["all", "any"]},
        "lineage": {"type": "object"},
    },
}

VIDEO_FILTER_SPEC = ToolSpec(
    tool_id="capicove.video.filter",
    contract_version=1,
    description="Evaluate local video quality criteria with one unified filter.",
    input_schema=VIDEO_FILTER_INPUT_SCHEMA,
    output_schema=VIDEO_FILTER_OUTPUT_SCHEMA,
    modalities=("video", "text"),
    required_permissions=frozenset({"video:read", "subprocess:execute"}),
    privacy_class=PrivacyClass.CONFIDENTIAL,
    side_effect=SideEffect.READ,
    network_access=False,
    deterministic=True,
    idempotent=True,
    timeout_seconds=300.0,
    redacted_fields=("sources.*.source.base64",),
    resources=ToolResources(filesystem_read=True, subprocesses=("ffprobe",)),
)


class VideoFilterPipelinePort(Protocol):
    """Narrow operation consumed by the video-filter Tool handler."""

    def evaluate(self, request: VideoFilterRequest) -> VideoFilterResult: ...


class VideoFilterToolHandler:
    """Validate canonical arguments and preserve filter diagnostics."""

    def __init__(self, pipeline: VideoFilterPipelinePort, execution_policy_ref: str) -> None:
        if not execution_policy_ref.strip():
            raise ValueError("execution_policy_ref must not be blank")
        self._pipeline = pipeline
        self._execution_policy_ref = execution_policy_ref

    def __call__(
        self,
        context: InvocationContext,
        arguments: Mapping[str, object],
        /,
    ) -> ToolResult:
        """Execute one validated local video-filter invocation."""
        started = datetime.now(UTC)
        missing = VIDEO_FILTER_SPEC.required_permissions.difference(context.permissions)
        if missing:
            return _tool_failure(
                context,
                started,
                ToolStatus.REJECTED,
                ToolErrorCode.PERMISSION_DENIED,
                f"Missing required permissions: {', '.join(sorted(missing))}.",
            )
        try:
            request = _parse_request(context, arguments, self._execution_policy_ref)
            processing_result = self._pipeline.evaluate(request)
        except (ValueError, TypeError, binascii.Error):
            return _tool_failure(
                context,
                started,
                ToolStatus.FAILED,
                ToolErrorCode.INVALID_INPUT,
                "Video filter arguments are invalid.",
            )

        serialized = _serialize_result(processing_result)
        evidence_refs = tuple(
            dict.fromkeys(item.evidence_ref for item in processing_result.item_results)
        )
        if processing_result.status is VideoFilterStatus.SUCCEEDED:
            return ToolResult(
                request_id=context.request_id,
                trace_id=context.trace_id,
                tool_id=VIDEO_FILTER_SPEC.tool_id,
                tool_version=VIDEO_FILTER_SPEC.contract_version,
                status=ToolStatus.SUCCEEDED,
                result=serialized,
                evidence_refs=evidence_refs,
                started_at=started,
                completed_at=datetime.now(UTC),
            )
        first_failure = processing_result.failures[0]
        return ToolResult(
            request_id=context.request_id,
            trace_id=context.trace_id,
            tool_id=VIDEO_FILTER_SPEC.tool_id,
            tool_version=VIDEO_FILTER_SPEC.contract_version,
            status=ToolStatus.FAILED,
            result=serialized,
            evidence_refs=evidence_refs,
            error=ToolError(
                _tool_error_code(first_failure.code),
                first_failure.message,
                retryable=False,
            ),
            started_at=started,
            completed_at=datetime.now(UTC),
        )


def register_video_filter_tool(
    registry: ToolRegistry,
    pipeline: VideoFilterPipelinePort,
    *,
    execution_policy_ref: str,
) -> VideoFilterToolHandler:
    """Register one canonical video-filter Tool handler."""
    handler = VideoFilterToolHandler(pipeline, execution_policy_ref)
    registry.register(VIDEO_FILTER_SPEC, handler)
    return handler


def build_video_filter_tool_handler(
    execution_policy_ref: str,
    read_roots: tuple[Path, ...],
    *,
    ocr_configuration: ImageOcrConfiguration | None = None,
    model_factory: ModelFactory | None = None,
    ocr_binding: str = "video.ocr",
) -> VideoFilterToolHandler:
    """Compose lazy local adapters without loading optional model packages."""
    classifier = HuggingFaceClassificationEvaluator()
    pipeline = VideoFilterPipeline(
        metadata_probe=FfprobeVideoMetadataProbe(),
        frame_sampler=PyavVideoFrameSampler(),
        evaluators={
            "motion": OpenCvFarnebackMotionEvaluator(),
            "ocr_area": PaddleOcrTextAreaEvaluator(
                ocr_configuration, model_factory=model_factory, binding=ocr_binding
            ),
            "aesthetics": classifier,
            "nsfw": classifier,
            "watermark": classifier,
            "tags": RamPlusLocalTagCriterionEvaluator(),
            "frame_text_similarity": HuggingFaceClipSimilarityEvaluator(),
        },
        path_authorizer=ExplicitPathAuthorizer({execution_policy_ref: (read_roots, ())}),
    )
    return VideoFilterToolHandler(pipeline, execution_policy_ref)


def sanitize_video_filter_arguments(arguments: Mapping[str, object]) -> dict[str, object]:
    """Replace inline media bytes with size and hash in persisted traces."""
    sanitized = dict(arguments)
    source_values = _require_sequence(arguments.get("sources"), "sources")
    sanitized_sources: list[dict[str, object]] = []
    for item_value in source_values:
        item = dict(_require_mapping(item_value, "source item"))
        source = _require_mapping(item.get("source"), "source")
        if source.get("type") == "bytes":
            payload = base64.b64decode(
                _require_string(source.get("base64"), "source.base64"),
                validate=True,
            )
            item["source"] = {
                "type": "bytes",
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        else:
            item["source"] = dict(source)
        sanitized_sources.append(item)
    sanitized["sources"] = sanitized_sources
    return sanitized


def _parse_request(
    context: InvocationContext,
    arguments: Mapping[str, object],
    execution_policy_ref: str,
) -> VideoFilterRequest:
    allowed = {"sources", "criteria", "aggregation"}
    if set(arguments).difference(allowed) or not {"sources", "criteria"}.issubset(arguments):
        raise ValueError("arguments contain unknown or missing fields")
    source_values = _require_sequence(arguments["sources"], "sources")
    criterion_values = _require_sequence(arguments["criteria"], "criteria")
    items = tuple(
        _parse_item(_require_mapping(value, "source item"), index)
        for index, value in enumerate(source_values, start=1)
    )
    criteria = tuple(
        _parse_criterion(_require_mapping(value, "criterion")) for value in criterion_values
    )
    attempt_id = hashlib.sha256(f"{context.request_id}:1:attempt".encode()).hexdigest()
    operation_id = hashlib.sha256(
        f"{context.request_id}:capicove.video.filter:1".encode()
    ).hexdigest()
    return VideoFilterRequest(
        request_id=context.request_id,
        trace_id=context.trace_id,
        attempt_id=attempt_id,
        operation_id=operation_id,
        execution_policy_ref=execution_policy_ref,
        items=items,
        criteria=criteria,
        deadline=context.deadline,
        aggregation=Aggregation(
            _require_string(arguments.get("aggregation", "all"), "aggregation")
        ),
    )


def _parse_item(values: Mapping[str, object], index: int) -> VideoFilterItem:
    allowed = {"item_id", "source", "text"}
    if set(values).difference(allowed) or "source" not in values:
        raise ValueError("source item is invalid")
    item_id = _require_string(values.get("item_id", f"video-{index}"), "item_id")
    text_value = values.get("text")
    text = _require_string(text_value, "text") if text_value is not None else None
    if text is not None and len(text) > 8192:
        raise ValueError("text exceeds the 8192-character limit")
    return VideoFilterItem(
        item_id,
        _parse_source(_require_mapping(values["source"], "source")),
        text=text,
    )


def _parse_source(values: Mapping[str, object]) -> VideoSource:
    source_type = _require_string(values.get("type"), "source.type")
    if source_type == "path" and set(values) == {"type", "path"}:
        return VideoSource(path=Path(_require_string(values["path"], "source.path")))
    if source_type == "bytes" and set(values) == {"type", "base64"}:
        encoded = _require_string(values["base64"], "source.base64")
        return VideoSource(data=base64.b64decode(encoded, validate=True))
    if source_type == "artifact" and set(values) == {"type", "ref", "sha256"}:
        return VideoSource(
            artifact_ref=_require_string(values["ref"], "source.ref"),
            artifact_sha256=_require_string(values["sha256"], "source.sha256"),
        )
    raise ValueError("source representation is invalid")


def _parse_criterion(values: Mapping[str, object]) -> VideoFilterCriterion:
    criterion_type = _require_string(values.get("type"), "criterion.type")
    criterion_id = _require_string(values.get("criterion_id"), "criterion_id")
    videos_aggregation = Aggregation(
        _require_string(values.get("videos_aggregation", "all"), "videos_aggregation")
    )
    common = {"type", "criterion_id", "videos_aggregation"}
    if criterion_type == "metadata":
        fields = {
            "min_duration_seconds",
            "max_duration_seconds",
            "min_width",
            "max_width",
            "min_height",
            "max_height",
            "min_display_width",
            "max_display_width",
            "min_display_height",
            "max_display_height",
            "min_aspect_ratio",
            "max_aspect_ratio",
        }
        _require_allowed(values, common | fields)
        return MetadataCriterion(
            criterion_id,
            min_duration_seconds=_optional_number(values, "min_duration_seconds"),
            max_duration_seconds=_optional_number(values, "max_duration_seconds"),
            min_width=_optional_integer(values, "min_width"),
            max_width=_optional_integer(values, "max_width"),
            min_height=_optional_integer(values, "min_height"),
            max_height=_optional_integer(values, "max_height"),
            min_display_width=_optional_integer(values, "min_display_width"),
            max_display_width=_optional_integer(values, "max_display_width"),
            min_display_height=_optional_integer(values, "min_display_height"),
            max_display_height=_optional_integer(values, "max_display_height"),
            min_aspect_ratio=_optional_number(values, "min_aspect_ratio"),
            max_aspect_ratio=_optional_number(values, "max_aspect_ratio"),
            videos_aggregation=videos_aggregation,
        )

    sampling = _parse_sampling(values.get("sampling", {}))
    if criterion_type == "motion":
        _require_allowed(
            values,
            common | {"sampling", "min_score", "max_score", "sampling_fps", "resize", "relative"},
        )
        return MotionCriterion(
            criterion_id,
            sampling=sampling,
            min_score=_optional_number(values, "min_score", default=0.0),
            max_score=_optional_number(values, "max_score"),
            sampling_fps=_number(values.get("sampling_fps", 2.0), "sampling_fps"),
            resize=_parse_resize(values.get("resize")),
            relative=_boolean(values.get("relative", False), "relative"),
            videos_aggregation=videos_aggregation,
        )
    if criterion_type == "ocr_area":
        _require_allowed(
            values,
            common | {"sampling", "min_area_ratio", "max_area_ratio", "languages", "reduction"},
        )
        return OcrAreaCriterion(
            criterion_id,
            sampling=sampling,
            min_area_ratio=_optional_number(values, "min_area_ratio", default=0.0),
            max_area_ratio=_optional_number(values, "max_area_ratio", default=1.0),
            languages=_string_tuple(values.get("languages", ("ch_sim", "en")), "languages"),
            reduction=_reduction(values),
            videos_aggregation=videos_aggregation,
        )

    model = _parse_model(values.get("model"))
    if criterion_type == "aesthetics":
        _require_allowed(
            values,
            common | {"model", "sampling", "min_score", "max_score", "reduction"},
        )
        return AestheticsCriterion(
            criterion_id,
            model,
            sampling,
            min_score=_optional_number(values, "min_score", default=0.4),
            max_score=_optional_number(values, "max_score", default=1.0),
            reduction=_reduction(values),
            videos_aggregation=videos_aggregation,
        )
    if criterion_type == "nsfw":
        _require_allowed(
            values,
            common | {"model", "sampling", "target_label", "min_score", "max_score", "reduction"},
        )
        return NsfwCriterion(
            criterion_id,
            model,
            sampling,
            target_label=_optional_string(values, "target_label"),
            min_score=_optional_number(values, "min_score", default=0.0),
            max_score=_optional_number(values, "max_score", default=0.5),
            reduction=_reduction(values, Reduction.MAX),
            videos_aggregation=videos_aggregation,
        )
    if criterion_type == "watermark":
        _require_allowed(
            values,
            common | {"model", "sampling", "target_label", "max_probability", "reduction"},
        )
        return WatermarkCriterion(
            criterion_id,
            model,
            sampling,
            target_label=_optional_string(values, "target_label"),
            max_probability=_number(values.get("max_probability", 0.8), "max_probability"),
            reduction=_reduction(values, Reduction.MAX),
            videos_aggregation=videos_aggregation,
        )
    if criterion_type == "tags":
        _require_allowed(
            values,
            common | {"model", "sampling", "tags", "vocabulary_path", "contain"},
        )
        return TagCriterion(
            criterion_id,
            model,
            sampling,
            tags=_string_tuple(values.get("tags"), "tags"),
            vocabulary_path=_require_string(values.get("vocabulary_path"), "vocabulary_path"),
            contain=Aggregation(_require_string(values.get("contain", "any"), "contain")),
            videos_aggregation=videos_aggregation,
        )
    if criterion_type == "frame_text_similarity":
        _require_allowed(
            values,
            common | {"model", "sampling", "min_score", "max_score", "reduction"},
        )
        return FrameTextSimilarityCriterion(
            criterion_id,
            model,
            sampling,
            min_score=_optional_number(values, "min_score", default=0.1),
            max_score=_optional_number(values, "max_score", default=1.0),
            reduction=_reduction(values),
            videos_aggregation=videos_aggregation,
        )
    raise ValueError(f"unsupported criterion type: {criterion_type}")


def _parse_sampling(value: object) -> FrameSamplingConfig:
    values = _require_mapping(value, "sampling")
    _require_allowed(values, {"method", "frame_count", "max_frames"})
    return FrameSamplingConfig(
        method=SamplingMethod(_require_string(values.get("method", "uniform"), "method")),
        frame_count=_integer(values.get("frame_count", 3), "frame_count"),
        max_frames=_integer(values.get("max_frames", 64), "max_frames"),
    )


def _parse_model(value: object) -> LocalModelReference:
    values = _require_mapping(value, "model")
    _require_allowed(values, {"reference", "revision", "cache_directory"})
    return LocalModelReference(
        reference=_require_string(values.get("reference"), "model.reference"),
        revision=_optional_string(values, "revision"),
        cache_directory=_optional_string(values, "cache_directory"),
    )


def _serialize_result(result: VideoFilterResult) -> dict[str, object]:
    return {
        "status": result.status.value,
        "keep": result.keep,
        "items": [
            {
                "item_id": item.item_id,
                "source_id": item.source_id,
                "source_sha256": item.source_sha256,
                "evidence_ref": item.evidence_ref,
                "metadata": (
                    {
                        "duration_seconds": item.metadata.duration_seconds,
                        "encoded_width": item.metadata.encoded_width,
                        "encoded_height": item.metadata.encoded_height,
                        "display_width": item.metadata.display_width,
                        "display_height": item.metadata.display_height,
                        "rotation_degrees": item.metadata.rotation_degrees,
                        "sample_aspect_ratio": item.metadata.sample_aspect_ratio,
                        "display_aspect_ratio": item.metadata.display_aspect_ratio,
                        "aspect_ratio": item.metadata.aspect_ratio,
                        "backend": item.metadata.backend,
                        "backend_version": item.metadata.backend_version,
                    }
                    if item.metadata is not None
                    else None
                ),
            }
            for item in result.item_results
        ],
        "criteria": [
            {
                "criterion_id": criterion.criterion_id,
                "criterion_type": criterion.criterion_type.value,
                "aggregation": criterion.aggregation.value,
                "passed": criterion.passed,
                "video_results": [
                    {
                        "item_id": video.item_id,
                        "criterion_id": video.criterion_id,
                        "passed": video.passed,
                        "score": video.score,
                        "frame_metrics": [
                            {
                                "frame_index": metric.frame_index,
                                "timestamp_seconds": metric.timestamp_seconds,
                                "value": metric.value,
                                "raw_value": metric.raw_value,
                            }
                            for metric in video.frame_metrics
                        ],
                        "tags": list(video.tags),
                        "details": dict(video.details),
                    }
                    for video in criterion.video_results
                ],
            }
            for criterion in result.criterion_results
        ],
        "warnings": [
            {
                "code": warning.code.value,
                "message": warning.message,
                "item_id": warning.item_id,
                "criterion_id": warning.criterion_id,
            }
            for warning in result.warnings
        ],
        "failures": [
            {
                "code": failure.code.value,
                "message": failure.message,
                "item_id": failure.item_id,
                "criterion_id": failure.criterion_id,
            }
            for failure in result.failures
        ],
        "aggregation": result.aggregation.value,
        "lineage": {
            "request_id": result.request_id,
            "trace_id": result.trace_id,
            "attempt_id": result.attempt_id,
            "operation_id": result.operation_id,
        },
    }


def _tool_failure(
    context: InvocationContext,
    started: datetime,
    status: ToolStatus,
    code: ToolErrorCode,
    message: str,
) -> ToolResult:
    return ToolResult(
        request_id=context.request_id,
        trace_id=context.trace_id,
        tool_id=VIDEO_FILTER_SPEC.tool_id,
        tool_version=VIDEO_FILTER_SPEC.contract_version,
        status=status,
        error=ToolError(code, message, retryable=False),
        started_at=started,
        completed_at=datetime.now(UTC),
    )


def _tool_error_code(code: VideoFilterFailureCode) -> ToolErrorCode:
    return {
        VideoFilterFailureCode.INVALID_INPUT: ToolErrorCode.INVALID_INPUT,
        VideoFilterFailureCode.PERMISSION_DENIED: ToolErrorCode.PERMISSION_DENIED,
        VideoFilterFailureCode.BACKEND_UNAVAILABLE: ToolErrorCode.UNAVAILABLE,
        VideoFilterFailureCode.TIMEOUT: ToolErrorCode.TIMEOUT,
        VideoFilterFailureCode.SOURCE_CHANGED: ToolErrorCode.INVALID_INPUT,
        VideoFilterFailureCode.UNSUPPORTED_MEDIA: ToolErrorCode.INVALID_INPUT,
        VideoFilterFailureCode.MODEL_INCOMPATIBLE: ToolErrorCode.INVALID_INPUT,
        VideoFilterFailureCode.DECODE_FAILED: ToolErrorCode.INVALID_INPUT,
        VideoFilterFailureCode.INTERNAL: ToolErrorCode.INTERNAL,
    }[code]


def _require_allowed(values: Mapping[str, object], allowed: set[str]) -> None:
    if set(values).difference(allowed):
        raise ValueError("object contains unknown fields")


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be an object")
    return value


def _require_sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise TypeError(f"{name} must be a non-empty array")
    return value


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _optional_string(values: Mapping[str, object], key: str) -> str | None:
    value = values.get(key)
    return _require_string(value, key) if value is not None else None


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    return float(value)


def _optional_number(
    values: Mapping[str, object],
    key: str,
    *,
    default: float | None = None,
) -> float | None:
    value = values.get(key, default)
    return _number(value, key) if value is not None else None


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _optional_integer(values: Mapping[str, object], key: str) -> int | None:
    value = values.get(key)
    return _integer(value, key) if value is not None else None


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    values = _require_sequence(value, name)
    return tuple(_require_string(item, name) for item in values)


def _parse_resize(value: object) -> tuple[int, int] | None:
    if value is None:
        return None
    values = _require_sequence(value, "resize")
    if len(values) != 2:
        raise ValueError("resize must contain width and height")
    return _integer(values[0], "resize width"), _integer(values[1], "resize height")


def _reduction(
    values: Mapping[str, object],
    default: Reduction = Reduction.AVG,
) -> Reduction:
    return Reduction(_require_string(values.get("reduction", default.value), "reduction"))
