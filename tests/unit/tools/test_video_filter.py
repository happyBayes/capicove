import base64
import hashlib
import importlib
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from capicove.contracts import (
    InvocationContext,
    PrivacyClass,
    SideEffect,
    ToolErrorCode,
    ToolResources,
    ToolStatus,
)
from capicove.processing.video.filtering import (
    AestheticsCriterion,
    Aggregation,
    CriterionResult,
    CriterionType,
    FrameTextSimilarityCriterion,
    MetadataCriterion,
    MotionCriterion,
    NsfwCriterion,
    OcrAreaCriterion,
    TagCriterion,
    VideoCriterionResult,
    VideoFilterFailure,
    VideoFilterFailureCode,
    VideoFilterItemResult,
    VideoFilterMetadata,
    VideoFilterRequest,
    VideoFilterResult,
    VideoFilterStatus,
    WatermarkCriterion,
)
from capicove.tools import ToolRegistry
from capicove.tools.video_filter import (
    VIDEO_FILTER_SPEC,
    VideoFilterToolHandler,
    build_video_filter_tool_handler,
    register_video_filter_tool,
    sanitize_video_filter_arguments,
)


class StubPipeline:
    def __init__(self, result: VideoFilterResult) -> None:
        self.result = result
        self.requests: list[VideoFilterRequest] = []

    def evaluate(self, request: VideoFilterRequest) -> VideoFilterResult:
        self.requests.append(request)
        return self.result


def _context(*, permissions: frozenset[str] | None = None) -> InvocationContext:
    return InvocationContext(
        request_id="request-1",
        trace_id="trace-1",
        workspace_id="workspace-1",
        actor_id="agent-1",
        environment_id="environment-1",
        deadline=datetime.now(UTC) + timedelta(minutes=1),
        permissions=permissions or frozenset({"video:read", "subprocess:execute"}),
        privacy_class=PrivacyClass.CONFIDENTIAL,
    )


def _arguments() -> dict[str, object]:
    return {
        "sources": [
            {
                "item_id": "item-1",
                "source": {
                    "type": "bytes",
                    "base64": base64.b64encode(b"video").decode("ascii"),
                },
            }
        ],
        "criteria": [
            {
                "type": "metadata",
                "criterion_id": "metadata",
                "min_duration_seconds": 1.0,
                "videos_aggregation": "all",
            }
        ],
        "aggregation": "all",
    }


def _result(
    *,
    status: VideoFilterStatus = VideoFilterStatus.SUCCEEDED,
    keep: bool | None = True,
    failure_code: VideoFilterFailureCode | None = None,
) -> VideoFilterResult:
    metadata = VideoFilterMetadata(2.0, 16, 9, 16, 9, backend="fake")
    item = VideoFilterItemResult(
        item_id="item-1",
        source_id="source-1",
        source_sha256="a" * 64,
        evidence_ref="source-1",
        metadata=metadata,
    )
    observation = VideoCriterionResult(
        item_id="item-1",
        criterion_id="metadata",
        passed=True,
        score=2.0,
        details={"duration_seconds": 2.0},
    )
    criterion = CriterionResult(
        criterion_id="metadata",
        criterion_type=CriterionType.METADATA,
        video_results=(observation,),
        aggregation=Aggregation.ALL,
        passed=True,
    )
    failures = (
        (
            VideoFilterFailure(
                failure_code,
                "backend unavailable",
                item_id="item-1",
                criterion_id="aesthetics",
            ),
        )
        if failure_code is not None
        else ()
    )
    return VideoFilterResult(
        status=status,
        keep=keep,
        item_results=(item,),
        criterion_results=(criterion,),
        warnings=(),
        failures=failures,
        aggregation=Aggregation.ALL,
        request_id="request-1",
        trace_id="trace-1",
        attempt_id="attempt-1",
        operation_id="operation-1",
    )


def test_video_filter_spec_declares_read_only_local_boundary() -> None:
    assert VIDEO_FILTER_SPEC.tool_id == "capicove.video.filter"
    assert VIDEO_FILTER_SPEC.contract_version == 1
    assert VIDEO_FILTER_SPEC.required_permissions == frozenset({"video:read", "subprocess:execute"})
    assert VIDEO_FILTER_SPEC.side_effect is SideEffect.READ
    assert VIDEO_FILTER_SPEC.network_access is False
    assert VIDEO_FILTER_SPEC.resources == ToolResources(
        filesystem_read=True,
        subprocesses=("ffprobe",),
    )
    assert VIDEO_FILTER_SPEC.input_schema["additionalProperties"] is False
    assert VIDEO_FILTER_SPEC.output_schema["additionalProperties"] is False


def test_handler_parses_defaults_and_serializes_completed_diagnostics() -> None:
    pipeline = StubPipeline(_result())

    result = VideoFilterToolHandler(pipeline, "policy-1")(_context(), _arguments())

    assert result.status is ToolStatus.SUCCEEDED
    assert result.result is not None
    assert result.result["keep"] is True
    criteria = cast(list[Mapping[str, object]], result.result["criteria"])
    assert criteria[0]["criterion_type"] == "metadata"
    assert result.evidence_refs == ("source-1",)
    request = pipeline.requests[0]
    assert request.aggregation is Aggregation.ALL
    assert isinstance(request.criteria[0], MetadataCriterion)
    assert request.criteria[0].videos_aggregation is Aggregation.ALL


def test_handler_preserves_false_as_successful_filter_decision() -> None:
    result = VideoFilterToolHandler(StubPipeline(_result(keep=False)), "policy-1")(
        _context(), _arguments()
    )

    assert result.status is ToolStatus.SUCCEEDED
    assert result.result is not None
    assert result.result["keep"] is False


def test_handler_maps_failure_and_keeps_completed_diagnostics() -> None:
    processing_result = _result(
        status=VideoFilterStatus.FAILED,
        keep=None,
        failure_code=VideoFilterFailureCode.BACKEND_UNAVAILABLE,
    )

    result = VideoFilterToolHandler(StubPipeline(processing_result), "policy-1")(
        _context(), _arguments()
    )

    assert result.status is ToolStatus.FAILED
    assert result.error is not None
    assert result.error.code is ToolErrorCode.UNAVAILABLE
    assert result.result is not None
    assert result.result["keep"] is None
    assert len(cast(list[object], result.result["criteria"])) == 1
    failures = cast(list[Mapping[str, object]], result.result["failures"])
    assert failures[0]["code"] == "backend_unavailable"


def test_handler_rejects_missing_permission_before_evaluation() -> None:
    pipeline = StubPipeline(_result())

    result = VideoFilterToolHandler(pipeline, "policy-1")(
        _context(permissions=frozenset({"video:read"})),
        _arguments(),
    )

    assert result.status is ToolStatus.REJECTED
    assert result.error is not None
    assert result.error.code is ToolErrorCode.PERMISSION_DENIED
    assert pipeline.requests == []


def test_handler_rejects_unknown_or_invalid_criterion_fields() -> None:
    arguments = _arguments()
    criterion = cast(list[dict[str, object]], arguments["criteria"])[0]
    criterion["provider_kwargs"] = {"trust_remote_code": True}

    result = VideoFilterToolHandler(StubPipeline(_result()), "policy-1")(_context(), arguments)

    assert result.status is ToolStatus.FAILED
    assert result.error is not None
    assert result.error.code is ToolErrorCode.INVALID_INPUT


def test_trace_sanitization_hashes_every_inline_video() -> None:
    sanitized = sanitize_video_filter_arguments(_arguments())
    sources = cast(list[Mapping[str, object]], sanitized["sources"])
    source = cast(Mapping[str, object], sources[0]["source"])

    assert source["size"] == len(b"video")
    assert source["sha256"] == hashlib.sha256(b"video").hexdigest()
    assert "base64" not in source


def test_registration_uses_canonical_registry_key() -> None:
    registry = ToolRegistry()
    handler = register_video_filter_tool(
        registry,
        StubPipeline(_result()),
        execution_policy_ref="policy-1",
    )

    registered = registry.resolve("capicove.video.filter", 1)
    assert registered.spec is VIDEO_FILTER_SPEC
    assert registered.handler is handler


def test_filter_tool_is_exported_from_public_tools_package() -> None:
    tools = importlib.import_module("capicove.tools")

    assert tools.VIDEO_FILTER_SPEC is VIDEO_FILTER_SPEC
    assert tools.VideoFilterToolHandler is VideoFilterToolHandler


@pytest.mark.parametrize(
    ("criterion", "expected_type"),
    [
        (
            {
                "type": "motion",
                "criterion_id": "motion",
                "sampling": {"frame_count": 4, "max_frames": 8},
                "min_score": 0.1,
                "sampling_fps": 1.5,
                "resize": [320, 180],
                "relative": True,
            },
            MotionCriterion,
        ),
        (
            {
                "type": "ocr_area",
                "criterion_id": "ocr",
                "languages": ["en"],
                "max_area_ratio": 0.4,
                "reduction": "max",
            },
            OcrAreaCriterion,
        ),
        (
            {
                "type": "aesthetics",
                "criterion_id": "aesthetics",
                "model": {"reference": "local/aesthetics", "revision": "v1"},
                "min_score": 0.5,
            },
            AestheticsCriterion,
        ),
        (
            {
                "type": "nsfw",
                "criterion_id": "nsfw",
                "model": {"reference": "local/nsfw"},
                "target_label": "nsfw",
                "max_score": 0.2,
            },
            NsfwCriterion,
        ),
        (
            {
                "type": "watermark",
                "criterion_id": "watermark",
                "model": {"reference": "local/watermark"},
                "target_label": "watermark",
                "max_probability": 0.7,
            },
            WatermarkCriterion,
        ),
        (
            {
                "type": "tags",
                "criterion_id": "tags",
                "model": {"reference": "local/ram-plus.pth"},
                "tags": ["person", "car"],
                "vocabulary_path": "local/tags.txt",
                "contain": "all",
            },
            TagCriterion,
        ),
        (
            {
                "type": "frame_text_similarity",
                "criterion_id": "similarity",
                "model": {
                    "reference": "local/clip",
                    "cache_directory": "local/cache",
                },
                "min_score": 0.2,
                "reduction": "min",
            },
            FrameTextSimilarityCriterion,
        ),
    ],
)
def test_handler_parses_every_supported_non_metadata_criterion(
    criterion: dict[str, object],
    expected_type: type[object],
) -> None:
    arguments = _arguments()
    arguments["criteria"] = [criterion]
    if criterion["type"] == "frame_text_similarity":
        sources = cast(list[dict[str, object]], arguments["sources"])
        sources[0]["text"] = "a person beside a car"
    pipeline = StubPipeline(_result())

    result = VideoFilterToolHandler(pipeline, "policy-1")(_context(), arguments)

    assert result.status is ToolStatus.SUCCEEDED
    assert isinstance(pipeline.requests[0].criteria[0], expected_type)


def test_default_tool_composition_does_not_eagerly_load_optional_providers(
    tmp_path: Path,
) -> None:
    handler = build_video_filter_tool_handler("policy-1", (tmp_path.resolve(),))

    assert isinstance(handler, VideoFilterToolHandler)
