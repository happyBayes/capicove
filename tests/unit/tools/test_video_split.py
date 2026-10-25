import base64
import hashlib
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import cast

from capicove.contracts import (
    InvocationContext,
    PrivacyClass,
    SideEffect,
    ToolErrorCode,
    ToolResources,
    ToolStatus,
)
from capicove.processing.video import (
    KeyframeSplitConfig,
    TimeRange,
    VideoAsset,
    VideoSegment,
    VideoSplitFailureCode,
    VideoSplitRequest,
    VideoSplitResult,
    VideoSplitStatus,
    VideoSplitStrategy,
    VideoSplitWarning,
    VideoSplitWarningCode,
    VideoWriteMode,
)
from capicove.processing.video.ports import PathPolicyError
from capicove.tools import ToolRegistry
from capicove.tools.video_split import (
    VIDEO_SPLIT_SPEC,
    VideoSplitToolHandler,
    register_video_split_tool,
    sanitize_video_split_arguments,
)


class StubSplitter:
    def __init__(self, result: VideoSplitResult) -> None:
        self.result = result
        self.requests: list[object] = []

    def split(self, request: object) -> VideoSplitResult:
        self.requests.append(request)
        return self.result


class PolicyRejectedSplitter:
    def split(self, request: object) -> VideoSplitResult:
        raise PathPolicyError("path denied")


def make_context(*, permissions: frozenset[str] | None = None) -> InvocationContext:
    return InvocationContext(
        request_id="request-1",
        trace_id="trace-1",
        workspace_id="workspace-1",
        actor_id="agent-1",
        environment_id="environment-1",
        deadline=datetime.now(UTC) + timedelta(minutes=1),
        permissions=permissions
        or frozenset({"video:read", "artifact:write", "subprocess:execute"}),
        privacy_class=PrivacyClass.CONFIDENTIAL,
    )


def make_arguments() -> dict[str, object]:
    return {
        "source": {"type": "bytes", "base64": base64.b64encode(b"video").decode()},
        "strategy": "duration",
        "config": {
            "split_duration": 5.0,
            "overlap_duration": 0.0,
            "min_last_split_duration": 0.0,
        },
        "output_format": "bytes",
        "backend": "auto",
    }


def make_processing_result(
    *,
    status: VideoSplitStatus = VideoSplitStatus.SUCCEEDED,
    warnings: tuple[VideoSplitWarning, ...] = (),
    degraded: bool = False,
    failure_code: VideoSplitFailureCode | None = None,
) -> VideoSplitResult:
    source_hash = hashlib.sha256(b"video").hexdigest()
    segment_payload = b"segment"
    segment_hash = hashlib.sha256(segment_payload).hexdigest()
    return VideoSplitResult(
        status=status,
        original=VideoAsset(
            source_id=f"sha256:{source_hash}",
            sha256=source_hash,
            mime_type="video/mp4",
            data=b"video",
        ),
        segments=(
            VideoSegment(
                index=0,
                time_range=TimeRange(0.0, 5.0),
                output_id="output-1",
                sha256=segment_hash,
                mime_type="video/mp4",
                source_id=f"sha256:{source_hash}",
                strategy=VideoSplitStrategy.DURATION,
                operation_id="operation-1",
                configuration_hash="c" * 64,
                backend="fake:1",
                backend_version="1",
                data=segment_payload,
            ),
        )
        if status is VideoSplitStatus.SUCCEEDED
        else (),
        warnings=warnings,
        degraded=degraded,
        strategy=VideoSplitStrategy.DURATION,
        normalized_config={"split_duration": 5.0},
        configuration_hash="c" * 64,
        backend="fake:1",
        backend_version="1",
        request_id="request-1",
        trace_id="trace-1",
        attempt_id="attempt-1",
        operation_id="operation-1",
        failure_code=failure_code,
        failure_message="split failed" if failure_code else None,
    )


def test_video_split_spec_declares_the_security_boundary() -> None:
    assert VIDEO_SPLIT_SPEC.tool_id == "capicove.video.split"
    assert VIDEO_SPLIT_SPEC.contract_version == 1
    assert VIDEO_SPLIT_SPEC.modalities == ("video",)
    assert VIDEO_SPLIT_SPEC.required_permissions == frozenset(
        {"video:read", "artifact:write", "subprocess:execute"}
    )
    assert VIDEO_SPLIT_SPEC.side_effect is SideEffect.WRITE
    assert VIDEO_SPLIT_SPEC.privacy_class is PrivacyClass.CONFIDENTIAL
    assert VIDEO_SPLIT_SPEC.network_access is False
    assert VIDEO_SPLIT_SPEC.deterministic is True
    assert VIDEO_SPLIT_SPEC.idempotent is True
    assert VIDEO_SPLIT_SPEC.resources == ToolResources(
        filesystem_read=True,
        filesystem_write=True,
        subprocesses=("ffmpeg", "ffprobe"),
    )
    assert VIDEO_SPLIT_SPEC.input_schema["additionalProperties"] is False
    assert VIDEO_SPLIT_SPEC.output_schema["additionalProperties"] is False


def test_handler_maps_success_and_lineage_to_canonical_result() -> None:
    splitter = StubSplitter(make_processing_result())
    handler = VideoSplitToolHandler(splitter, execution_policy_ref="policy-1")

    result = handler(make_context(), make_arguments())

    assert result.status is ToolStatus.SUCCEEDED
    assert result.evidence_refs[1] == "output-1"
    assert result.result is not None
    assert result.result["degraded"] is False
    segments = result.result["segments"]
    assert isinstance(segments, list)
    assert segments[0]["start_seconds"] == 0.0
    assert segments[0]["configuration_hash"] == "c" * 64
    assert segments[0]["base64"] == base64.b64encode(b"segment").decode()
    assert len(splitter.requests) == 1


def test_handler_accepts_explicit_stream_copy_mode() -> None:
    splitter = StubSplitter(make_processing_result())
    arguments = make_arguments()
    arguments["write_mode"] = "copy"

    result = VideoSplitToolHandler(splitter, execution_policy_ref="policy-1")(
        make_context(),
        arguments,
    )

    assert result.status is ToolStatus.SUCCEEDED
    request = cast(VideoSplitRequest, splitter.requests[0])
    assert request.write_mode is VideoWriteMode.COPY


def test_handler_accepts_keyframe_minimum_segment_duration() -> None:
    splitter = StubSplitter(make_processing_result())
    arguments = make_arguments()
    arguments["strategy"] = "keyframe"
    arguments["config"] = {"min_segment_duration": 6.0}

    result = VideoSplitToolHandler(splitter, execution_policy_ref="policy-1")(
        make_context(),
        arguments,
    )

    assert result.status is ToolStatus.SUCCEEDED
    request = cast(VideoSplitRequest, splitter.requests[0])
    assert isinstance(request.config, KeyframeSplitConfig)
    assert request.config.min_segment_duration == 6.0


def test_handler_preserves_degraded_success_warnings() -> None:
    warning = VideoSplitWarning(
        VideoSplitWarningCode.SEGMENT_FAILED,
        "Segment 1 failed.",
        segment_index=1,
        time_range=TimeRange(5.0, 10.0),
    )
    handler = VideoSplitToolHandler(
        StubSplitter(make_processing_result(warnings=(warning,), degraded=True)),
        execution_policy_ref="policy-1",
    )

    result = handler(make_context(), make_arguments())

    assert result.status is ToolStatus.SUCCEEDED
    assert result.result is not None
    assert result.result["degraded"] is True
    warnings = result.result["warnings"]
    assert isinstance(warnings, list)
    assert warnings[0]["code"] == "segment_failed"


def test_handler_maps_processing_failure_and_retains_original() -> None:
    handler = VideoSplitToolHandler(
        StubSplitter(
            make_processing_result(
                status=VideoSplitStatus.FAILED,
                failure_code=VideoSplitFailureCode.BACKEND_UNAVAILABLE,
                degraded=True,
            )
        ),
        execution_policy_ref="policy-1",
    )

    result = handler(make_context(), make_arguments())

    assert result.status is ToolStatus.FAILED
    assert result.error is not None
    assert result.error.code is ToolErrorCode.UNAVAILABLE
    assert result.result is not None
    original = result.result["original"]
    assert isinstance(original, Mapping)
    assert original["sha256"]


def test_handler_rejects_missing_permission_before_split() -> None:
    splitter = StubSplitter(make_processing_result())
    handler = VideoSplitToolHandler(splitter, execution_policy_ref="policy-1")

    result = handler(make_context(permissions=frozenset({"video:read"})), make_arguments())

    assert result.status is ToolStatus.REJECTED
    assert result.error is not None
    assert result.error.code is ToolErrorCode.PERMISSION_DENIED
    assert splitter.requests == []


def test_handler_rejects_unknown_input_fields() -> None:
    arguments = make_arguments()
    arguments["ffmpeg_flags"] = ["-f", "rawvideo"]
    result = VideoSplitToolHandler(
        StubSplitter(make_processing_result()),
        execution_policy_ref="policy-1",
    )(make_context(), arguments)

    assert result.status is ToolStatus.FAILED
    assert result.error is not None
    assert result.error.code is ToolErrorCode.INVALID_INPUT


def test_handler_maps_runtime_path_policy_rejection() -> None:
    result = VideoSplitToolHandler(
        PolicyRejectedSplitter(),
        execution_policy_ref="policy-1",
    )(make_context(), make_arguments())

    assert result.status is ToolStatus.REJECTED
    assert result.error is not None
    assert result.error.code is ToolErrorCode.PERMISSION_DENIED


def test_trace_sanitization_hashes_inline_video_without_content() -> None:
    sanitized = sanitize_video_split_arguments(make_arguments())
    source = sanitized["source"]

    assert isinstance(source, Mapping)
    assert source["type"] == "bytes"
    assert source["size"] == len(b"video")
    assert source["sha256"] == hashlib.sha256(b"video").hexdigest()
    assert "base64" not in source


def test_registration_uses_the_canonical_registry_key() -> None:
    registry = ToolRegistry()
    handler = register_video_split_tool(
        registry,
        StubSplitter(make_processing_result()),
        execution_policy_ref="policy-1",
    )

    registered = registry.resolve("capicove.video.split", 1)
    assert registered.spec is VIDEO_SPLIT_SPEC
    assert registered.handler is handler
