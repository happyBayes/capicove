"""Tests for canonical image and video Tool batch adapters."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import pytest

from capicove.batch import (
    BatchItem,
    BatchModality,
    BatchOperationError,
    ImageToolBatchOperation,
    SourceSnapshot,
    VideoFilterBatchOperation,
    VideoSplitBatchOperation,
)
from capicove.batch.planner import BatchPlanner
from capicove.contracts import (
    InvocationContext,
    ToolError,
    ToolErrorCode,
    ToolHandler,
    ToolResult,
    ToolSpec,
    ToolStatus,
)
from capicove.tools.image import IMAGE_TOOL_SPECS_BY_NAME
from capicove.tools.video_filter import VIDEO_FILTER_SPEC
from capicove.tools.video_split import VIDEO_SPLIT_SPEC


_NOW = datetime(2026, 8, 19, 12, tzinfo=UTC)
_PROVIDER_SHA = "b" * 64


class _Handler:
    def __init__(
        self,
        *,
        status: ToolStatus = ToolStatus.SUCCEEDED,
        error: ToolError | None = None,
        result: Mapping[str, object] | None = None,
    ) -> None:
        self.status = status
        self.error = error
        self.result = result if result is not None else {"lineage": {"provider": "local"}}
        self.calls: list[tuple[InvocationContext, Mapping[str, object]]] = []

    def __call__(
        self,
        context: InvocationContext,
        arguments: Mapping[str, object],
        /,
    ) -> ToolResult:
        self.calls.append((context, arguments))
        spec = _spec_for_context(context, arguments)
        return ToolResult(
            request_id=context.request_id,
            trace_id=context.trace_id,
            tool_id=spec.tool_id,
            tool_version=spec.contract_version,
            status=self.status,
            result=self.result if self.status is ToolStatus.SUCCEEDED else None,
            evidence_refs=("sha256:" + "c" * 64,),
            error=self.error,
            metrics={"duration_seconds": 0.25},
            started_at=_NOW,
            completed_at=_NOW,
        )


def _spec_for_context(
    context: InvocationContext,
    arguments: Mapping[str, object],
) -> ToolSpec:
    del context
    if "criteria" in arguments:
        return VIDEO_FILTER_SPEC
    if "strategy" in arguments:
        return VIDEO_SPLIT_SPEC
    return IMAGE_TOOL_SPECS_BY_NAME["image_shape_filter"]


def _item(tmp_path: Path, modality: BatchModality, name: str) -> BatchItem:
    source = (tmp_path / name).resolve()
    source.write_bytes(b"media")
    return BatchItem(
        item_id="item:" + "a" * 64,
        modality=modality,
        source_path=source,
        relative_path=PurePosixPath(name),
        media_type="image/png" if modality is BatchModality.IMAGE else "video/mp4",
        sha256="c" * 64,
        snapshot=SourceSnapshot(1, 2, 5, 3, 4),
    )


def _output_digest(operation: VideoSplitBatchOperation, item: BatchItem) -> str:
    digest = hashlib.sha256()
    for value in (
        "capicove.batch.tool-identity.v1",
        "output",
        operation.fingerprint.operation_id,
        operation.fingerprint.operation_version,
        operation.fingerprint.parameters_sha256,
        operation.fingerprint.provider_fingerprint,
        item.item_id,
        item.sha256,
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def test_image_adapter_invokes_one_source_tool_and_preserves_canonical_result(
    tmp_path: Path,
) -> None:
    handler = _Handler(result={"applicable": True, "lineage": {"operation_version": 1}})
    operation = ImageToolBatchOperation(
        "image_shape_filter",
        handler,
        operation_parameters={"minimum_width": 128},
        provider_fingerprint=_PROVIDER_SHA,
        clock=lambda: _NOW,
    )
    item = _item(tmp_path, BatchModality.IMAGE, "frame.png")

    encoded = operation.execute(item)

    context, arguments = handler.calls[0]
    assert arguments == {"sources": [{"path": str(item.source_path)}]}
    assert (
        context.permissions == IMAGE_TOOL_SPECS_BY_NAME["image_shape_filter"].required_permissions
    )
    assert context.request_id == operation.request_id_for(item)
    assert context.trace_id == operation.trace_id_for(item)
    assert encoded == {
        "source": {
            "item_id": item.item_id,
            "relative_path": "frame.png",
            "media_type": "image/png",
            "size_bytes": 5,
            "sha256": "c" * 64,
        },
        "tool": {
            "request_id": context.request_id,
            "trace_id": context.trace_id,
            "tool_id": "capicove.image.filter.shape",
            "tool_version": 1,
            "status": "succeeded",
            "result": {"applicable": True, "lineage": {"operation_version": 1}},
            "evidence_refs": ["sha256:" + "c" * 64],
            "error": None,
            "metrics": {"duration_seconds": 0.25},
            "started_at": "2026-08-19T12:00:00.000000Z",
            "completed_at": "2026-08-19T12:00:00.000000Z",
        },
    }


def test_operation_fingerprint_changes_with_arguments_and_provider() -> None:
    handler = _Handler()
    first = ImageToolBatchOperation(
        "image_shape_filter",
        handler,
        operation_parameters={"height": 256, "width": 512},
        provider_fingerprint="a" * 64,
        clock=lambda: _NOW,
    )
    reordered = ImageToolBatchOperation(
        "image_shape_filter",
        handler,
        operation_parameters={"width": 512, "height": 256},
        provider_fingerprint="a" * 64,
        clock=lambda: _NOW,
    )
    changed_arguments = ImageToolBatchOperation(
        "image_shape_filter",
        handler,
        operation_parameters={"height": 257, "width": 512},
        provider_fingerprint="a" * 64,
        clock=lambda: _NOW,
    )
    changed_provider = ImageToolBatchOperation(
        "image_shape_filter",
        handler,
        operation_parameters={"height": 256, "width": 512},
        provider_fingerprint="d" * 64,
        clock=lambda: _NOW,
    )

    assert first.fingerprint == reordered.fingerprint
    assert first.fingerprint.parameters_sha256 != changed_arguments.fingerprint.parameters_sha256
    assert (
        first.fingerprint.provider_fingerprint != changed_provider.fingerprint.provider_fingerprint
    )


@pytest.mark.parametrize(
    "tool_name",
    [
        "image_pair_similarity_filter",
        "image_phash",
        "image_lsh_search",
    ],
)
def test_image_adapter_rejects_multi_item_tools(tool_name: str) -> None:
    with pytest.raises(ValueError, match="single-item"):
        ImageToolBatchOperation(
            tool_name,
            _Handler(),
            provider_fingerprint=_PROVIDER_SHA,
            clock=lambda: _NOW,
        )


def test_video_filter_adapter_builds_one_item_request(tmp_path: Path) -> None:
    handler = _Handler(result={"keep": True, "lineage": {"backend": "local"}})
    criteria = (
        {
            "type": "metadata",
            "criterion_id": "duration",
            "min_duration_seconds": 1,
        },
    )
    operation = VideoFilterBatchOperation(
        handler,
        criteria=criteria,
        aggregation="all",
        provider_fingerprint=_PROVIDER_SHA,
        clock=lambda: _NOW,
    )
    item = _item(tmp_path, BatchModality.VIDEO, "clip.mp4")

    operation.execute(item)

    _, arguments = handler.calls[0]
    assert arguments == {
        "sources": [
            {
                "item_id": item.item_id,
                "source": {"type": "path", "path": str(item.source_path)},
            }
        ],
        "criteria": [
            {
                "type": "metadata",
                "criterion_id": "duration",
                "min_duration_seconds": 1,
            }
        ],
        "aggregation": "all",
    }


def test_video_split_adapter_uses_isolated_content_stable_output_directory(
    tmp_path: Path,
) -> None:
    handler = _Handler(result={"segments": [], "lineage": {"backend": "local"}})
    output_root = (tmp_path / "segments").resolve()
    operation = VideoSplitBatchOperation(
        handler,
        strategy="duration",
        config={"split_duration": 30.0},
        output_format="path",
        output_root=output_root,
        backend="ffmpeg",
        write_mode="copy",
        provider_fingerprint=_PROVIDER_SHA,
        clock=lambda: _NOW,
    )
    item = _item(tmp_path, BatchModality.VIDEO, "movie.mp4")

    operation.execute(item)

    _, arguments = handler.calls[0]
    assert arguments == {
        "source": {"type": "path", "path": str(item.source_path)},
        "strategy": "duration",
        "config": {"split_duration": 30.0},
        "output_format": "path",
        "output_directory": str(output_root / _output_digest(operation, item)),
        "backend": "ffmpeg",
        "write_mode": "copy",
    }
    assert operation.fingerprint.modalities == frozenset({BatchModality.VIDEO})


def test_tool_adapter_rebinds_content_cache_source_identity(tmp_path: Path) -> None:
    operation = ImageToolBatchOperation(
        "image_shape_filter",
        _Handler(),
        provider_fingerprint=_PROVIDER_SHA,
        clock=lambda: _NOW,
    )
    first = _item(tmp_path, BatchModality.IMAGE, "first.png")
    second = replace(
        _item(tmp_path, BatchModality.IMAGE, "second.png"),
        item_id="item:" + "d" * 64,
    )
    cached = operation.execute(first)

    rebound = operation.rebind_cached_result(second, cached)

    assert rebound["source"] == {
        "item_id": second.item_id,
        "relative_path": "second.png",
        "media_type": second.media_type,
        "size_bytes": second.size_bytes,
        "sha256": second.sha256,
    }
    assert rebound["tool"] == cached["tool"]


def test_video_filter_adapter_rebinds_nested_content_cache_item_identity(
    tmp_path: Path,
) -> None:
    first = _item(tmp_path, BatchModality.VIDEO, "first.mp4")
    second = replace(
        _item(tmp_path, BatchModality.VIDEO, "second.mp4"),
        item_id="item:" + "d" * 64,
    )
    result = {
        "items": [{"item_id": first.item_id, "keep": True}],
        "criteria": [
            {
                "criterion_id": "duration",
                "video_results": [{"item_id": first.item_id, "passed": True}],
            }
        ],
        "warnings": [{"item_id": first.item_id, "code": "example"}],
        "failures": [{"item_id": first.item_id, "code": "example"}],
    }
    operation = VideoFilterBatchOperation(
        _Handler(result=result),
        criteria=({"type": "metadata", "criterion_id": "duration"},),
        provider_fingerprint=_PROVIDER_SHA,
        clock=lambda: _NOW,
    )
    cached = operation.execute(first)

    rebound = operation.rebind_cached_result(second, cached)

    tool = rebound["tool"]
    assert isinstance(tool, Mapping)
    rebound_result = tool["result"]
    assert isinstance(rebound_result, Mapping)
    assert rebound_result["items"][0]["item_id"] == second.item_id
    assert rebound_result["criteria"][0]["video_results"][0]["item_id"] == second.item_id
    assert rebound_result["warnings"][0]["item_id"] == second.item_id
    assert rebound_result["failures"][0]["item_id"] == second.item_id
    assert cached["tool"]["result"]["items"][0]["item_id"] == first.item_id  # type: ignore[index]


def test_video_split_path_cache_is_item_scoped(tmp_path: Path) -> None:
    operation = VideoSplitBatchOperation(
        _Handler(),
        strategy="duration",
        config={"split_duration": 30.0},
        output_format="path",
        output_root=(tmp_path / "segments").resolve(),
        provider_fingerprint=_PROVIDER_SHA,
        clock=lambda: _NOW,
    )
    first = _item(tmp_path, BatchModality.VIDEO, "first.mp4")
    second = replace(
        _item(tmp_path, BatchModality.VIDEO, "second.mp4"),
        item_id="item:" + "d" * 64,
    )

    plan = BatchPlanner().plan((first, second), operation.fingerprint)

    assert plan.items[0].cache_key != plan.items[1].cache_key


def test_video_split_rejects_missing_or_modified_cached_path_artifacts(
    tmp_path: Path,
) -> None:
    output_root = (tmp_path / "segments").resolve()
    operation = VideoSplitBatchOperation(
        _Handler(),
        strategy="duration",
        config={"split_duration": 30.0},
        output_format="path",
        output_root=output_root,
        provider_fingerprint=_PROVIDER_SHA,
        clock=lambda: _NOW,
    )
    item = _item(tmp_path, BatchModality.VIDEO, "movie.mp4")
    segment = output_root / _output_digest(operation, item) / "segment-0.mp4"
    segment.parent.mkdir(parents=True)
    segment.write_bytes(b"segment")
    cached = {
        "source": {"item_id": item.item_id},
        "tool": {
            "result": {
                "segments": [
                    {
                        "type": "path",
                        "path": str(segment),
                        "sha256": hashlib.sha256(b"segment").hexdigest(),
                    }
                ]
            }
        },
    }

    assert operation.is_cached_result_valid(item, cached)
    segment.write_bytes(b"modified")
    assert not operation.is_cached_result_valid(item, cached)
    segment.unlink()
    assert not operation.is_cached_result_valid(item, cached)


def test_video_split_rejects_cached_path_parent_traversal(tmp_path: Path) -> None:
    output_root = (tmp_path / "segments").resolve()
    operation = VideoSplitBatchOperation(
        _Handler(),
        strategy="duration",
        config={"split_duration": 30.0},
        output_format="path",
        output_root=output_root,
        provider_fingerprint=_PROVIDER_SHA,
        clock=lambda: _NOW,
    )
    item = _item(tmp_path, BatchModality.VIDEO, "movie.mp4")
    expected_directory = output_root / _output_digest(operation, item)
    expected_directory.mkdir(parents=True)
    outside = output_root / "outside.mp4"
    outside.write_bytes(b"outside")
    escaped = expected_directory / "nested" / ".." / ".." / outside.name
    cached = {
        "tool": {
            "result": {
                "segments": [
                    {
                        "type": "path",
                        "path": str(escaped),
                        "sha256": hashlib.sha256(b"outside").hexdigest(),
                    }
                ]
            }
        }
    }

    assert not operation.is_cached_result_valid(item, cached)


def test_video_split_rejects_cached_path_through_symlink_directory(
    tmp_path: Path,
) -> None:
    output_root = (tmp_path / "segments").resolve()
    operation = VideoSplitBatchOperation(
        _Handler(),
        strategy="duration",
        config={"split_duration": 30.0},
        output_format="path",
        output_root=output_root,
        provider_fingerprint=_PROVIDER_SHA,
        clock=lambda: _NOW,
    )
    item = _item(tmp_path, BatchModality.VIDEO, "movie.mp4")
    expected_directory = output_root / _output_digest(operation, item)
    expected_directory.mkdir(parents=True)
    outside_directory = output_root / "outside"
    outside_directory.mkdir()
    outside = outside_directory / "segment.mp4"
    outside.write_bytes(b"outside")
    linked_directory = expected_directory / "linked"
    linked_directory.symlink_to(outside_directory, target_is_directory=True)
    cached = {
        "tool": {
            "result": {
                "segments": [
                    {
                        "type": "path",
                        "path": str(linked_directory / outside.name),
                        "sha256": hashlib.sha256(b"outside").hexdigest(),
                    }
                ]
            }
        }
    }

    assert not operation.is_cached_result_valid(item, cached)


@pytest.mark.parametrize("status", [ToolStatus.FAILED, ToolStatus.REJECTED])
def test_tool_failure_maps_to_batch_retry_classification(
    tmp_path: Path,
    status: ToolStatus,
) -> None:
    handler = _Handler(
        status=status,
        error=ToolError(ToolErrorCode.RETRYABLE, "provider busy", retryable=True),
    )
    operation = ImageToolBatchOperation(
        "image_shape_filter",
        handler,
        provider_fingerprint=_PROVIDER_SHA,
        clock=lambda: _NOW,
    )

    with pytest.raises(BatchOperationError) as captured:
        operation.execute(_item(tmp_path, BatchModality.IMAGE, "frame.png"))

    assert captured.value.failure.code == "retryable"
    assert captured.value.failure.message == "provider busy"
    assert captured.value.failure.retryable is (status is ToolStatus.FAILED)


def test_adapter_rejects_tool_result_identity_mismatch(tmp_path: Path) -> None:
    class WrongIdentityHandler:
        def __call__(
            self,
            context: InvocationContext,
            arguments: Mapping[str, object],
            /,
        ) -> ToolResult:
            del arguments
            return ToolResult(
                request_id="wrong-request",
                trace_id=context.trace_id,
                tool_id="capicove.image.filter.shape",
                tool_version=1,
                status=ToolStatus.SUCCEEDED,
                result={},
                started_at=_NOW,
                completed_at=_NOW,
            )

    operation = ImageToolBatchOperation(
        "image_shape_filter",
        WrongIdentityHandler(),
        provider_fingerprint=_PROVIDER_SHA,
        clock=lambda: _NOW,
    )

    with pytest.raises(BatchOperationError) as captured:
        operation.execute(_item(tmp_path, BatchModality.IMAGE, "frame.png"))

    assert captured.value.failure.code == "PROTOCOL_ERROR"
    assert not captured.value.failure.retryable


def test_adapter_rejects_wrong_item_type_modality_and_clock(tmp_path: Path) -> None:
    operation = ImageToolBatchOperation(
        "image_shape_filter",
        _Handler(),
        provider_fingerprint=_PROVIDER_SHA,
        clock=lambda: _NOW,
    )
    with pytest.raises(ValueError, match="item must be"):
        operation.execute("not-an-item")  # type: ignore[arg-type]
    with pytest.raises(BatchOperationError) as wrong_modality:
        operation.execute(_item(tmp_path, BatchModality.VIDEO, "movie.mp4"))
    assert wrong_modality.value.failure.code == "UNSUPPORTED_MODALITY"

    invalid_clock = ImageToolBatchOperation(
        "image_shape_filter",
        _Handler(),
        provider_fingerprint=_PROVIDER_SHA,
        clock=lambda: datetime(2026, 8, 19, 12),
    )
    with pytest.raises(BatchOperationError) as captured:
        invalid_clock.execute(_item(tmp_path, BatchModality.IMAGE, "frame.png"))
    assert captured.value.failure.code == "INVALID_CLOCK"


def test_adapter_rejects_non_tool_result(tmp_path: Path) -> None:
    def invalid_handler(
        context: InvocationContext,
        arguments: Mapping[str, object],
        /,
    ) -> object:
        del context, arguments
        return object()

    operation = ImageToolBatchOperation(
        "image_shape_filter",
        invalid_handler,  # type: ignore[arg-type]
        provider_fingerprint=_PROVIDER_SHA,
        clock=lambda: _NOW,
    )

    with pytest.raises(BatchOperationError) as captured:
        operation.execute(_item(tmp_path, BatchModality.IMAGE, "frame.png"))
    assert captured.value.failure.code == "PROTOCOL_ERROR"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"sources": []}, "must not override sources"),
        ({"value": float("nan")}, "finite"),
        ({"value": Path("not-json")}, "JSON-compatible"),
        ({1: "not-a-string-key"}, "keys must be strings"),
    ],
)
def test_image_adapter_rejects_unsafe_or_non_json_arguments(
    arguments: Mapping[object, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ImageToolBatchOperation(
            "image_shape_filter",
            _Handler(),
            arguments=arguments,  # type: ignore[arg-type]
            provider_fingerprint=_PROVIDER_SHA,
            clock=lambda: _NOW,
        )


def test_image_adapter_rejects_unknown_tool_and_invalid_identity_configuration() -> None:
    with pytest.raises(ValueError, match="unknown image Tool"):
        ImageToolBatchOperation(
            "unknown",
            _Handler(),
            provider_fingerprint=_PROVIDER_SHA,
            clock=lambda: _NOW,
        )
    with pytest.raises(ValueError, match="provider_fingerprint"):
        ImageToolBatchOperation(
            "image_shape_filter",
            _Handler(),
            provider_fingerprint="not-a-hash",
            clock=lambda: _NOW,
        )
    with pytest.raises(ValueError, match="workspace_id"):
        ImageToolBatchOperation(
            "image_shape_filter",
            _Handler(),
            provider_fingerprint=_PROVIDER_SHA,
            workspace_id=" ",
            clock=lambda: _NOW,
        )


def test_image_adapter_rejects_excessively_nested_parameters() -> None:
    nested: object = None
    for _ in range(34):
        nested = [nested]

    with pytest.raises(ValueError, match="maximum depth"):
        ImageToolBatchOperation(
            "image_shape_filter",
            _Handler(),
            operation_parameters={"nested": nested},
            provider_fingerprint=_PROVIDER_SHA,
            clock=lambda: _NOW,
        )


@pytest.mark.parametrize(
    ("criteria", "aggregation", "message"),
    [
        ((), "all", "must not be empty"),
        (({"type": "metadata"},), "neither", "all or any"),
    ],
)
def test_video_filter_adapter_rejects_invalid_configuration(
    criteria: tuple[Mapping[str, object], ...],
    aggregation: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        VideoFilterBatchOperation(
            _Handler(),
            criteria=criteria,
            aggregation=aggregation,
            provider_fingerprint=_PROVIDER_SHA,
            clock=lambda: _NOW,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"strategy": "unknown"}, "strategy"),
        ({"output_format": "unknown"}, "output_format"),
        ({"backend": "unknown"}, "backend"),
        ({"write_mode": "unknown"}, "write_mode"),
        ({"output_root": None}, "absolute output_root"),
        ({"output_root": Path("relative")}, "absolute output_root"),
        ({"output_format": "bytes", "output_root": Path("/tmp/output")}, "must not define"),
    ],
)
def test_video_split_adapter_rejects_invalid_configuration(
    overrides: Mapping[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "strategy": "duration",
        "config": {"split_duration": 30.0},
        "output_format": "path",
        "output_root": Path("/tmp/output"),
        "backend": "auto",
        "write_mode": "accurate",
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        VideoSplitBatchOperation(
            _Handler(),
            provider_fingerprint=_PROVIDER_SHA,
            clock=lambda: _NOW,
            **values,  # type: ignore[arg-type]
        )


def test_video_split_bytes_mode_omits_output_directory(tmp_path: Path) -> None:
    handler = _Handler(result={"segments": [], "lineage": {"backend": "local"}})
    operation = VideoSplitBatchOperation(
        handler,
        strategy="keyframe",
        config={"min_segment_duration": 5.0},
        output_format="bytes",
        backend="av",
        provider_fingerprint=_PROVIDER_SHA,
        clock=lambda: _NOW,
    )

    operation.execute(_item(tmp_path, BatchModality.VIDEO, "movie.mp4"))

    assert "output_directory" not in handler.calls[0][1]


def test_adapter_invoker_observes_real_attempt_and_attempt_scoped_request_id(
    tmp_path: Path,
) -> None:
    handler = _Handler()
    observed: list[tuple[ToolSpec, int, str]] = []

    def invoker(
        spec: ToolSpec,
        wrapped: ToolHandler,
        context: InvocationContext,
        arguments: Mapping[str, object],
        attempt: int,
        /,
    ) -> ToolResult:
        observed.append((spec, attempt, context.request_id))
        return wrapped(context, arguments)

    operation = ImageToolBatchOperation(
        "image_shape_filter",
        handler,
        provider_fingerprint=_PROVIDER_SHA,
        invoker=invoker,
        clock=lambda: _NOW,
    )
    item = _item(tmp_path, BatchModality.IMAGE, "frame.png")

    operation.execute_attempt(item, 3)

    assert observed == [
        (
            IMAGE_TOOL_SPECS_BY_NAME["image_shape_filter"],
            3,
            operation.request_id_for(item, attempt=3),
        )
    ]
    assert operation.request_id_for(item, attempt=3) != operation.request_id_for(item)
