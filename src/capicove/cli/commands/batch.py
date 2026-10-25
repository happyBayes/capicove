"""Optional unified CLI for resumable image and video directory batches."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from capicove.batch import (
    BatchManifest,
    BatchModality,
    BatchOperation,
    BatchPlanner,
    BatchState,
    BatchToolInvoker,
    DirectoryDiscoverer,
    DirectoryDiscoveryError,
    ImageToolBatchOperation,
    RetryPolicy,
    VideoFilterBatchOperation,
    VideoSplitBatchOperation,
    build_batch_manifest,
)
from capicove.batch.compose import (
    IMAGE_OPERATIONS,
    OPERATIONS,
    VIDEO_OPERATIONS,
    build_image_batch_operation,
    media_types,
)
from capicove.contracts import InvocationContext, ToolHandler, ToolResult, ToolSpec


if TYPE_CHECKING:
    from capicove.traces import TraceRecorder


_IMAGE_OPERATIONS = IMAGE_OPERATIONS
_VIDEO_OPERATIONS = VIDEO_OPERATIONS
_OPERATIONS = OPERATIONS
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def add_parser(parent: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register `capicove batch run`."""
    batch = parent.add_parser(
        "batch",
        help="Run resumable image or video directory batches.",
    )
    commands = batch.add_subparsers(dest="batch_command")
    parser = commands.add_parser(
        "run",
        help="Discover a directory and invoke one canonical Tool per media item.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Input directory root.")
    parser.add_argument("--operation", choices=_OPERATIONS, required=True)
    parser.add_argument(
        "--parameters",
        type=Path,
        required=True,
        help="UTF-8 JSON object containing operation parameters.",
    )
    parser.add_argument(
        "--state",
        type=Path,
        required=True,
        help="SQLite checkpoint and content-cache file.",
    )
    parser.add_argument(
        "--provider-fingerprint",
        type=_sha256,
        required=True,
        help="SHA-256 identity of the configured provider and runtime.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Output root required by video_split; keep it outside the input tree.",
    )
    parser.add_argument("--workers", type=_worker_count, default=4)
    parser.add_argument("--modality-workers", type=_worker_count)
    parser.add_argument("--max-retries", type=_retry_count, default=2)
    parser.add_argument("--initial-backoff", type=_non_negative_float, default=0.5)
    parser.add_argument("--max-backoff", type=_positive_float, default=30.0)
    parser.add_argument("--output", choices=("human", "json"), default="human")


def run(
    arguments: argparse.Namespace,
    *,
    operation: BatchOperation | None = None,
    recorder: TraceRecorder | None = None,
) -> int:
    """Execute one resumable directory batch and render its terminal manifest."""
    if arguments.modality_workers is not None and arguments.modality_workers > arguments.workers:
        raise ValueError("--modality-workers must not exceed --workers")
    parameters = _read_parameters(arguments.parameters)
    input_root = arguments.input.expanduser().resolve()
    composed = operation or _build_default_operation(
        arguments.operation,
        parameters,
        input_root=input_root,
        artifact_root=arguments.artifact_root,
        provider_fingerprint=arguments.provider_fingerprint,
        recorder=recorder,
    )
    modality = _single_modality(composed)
    try:
        items = DirectoryDiscoverer(_media_types(modality)).discover(input_root, modality)
    except DirectoryDiscoveryError as error:
        raise ValueError(str(error)) from error
    plan = BatchPlanner().plan(items, composed.fingerprint)

    from capicove.batch._adapters.sqlite import SqliteBatchStore, SqliteBatchStoreError
    from capicove.batch.executor import BatchExecutor

    try:
        store = SqliteBatchStore(arguments.state.expanduser().resolve())
    except SqliteBatchStoreError as error:
        raise ValueError(str(error)) from error
    retry_policy = RetryPolicy(
        max_retries=arguments.max_retries,
        initial_backoff_seconds=arguments.initial_backoff,
        max_backoff_seconds=arguments.max_backoff,
    )
    modality_workers = arguments.modality_workers or arguments.workers
    execution = BatchExecutor(
        store,
        store,
        max_workers=arguments.workers,
        modality_limits={modality: modality_workers},
        retry_policy=retry_policy,
    ).execute(plan, composed)
    manifest = build_batch_manifest(execution)
    payload = _manifest_payload(manifest)
    if arguments.output == "json":
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        _render_human(payload)
    failed = manifest.counts[BatchState.FAILED] + manifest.counts[BatchState.SOURCE_CHANGED]
    return 1 if failed else 0


def _build_default_operation(
    name: str,
    parameters: Mapping[str, object],
    *,
    input_root: Path,
    artifact_root: Path | None,
    provider_fingerprint: str,
    recorder: TraceRecorder | None = None,
) -> BatchOperation:
    if name in _IMAGE_OPERATIONS:
        return _build_image_operation(
            name,
            parameters,
            input_root=input_root,
            provider_fingerprint=provider_fingerprint,
            recorder=recorder,
        )
    if name == "video_filter":
        criteria = _criteria(parameters.get("criteria"))
        aggregation = _optional_string(parameters, "aggregation", "all")
        _require_exact_keys(parameters, {"aggregation", "criteria"})
        from capicove.tools.video_filter import build_video_filter_tool_handler

        video_filter_handler = build_video_filter_tool_handler(
            "batch-cli-local",
            (input_root,),
        )
        return VideoFilterBatchOperation(
            video_filter_handler,
            criteria=criteria,
            aggregation=aggregation,
            provider_fingerprint=provider_fingerprint,
            invoker=_tool_invoker(recorder),
        )
    if name == "video_split":
        _require_exact_keys(
            parameters,
            {"backend", "config", "strategy", "write_mode"},
        )
        if artifact_root is None:
            raise ValueError("--artifact-root is required for video_split")
        output_root = artifact_root.expanduser().resolve()
        if output_root == input_root or output_root.is_relative_to(input_root):
            raise ValueError("--artifact-root must remain outside the input tree")
        strategy = _required_string(parameters, "strategy")
        config = _required_object(parameters, "config")
        backend = _optional_string(parameters, "backend", "auto")
        write_mode = _optional_string(parameters, "write_mode", "accurate")
        from capicove.tools.video_split import build_video_split_tool_handler

        video_split_handler = build_video_split_tool_handler(
            "batch-cli-local",
            (input_root,),
            (output_root,),
            output_root,
        )
        return VideoSplitBatchOperation(
            video_split_handler,
            strategy=strategy,
            config=config,
            output_format="path",
            output_root=output_root,
            backend=backend,
            write_mode=write_mode,
            provider_fingerprint=provider_fingerprint,
            invoker=_tool_invoker(recorder),
        )
    raise ValueError("unsupported batch operation")


def _build_image_operation(
    name: str,
    parameters: Mapping[str, object],
    *,
    input_root: Path,
    provider_fingerprint: str,
    recorder: TraceRecorder | None = None,
) -> ImageToolBatchOperation:
    return build_image_batch_operation(
        name,
        parameters,
        input_root=input_root,
        provider_fingerprint=provider_fingerprint,
        invoker=_tool_invoker(recorder),
    )


def _tool_invoker(recorder: TraceRecorder | None) -> BatchToolInvoker | None:
    if recorder is None:
        return None

    def invoke(
        spec: ToolSpec,
        handler: ToolHandler,
        context: InvocationContext,
        arguments: Mapping[str, object],
        attempt: int,
        /,
    ) -> ToolResult:
        from capicove.tools.tracing import invoke_one_step_tool

        return invoke_one_step_tool(
            recorder=recorder,
            spec=spec,
            handler=handler,
            context=context,
            arguments=arguments,
            attempt=attempt,
            episode_id=f"{context.trace_id}:attempt:{attempt}",
        )

    return invoke


def _read_parameters(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("parameters must be a readable UTF-8 JSON file") from error
    if not isinstance(value, dict):
        raise ValueError("parameters JSON must contain an object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError("parameters JSON keys must be strings")
    return value


def _single_modality(operation: BatchOperation) -> BatchModality:
    modalities = operation.fingerprint.modalities
    if len(modalities) != 1:
        raise ValueError("batch CLI operations must select exactly one modality")
    return next(iter(modalities))


def _media_types(modality: BatchModality) -> Mapping[str, str]:
    return media_types(modality)


def _manifest_payload(manifest: BatchManifest) -> dict[str, object]:
    return {
        "plan_id": manifest.plan_id,
        "total": manifest.total,
        "cache_hits": manifest.cache_hits,
        "counts": {state.value: manifest.counts[state] for state in BatchState},
        "items": [
            {
                "item_id": item.item_id,
                "state": item.state.value,
                "attempts": item.attempts,
                "cache_hit": item.cache_hit,
                "result": dict(item.result) if item.result is not None else None,
                "failure": (
                    {
                        "code": item.failure.code,
                        "message": item.failure.message,
                        "retryable": item.failure.retryable,
                    }
                    if item.failure is not None
                    else None
                ),
                "detail": item.detail,
            }
            for item in manifest.items
        ],
    }


def _render_human(payload: Mapping[str, object]) -> None:
    print(f"plan: {payload['plan_id']}")
    print(f"total: {payload['total']}")
    print(f"cache hits: {payload['cache_hits']}")
    counts = payload["counts"]
    if isinstance(counts, Mapping):
        for state in BatchState:
            print(f"{state.value}: {counts[state.value]}")


def _criteria(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError("video_filter criteria must be a non-empty array")
    if any(not isinstance(item, Mapping) for item in value):
        raise ValueError("every video_filter criterion must be an object")
    return tuple(item for item in value if isinstance(item, Mapping))


def _required_object(parameters: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = parameters.get(name)
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object")
    return value


def _required_string(parameters: Mapping[str, object], name: str) -> str:
    value = parameters.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value


def _optional_string(parameters: Mapping[str, object], name: str, default: str) -> str:
    value = parameters.get(name, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value


def _require_exact_keys(parameters: Mapping[str, object], allowed: set[str]) -> None:
    unknown = set(parameters).difference(allowed)
    if unknown:
        raise ValueError(f"unknown operation parameters: {', '.join(sorted(unknown))}")


def _sha256(value: str) -> str:
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be a lowercase SHA-256 hash")
    return value


def _worker_count(value: str) -> int:
    parsed = _integer(value, "worker count")
    if not 1 <= parsed <= 64:
        raise argparse.ArgumentTypeError("worker count must be between 1 and 64")
    return parsed


def _retry_count(value: str) -> int:
    parsed = _integer(value, "retry count")
    if not 0 <= parsed <= 10:
        raise argparse.ArgumentTypeError("retry count must be between 0 and 10")
    return parsed


def _integer(value: str, name: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must be an integer") from error


def _non_negative_float(value: str) -> float:
    parsed = _float(value, "initial backoff")
    if parsed < 0:
        raise argparse.ArgumentTypeError("initial backoff must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = _float(value, "maximum backoff")
    if parsed <= 0:
        raise argparse.ArgumentTypeError("maximum backoff must be positive")
    return parsed


def _float(value: str, name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must be a number") from error
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError(f"{name} must be finite")
    return parsed
