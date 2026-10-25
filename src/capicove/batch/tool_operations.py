"""BatchOperation adapters for canonical image and video Tool handlers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol, cast

from capicove.batch.models import (
    BatchCacheScope,
    BatchItem,
    BatchModality,
    OperationFingerprint,
)
from capicove.batch.retry import BatchOperationError
from capicove.contracts import InvocationContext, ToolHandler, ToolResult, ToolSpec, ToolStatus


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_HASH_CHUNK_SIZE = 1024 * 1024
_UNSUPPORTED_IMAGE_TOOLS = frozenset(
    {
        "imgdiff_difference_area_generator_mapper",
        "imgdiff_difference_caption_generator_mapper",
        "image_pair_similarity_filter",
        "image_phash",
        "image_lsh_search",
        "image_perceptual_deduplicator",
        "ray_image_perceptual_deduplicator",
    }
)
_JsonValue = None | bool | int | float | str | list["_JsonValue"] | dict[str, "_JsonValue"]


class BatchToolInvoker(Protocol):
    """Invoke one canonical Tool with its checkpoint-owned attempt number."""

    def __call__(
        self,
        spec: ToolSpec,
        handler: ToolHandler,
        context: InvocationContext,
        arguments: Mapping[str, object],
        attempt: int,
        /,
    ) -> ToolResult: ...


class _ToolBatchOperation:
    """Translate one immutable BatchItem through one canonical Tool boundary."""

    def __init__(
        self,
        spec: ToolSpec,
        handler: ToolHandler,
        *,
        modality: BatchModality,
        parameters: Mapping[str, object],
        provider_fingerprint: str,
        workspace_id: str,
        actor_id: str,
        environment_id: str,
        clock: Callable[[], datetime],
        invoker: BatchToolInvoker | None,
        cache_scope: BatchCacheScope = BatchCacheScope.CONTENT,
    ) -> None:
        if not isinstance(spec, ToolSpec):
            raise ValueError("spec must be a ToolSpec")
        if not callable(handler):
            raise ValueError("handler must be callable")
        if not isinstance(modality, BatchModality):
            raise ValueError("modality must be a BatchModality")
        if _SHA256_PATTERN.fullmatch(provider_fingerprint) is None:
            raise ValueError("provider_fingerprint must be a lowercase SHA-256 hash")
        for field_name, value in (
            ("workspace_id", workspace_id),
            ("actor_id", actor_id),
            ("environment_id", environment_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be blank")
        if not callable(clock):
            raise ValueError("clock must be callable")
        if invoker is not None and not callable(invoker):
            raise ValueError("invoker must be callable")
        normalized_parameters = _json_object(parameters, "parameters")
        self._spec = spec
        self._handler = handler
        self._modality = modality
        self._parameters = normalized_parameters
        self._workspace_id = workspace_id
        self._actor_id = actor_id
        self._environment_id = environment_id
        self._clock = clock
        self._invoker = invoker
        self._fingerprint = OperationFingerprint(
            operation_id=spec.tool_id,
            operation_version=str(spec.contract_version),
            modalities=frozenset({modality}),
            parameters_sha256=_json_sha256(normalized_parameters),
            provider_fingerprint=provider_fingerprint,
            cache_scope=cache_scope,
        )

    @property
    def fingerprint(self) -> OperationFingerprint:
        """Return the complete content-cache operation fingerprint."""
        return self._fingerprint

    def request_id_for(self, item: BatchItem, /, *, attempt: int = 1) -> str:
        """Return the deterministic request identity for one item and operation."""
        _require_attempt(attempt)
        return f"batch-request:{self._identity_digest(f'request:{attempt}', item)}"

    def trace_id_for(self, item: BatchItem, /) -> str:
        """Return the deterministic trace identity for one item and operation."""
        return f"batch-trace:{self._identity_digest('trace', item)}"

    def execute(self, item: BatchItem, /) -> Mapping[str, object]:
        """Invoke the Tool and preserve its canonical successful result envelope."""
        return self.execute_attempt(item, 1)

    def execute_attempt(
        self,
        item: BatchItem,
        attempt: int,
        /,
    ) -> Mapping[str, object]:
        """Invoke the Tool with one checkpoint-owned retry attempt identity."""
        _require_attempt(attempt)
        if not isinstance(item, BatchItem):
            raise ValueError("item must be a BatchItem")
        if item.modality is not self._modality:
            raise BatchOperationError(
                "UNSUPPORTED_MODALITY",
                "batch item modality does not match the Tool adapter",
                retryable=False,
            )
        now = self._clock()
        if not isinstance(now, datetime) or now.utcoffset() is None:
            raise BatchOperationError(
                "INVALID_CLOCK",
                "batch Tool clock must return a timezone-aware datetime",
                retryable=False,
            )
        context = InvocationContext(
            request_id=self.request_id_for(item, attempt=attempt),
            trace_id=self.trace_id_for(item),
            workspace_id=self._workspace_id,
            actor_id=self._actor_id,
            environment_id=self._environment_id,
            deadline=now + timedelta(seconds=self._spec.timeout_seconds),
            permissions=self._spec.required_permissions,
            privacy_class=self._spec.privacy_class,
        )
        arguments = self._arguments_for(item)
        if self._invoker is None:
            result = self._handler(context, arguments)
        else:
            result = self._invoker(
                self._spec,
                self._handler,
                context,
                arguments,
                attempt,
            )
        self._validate_result_identity(context, result)
        if result.status is not ToolStatus.SUCCEEDED:
            error = result.error
            if error is None:
                raise BatchOperationError(
                    "PROTOCOL_ERROR",
                    "Tool failure omitted its canonical error",
                    retryable=False,
                )
            raise BatchOperationError(
                error.code.value,
                error.message,
                retryable=(error.retryable and result.status is ToolStatus.FAILED),
            )
        return _encode_success(item, result)

    def rebind_cached_result(
        self,
        item: BatchItem,
        cached: Mapping[str, object],
        /,
    ) -> Mapping[str, object]:
        """Return a copy with source metadata bound to the current batch item."""
        rebound = _json_object(cached, "cached result")
        rebound["source"] = _encode_source(item)
        return rebound

    def _arguments_for(self, item: BatchItem) -> Mapping[str, object]:
        raise NotImplementedError

    def _identity_digest(self, kind: str, item: BatchItem) -> str:
        digest = hashlib.sha256()
        for value in (
            "capicove.batch.tool-identity.v1",
            kind,
            self._fingerprint.operation_id,
            self._fingerprint.operation_version,
            self._fingerprint.parameters_sha256,
            self._fingerprint.provider_fingerprint,
            item.item_id,
            item.sha256,
        ):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _validate_result_identity(
        self,
        context: InvocationContext,
        result: object,
    ) -> None:
        if not isinstance(result, ToolResult) or (
            result.request_id != context.request_id
            or result.trace_id != context.trace_id
            or (result.tool_id, result.tool_version) != self._spec.registry_key
        ):
            raise BatchOperationError(
                "PROTOCOL_ERROR",
                "Tool result identity does not match its batch invocation",
                retryable=False,
            )


class ImageToolBatchOperation(_ToolBatchOperation):
    """Invoke one explicitly supplied single-image Tool handler per BatchItem."""

    def __init__(
        self,
        tool_name: str,
        handler: ToolHandler,
        *,
        arguments: Mapping[str, object] | None = None,
        operation_parameters: Mapping[str, object] | None = None,
        provider_fingerprint: str,
        workspace_id: str = "batch-local",
        actor_id: str = "batch-worker",
        environment_id: str = "batch-local",
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        invoker: BatchToolInvoker | None = None,
    ) -> None:
        from capicove.tools.image import IMAGE_TOOL_SPECS_BY_NAME

        if tool_name not in IMAGE_TOOL_SPECS_BY_NAME:
            raise ValueError("unknown image Tool name")
        if tool_name in _UNSUPPORTED_IMAGE_TOOLS:
            raise ValueError("image Tool is not compatible with single-item batch execution")
        normalized = _json_object(arguments or {}, "arguments")
        normalized_operation = _json_object(
            operation_parameters or {},
            "operation_parameters",
        )
        if "sources" in normalized:
            raise ValueError("image batch arguments must not override sources")
        self._arguments = normalized
        super().__init__(
            IMAGE_TOOL_SPECS_BY_NAME[tool_name],
            handler,
            modality=BatchModality.IMAGE,
            parameters={
                "arguments": normalized,
                "operation_parameters": normalized_operation,
            },
            provider_fingerprint=provider_fingerprint,
            workspace_id=workspace_id,
            actor_id=actor_id,
            environment_id=environment_id,
            clock=clock,
            invoker=invoker,
        )

    def _arguments_for(self, item: BatchItem) -> Mapping[str, object]:
        arguments = cast(dict[str, object], _json_copy(self._arguments))
        arguments["sources"] = [{"path": str(item.source_path)}]
        return arguments


class VideoFilterBatchOperation(_ToolBatchOperation):
    """Invoke the canonical video-filter Tool once per video BatchItem."""

    def __init__(
        self,
        handler: ToolHandler,
        *,
        criteria: Sequence[Mapping[str, object]],
        aggregation: str = "all",
        provider_fingerprint: str,
        workspace_id: str = "batch-local",
        actor_id: str = "batch-worker",
        environment_id: str = "batch-local",
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        invoker: BatchToolInvoker | None = None,
    ) -> None:
        from capicove.tools.video_filter import VIDEO_FILTER_SPEC

        if not criteria:
            raise ValueError("criteria must not be empty")
        if aggregation not in {"all", "any"}:
            raise ValueError("aggregation must be all or any")
        normalized_criteria = tuple(_json_object(value, "criterion") for value in criteria)
        self._criteria = normalized_criteria
        self._aggregation = aggregation
        super().__init__(
            VIDEO_FILTER_SPEC,
            handler,
            modality=BatchModality.VIDEO,
            parameters={"criteria": normalized_criteria, "aggregation": aggregation},
            provider_fingerprint=provider_fingerprint,
            workspace_id=workspace_id,
            actor_id=actor_id,
            environment_id=environment_id,
            clock=clock,
            invoker=invoker,
        )

    def _arguments_for(self, item: BatchItem) -> Mapping[str, object]:
        return {
            "sources": [
                {
                    "item_id": item.item_id,
                    "source": {"type": "path", "path": str(item.source_path)},
                }
            ],
            "criteria": _json_copy(self._criteria),
            "aggregation": self._aggregation,
        }

    def rebind_cached_result(
        self,
        item: BatchItem,
        cached: Mapping[str, object],
        /,
    ) -> Mapping[str, object]:
        """Rebind every single-item identity in a cached filter result."""
        rebound = cast(dict[str, object], super().rebind_cached_result(item, cached))
        tool = rebound.get("tool")
        if isinstance(tool, dict):
            result = tool.get("result")
            if isinstance(result, dict):
                _rebind_item_id_fields(result, item.item_id)
        return rebound


class VideoSplitBatchOperation(_ToolBatchOperation):
    """Invoke the canonical video-split Tool with isolated per-item outputs."""

    def __init__(
        self,
        handler: ToolHandler,
        *,
        strategy: str,
        config: Mapping[str, object],
        output_format: str = "path",
        output_root: Path | None = None,
        backend: str = "auto",
        write_mode: str = "accurate",
        provider_fingerprint: str,
        workspace_id: str = "batch-local",
        actor_id: str = "batch-worker",
        environment_id: str = "batch-local",
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        invoker: BatchToolInvoker | None = None,
    ) -> None:
        from capicove.tools.video_split import VIDEO_SPLIT_SPEC

        if strategy not in {"duration", "keyframe", "scene"}:
            raise ValueError("unsupported video split strategy")
        if output_format not in {"path", "bytes"}:
            raise ValueError("output_format must be path or bytes")
        if backend not in {"auto", "ffmpeg", "av"}:
            raise ValueError("unsupported video split backend")
        if write_mode not in {"accurate", "copy"}:
            raise ValueError("write_mode must be accurate or copy")
        if output_format == "path":
            if output_root is None or not output_root.is_absolute():
                raise ValueError("path output requires an absolute output_root")
        elif output_root is not None:
            raise ValueError("bytes output must not define output_root")
        normalized_config = _json_object(config, "config")
        self._strategy = strategy
        self._config = normalized_config
        self._output_format = output_format
        self._output_root = output_root
        self._backend = backend
        self._write_mode = write_mode
        super().__init__(
            VIDEO_SPLIT_SPEC,
            handler,
            modality=BatchModality.VIDEO,
            parameters={
                "strategy": strategy,
                "config": normalized_config,
                "output_format": output_format,
                "output_root": str(output_root) if output_root is not None else None,
                "backend": backend,
                "write_mode": write_mode,
            },
            provider_fingerprint=provider_fingerprint,
            workspace_id=workspace_id,
            actor_id=actor_id,
            environment_id=environment_id,
            clock=clock,
            invoker=invoker,
            cache_scope=BatchCacheScope.ITEM,
        )

    def _arguments_for(self, item: BatchItem) -> Mapping[str, object]:
        arguments: dict[str, object] = {
            "source": {"type": "path", "path": str(item.source_path)},
            "strategy": self._strategy,
            "config": _json_copy(self._config),
            "output_format": self._output_format,
            "backend": self._backend,
            "write_mode": self._write_mode,
        }
        if self._output_root is not None:
            arguments["output_directory"] = str(
                self._output_root / self._identity_digest("output", item)
            )
        return arguments

    def is_cached_result_valid(
        self,
        item: BatchItem,
        cached: Mapping[str, object],
        /,
    ) -> bool:
        """Revalidate path-backed split artifacts before cache reuse."""
        if self._output_format == "bytes":
            return True
        if self._output_root is None:
            return False
        expected_directory = self._output_root / self._identity_digest("output", item)
        segments = _cached_segments(cached)
        if segments is None:
            return False
        for segment in segments:
            if not isinstance(segment, Mapping):
                return False
            path_value = segment.get("path")
            expected_sha256 = segment.get("sha256")
            if (
                segment.get("type") != "path"
                or not isinstance(path_value, str)
                or not isinstance(expected_sha256, str)
                or _SHA256_PATTERN.fullmatch(expected_sha256) is None
            ):
                return False
            observed_sha256 = _observe_confined_regular_file(
                Path(path_value),
                expected_directory,
            )
            if observed_sha256 != expected_sha256:
                return False
        return True


def _encode_success(item: BatchItem, result: ToolResult) -> dict[str, object]:
    result_value = None if result.result is None else _json_object(result.result, "Tool result")
    return {
        "source": _encode_source(item),
        "tool": {
            "request_id": result.request_id,
            "trace_id": result.trace_id,
            "tool_id": result.tool_id,
            "tool_version": result.tool_version,
            "status": result.status.value,
            "result": result_value,
            "evidence_refs": list(result.evidence_refs),
            "error": None,
            "metrics": dict(result.metrics),
            "started_at": _timestamp(result.started_at),
            "completed_at": _timestamp(result.completed_at),
        },
    }


def _encode_source(item: BatchItem) -> dict[str, object]:
    return {
        "item_id": item.item_id,
        "relative_path": item.relative_path.as_posix(),
        "media_type": item.media_type,
        "size_bytes": item.size_bytes,
        "sha256": item.sha256,
    }


def _cached_segments(cached: Mapping[str, object]) -> list[object] | None:
    tool = cached.get("tool")
    if not isinstance(tool, Mapping):
        return None
    result = tool.get("result")
    if not isinstance(result, Mapping):
        return None
    segments = result.get("segments")
    if not isinstance(segments, list):
        return None
    return segments


def _json_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_object(value: Mapping[str, object], field_name: str) -> dict[str, object]:
    normalized = _json_copy(value)
    if not isinstance(normalized, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return cast(dict[str, object], normalized)


def _json_copy(value: object, *, depth: int = 0) -> _JsonValue:
    if depth > 32:
        raise ValueError("JSON value exceeds the maximum depth")
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, _JsonValue] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            normalized[key] = _json_copy(child, depth=depth + 1)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_json_copy(child, depth=depth + 1) for child in value]
    raise ValueError("value must be JSON-compatible")


def _rebind_item_id_fields(value: object, item_id: str, *, depth: int = 0) -> None:
    """Replace string item identity fields in one bounded JSON value."""
    if depth > 32:
        raise ValueError("JSON value exceeds the maximum depth")
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "item_id" and isinstance(child, str):
                value[key] = item_id
            else:
                _rebind_item_id_fields(child, item_id, depth=depth + 1)
    elif isinstance(value, list):
        for child in value:
            _rebind_item_id_fields(child, item_id, depth=depth + 1)


def _observe_confined_regular_file(path: Path, root: Path) -> str | None:
    """Hash one regular descendant without following traversal or symlinks."""
    if not path.is_absolute() or ".." in path.parts:
        return None
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    if not relative.parts:
        return None
    descriptors: list[int] = []
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | no_follow
    try:
        root_stat = root.lstat()
        if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
            return None
        current = os.open(root, directory_flags)
        descriptors.append(current)
        for component in relative.parts[:-1]:
            component_stat = os.stat(component, dir_fd=current, follow_symlinks=False)
            if stat.S_ISLNK(component_stat.st_mode) or not stat.S_ISDIR(component_stat.st_mode):
                return None
            current = os.open(component, directory_flags, dir_fd=current)
            descriptors.append(current)
        file_name = relative.parts[-1]
        path_stat = os.stat(file_name, dir_fd=current, follow_symlinks=False)
        if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
            return None
        file_descriptor = os.open(file_name, os.O_RDONLY | no_follow, dir_fd=current)
        descriptors.append(file_descriptor)
        opened_before = os.fstat(file_descriptor)
        if _stat_identity(path_stat) != _stat_identity(opened_before):
            return None
        digest = hashlib.sha256()
        while chunk := os.read(file_descriptor, _HASH_CHUNK_SIZE):
            digest.update(chunk)
        opened_after = os.fstat(file_descriptor)
        path_after = os.stat(file_name, dir_fd=current, follow_symlinks=False)
        if not (
            _stat_identity(opened_before)
            == _stat_identity(opened_after)
            == _stat_identity(path_after)
        ):
            return None
        return digest.hexdigest()
    except OSError:
        return None
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _require_attempt(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("attempt must be a positive integer")
