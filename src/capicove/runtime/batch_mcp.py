"""Operator batch jobs for the version-2 workspace host (RFC-0016 amendment 1).

`capabilities.batch` fixes every input and output root, the operation allowlist,
provider identities, and worker bounds at startup. MCP, HTTP, and configured
CLI calls plan, enqueue, observe, and cancel durable jobs; only the separately
started `capicove jobs worker` process executes them. Video operations reuse
`capabilities.video` pinned executables and descriptor-bound staging (ADR-0026).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from types import MappingProxyType
from typing import TYPE_CHECKING, NoReturn

from capicove.batch.compose import IMAGE_OPERATIONS, OPERATIONS
from capicove.contracts import (
    InvocationContext,
    ToolErrorCode,
    ToolHandler,
    ToolResult,
    ToolSpec,
)
from capicove.mcp.bindings import McpToolBinding
from capicove.runtime._operator import Rejected, catalog_only_handler, operator_handler
from capicove.runtime.mcp_config import WorkspaceHostConfiguration
from capicove.runtime.video_mcp import (
    FILTER as VIDEO_FILTER,
)
from capicove.runtime.video_mcp import (
    HOST_FILTER_CRITERIA,
    VideoSettings,
)
from capicove.runtime.video_mcp import (
    SPLIT as VIDEO_SPLIT,
)


if TYPE_CHECKING:
    from capicove.batch.job_registry import JobRecord, SqliteJobRegistry
    from capicove.batch.job_service import BatchJobService
    from capicove.batch.job_worker import JobWorker
    from capicove.batch.models import BatchPlan
    from capicove.batch.ports import BatchOperation
    from capicove.runtime.executables import PinnedExecutables


PLAN = "capicove.batch.plan"
ENQUEUE = "capicove.batch.enqueue"
STATUS = "capicove.batch.status"
CANCEL = "capicove.batch.cancel"
TOOLS = (CANCEL, ENQUEUE, PLAN, STATUS)
MAX_ITEMS = 100_000
_FIELDS = frozenset(
    {
        "schema_version",
        "input_roots",
        "output_roots",
        "registry_root",
        "operations",
        "provider_fingerprints",
        "max_items",
        "worker",
    }
)
_WORKER_FIELDS = frozenset({"concurrency", "lease_seconds", "poll_seconds"})
_ALIAS = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_JOB_ID = re.compile(r"job:([0-9a-f]{24})\Z")
_DESCRIPTOR_LIMIT = 64 * 1024
_QUEUE_SCAN = 100
_REGISTRY = "jobs.sqlite3"
_CACHE = "batch.sqlite3"
_DESCRIPTORS = "operations"
_WORKER_HINT = "Waiting for a local worker; start `capicove jobs worker --config PATH`."
_RESUME_HINT = "The worker lease expired; resume with `capicove jobs resume --config PATH`."


class BatchUnavailable(ValueError):
    """A request selected a job, root, or operation this host cannot serve."""


@dataclass(frozen=True, slots=True)
class BatchSettings:
    """Validated startup authority for operator batch jobs."""

    input_roots: Mapping[str, Path]
    output_roots: Mapping[str, Path]
    registry_root: Path
    operations: tuple[str, ...]
    provider_fingerprints: Mapping[str, str]
    max_items: int
    concurrency: int
    lease_seconds: float
    poll_seconds: float
    video: VideoSettings | None


def batch_settings(config: WorkspaceHostConfiguration) -> BatchSettings:
    """Validate `capabilities.batch` and, for video operations, `capabilities.video`."""
    from capicove.runtime.video_mcp import _existing_directory, _private_child

    if os.name != "posix":
        raise ValueError("batch host Tools require POSIX no-follow file APIs in this release")
    raw = config.capabilities.get("batch")
    if raw is None:
        raise ValueError("capabilities.batch is not configured")
    if set(raw) != _FIELDS:
        raise ValueError("capabilities.batch requires exactly the documented fields")
    if raw["schema_version"] != 1:
        raise ValueError("capabilities.batch schema_version must be 1")
    workspace = config.workspace_root
    inputs = _aliases(raw["input_roots"], "batch input_roots")
    outputs = _aliases(raw["output_roots"], "batch output_roots")
    input_roots = {
        alias: _existing_directory(value, "batch input_roots") for alias, value in inputs.items()
    }
    for path in input_roots.values():
        if path.resolve(strict=True) != path:
            raise ValueError("batch input_roots must be resolved paths")
    output_roots = {
        alias: _private_child(value, workspace, "batch output_roots")
        for alias, value in outputs.items()
    }
    registry_root = _private_child(raw["registry_root"], workspace, "batch registry_root")
    owned = [*output_roots.values(), registry_root]
    for index, path in enumerate(owned):
        for other in [*input_roots.values(), *owned[index + 1 :]]:
            if path.is_relative_to(other) or other.is_relative_to(path):
                raise ValueError("batch output, registry, and input roots must be disjoint")
    operations = raw["operations"]
    if (
        not isinstance(operations, tuple)
        or not operations
        or len(set(operations)) != len(operations)
        or any(item not in OPERATIONS for item in operations)
    ):
        raise ValueError("batch operations must be a nonempty unique allowlist")
    fingerprints = raw["provider_fingerprints"]
    if (
        not isinstance(fingerprints, Mapping)
        or set(fingerprints) != set(operations)
        or any(
            not isinstance(item, str) or not _SHA256.match(item) for item in fingerprints.values()
        )
    ):
        raise ValueError("batch provider_fingerprints must map every operation to a SHA-256")
    max_items = raw["max_items"]
    if type(max_items) is not int or not 1 <= max_items <= MAX_ITEMS:
        raise ValueError("batch max_items must be an integer from 1 through 100000")
    worker = raw["worker"]
    if not isinstance(worker, Mapping) or set(worker) != _WORKER_FIELDS:
        raise ValueError("batch worker requires concurrency, lease_seconds, and poll_seconds")
    concurrency = worker["concurrency"]
    if type(concurrency) is not int or not 1 <= concurrency <= 8:
        raise ValueError("batch worker concurrency must be an integer from 1 through 8")
    lease = _number(worker["lease_seconds"], 30, 3600, "batch worker lease_seconds")
    poll = _number(worker["poll_seconds"], 0.5, 60, "batch worker poll_seconds")
    video = _video_settings(config, operations, (*input_roots.values(), *owned))
    return BatchSettings(
        MappingProxyType(input_roots),
        MappingProxyType(output_roots),
        registry_root,
        tuple(sorted(operations)),
        MappingProxyType(dict(fingerprints)),
        max_items,
        concurrency,
        lease,
        poll,
        video,
    )


def _video_settings(
    config: WorkspaceHostConfiguration, operations: tuple[str, ...], batch_roots: tuple[Path, ...]
) -> VideoSettings | None:
    from capicove.runtime.video_mcp import _check_dependencies, video_settings

    selected = {f"capicove.video.{name.removeprefix('video_')}" for name in operations}
    selected &= {VIDEO_SPLIT, VIDEO_FILTER}
    if not selected:
        return None
    if "video" not in config.capabilities:
        raise ValueError("batch video operations require capabilities.video")
    settings = video_settings(replace(config, enabled_tools=("capicove.video.deduplicate",)))
    for root in batch_roots:
        staging = settings.staging_root
        if staging.is_relative_to(root) or root.is_relative_to(staging):
            raise ValueError("batch roots must be disjoint from the video staging_root")
    # Split writes into the batch output roots, so any configured root satisfies
    # the video dependency check's output requirement.
    _check_dependencies(replace(settings, output_root=batch_roots[-1]), selected)
    return settings


def _aliases(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not 1 <= len(value) <= 16:
        raise ValueError(f"{field} must map 1 to 16 aliases to paths")
    if any(not isinstance(alias, str) or not _ALIAS.match(alias) for alias in value):
        raise ValueError(f"{field} aliases must be lowercase identifiers")
    if len(set(value.values())) != len(value):
        raise ValueError(f"{field} must name distinct paths")
    return dict(value)


def _number(value: object, low: float, high: float, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not low <= value <= high:
        raise ValueError(f"{field} must be a number from {low:g} through {high:g}")
    return float(value)


def root_sha256(alias: str, path: Path) -> str:
    """Bind a job to the alias and exact path configured when it was queued."""
    return hashlib.sha256(f"capicove.batch.root.v1\0{alias}\0{path}".encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class _Selection:
    operation: str
    parameters: Mapping[str, object]
    input_alias: str
    output_alias: str | None


class BatchRuntime:
    """Build services and operations only from configured authority."""

    def __init__(
        self,
        config: WorkspaceHostConfiguration,
        settings: BatchSettings,
        *,
        pins: Mapping[str, PinnedExecutables] | None = None,
    ) -> None:
        self.config = config
        self.settings = settings
        self._pins = pins

    def _identity(self) -> tuple[str, str, str]:
        return self.config.workspace_id, self.config.actor_id, self.config.environment_id

    @property
    def registry_path(self) -> Path:
        """Return the durable registry location inside the registry root."""
        return self.settings.registry_root / _REGISTRY

    def registry(self) -> SqliteJobRegistry | None:
        """Open the registry only when an authorized enqueue already created it."""
        from capicove.batch.job_registry import SqliteJobRegistry

        if not self.registry_path.exists():
            return None
        return SqliteJobRegistry(self.registry_path)

    def operation(self, selection: _Selection) -> BatchOperation:
        """Compose the allowlisted batch operation for one selection."""
        from capicove.batch.compose import build_image_batch_operation
        from capicove.batch.tool_operations import (
            VideoFilterBatchOperation,
            VideoSplitBatchOperation,
        )

        name = selection.operation
        if name not in self.settings.operations:
            raise BatchUnavailable("operation_not_configured")
        input_root = self.settings.input_roots[selection.input_alias]
        identity = self._identity()
        fingerprint = self.settings.provider_fingerprints[name]
        parameters = dict(selection.parameters)
        if name in IMAGE_OPERATIONS:
            return build_image_batch_operation(
                name,
                parameters,
                input_root=input_root,
                provider_fingerprint=fingerprint,
                workspace_id=identity[0],
                actor_id=identity[1],
                environment_id=identity[2],
            )
        video = self.settings.video
        assert video is not None
        if name == "video_filter":
            if set(parameters) - {"criteria", "aggregation"}:
                raise ValueError("unknown video_filter parameters")
            criteria = parameters.get("criteria")
            if (
                not isinstance(criteria, (list, tuple))
                or not criteria
                or any(
                    not isinstance(item, Mapping) or item.get("type") not in HOST_FILTER_CRITERIA
                    for item in criteria
                )
            ):
                raise ValueError("video_filter criteria must be metadata or motion objects")
            aggregation = parameters.get("aggregation", "all")
            if not isinstance(aggregation, str):
                raise ValueError("video_filter aggregation must be a string")
            handler = self._video_handler(name, input_root, None)
            return VideoFilterBatchOperation(
                handler,
                criteria=criteria,
                aggregation=aggregation,
                provider_fingerprint=fingerprint,
                workspace_id=identity[0],
                actor_id=identity[1],
                environment_id=identity[2],
            )
        if selection.output_alias is None:
            raise ValueError("video_split plans require output_root")
        if set(parameters) - {"strategy", "config", "write_mode"}:
            raise ValueError("unknown video_split parameters")
        strategy, config = parameters.get("strategy"), parameters.get("config")
        write_mode = parameters.get("write_mode", "accurate")
        if not isinstance(strategy, str) or not isinstance(config, Mapping):
            raise ValueError("video_split requires a strategy and a config object")
        if not isinstance(write_mode, str):
            raise ValueError("video_split write_mode must be a string")
        output_root = self.settings.output_roots[selection.output_alias]
        return VideoSplitBatchOperation(
            self._video_handler(name, input_root, output_root),
            strategy=strategy,
            config=config,
            output_format="path",
            output_root=output_root,
            backend=video.backend,
            write_mode=write_mode,
            provider_fingerprint=fingerprint,
            workspace_id=identity[0],
            actor_id=identity[1],
            environment_id=identity[2],
        )

    def service(
        self, selection: _Selection, operation: BatchOperation, output_alias: str
    ) -> BatchJobService:
        """Bind one job service to a configured input, output, and operation."""
        from capicove.batch.compose import media_types
        from capicove.batch.job_service import BatchJobService

        fingerprint = operation.fingerprint
        modality = next(iter(fingerprint.modalities))
        input_root = self.settings.input_roots[selection.input_alias]
        output_root = self.settings.output_roots[output_alias]
        return BatchJobService(
            database_path=self.registry_path,
            workspace_id=self.config.workspace_id,
            input_root=input_root,
            input_root_alias=selection.input_alias,
            input_root_sha256=root_sha256(selection.input_alias, input_root),
            output_root_alias=output_alias,
            output_root_sha256=root_sha256(output_alias, output_root),
            modality=modality,
            media_types=media_types(modality),
            operation=fingerprint,
        )

    def _video_handler(self, name: str, input_root: Path, output_root: Path | None) -> ToolHandler:
        video = self.settings.video
        assert video is not None
        pins = self._pins

        def handler(context: InvocationContext, arguments: Mapping[str, object], /) -> ToolResult:
            if pins is None:
                raise RuntimeError("batch video operations run only in the local worker")
            from capicove.runtime.video_mcp import _filter_handler, _split_handler

            scoped = replace(video, read_roots=(input_root,))
            if name == "video_filter":
                return _filter_handler(scoped, pins[name])(context, arguments)
            assert output_root is not None
            directory = Path(str(arguments.get("output_directory", "")))
            if directory.parent != output_root or not _SHA256.match(directory.name):
                raise RuntimeError("batch split output directory is unmanaged")
            from capicove.runtime.media_handoff import open_private_directory

            os.close(open_private_directory(output_root))
            os.close(open_private_directory(directory))
            forwarded = {
                key: value for key, value in arguments.items() if key != "output_directory"
            }
            return _split_handler(replace(scoped, output_root=directory), pins[name])(
                context, forwarded
            )

        return handler

    # Durable operation descriptors let the worker rebuild an allowlisted operation.

    def write_descriptor(self, digest: str, selection: _Selection) -> None:
        """Record a job's operation name and parameters, create-only and private."""
        from capicove.runtime.media_handoff import open_private_directory

        payload = _canonical(
            {
                "schema_version": 1,
                "operation": selection.operation,
                "parameters": dict(selection.parameters),
            }
        ).encode("utf-8")
        if len(payload) > _DESCRIPTOR_LIMIT:
            raise ValueError("operation parameters are too large")
        os.close(open_private_directory(self.settings.registry_root))
        directory = open_private_directory(self.settings.registry_root / _DESCRIPTORS)
        try:
            existing = _read_file(directory, f"{digest}.json")
            if existing is not None:
                if existing != payload:
                    raise ValueError("operation descriptor conflict")
                return
            temporary = f".{digest}.{secrets.token_hex(8)}.tmp"
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory,
            )
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.link(temporary, f"{digest}.json", src_dir_fd=directory, dst_dir_fd=directory)
                except FileExistsError:
                    if _read_file(directory, f"{digest}.json") != payload:
                        raise ValueError("operation descriptor conflict") from None
            finally:
                os.unlink(temporary, dir_fd=directory)
        finally:
            os.close(directory)

    def read_descriptor(self, digest: str) -> tuple[str, Mapping[str, object]] | None:
        """Return a recorded operation, or None when it is absent or malformed."""
        from capicove.runtime.media_handoff import MediaDenied, open_private_directory

        if not _SHA256.match(digest):
            return None
        try:
            directory = open_private_directory(
                self.settings.registry_root / _DESCRIPTORS, create=False
            )
        except (MediaDenied, OSError):
            return None
        try:
            payload = _read_file(directory, f"{digest}.json")
        finally:
            os.close(directory)
        if payload is None:
            return None
        try:
            body = json.loads(payload)
        except (UnicodeError, json.JSONDecodeError):
            return None
        if (
            not isinstance(body, dict)
            or set(body) != {"schema_version", "operation", "parameters"}
            or body["schema_version"] != 1
            or body["operation"] not in OPERATIONS
            or not isinstance(body["parameters"], dict)
        ):
            return None
        return body["operation"], body["parameters"]

    def route(self, record: JobRecord) -> tuple[BatchJobService, BatchOperation]:
        """Rebuild the exact service and operation a queued job was planned with."""
        from capicove.batch.job_plan_receipt import operation_sha256

        job = record
        described = self.read_descriptor(job.operation_fingerprint)
        if described is None:
            raise BatchUnavailable("operation_descriptor_unavailable")
        name, parameters = described
        input_alias = job.input_root_alias
        output_alias = job.output_root_alias
        if (
            input_alias not in self.settings.input_roots
            or output_alias not in self.settings.output_roots
        ):
            raise BatchUnavailable("root_not_configured")
        selection = _Selection(name, parameters, input_alias, output_alias)
        operation = self.operation(selection)
        if operation_sha256(operation.fingerprint) != job.operation_fingerprint:
            raise BatchUnavailable("operation_changed")
        return self.service(selection, operation, output_alias), operation


def _read_file(directory: int, name: str) -> bytes | None:
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    except FileNotFoundError:
        return None
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("operation descriptor must be a regular file")
        payload = stream.read(_DESCRIPTOR_LIMIT + 1)
    if len(payload) > _DESCRIPTOR_LIMIT:
        raise ValueError("operation descriptor is too large")
    return payload


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@contextmanager
def compose_batch_mcp(
    config: WorkspaceHostConfiguration,
    context_factory: Callable[[ToolSpec], InvocationContext],
    *,
    catalog_only: bool = False,
) -> Iterator[tuple[McpToolBinding, ...]]:
    """Validate batch authority and bind the selected operator Tools."""
    from capicove.tools.batch_jobs import BATCH_SPECS

    if any(tool_id not in BATCH_SPECS for tool_id in config.enabled_tools):
        raise ValueError("batch profile includes an unavailable Tool")
    for tool_id in config.enabled_tools:
        if not BATCH_SPECS[tool_id].required_permissions.issubset(config.permissions):
            raise ValueError("configured permissions do not authorize every enabled Tool")
    runtime = BatchRuntime(config, batch_settings(config))
    handlers: dict[str, Callable[[], ToolHandler]] = {
        PLAN: lambda: operator_handler(
            BATCH_SPECS[PLAN], lambda _, arguments: _plan(runtime, arguments)
        ),
        ENQUEUE: lambda: operator_handler(
            BATCH_SPECS[ENQUEUE], lambda _, arguments: _enqueue(runtime, arguments)
        ),
        STATUS: lambda: operator_handler(
            BATCH_SPECS[STATUS], lambda _, arguments: _status(runtime, arguments)
        ),
        CANCEL: lambda: operator_handler(
            BATCH_SPECS[CANCEL], lambda _, arguments: _cancel(runtime, arguments)
        ),
    }
    yield tuple(
        McpToolBinding(
            "capicove_batch_" + tool_id.rsplit(".", 1)[1],
            BATCH_SPECS[tool_id],
            catalog_only_handler if catalog_only else handlers[tool_id](),
            context_factory,
        )
        for tool_id in sorted(config.enabled_tools)
    )


_DENIED = ToolErrorCode.PERMISSION_DENIED
_INVALID = ToolErrorCode.INVALID_INPUT
_POLICY = ToolErrorCode.POLICY_REJECTED
_UNAVAILABLE = ToolErrorCode.UNAVAILABLE
_MESSAGES = {
    "permission_denied": "The request is outside the configured batch authority.",
    "unknown_root": "The selected root alias is not configured.",
    "operation_not_configured": "The operation is not allowlisted by this host.",
    "provider_mismatch": "The provider fingerprint does not match the configured provider.",
    "invalid_parameters": "The operation parameters are not accepted.",
    "discovery_failed": "The configured input root could not be planned.",
    "max_items_exceeded": "The plan exceeds the configured max_items bound.",
    "plan_changed": "plan_changed: the current plan differs from plan_id; plan again.",
    "idempotency_conflict": "idempotency_conflict: the key was used for a different job.",
    "plan_conflict": "plan_conflict: the stored plan receipt differs.",
    "source_changed": "source_changed: a source changed while it was being queued.",
    "job_unavailable": "The job is unavailable in this workspace.",
    "revision_conflict": "revision_conflict: the job changed; read its status again.",
    "terminal_job": "A terminal job cannot be cancelled.",
    "job_state_invalid": "The durable job state failed verification.",
}


def _reject(code: ToolErrorCode, reason: str) -> NoReturn:
    raise Rejected(code, _MESSAGES[reason])


def _selection(runtime: BatchRuntime, arguments: Mapping[str, object]) -> _Selection:
    settings = runtime.settings
    input_alias = str(arguments["input_root"])
    output = arguments.get("output_root")
    if input_alias not in settings.input_roots or (
        output is not None and output not in settings.output_roots
    ):
        _reject(_INVALID, "unknown_root")
    operation = str(arguments["operation"])
    if operation not in settings.operations:
        _reject(_POLICY, "operation_not_configured")
    if arguments["provider_fingerprint"] != settings.provider_fingerprints[operation]:
        _reject(_POLICY, "provider_mismatch")
    parameters = arguments["parameters"]
    assert isinstance(parameters, Mapping)
    return _Selection(
        operation, dict(parameters), input_alias, None if output is None else str(output)
    )


def _planned(
    runtime: BatchRuntime, selection: _Selection
) -> tuple[BatchOperation, BatchJobService, BatchPlan]:
    from capicove.batch.discovery import DirectoryDiscoveryError

    try:
        operation = runtime.operation(selection)
        _check_video_arguments(selection)
    except (TypeError, ValueError):
        _reject(_INVALID, "invalid_parameters")
    output_alias = selection.output_alias or next(iter(runtime.settings.output_roots))
    service = runtime.service(selection, operation, output_alias)
    try:
        plan = service.plan()
    except (DirectoryDiscoveryError, OSError, ValueError):
        _reject(_INVALID, "discovery_failed")
    if len(plan.items) > runtime.settings.max_items:
        _reject(_POLICY, "max_items_exceeded")
    return operation, service, plan


def _check_video_arguments(selection: _Selection) -> None:
    """Validate video parameters against the Tool schema before any job exists.

    Otherwise a malformed criterion would only surface as one failure per item
    after the worker has staged every source.
    """
    import importlib

    from capicove.tools.video_filter import VIDEO_FILTER_SPEC
    from capicove.tools.video_split import VIDEO_SPLIT_SPEC

    parameters = dict(selection.parameters)
    source = {"type": "path", "path": "/batch-item"}
    if selection.operation == "video_filter":
        spec, arguments = VIDEO_FILTER_SPEC, {"sources": [{"source": source}], **parameters}
    elif selection.operation == "video_split":
        spec = VIDEO_SPLIT_SPEC
        arguments = {
            "source": source,
            "output_format": "path",
            "backend": "auto",
            "write_mode": "accurate",
            **parameters,
        }
    else:
        return
    # Loaded dynamically: jsonschema ships without type stubs in this environment.
    jsonschema = importlib.import_module("jsonschema")
    try:
        jsonschema.validate(arguments, dict(spec.input_schema))
    except jsonschema.ValidationError as error:
        raise ValueError("invalid video parameters") from error


def _plan(runtime: BatchRuntime, arguments: Mapping[str, object]) -> dict[str, object]:
    from capicove.tools.batch_jobs import MAX_PLAN_PREVIEW_ITEMS

    selection = _selection(runtime, arguments)
    _, _, plan = _planned(runtime, selection)
    items = plan.items
    fingerprint = plan.operation
    return {
        "plan_id": plan.plan_id,
        "input_root": selection.input_alias,
        "output_root": selection.output_alias,
        "operation": selection.operation,
        "operation_fingerprint": {
            "operation_id": fingerprint.operation_id,
            "operation_version": fingerprint.operation_version,
            "parameters_sha256": fingerprint.parameters_sha256,
            "provider_fingerprint": fingerprint.provider_fingerprint,
        },
        "item_count": len(items),
        "total_bytes": sum(entry.item.size_bytes for entry in items),
        "items": [
            {
                "item_id": entry.item.item_id,
                "relative_path": entry.item.relative_path.as_posix(),
                "media_type": entry.item.media_type,
                "size_bytes": entry.item.size_bytes,
                "sha256": entry.item.sha256,
            }
            for entry in items[:MAX_PLAN_PREVIEW_ITEMS]
        ],
        "truncated": len(items) > MAX_PLAN_PREVIEW_ITEMS,
    }


def _enqueue(runtime: BatchRuntime, arguments: Mapping[str, object]) -> dict[str, object]:
    from capicove.batch.job_plan_receipt import operation_sha256
    from capicove.batch.job_registry import JobConflictError, SqliteJobRegistry
    from capicove.runtime.media_handoff import MediaDenied, open_private_directory

    selection = _selection(runtime, arguments)
    assert selection.output_alias is not None
    _, _, plan = _planned(runtime, selection)
    if plan.plan_id != arguments["plan_id"]:
        _reject(_INVALID, "plan_changed")
    settings = runtime.settings
    input_root = settings.input_roots[selection.input_alias]
    output_root = settings.output_roots[selection.output_alias]
    try:
        os.close(open_private_directory(settings.registry_root))
        os.close(open_private_directory(output_root))
    except MediaDenied:
        _reject(_DENIED, "permission_denied")
    try:
        runtime.write_descriptor(operation_sha256(plan.operation), selection)
    except (OSError, ValueError):
        _reject(_UNAVAILABLE, "job_state_invalid")
    try:
        record = SqliteJobRegistry(runtime.registry_path).enqueue_plan(
            workspace_id=runtime.config.workspace_id,
            plan=plan,
            input_root=input_root,
            input_root_alias=selection.input_alias,
            input_root_sha256=root_sha256(selection.input_alias, input_root),
            output_root_alias=selection.output_alias,
            output_root_sha256=root_sha256(selection.output_alias, output_root),
            idempotency_key=str(arguments["idempotency_key"]),
        )
    except JobConflictError as error:
        _reject(_INVALID, str(error) if str(error) in _MESSAGES else "idempotency_conflict")
    except ValueError as error:
        if str(error) == "source_changed":
            _reject(_INVALID, "source_changed")
        _reject(_UNAVAILABLE, "job_state_invalid")
    return {"job": _job_payload(runtime, record)}


def _job(runtime: BatchRuntime, job_id: str) -> JobRecord:
    registry = runtime.registry()
    if registry is None:
        _reject(_INVALID, "job_unavailable")
    try:
        return registry.status(workspace_id=runtime.config.workspace_id, job_id=job_id)
    except KeyError:
        _reject(_INVALID, "job_unavailable")
    except ValueError:
        _reject(_UNAVAILABLE, "job_state_invalid")


def _job_payload(runtime: BatchRuntime, record: JobRecord) -> dict[str, object]:
    job = record
    state = job.state
    described = runtime.read_descriptor(job.operation_fingerprint)
    lease = job.lease_until
    return {
        "job_id": job.job_id,
        "plan_id": job.plan_id,
        "operation": described[0] if described is not None else None,
        "input_root": job.input_root_alias,
        "output_root": job.output_root_alias,
        "state": state,
        "revision": job.revision,
        "total_items": job.total_items,
        "completed_items": job.completed_items,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "lease_until": lease.isoformat() if lease is not None else None,
        "terminal_manifest_sha256": job.terminal_manifest_sha256,
        "error_code": job.error_code,
        "diagnostic": {"queued": _WORKER_HINT, "interrupted": _RESUME_HINT}.get(state),
    }


def _status(runtime: BatchRuntime, arguments: Mapping[str, object]) -> dict[str, object]:
    from capicove.tools.batch_jobs import MAX_STATUS_ITEMS

    job_id = str(arguments["job_id"])
    offset = _integer(arguments.get("offset", 0))
    limit = _integer(arguments.get("limit", MAX_STATUS_ITEMS))
    record = _job(runtime, job_id)
    items = _status_items(runtime, record)
    page = items[offset : offset + limit]
    return {
        "job": _job_payload(runtime, record),
        "items": page,
        "offset": offset,
        "limit": limit,
        "next_offset": offset + limit if offset + limit < len(items) else None,
    }


def _status_items(runtime: BatchRuntime, record: JobRecord) -> list[dict[str, object]]:
    """List item checkpoints in plan order; items never claimed report pending."""
    from capicove.batch._adapters.sqlite import SqliteBatchStore

    job = record
    settings = runtime.settings
    alias = job.input_root_alias
    input_root = settings.input_roots.get(alias)
    registry = runtime.registry()
    if input_root is None or registry is None:
        return []
    try:
        plan = registry.load_plan(
            workspace_id=runtime.config.workspace_id,
            plan_id=job.plan_id,
            input_root=input_root,
            input_root_alias=alias,
            input_root_sha256=root_sha256(alias, input_root),
        )
    except (KeyError, ValueError):
        return []
    matched = _JOB_ID.fullmatch(job.job_id)
    assert matched is not None
    checkpoint_path = settings.registry_root / "batch-jobs" / f"{matched.group(1)}.sqlite3"
    checkpoints = {}
    if checkpoint_path.is_file() and not checkpoint_path.is_symlink():
        checkpoints = {
            item.item_id: item for item in SqliteBatchStore(checkpoint_path).list(plan.plan_id)
        }
    cache_path = settings.registry_root / _CACHE
    cache = (
        SqliteBatchStore(cache_path)
        if cache_path.is_file() and not cache_path.is_symlink()
        else None
    )
    items: list[dict[str, object]] = []
    for entry in plan.items:
        checkpoint = checkpoints.get(entry.item.item_id)
        state = checkpoint.state.value if checkpoint is not None else "pending"
        result = None
        if checkpoint is not None and state == "succeeded" and cache is not None:
            cached = cache.get(entry.cache_key)
            tool = cached.get("tool") if cached is not None else None
            if isinstance(tool, Mapping) and isinstance(tool.get("result"), Mapping):
                result = json.loads(json.dumps(tool["result"]))
        items.append(
            {
                "item_id": entry.item.item_id,
                "relative_path": entry.item.relative_path.as_posix(),
                "state": state,
                "attempts": checkpoint.attempts if checkpoint is not None else 0,
                "failure_code": (
                    checkpoint.failure.code[:64]
                    if checkpoint is not None and checkpoint.failure is not None
                    else None
                ),
                "result": result,
            }
        )
    return items


def _cancel(runtime: BatchRuntime, arguments: Mapping[str, object]) -> dict[str, object]:
    from capicove.batch.job_registry import JobTransitionError

    job_id = str(arguments["job_id"])
    _job(runtime, job_id)
    registry = runtime.registry()
    try:
        assert registry is not None
        record = registry.cancel(
            workspace_id=runtime.config.workspace_id,
            job_id=job_id,
            expected_revision=_integer(arguments["expected_revision"]),
        )
    except KeyError:
        _reject(_INVALID, "job_unavailable")
    except JobTransitionError as error:
        _reject(
            _INVALID, "revision_conflict" if str(error) == "revision_conflict" else "terminal_job"
        )
    return {"job": _job_payload(runtime, record)}


def _integer(value: object) -> int:
    if type(value) is not int:
        _reject(_INVALID, "invalid_parameters")
    return value


# Operator-started worker and explicit resume.


def worker_pins(settings: BatchSettings) -> dict[str, PinnedExecutables]:
    """Snapshot executable identities for the configured video operations."""
    from capicove.runtime.executables import PinnedExecutables
    from capicove.runtime.video_mcp import _split_executables

    video = settings.video
    pins: dict[str, PinnedExecutables] = {}
    if video is None:
        return pins
    if "video_split" in settings.operations:
        pins["video_split"] = PinnedExecutables(_split_executables(video))
    if "video_filter" in settings.operations:
        assert video.ffprobe is not None
        pins["video_filter"] = PinnedExecutables({"ffprobe": video.ffprobe})
    return pins


def worker_runtime(config: WorkspaceHostConfiguration) -> BatchRuntime:
    """Validate the configuration and pin executables for one worker lifetime."""
    if "batch" not in config.capabilities:
        raise ValueError("capabilities.batch is not configured")
    if "batch:write" not in config.permissions:
        raise ValueError("the batch worker requires the batch:write permission")
    settings = batch_settings(config)
    return BatchRuntime(config, settings, pins=worker_pins(settings))


def _job_worker(
    runtime: BatchRuntime, record: JobRecord, *, worker_id: str, lease_seconds: float | None = None
) -> JobWorker:
    from capicove.batch._adapters.sqlite import SqliteBatchStore
    from capicove.batch.job_worker import JobWorker
    from capicove.batch.retry import RetryPolicy

    service, operation = runtime.route(record)
    return JobWorker(
        service,
        SqliteBatchStore(runtime.settings.registry_root / _CACHE),
        operation,
        worker_id=worker_id,
        max_workers=runtime.settings.concurrency,
        retry_policy=RetryPolicy(max_retries=2),
        lease_seconds=lease_seconds or runtime.settings.lease_seconds,
    )


def run_worker(
    runtime: BatchRuntime,
    stop: Event,
    *,
    worker_id: str | None = None,
    until_idle: bool = False,
    report: Callable[[str], None] = lambda line: print(line, file=sys.stderr, flush=True),
) -> None:
    """Recover expired leases and run queued jobs until ``stop`` is set.

    With ``until_idle`` the worker also returns once no routable job is queued
    and no lease has expired.

    A job that cannot be routed to the current configuration (for example an
    operation removed since it was queued) stays durable and is skipped, so it
    never blocks later jobs; the reason is reported once per worker lifetime.
    """
    from capicove.batch.job_registry import JobClaimLostError

    identity = worker_id or f"worker-{os.getpid()}-{secrets.token_hex(4)}"
    workspace_id = runtime.config.workspace_id
    skipped: set[str] = set()

    def skip(job_id: str, reason: str) -> None:
        if job_id not in skipped:
            skipped.add(job_id)
            report(json.dumps({"event": "job_skipped", "job_id": job_id, "reason": reason}))

    report(json.dumps({"event": "worker_started", "worker_id": identity}))
    while not stop.is_set():
        registry = runtime.registry()
        if registry is None:
            if until_idle:
                break
            stop.wait(runtime.settings.poll_seconds)
            continue
        now = datetime.now(UTC)
        expired = registry.next_expired(workspace_id=workspace_id, now=now)
        if expired is not None and expired.job_id not in skipped:
            try:
                worker = _job_worker(runtime, expired, worker_id=identity)
                recovered = worker.recover_next(now=now)
            except Exception as error:  # One job must not stop the worker.
                skip(expired.job_id, _reason(error))
            else:
                if recovered is not None:
                    report(_event("job_recovered", recovered))
                continue
        ran = False
        for queued in registry.queued_jobs(workspace_id=workspace_id, limit=_QUEUE_SCAN):
            if queued.job_id in skipped:
                continue
            try:
                worker = _job_worker(runtime, queued, worker_id=identity)
                report(json.dumps({"event": "job_started", "job_id": queued.job_id}))
                finished = worker.run_one(queued.job_id)
            except JobClaimLostError:
                continue
            except Exception as error:  # One job must not stop the worker.
                skip(queued.job_id, _reason(error))
                continue
            report(_event("job_finished", finished))
            ran = True
            break
        if not ran:
            if until_idle:
                break
            stop.wait(runtime.settings.poll_seconds)
    report(json.dumps({"event": "worker_stopped", "worker_id": identity}))


def resume_job(runtime: BatchRuntime, job_id: str, expected_revision: int) -> JobRecord:
    """Requeue one interrupted job after checkpoint and source verification."""
    from capicove.batch._adapters.sqlite import SqliteBatchStore
    from capicove.batch.job_checkpoint_store import JobCheckpointStores

    registry = runtime.registry()
    if registry is None:
        raise BatchUnavailable("job is unavailable")
    record = registry.status(workspace_id=runtime.config.workspace_id, job_id=job_id)
    service, _ = runtime.route(record)
    stores = JobCheckpointStores(SqliteBatchStore(runtime.settings.registry_root / _CACHE))
    return service.resume(
        job_id=job_id,
        expected_revision=expected_revision,
        checkpoint_store=stores.for_job(job_id, require_existing=record.checkpoint_prepared),
    )


def job_payload(runtime: BatchRuntime, record: JobRecord) -> dict[str, object]:
    """Encode one job record exactly as the status Tool does."""
    return _job_payload(runtime, record)


def _reason(error: Exception) -> str:
    if isinstance(error, BatchUnavailable):
        return str(error)
    message = str(error)
    return message if re.fullmatch(r"[a-z_ ]{1,64}", message) else type(error).__name__


def _event(name: str, record: JobRecord) -> str:
    return json.dumps(
        {
            "event": name,
            "job_id": record.job_id,
            "state": record.state,
            "completed_items": record.completed_items,
            "total_items": record.total_items,
        }
    )


__all__ = [
    "BatchRuntime",
    "BatchSettings",
    "BatchUnavailable",
    "batch_settings",
    "compose_batch_mcp",
    "job_payload",
    "resume_job",
    "root_sha256",
    "run_worker",
    "worker_runtime",
]
