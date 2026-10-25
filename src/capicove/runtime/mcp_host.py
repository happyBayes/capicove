"""Closed, process-owned MCP composition with startup-bound identity."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from capicove.contracts import (
    InvocationContext,
    SideEffect,
    ToolError,
    ToolErrorCode,
    ToolResult,
    ToolSpec,
    ToolStatus,
)
from capicove.mcp.bindings import McpToolBinding, validate_bindings
from capicove.runtime.mcp_config import (
    McpHostConfiguration,
    WorkspaceHostConfiguration,
    _require_secure_file_apis,
    tool_capability,
)


@contextmanager
def compose_mcp_host(
    config: McpHostConfiguration | WorkspaceHostConfiguration, *, catalog_only: bool = False
) -> Iterator[tuple[McpToolBinding, ...]]:
    """Own bindings for one server lifetime; catalog discovery never opens a trace DB."""
    _require_secure_file_apis()

    def context_factory(spec: ToolSpec) -> InvocationContext:
        return InvocationContext(
            str(uuid.uuid4()),
            str(uuid.uuid4()),
            config.workspace_id,
            config.actor_id,
            config.environment_id,
            datetime.now(UTC) + timedelta(seconds=spec.timeout_seconds),
            config.permissions,
            spec.privacy_class,
        )

    composed: AbstractContextManager[tuple[McpToolBinding, ...]]
    if isinstance(config, WorkspaceHostConfiguration):
        composed = _compose_workspace(config, context_factory, catalog_only=catalog_only)
    else:
        composed = _compose_one(config, context_factory, catalog_only=catalog_only)
    with composed as bindings:
        validate_bindings(bindings)
        if config.trace_store is not None:
            path = config.trace_store
            if os.name == "nt":
                from capicove._windows_files import (
                    WindowsPathPolicyError,
                    validate_windows_file_target,
                )

                try:
                    validate_windows_file_target(path)
                except (OSError, WindowsPathPolicyError) as error:
                    raise ValueError(
                        "trace_store requires an existing private Windows parent directory"
                    ) from error
            else:
                if any(part.is_symlink() for part in (path, *path.parents)):
                    raise ValueError("trace_store must not contain symlinks")
                if not path.parent.is_dir() or path.parent.stat().st_mode & 0o077:
                    raise ValueError("trace_store requires an existing owner-only parent directory")
                if path.exists() and not path.is_file():
                    raise ValueError("trace_store must be a regular file")
            if not catalog_only:
                from capicove.traces import TraceRecorder, TraceSanitizer
                from capicove.traces._adapters.sqlite import SqliteTraceStore

                # The existing store opens and closes a connection per operation;
                # it has no persistent connection or shutdown callback.
                recorder = TraceRecorder(
                    SqliteTraceStore(path),
                    sanitizer=TraceSanitizer(),
                    clock=lambda: datetime.now(UTC),
                )
                bindings = tuple(replace(item, recorder=recorder) for item in bindings)
        yield bindings


def _retrieval_factory() -> Callable[..., AbstractContextManager[tuple[McpToolBinding, ...]]]:
    from capicove.runtime.retrieval_mcp import compose_retrieval_mcp

    return compose_retrieval_mcp


def _video_factory() -> Callable[..., AbstractContextManager[tuple[McpToolBinding, ...]]]:
    from capicove.runtime.video_mcp import compose_video_mcp

    return compose_video_mcp


def _audio_factory() -> Callable[..., AbstractContextManager[tuple[McpToolBinding, ...]]]:
    from capicove.runtime.audio_mcp import compose_audio_mcp

    return compose_audio_mcp


def _batch_factory() -> Callable[..., AbstractContextManager[tuple[McpToolBinding, ...]]]:
    from capicove.runtime.batch_mcp import compose_batch_mcp

    return compose_batch_mcp


def _trace_factory() -> Callable[..., AbstractContextManager[tuple[McpToolBinding, ...]]]:
    from capicove.runtime.trace_mcp import compose_trace_mcp

    return compose_trace_mcp


def _model_factory() -> Callable[..., AbstractContextManager[tuple[McpToolBinding, ...]]]:
    from capicove.runtime.model_mcp import compose_model_mcp

    return compose_model_mcp


def _multimodal_factory() -> Callable[..., AbstractContextManager[tuple[McpToolBinding, ...]]]:
    from capicove.runtime.multimodal_mcp import compose_multimodal_mcp

    return compose_multimodal_mcp


def _knowledge_factory() -> Callable[..., AbstractContextManager[tuple[McpToolBinding, ...]]]:
    from capicove.runtime.knowledge_mcp import compose_knowledge_mcp

    return compose_knowledge_mcp


_WORKSPACE_CAPABILITIES: dict[
    str, Callable[[], Callable[..., AbstractContextManager[tuple[McpToolBinding, ...]]]]
] = {
    "retrieval": _retrieval_factory,
    "video": _video_factory,
    "audio": _audio_factory,
    "batch": _batch_factory,
    "trace": _trace_factory,
    "model": _model_factory,
    "knowledge": _knowledge_factory,
    "multimodal": _multimodal_factory,
}


def _compose_one(
    config: McpHostConfiguration,
    context_factory: Callable[[ToolSpec], InvocationContext],
    *,
    catalog_only: bool,
) -> AbstractContextManager[tuple[McpToolBinding, ...]]:
    if config.capability == "memory" and config.trace_store is not None:
        raise ValueError("memory MCP tracing is unavailable in this release")
    if config.capability == "image":
        from capicove.runtime.image_mcp import compose_image_mcp

        return compose_image_mcp(config, context_factory, None)
    if config.capability == "memory":
        from capicove.runtime.memory_mcp import compose_memory_mcp

        return compose_memory_mcp(config, context_factory, catalog_only=catalog_only)
    raise ValueError("MCP capability is unavailable; select image or memory")


def _open_posix_workspace_root(root: Path) -> int:
    """Pin every directory component without following a replacement link."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        descriptor = os.open(root.anchor, flags)
    except OSError:
        raise ValueError(
            "workspace_root must be an existing local directory without symlinks"
        ) from None
    try:
        for component in root.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except OSError:
        os.close(descriptor)
        raise ValueError(
            "workspace_root must be an existing local directory without symlinks"
        ) from None


def _require_current_posix_workspace_root(root: Path, pinned: int) -> None:
    """Reject a selected workspace path replaced during factory composition."""
    try:
        current = _open_posix_workspace_root(root)
    except ValueError:
        raise ValueError("workspace_root changed during startup") from None
    try:
        before = os.fstat(pinned)
        after = os.fstat(current)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise ValueError("workspace_root changed during startup")
    finally:
        os.close(current)


def _guard_posix_workspace_binding(item: McpToolBinding, root: Path, pinned: int) -> McpToolBinding:
    """Deny a selected Tool if the startup workspace path was replaced."""
    handler = item.handler

    def denied(context: InvocationContext) -> ToolResult:
        return ToolResult(
            request_id=context.request_id,
            trace_id=context.trace_id,
            tool_id=item.spec.tool_id,
            tool_version=item.spec.contract_version,
            status=ToolStatus.REJECTED,
            error=ToolError(
                ToolErrorCode.PERMISSION_DENIED,
                "The configured workspace root changed; restart the MCP host.",
                False,
            ),
        )

    def guarded(context: InvocationContext, arguments: Mapping[str, object]) -> ToolResult:
        try:
            _require_current_posix_workspace_root(root, pinned)
        except ValueError:
            return denied(context)
        return handler(context, arguments)

    return replace(item, handler=guarded)


@contextmanager
def _compose_workspace(
    config: WorkspaceHostConfiguration,
    context_factory: Callable[[ToolSpec], InvocationContext],
    *,
    catalog_only: bool,
) -> Iterator[tuple[McpToolBinding, ...]]:
    root = config.workspace_root
    with ExitStack() as stack:
        root_fd: int | None = None
        if os.name == "nt":
            from capicove._windows_files import validate_windows_path

            validate_windows_path(root, directory=True, private_acl=config.profile != "read")
        else:
            root_fd = _open_posix_workspace_root(root)
            stack.callback(os.close, root_fd)
            if config.profile != "read" and os.fstat(root_fd).st_mode & 0o077:
                raise ValueError("writable workspace_root must be owner-private")
        if config.trace_store is not None and any(
            tool.startswith("capicove.memory.") for tool in config.enabled_tools
        ):
            raise ValueError("memory MCP tracing is unavailable in this release")
        all_bindings: list[McpToolBinding] = []
        for capability in sorted(config.capabilities):
            selected = tuple(
                tool for tool in config.enabled_tools if tool_capability(tool) == capability
            )
            if not selected:
                continue
            if capability in _WORKSPACE_CAPABILITIES:
                # These capabilities need the version-2 workspace root, so they never
                # run under a version-1 configuration.
                compose = _WORKSPACE_CAPABILITIES[capability]()
                all_bindings.extend(
                    stack.enter_context(
                        compose(
                            replace(config, enabled_tools=selected),
                            context_factory,
                            catalog_only=catalog_only,
                        )
                    )
                )
                continue
            raw = config.capabilities[capability]
            subconfig = McpHostConfiguration(
                1,
                config.server_name,
                capability,
                config.workspace_id,
                config.actor_id,
                config.environment_id,
                config.permissions,
                selected,
                None,
                {key: value for key, value in raw.items() if key != "schema_version"},
            )
            all_bindings.extend(
                stack.enter_context(
                    _compose_one(subconfig, context_factory, catalog_only=catalog_only)
                )
            )
        if config.profile == "read" and any(
            item.spec.side_effect not in {SideEffect.NONE, SideEffect.READ} for item in all_bindings
        ):
            raise ValueError("read profile cannot enable a writing Tool")
        if any(item.spec.network_access for item in all_bindings) and not config.network["enabled"]:
            raise ValueError("network Tool requires explicit network authority")
        validate_bindings(tuple(all_bindings))
        if root_fd is not None:
            _require_current_posix_workspace_root(root, root_fd)
            all_bindings = [
                _guard_posix_workspace_binding(item, root, root_fd) for item in all_bindings
            ]
        yield tuple(sorted(all_bindings, key=lambda item: item.spec.tool_id))
