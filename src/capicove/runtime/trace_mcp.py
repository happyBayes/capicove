"""Operator trace inspection and export for the workspace host (RFC-0016 amendment 1).

`capabilities.trace` names every trace store and private export root at startup.
Handlers delegate to the capability-owned `TraceOperatorReader` and
`TraceOperatorExporter`, so results stay content-free and exports stay
workspace-bound, idempotent, and confined to configured roots.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, NoReturn, TypeVar

from capicove.contracts import InvocationContext, ToolErrorCode, ToolHandler, ToolSpec
from capicove.mcp.bindings import McpToolBinding
from capicove.runtime._operator import Rejected, catalog_only_handler, operator_handler
from capicove.runtime.mcp_config import WorkspaceHostConfiguration


if TYPE_CHECKING:
    from capicove.export.validation import TraceValidationReport
    from capicove.traces._adapters.sqlite import SqliteTraceStore


_EnumT = TypeVar("_EnumT")
INSPECT = "capicove.trace.inspect"
VALIDATE = "capicove.trace.validate"
EXPORT = "capicove.trace.export"
_FIELDS = frozenset({"schema_version", "stores", "export_roots", "media_root"})
_ALIAS = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_EXPORT_ID = re.compile(r"export:([0-9a-f]{32})\Z")
_MESSAGES = {
    "unknown_store": "The trace store alias is not configured.",
    "unknown_root": "The export root alias is not configured.",
    "unknown_export": "The export is not present in a configured export root.",
    "inspection_failed": "The configured trace resource could not be read.",
    "restricted_denied": "Restricted Episodes need the admin profile and trace:export_restricted.",
    "copy_denied": "Copy mode needs the admin profile, trace:copy_media, and a media_root.",
    "invalid_filter": "The export filter is not accepted.",
    "idempotency_conflict": "idempotency_conflict: the key was used for a different export.",
    "export_failed": "The trace export failed; no partial package is reported as complete.",
}


@dataclass(frozen=True, slots=True)
class TraceSettings:
    """Validated startup authority for operator trace access."""

    stores: Mapping[str, Path]
    export_roots: Mapping[str, Path]
    media_root: Path | None


def trace_settings(config: WorkspaceHostConfiguration) -> TraceSettings:
    """Validate `capabilities.trace` without opening any store."""
    from capicove.runtime.video_mcp import _absolute, _existing_directory, _private_child

    if os.name != "posix":
        raise ValueError("trace host Tools require POSIX no-follow file APIs in this release")
    raw = config.capabilities.get("trace")
    if raw is None:
        raise ValueError("capabilities.trace is not configured")
    if set(raw) != _FIELDS:
        raise ValueError("capabilities.trace requires exactly the documented fields")
    if raw["schema_version"] != 1:
        raise ValueError("capabilities.trace schema_version must be 1")
    stores = _aliases(raw["stores"], "trace stores")
    roots = _aliases(raw["export_roots"], "trace export_roots")
    store_paths: dict[str, Path] = {}
    for alias, value in stores.items():
        path = _absolute(value, "trace stores")
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError("trace stores must not contain symlinks")
        try:
            metadata = path.lstat()
            parent = path.parent.lstat()
        except OSError:
            raise ValueError("trace stores must name existing SQLite files") from None
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
            raise ValueError("trace stores must be owner-private regular files")
        if parent.st_mode & 0o077:
            raise ValueError("trace stores need owner-private parent directories")
        store_paths[alias] = path
    export_roots = {
        alias: _private_child(value, config.workspace_root, "trace export_roots")
        for alias, value in roots.items()
    }
    media = raw["media_root"]
    media_root = None if media is None else _existing_directory(media, "trace media_root")
    for path in export_roots.values():
        for store in store_paths.values():
            if store.is_relative_to(path):
                raise ValueError("trace stores must be outside export_roots")
        if media_root is not None and (
            path.is_relative_to(media_root) or media_root.is_relative_to(path)
        ):
            raise ValueError("trace export_roots must be disjoint from media_root")
    return TraceSettings(MappingProxyType(store_paths), MappingProxyType(export_roots), media_root)


def _aliases(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not 1 <= len(value) <= 16:
        raise ValueError(f"{field} must map 1 to 16 aliases to paths")
    if any(not isinstance(alias, str) or not _ALIAS.match(alias) for alias in value):
        raise ValueError(f"{field} aliases must be lowercase identifiers")
    if len(set(value.values())) != len(value):
        raise ValueError(f"{field} must name distinct paths")
    return dict(value)


@contextmanager
def compose_trace_mcp(
    config: WorkspaceHostConfiguration,
    context_factory: Callable[[ToolSpec], InvocationContext],
    *,
    catalog_only: bool = False,
) -> Iterator[tuple[McpToolBinding, ...]]:
    """Validate trace authority and bind the selected operator Tools."""
    from capicove.tools.operator_traces import TRACE_SPECS

    if any(tool_id not in TRACE_SPECS for tool_id in config.enabled_tools):
        raise ValueError("trace profile includes an unavailable Tool")
    for tool_id in config.enabled_tools:
        if not TRACE_SPECS[tool_id].required_permissions.issubset(config.permissions):
            raise ValueError("configured permissions do not authorize every enabled Tool")
    settings = trace_settings(config)
    administrative = config.profile == "admin"
    bodies: dict[str, Callable[[InvocationContext, Mapping[str, object]], dict[str, object]]] = {
        INSPECT: lambda context, arguments: _inspect(settings, arguments),
        VALIDATE: lambda context, arguments: _validate(settings, arguments),
        EXPORT: lambda context, arguments: _export(
            settings, config.workspace_id, administrative, context, arguments
        ),
    }
    handlers: dict[str, ToolHandler] = {
        tool_id: catalog_only_handler
        if catalog_only
        else operator_handler(TRACE_SPECS[tool_id], bodies[tool_id])
        for tool_id in config.enabled_tools
    }
    yield tuple(
        McpToolBinding(
            "capicove_trace_" + tool_id.rsplit(".", 1)[1],
            TRACE_SPECS[tool_id],
            handlers[tool_id],
            context_factory,
        )
        for tool_id in sorted(config.enabled_tools)
    )


def _reject(code: ToolErrorCode, reason: str) -> NoReturn:
    raise Rejected(code, _MESSAGES[reason])


def _store(settings: TraceSettings, alias: object) -> SqliteTraceStore:
    from capicove.traces import TraceStoreError
    from capicove.traces._adapters.sqlite import SqliteTraceStore

    path = settings.stores.get(str(alias))
    if path is None:
        _reject(ToolErrorCode.INVALID_INPUT, "unknown_store")
    try:
        return SqliteTraceStore(path)
    except (TraceStoreError, OSError, ValueError):
        _reject(ToolErrorCode.UNAVAILABLE, "inspection_failed")


def _export_directory(settings: TraceSettings, export_id: str) -> Path:
    """Locate one package by ID strictly inside the configured export roots."""
    matched = _EXPORT_ID.fullmatch(export_id)
    if matched is None:
        _reject(ToolErrorCode.INVALID_INPUT, "unknown_export")
    found = [
        root / f"trace-{matched.group(1)}"
        for root in settings.export_roots.values()
        if (root / f"trace-{matched.group(1)}").is_dir()
        and not (root / f"trace-{matched.group(1)}").is_symlink()
    ]
    if len(found) != 1:
        _reject(ToolErrorCode.INVALID_INPUT, "unknown_export")
    return found[0]


def _inspect(settings: TraceSettings, arguments: Mapping[str, object]) -> dict[str, object]:
    from capicove.traces.operator import TraceOperatorReader

    if "store" in arguments:
        alias = str(arguments["store"])
        reader = TraceOperatorReader(stores={alias: _store(settings, alias)})
        try:
            inspection = reader.inspect_store(alias)
        except ValueError:
            _reject(ToolErrorCode.UNAVAILABLE, "inspection_failed")
        return {
            "kind": "store",
            "store": alias,
            "episode_count": inspection.episode_count,
            "incomplete_episode_count": inspection.incomplete_episode_count,
            "omitted_restricted_episode_count": inspection.omitted_restricted_episode_count,
        }
    return _validate(settings, {"export_id": arguments["export_id"], "verify_media": False})


def _validate(settings: TraceSettings, arguments: Mapping[str, object]) -> dict[str, object]:
    from capicove.export import inspect_trace_export
    from capicove.traces.operator import TraceOperatorReader

    export_id = str(arguments["export_id"])
    directory = _export_directory(settings, export_id)

    def validator(verify_media: bool) -> TraceValidationReport:
        manifest, report = inspect_trace_export(directory, verify_media=verify_media)
        if manifest.export_id != export_id:
            raise ValueError("export identity mismatch")
        return report

    reader = TraceOperatorReader(stores={}, exports={"package": validator})
    try:
        inspection = reader.validate_export(
            "package", verify_media=bool(arguments.get("verify_media", True))
        )
    except ValueError:
        _reject(ToolErrorCode.UNAVAILABLE, "inspection_failed")
    return {
        "kind": "export",
        "export_id": export_id,
        "valid": inspection.valid,
        "records_checked": inspection.records_checked,
        "episodes_checked": inspection.episodes_checked,
        "media_checked": inspection.media_checked,
        "findings": [
            {"severity": item.severity, "code": item.code} for item in inspection.findings
        ],
        "findings_truncated": inspection.findings_truncated,
    }


def _export(
    settings: TraceSettings,
    workspace_id: str,
    administrative: bool,
    context: InvocationContext,
    arguments: Mapping[str, object],
) -> dict[str, object]:
    from capicove.contracts import ToolStatus
    from capicove.export import MediaMode
    from capicove.runtime.media_handoff import MediaDenied, open_private_directory
    from capicove.traces import EpisodeStatus, TraceModality, TraceQuery
    from capicove.traces.operator import TraceOperatorExporter

    alias = str(arguments["output_root"])
    root = settings.export_roots.get(alias)
    if root is None:
        _reject(ToolErrorCode.INVALID_INPUT, "unknown_root")
    mode = MediaMode(str(arguments.get("media_mode", "reference")))
    restricted = arguments.get("include_restricted") is True
    permissions = frozenset(context.permissions)
    # Deny before opening the store or creating the export root.
    if restricted and not (administrative and "trace:export_restricted" in permissions):
        _reject(ToolErrorCode.PERMISSION_DENIED, "restricted_denied")
    if mode is MediaMode.COPY and not (
        administrative and "trace:copy_media" in permissions and settings.media_root is not None
    ):
        _reject(ToolErrorCode.PERMISSION_DENIED, "copy_denied")

    def strings(name: str) -> frozenset[str] | None:
        value = arguments.get(name)
        return frozenset(str(item) for item in value) if isinstance(value, list) else None

    try:
        query = TraceQuery(
            episode_ids=strings("episode_ids"),
            started_after=_timestamp(arguments.get("started_after")),
            completed_before=_timestamp(arguments.get("completed_before")),
            modalities=_enums(strings("modalities"), TraceModality),
            tool_ids=strings("tool_ids"),
            tool_statuses=_enums(strings("tool_statuses"), ToolStatus),
            episode_statuses=_enums(strings("episode_statuses"), EpisodeStatus),
            include_incomplete=arguments.get("include_incomplete") is True,
            include_restricted=restricted,
        )
    except ValueError:
        _reject(ToolErrorCode.INVALID_INPUT, "invalid_filter")
    store = _store(settings, arguments["store"])
    try:
        os.close(open_private_directory(root))
    except MediaDenied:
        _reject(ToolErrorCode.PERMISSION_DENIED, "unknown_root")
    exporter = TraceOperatorExporter(
        reader=store,
        workspace_id=workspace_id,
        output_roots={alias: root},
        media_root=settings.media_root,
        administrative=administrative,
        permissions=permissions,
    )
    try:
        result = exporter.export(
            output_root_alias=alias,
            query=query,
            idempotency_key=str(arguments["idempotency_key"]),
            media_mode=mode,
        )
    except PermissionError:
        _reject(ToolErrorCode.PERMISSION_DENIED, "restricted_denied")
    except ValueError as error:
        if str(error) == "idempotency_conflict":
            _reject(ToolErrorCode.INVALID_INPUT, "idempotency_conflict")
        _reject(ToolErrorCode.UNAVAILABLE, "export_failed")
    return {
        "export_id": result.export_id,
        "output_root": alias,
        "included_episode_count": result.included_episode_count,
        "incomplete_episode_count": result.incomplete_episode_count,
        "omitted_restricted_episode_count": result.omitted_restricted_episode_count,
        "record_count": result.record_count,
        "shard_count": result.shard_count,
        "media_count": result.media_count,
        "missing_media_count": result.missing_media_count,
        "media_mode": result.media_mode,
    }


def _timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).removesuffix("Z")
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def _enums(
    values: frozenset[str] | None, kind: Callable[[str], _EnumT]
) -> frozenset[_EnumT] | None:
    return None if values is None else frozenset(kind(value) for value in values)


__all__ = ["TraceSettings", "compose_trace_mcp", "trace_settings"]
