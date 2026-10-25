"""Operator model plan, verification, and installation (RFC-0016 amendment 1).

`capabilities.model` names private destination roots and an install allowlist
of exact plan digests. Plans and verification never touch the network.
Installation requires the admin profile, `model:install`, enabled network
authority covering the plan's source hosts, and a digest the operator added to
the configuration; the capability-owned plugin installer does the transfer.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, NoReturn

from capicove.contracts import InvocationContext, ToolErrorCode, ToolHandler, ToolSpec
from capicove.mcp.bindings import McpToolBinding
from capicove.runtime._operator import Rejected, catalog_only_handler, operator_handler
from capicove.runtime.mcp_config import WorkspaceHostConfiguration


PLAN = "capicove.model.plan"
VERIFY = "capicove.model.verify"
INSTALL = "capicove.model.install"
_FIELDS = frozenset({"schema_version", "destinations", "install_allowlist"})
_ENTRY_FIELDS = frozenset({"plan_digest", "plugin", "profile", "destination", "source_hosts"})
_ALIAS = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_PLUGIN = re.compile(r"[a-z0-9][a-z0-9.-]{0,63}\Z")
_PROFILE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MESSAGES = {
    "unknown_destination": "The destination alias is not configured.",
    "plugin_unavailable": "The model installer plugin is unavailable.",
    "plan_failed": "The model plan could not be computed from the installed plugin.",
    "not_allowlisted": "The plan digest is not allowlisted for this destination.",
    "plan_changed": "plan_changed: the installed plugin now produces a different plan.",
    "network_denied": "The plan's source hosts are not authorized by network.source_hosts.",
    "install_failed": "Model installation failed; the destination is not qualified.",
}


def _load_plugin(name: str) -> object:
    from capicove.model.installers import load_model_installer

    return load_model_installer(name)


# Replaced in tests with a synthetic installer; production loads entry points.
PLUGIN_LOADER: Callable[[str], object] = _load_plugin


@dataclass(frozen=True, slots=True)
class AllowedInstall:
    """One operator-approved installation."""

    plan_digest: str
    plugin: str
    profile: str
    destination: str
    source_hosts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """Validated startup authority for model management."""

    destinations: Mapping[str, Path]
    install_allowlist: tuple[AllowedInstall, ...]


def model_settings(config: WorkspaceHostConfiguration) -> ModelSettings:
    """Validate `capabilities.model` without loading any plugin."""
    from capicove.runtime.video_mcp import _private_child

    if os.name != "posix":
        raise ValueError("model host Tools require POSIX no-follow file APIs in this release")
    raw = config.capabilities.get("model")
    if raw is None:
        raise ValueError("capabilities.model is not configured")
    if set(raw) != _FIELDS:
        raise ValueError("capabilities.model requires exactly the documented fields")
    if raw["schema_version"] != 1:
        raise ValueError("capabilities.model schema_version must be 1")
    destinations = raw["destinations"]
    if (
        not isinstance(destinations, Mapping)
        or not 1 <= len(destinations) <= 16
        or any(not isinstance(alias, str) or not _ALIAS.match(alias) for alias in destinations)
        or len(set(destinations.values())) != len(destinations)
    ):
        raise ValueError("model destinations must map 1 to 16 aliases to distinct paths")
    paths = {
        alias: _private_child(value, config.workspace_root, "model destinations")
        for alias, value in destinations.items()
    }
    entries = raw["install_allowlist"]
    if not isinstance(entries, tuple) or len(entries) > 64:
        raise ValueError("model install_allowlist must be an array of at most 64 entries")
    allowed: list[AllowedInstall] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != _ENTRY_FIELDS:
            raise ValueError(
                "model install_allowlist entries require exactly the documented fields"
            )
        hosts = entry["source_hosts"]
        if (
            not isinstance(entry["plan_digest"], str)
            or not _SHA256.match(entry["plan_digest"])
            or not isinstance(entry["plugin"], str)
            or not _PLUGIN.match(entry["plugin"])
            or not isinstance(entry["profile"], str)
            or not _PROFILE.match(entry["profile"])
            or entry["destination"] not in paths
            or not isinstance(hosts, tuple)
            or not hosts
            or any(not isinstance(host, str) for host in hosts)
            or tuple(sorted(set(hosts))) != hosts
        ):
            raise ValueError("model install_allowlist entry is invalid")
        allowed.append(
            AllowedInstall(
                entry["plan_digest"],
                entry["plugin"],
                entry["profile"],
                str(entry["destination"]),
                tuple(str(host) for host in hosts),
            )
        )
    return ModelSettings(MappingProxyType(paths), tuple(allowed))


@contextmanager
def compose_model_mcp(
    config: WorkspaceHostConfiguration,
    context_factory: Callable[[ToolSpec], InvocationContext],
    *,
    catalog_only: bool = False,
) -> Iterator[tuple[McpToolBinding, ...]]:
    """Validate model authority and bind the selected operator Tools."""
    from capicove.tools.operator_models import MODEL_SPECS

    if any(tool_id not in MODEL_SPECS for tool_id in config.enabled_tools):
        raise ValueError("model profile includes an unavailable Tool")
    for tool_id in config.enabled_tools:
        if not MODEL_SPECS[tool_id].required_permissions.issubset(config.permissions):
            raise ValueError("configured permissions do not authorize every enabled Tool")
    settings = model_settings(config)
    if INSTALL in config.enabled_tools:
        if config.profile != "admin" or not config.network["enabled"]:
            raise ValueError("model install requires an admin profile with network enabled")
        authorized = set(config.network["source_hosts"])  # type: ignore[call-overload]
        if any(not set(entry.source_hosts) <= authorized for entry in settings.install_allowlist):
            raise ValueError("model install_allowlist source_hosts must be in network.source_hosts")
    workspace_id = config.workspace_id
    bodies: dict[str, Callable[[InvocationContext, Mapping[str, object]], dict[str, object]]] = {
        PLAN: lambda context, arguments: _plan(settings, workspace_id, arguments),
        VERIFY: lambda context, arguments: _verify(settings, workspace_id, arguments),
        INSTALL: lambda context, arguments: _install(settings, workspace_id, arguments),
    }
    handlers: dict[str, ToolHandler] = {
        tool_id: catalog_only_handler
        if catalog_only
        else operator_handler(MODEL_SPECS[tool_id], bodies[tool_id])
        for tool_id in config.enabled_tools
    }
    yield tuple(
        McpToolBinding(
            "capicove_model_" + tool_id.rsplit(".", 1)[1],
            MODEL_SPECS[tool_id],
            handlers[tool_id],
            context_factory,
        )
        for tool_id in sorted(config.enabled_tools)
    )


def _reject(code: ToolErrorCode, reason: str) -> NoReturn:
    raise Rejected(code, _MESSAGES[reason])


def _installer(name: str) -> Any:
    from capicove.model.installers import ModelInstaller

    try:
        plugin = PLUGIN_LOADER(name)
    except Exception:
        _reject(ToolErrorCode.UNAVAILABLE, "plugin_unavailable")
    if not isinstance(plugin, ModelInstaller):
        _reject(ToolErrorCode.UNAVAILABLE, "plugin_unavailable")
    return plugin


def _destination(settings: ModelSettings, alias: str, plugin: str, profile: str) -> Path:
    root = settings.destinations.get(alias)
    if root is None:
        _reject(ToolErrorCode.INVALID_INPUT, "unknown_destination")
    return root / plugin / profile


def _plan_body(
    settings: ModelSettings, workspace_id: str, plugin: str, profile: str, destination: str
) -> tuple[Any, dict[str, object]]:
    path = _destination(settings, destination, plugin, profile)
    installer = _installer(plugin)
    try:
        plan = installer.model_plan(profile, path)
        models = [
            {
                "key": str(model.key),
                "repository": str(model.repository),
                "revision": str(model.revision),
                "size_bytes": int(model.size_bytes),
                "files": [
                    {
                        "path": str(item.path),
                        "size_bytes": int(item.size_bytes),
                        "sha256": item.sha256,
                    }
                    for item in model.files
                ],
            }
            for model in plan.models
        ]
        body: dict[str, object] = {
            "plugin": plugin,
            "profile": profile,
            "profile_version": int(plan.profile_version),
            "destination": destination,
            "manifest_sha256": str(plan.manifest_sha256),
            "models": models,
            "source_hosts": sorted(str(host) for host in plan.source_hosts),
            "total_size_bytes": int(plan.total_size_bytes),
        }
    except Rejected:
        raise
    except Exception:
        _reject(ToolErrorCode.INVALID_INPUT, "plan_failed")
    identity = {"schema": "capicove.model.plan.v1", "workspace_id": workspace_id, **body}
    canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    body["plan_digest"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return plan, body


def _plan(
    settings: ModelSettings, workspace_id: str, arguments: Mapping[str, object]
) -> dict[str, object]:
    _, body = _plan_body(
        settings,
        workspace_id,
        str(arguments["plugin"]),
        str(arguments["profile"]),
        str(arguments["destination"]),
    )
    body["install_allowlisted"] = any(
        entry.plan_digest == body["plan_digest"] and entry.destination == body["destination"]
        for entry in settings.install_allowlist
    )
    return body


def _verify(
    settings: ModelSettings, workspace_id: str, arguments: Mapping[str, object]
) -> dict[str, object]:
    plugin, profile = str(arguments["plugin"]), str(arguments["profile"])
    destination = str(arguments["destination"])
    _, body = _plan_body(settings, workspace_id, plugin, profile, destination)
    return _verification(body, _destination(settings, destination, plugin, profile))


def _verification(body: Mapping[str, object], root: Path) -> dict[str, object]:
    """Count planned files that are present, hash-identical, and not linked."""
    counts = {"verified": 0, "missing": 0, "mismatched": 0}
    models = body["models"]
    assert isinstance(models, list)
    for model in models:
        for item in model["files"]:
            relative = PurePosixPath(item["path"])
            if relative.is_absolute() or ".." in relative.parts:
                counts["mismatched"] += 1
                continue
            counts[_file_state(root / model["key"] / relative, item)] += 1
    return {
        "plan_digest": body["plan_digest"],
        "destination": body["destination"],
        "manifest_sha256": body["manifest_sha256"],
        "verified_files": counts["verified"],
        "missing_files": counts["missing"],
        "mismatched_files": counts["mismatched"],
        "complete": counts["missing"] == counts["mismatched"] == 0,
    }


def _file_state(path: Path, expected: Mapping[str, object]) -> str:
    if any(part.is_symlink() for part in (path, *path.parents)):
        return "mismatched"
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return "missing"
    except OSError:
        return "mismatched"
    digest = hashlib.sha256()
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != expected["size_bytes"]:
            return "mismatched"
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "verified" if digest.hexdigest() == expected["sha256"] else "mismatched"


def _install(
    settings: ModelSettings, workspace_id: str, arguments: Mapping[str, object]
) -> dict[str, object]:
    from capicove.runtime.media_handoff import MediaDenied, open_private_directory

    digest, destination = str(arguments["plan_digest"]), str(arguments["destination"])
    entry = next(
        (
            item
            for item in settings.install_allowlist
            if item.plan_digest == digest and item.destination == destination
        ),
        None,
    )
    if entry is None:
        _reject(ToolErrorCode.POLICY_REJECTED, "not_allowlisted")
    _, body = _plan_body(settings, workspace_id, entry.plugin, entry.profile, destination)
    if body["plan_digest"] != digest:
        _reject(ToolErrorCode.POLICY_REJECTED, "plan_changed")
    if tuple(body["source_hosts"]) != entry.source_hosts:  # type: ignore[arg-type]
        _reject(ToolErrorCode.PERMISSION_DENIED, "network_denied")
    target = _destination(settings, destination, entry.plugin, entry.profile)
    try:
        for path in (settings.destinations[destination], target.parent, target):
            os.close(open_private_directory(path))
    except MediaDenied:
        _reject(ToolErrorCode.PERMISSION_DENIED, "unknown_destination")
    try:
        _installer(entry.plugin).install_models(entry.profile, target, confirmed=True)
    except Rejected:
        raise
    except Exception:
        _reject(ToolErrorCode.UNAVAILABLE, "install_failed")
    result = _verification(body, target)
    if not result["complete"]:
        _reject(ToolErrorCode.UNAVAILABLE, "install_failed")
    return result


__all__ = ["AllowedInstall", "ModelSettings", "compose_model_mcp", "model_settings"]
