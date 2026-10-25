"""Bounded, explicit startup configuration for the local MCP host."""

from __future__ import annotations

import json
import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO, cast


MAX_CONFIG_BYTES = 64 * 1024
_FIELDS = frozenset(
    {
        "schema_version",
        "server_name",
        "capability",
        "workspace_id",
        "actor_id",
        "environment_id",
        "permissions",
        "enabled_tools",
        "trace_store",
        "capability_config",
    }
)
_WORKSPACE_FIELDS = frozenset(
    {
        "schema_version",
        "server_name",
        "workspace_id",
        "actor_id",
        "environment_id",
        "workspace_root",
        "profile",
        "permissions",
        "enabled_tools",
        "trace_store",
        "capabilities",
        "network",
    }
)


# Canonical Tool IDs owned by a capability whose name is not the ID's second segment.
_TOOL_CAPABILITIES = MappingProxyType(
    {
        "capicove.search": "retrieval",
        "capicove.get_evidence": "retrieval",
        "capicove.search.multimodal": "multimodal",
    }
)


def tool_capability(tool_id: str) -> str:
    """Return the version-2 capability that owns one canonical Tool ID."""
    return _TOOL_CAPABILITIES.get(tool_id, tool_id.split(".", 2)[1])


def _freeze(value: object, depth: int = 0) -> object:
    if depth > 32:
        raise ValueError("capability_config nesting exceeds 32 levels")
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("capability_config keys must be strings")
        return MappingProxyType({key: _freeze(item, depth + 1) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, depth + 1) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("capability_config numbers must be finite")
    if value is not None and not isinstance(value, (str, bool, int, float)):
        raise ValueError("capability_config must contain JSON values")
    return value


def _strings(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{field} must contain nonblank strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{field} must not contain duplicates")
    return tuple(value)


_CAPABILITY_NAMES = frozenset(
    {
        "image",
        "memory",
        "retrieval",
        "video",
        "audio",
        "batch",
        "trace",
        "model",
        "knowledge",
        "multimodal",
    }
)


@dataclass(frozen=True, slots=True)
class McpHostConfiguration:
    """Carry immutable authority chosen by the operator, never by Tool arguments."""

    schema_version: int
    server_name: str
    capability: str
    workspace_id: str
    actor_id: str
    environment_id: str
    permissions: frozenset[str]
    enabled_tools: tuple[str, ...]
    trace_store: Path | None
    capability_config: Mapping[str, object]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        for field in ("server_name", "capability", "workspace_id", "actor_id", "environment_id"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip() or len(value) > 128:
                raise ValueError(f"{field} must be a nonblank string of at most 128 characters")
        for value in (self.server_name, self.capability):
            if re.fullmatch(r"[a-z][a-z0-9_-]*", value) is None:
                raise ValueError("server_name and capability must be lowercase identifiers")
        if not isinstance(self.permissions, (set, frozenset)):
            raise ValueError("permissions must be a set of strings")
        permissions = _strings(tuple(self.permissions), "permissions")
        enabled = _strings(self.enabled_tools, "enabled_tools")
        if not enabled or any(
            re.fullmatch(r"capicove\.[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*", value) is None
            for value in enabled
        ):
            raise ValueError("enabled_tools must contain canonical Capicove Tool IDs")
        if self.trace_store is not None and (
            not isinstance(self.trace_store, Path) or not self.trace_store.is_absolute()
        ):
            raise ValueError("trace_store must be an absolute path or null")
        if not isinstance(self.capability_config, Mapping):
            raise ValueError("capability_config must be an object")
        object.__setattr__(self, "permissions", frozenset(permissions))
        object.__setattr__(self, "enabled_tools", enabled)
        object.__setattr__(self, "capability_config", _freeze(dict(self.capability_config)))


@dataclass(frozen=True, slots=True)
class WorkspaceHostConfiguration:
    """Carry version-2 multi-capability startup authority."""

    schema_version: int
    server_name: str
    workspace_id: str
    actor_id: str
    environment_id: str
    workspace_root: Path
    profile: str
    permissions: frozenset[str]
    enabled_tools: tuple[str, ...]
    trace_store: Path | None
    capabilities: Mapping[str, Mapping[str, object]]
    network: Mapping[str, object]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError("workspace schema_version must be 2")
        for field in ("server_name", "workspace_id", "actor_id", "environment_id"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > 128:
                raise ValueError(f"{field} must be a nonblank string of at most 128 bytes")
        if re.fullmatch(r"[a-z][a-z0-9_-]*", self.server_name) is None:
            raise ValueError("server_name must be a lowercase identifier")
        if not isinstance(self.workspace_root, Path) or not self.workspace_root.is_absolute():
            raise ValueError("workspace_root must be an absolute path")
        if not isinstance(self.profile, str) or self.profile not in {"read", "write", "admin"}:
            raise ValueError("profile must be read, write, or admin")
        if not isinstance(self.permissions, (set, frozenset)):
            raise ValueError("permissions must be a set of strings")
        permissions = _strings(tuple(self.permissions), "permissions")
        # Every profile may read evidence and run selected read-only media Tools;
        # artifact writes (for example published video segments) need write or admin.
        reads = {"image:read", "memory:read", "evidence:read", "evidence:search"}
        reads |= {"video:read", "subprocess:execute"}
        # RFC-0016 amendment 1: operator jobs, trace access, and model management.
        reads |= {"batch:read", "trace:read", "model:read"}
        writes = reads | {"memory:write", "artifact:write", "audio:read", "audio:write"}
        writes |= {"batch:write", "trace:export"}
        admin = writes | {"memory:erase", "trace:export_restricted", "trace:copy_media"}
        admin |= {"model:install"}
        profile_permissions = {
            "read": reads,
            "write": writes,
            "admin": admin,
        }
        if not set(permissions).issubset(profile_permissions[self.profile]):
            raise ValueError("permissions exceed the selected profile ceiling")
        enabled = _strings(self.enabled_tools, "enabled_tools")
        if not enabled or any(
            re.fullmatch(r"capicove\.[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*", item) is None
            for item in enabled
        ):
            raise ValueError("enabled_tools must contain canonical Capicove Tool IDs")
        if self.trace_store is not None and (
            not isinstance(self.trace_store, Path) or not self.trace_store.is_absolute()
        ):
            raise ValueError("trace_store must be an absolute path or null")
        if not isinstance(self.capabilities, Mapping) or not self.capabilities:
            raise ValueError("capabilities must be a nonempty object")
        if set(self.capabilities) - _CAPABILITY_NAMES:
            raise ValueError("capabilities contains an unavailable capability")
        capability_configs: dict[str, Mapping[str, object]] = {}
        for name, raw in self.capabilities.items():
            if not isinstance(raw, Mapping) or type(raw.get("schema_version")) is not int:
                raise ValueError("capability config requires an integer schema_version")
            if raw["schema_version"] not in ((1, 2) if name in {"audio", "retrieval"} else (1,)):
                raise ValueError("capability config version is unavailable")
            capability_configs[name] = cast(Mapping[str, object], _freeze(dict(raw)))
        if any(tool_capability(item) not in capability_configs for item in enabled):
            raise ValueError("enabled_tools must belong to configured capabilities")
        if not isinstance(self.network, Mapping) or set(self.network) != {
            "enabled",
            "source_hosts",
        }:
            raise ValueError("network requires enabled and source_hosts")
        network_enabled = self.network["enabled"]
        hosts = _strings(self.network["source_hosts"], "source_hosts")
        if type(network_enabled) is not bool or (network_enabled and self.profile != "admin"):
            raise ValueError("network requires an administrative profile")
        if tuple(sorted(hosts)) != hosts or (not network_enabled and hosts):
            raise ValueError("network source_hosts must be sorted and explicitly enabled")
        object.__setattr__(self, "permissions", frozenset(permissions))
        object.__setattr__(self, "enabled_tools", enabled)
        object.__setattr__(self, "capabilities", MappingProxyType(capability_configs))
        object.__setattr__(
            self, "network", MappingProxyType({"enabled": network_enabled, "source_hosts": hosts})
        )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("MCP config contains duplicate object keys")
        result[key] = value
    return result


def _invalid_constant(value: str) -> object:
    raise ValueError("MCP config must use finite JSON values")


def _require_secure_file_apis() -> None:
    if os.name == "posix":
        if not all(
            hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK", "geteuid")
        ):
            raise ValueError("MCP host requires POSIX secure file APIs")
    elif os.name == "nt":
        from capicove._windows_files import require_windows_file_apis

        require_windows_file_apis()
    else:
        raise ValueError("MCP host requires POSIX or Windows file security APIs")


def _open_posix_config(path: Path) -> BinaryIO:
    """Open a config beneath pinned, no-follow parent directory handles."""
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(path.anchor, directory_flags)
    try:
        for component in path.parts[1:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_fd,
        )
    finally:
        os.close(directory_fd)
    try:
        return os.fdopen(descriptor, "rb")
    except Exception:
        os.close(descriptor)
        raise


def load_mcp_host_config(path: Path) -> McpHostConfiguration | WorkspaceHostConfiguration:
    """Read a regular no-follow file with a hard size bound and closed fields."""
    return parse_mcp_host_config(read_mcp_host_config_bytes(path))


def read_mcp_host_config_bytes(path: Path) -> bytes:
    """Read the exact configuration bytes once, so a host can freeze its authority."""
    _require_secure_file_apis()
    if not path.is_absolute():
        raise ValueError("MCP config path must be absolute")
    try:
        if os.name == "nt":
            from capicove._windows_files import read_windows_secure_file

            payload = read_windows_secure_file(path, max_bytes=MAX_CONFIG_BYTES)
        else:
            with _open_posix_config(path) as source:
                metadata = os.fstat(source.fileno())
                if not stat.S_ISREG(metadata.st_mode):
                    raise ValueError("MCP config must be a regular file")
                if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022:
                    raise ValueError(
                        "MCP config must be owned by the process owner and not writable by others"
                    )
                payload = source.read(MAX_CONFIG_BYTES + 1)
        if len(payload) > MAX_CONFIG_BYTES:
            raise ValueError("MCP config exceeds the 64 KiB limit")
    except OSError as error:
        raise ValueError(
            "MCP config must be readable regular UTF-8 JSON without symlinks"
        ) from error
    return bytes(payload)


def parse_mcp_host_config(payload: bytes) -> McpHostConfiguration | WorkspaceHostConfiguration:
    """Parse configuration bytes previously read by ``read_mcp_host_config_bytes``."""
    if len(payload) > MAX_CONFIG_BYTES:
        raise ValueError("MCP config exceeds the 64 KiB limit")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
    except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise ValueError(
            "MCP config must be readable regular UTF-8 JSON without symlinks"
        ) from error
    if isinstance(value, dict) and value.get("schema_version") == 2:
        if set(value) != _WORKSPACE_FIELDS:
            raise ValueError("MCP config requires exactly the documented version 2 fields")
        permissions = _strings(value["permissions"], "permissions")
        enabled = _strings(value["enabled_tools"], "enabled_tools")
        trace = value["trace_store"]
        root = value["workspace_root"]
        if trace is not None and not isinstance(trace, str):
            raise ValueError("trace_store must be an absolute path or null")
        if not isinstance(root, str):
            raise ValueError("workspace_root must be an absolute path")
        return WorkspaceHostConfiguration(
            value["schema_version"],
            value["server_name"],
            value["workspace_id"],
            value["actor_id"],
            value["environment_id"],
            Path(root),
            value["profile"],
            frozenset(permissions),
            enabled,
            Path(trace) if trace is not None else None,
            cast(Mapping[str, Mapping[str, object]], value["capabilities"]),
            cast(Mapping[str, object], value["network"]),
        )
    if not isinstance(value, dict) or set(value) != _FIELDS:
        raise ValueError("MCP config requires exactly the documented version 1 fields")
    permissions = _strings(value["permissions"], "permissions")
    enabled = _strings(value["enabled_tools"], "enabled_tools")
    trace = value["trace_store"]
    if trace is not None and not isinstance(trace, str):
        raise ValueError("trace_store must be an absolute path or null")
    return McpHostConfiguration(
        value["schema_version"],
        value["server_name"],
        value["capability"],
        value["workspace_id"],
        value["actor_id"],
        value["environment_id"],
        frozenset(permissions),
        enabled,
        Path(trace) if trace is not None else None,
        cast(Mapping[str, object], value["capability_config"]),
    )
