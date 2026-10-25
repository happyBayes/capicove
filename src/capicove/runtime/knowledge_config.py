"""Closed operator-owned native knowledge-base configuration (RFC-0025)."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from capicove.runtime.media_handoff import MediaIntake


_FIELDS = {
    "schema_version",
    "workspace_id",
    "storage_root",
    "source_roots",
    "models",
    "generation",
    "audio",
    "graph",
    "network_enabled",
}


def absolute_path(value: object) -> Path:
    """Reject ambiguous operator paths before any filesystem access."""
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("invalid_knowledge_path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError("invalid_knowledge_path")
    return path


def read_configuration(path: Path) -> dict[str, Any]:
    """Read bounded configuration through no-follow handles without executing it."""
    path = absolute_path(str(path))
    intake = MediaIntake(
        read_roots=(path.parent,),
        staging_root=path.parent,
        max_input_bytes=65536,
        accepts=lambda _path, _head: True,
    )
    # A digest first rejects links and concurrent changes; the staged copy pins
    # the bytes subsequently decoded. Configurations must be privately owned.
    intake.digest(path)
    info = path.stat(follow_symlinks=False)
    if info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise ValueError("unsafe_knowledge_configuration")
    with intake.stage((path,)) as staged:
        raw = json.loads(staged[0].staged.read_bytes(), object_pairs_hook=_unique)
    if not isinstance(raw, dict):
        raise ValueError("invalid_knowledge_configuration")
    return raw


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_configuration_key")
        result[key] = value
    return result


@dataclass(frozen=True)
class KnowledgeConfiguration:
    """One single-owner local corpus with explicit optional model integrations."""

    workspace_id: str
    storage_root: Path
    source_roots: Mapping[str, Path]
    models: Path | None = None
    generation: Path | None = None
    audio: Path | None = None
    graph: bool = False
    network_enabled: bool = False

    @classmethod
    def load(cls, path: Path) -> KnowledgeConfiguration:
        """Load the closed version-one configuration."""
        return cls.from_mapping(read_configuration(path))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> KnowledgeConfiguration:
        """Validate authority independently of optional runtime availability."""
        if (
            set(raw) != _FIELDS
            or type(raw["schema_version"]) is not int
            or raw["schema_version"] != 1
        ):
            raise ValueError("invalid_knowledge_configuration")
        workspace = raw["workspace_id"]
        if not isinstance(workspace, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", workspace):
            raise ValueError("invalid_workspace_id")
        storage = absolute_path(raw["storage_root"])
        sources = raw["source_roots"]
        if not isinstance(sources, Mapping) or not 1 <= len(sources) <= 32:
            raise ValueError("invalid_source_roots")
        roots: dict[str, Path] = {}
        for alias, value in sources.items():
            if not isinstance(alias, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", alias):
                raise ValueError("invalid_source_alias")
            path = absolute_path(value)
            if not path.is_dir() or path.resolve() != path:
                raise ValueError("invalid_source_root")
            if storage.is_relative_to(path) or path.is_relative_to(storage):
                raise ValueError("storage_and_sources_must_be_disjoint")
            if any(
                path.is_relative_to(other) or other.is_relative_to(path) for other in roots.values()
            ):
                raise ValueError("source_roots_must_be_disjoint")
            roots[alias] = path
        selected = {
            name: None if raw[name] is None else absolute_path(raw[name])
            for name in ("models", "generation", "audio")
        }
        if type(raw["graph"]) is not bool or type(raw["network_enabled"]) is not bool:
            raise ValueError("invalid_knowledge_flags")
        if selected["generation"] and not raw["network_enabled"]:
            raise ValueError("generation_requires_network_opt_in")
        if raw["graph"] and not (selected["models"] and selected["generation"]):
            raise ValueError("graph_requires_models_and_generation")
        return cls(
            workspace,
            storage,
            MappingProxyType(roots),
            **selected,
            graph=raw["graph"],
            network_enabled=raw["network_enabled"],
        )

    def to_mapping(self) -> dict[str, Any]:
        """Serialize references only; credentials never belong in this document."""
        return {
            "schema_version": 1,
            "workspace_id": self.workspace_id,
            "storage_root": str(self.storage_root),
            "source_roots": {key: str(value) for key, value in self.source_roots.items()},
            "models": str(self.models) if self.models else None,
            "generation": str(self.generation) if self.generation else None,
            "audio": str(self.audio) if self.audio else None,
            "graph": self.graph,
            "network_enabled": self.network_enabled,
        }
