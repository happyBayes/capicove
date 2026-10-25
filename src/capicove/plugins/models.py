"""Dependency-light values for Capicove plugin discovery."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from capicove.processing.ports import ExecutionBackend, ExecutionBackendOperatorCatalog


if TYPE_CHECKING:
    from capicove.documents import DocumentParserBackend
    from capicove.retrieval.ports import TextRetrieverBackend


_PLUGIN_CAPABILITIES = frozenset({"document_parser", "execution_backend", "retriever"})


@dataclass(frozen=True, slots=True)
class DiscoveredPlugin:
    """Describe one entry point without importing its target."""

    name: str
    value: str

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z0-9][a-z0-9.-]*", self.name) is None:
            raise ValueError("plugin entry-point name is invalid")
        if not self.value.strip():
            raise ValueError("plugin entry-point value must not be blank")


@dataclass(frozen=True, slots=True)
class PluginMetadata:
    """Declare compatibility and side effects for one plugin."""

    plugin_id: str
    plugin_version: str
    capability: str
    contract_min: int
    contract_max: int
    optional_packages: tuple[str, ...]
    external_binaries: tuple[str, ...]
    network_access: bool

    def __post_init__(self) -> None:
        if re.fullmatch(r"capicove\.[a-z0-9][a-z0-9.-]*", self.plugin_id) is None:
            raise ValueError("plugin_id must be a canonical Capicove identifier")
        if not self.plugin_version.strip():
            raise ValueError("plugin_version must not be blank")
        if self.capability not in _PLUGIN_CAPABILITIES:
            raise ValueError(f"capability must be one of {', '.join(sorted(_PLUGIN_CAPABILITIES))}")
        for field_name, value in (
            ("contract_min", self.contract_min),
            ("contract_max", self.contract_max),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.contract_min > self.contract_max:
            raise ValueError("contract_min must not exceed contract_max")
        packages = tuple(self.optional_packages)
        binaries = tuple(self.external_binaries)
        if any(not item.strip() for item in (*packages, *binaries)):
            raise ValueError("package and binary names must not be blank")
        if len(set(packages)) != len(packages) or len(set(binaries)) != len(binaries):
            raise ValueError("package and binary names must be unique")
        if not isinstance(self.network_access, bool):
            raise ValueError("network_access must be a boolean")
        object.__setattr__(self, "optional_packages", packages)
        object.__setattr__(self, "external_binaries", binaries)


@dataclass(frozen=True, slots=True)
class PluginAvailability:
    """Report whether one plugin can be constructed in the current process."""

    available: bool
    detail: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.available, bool):
            raise ValueError("available must be a boolean")
        if self.detail is not None and (not self.detail.strip() or len(self.detail) > 500):
            raise ValueError("detail must be non-blank and bounded")
        if not self.available and self.detail is None:
            raise ValueError("unavailable plugin requires detail")


@runtime_checkable
class ExecutionBackendPlugin(ExecutionBackendOperatorCatalog, Protocol):
    """Construct one compatible execution backend explicitly."""

    @property
    def metadata(self) -> PluginMetadata:
        """Return static plugin metadata."""

    def availability(self) -> PluginAvailability:
        """Check local dependencies without creating a backend."""

    def create(self, configuration: Mapping[str, object]) -> ExecutionBackend:
        """Create one configured execution backend."""


@runtime_checkable
class RetrieverPlugin(Protocol):
    """Describe one retriever plugin before provider construction is defined."""

    @property
    def metadata(self) -> PluginMetadata:
        """Return static retriever plugin metadata."""

    def availability(self) -> PluginAvailability:
        """Check local dependencies without opening a network connection."""


@runtime_checkable
class ConfigurableRetrieverPlugin(RetrieverPlugin, Protocol):
    """Construct one explicitly configured text retriever runtime."""

    def create(self, configuration: Mapping[str, object]) -> TextRetrieverBackend:
        """Create one lazy provider runtime from closed startup configuration."""


@runtime_checkable
class DocumentParserPlugin(Protocol):
    """Construct one compatible document parser explicitly."""

    @property
    def metadata(self) -> PluginMetadata:
        """Return static document-parser metadata."""

    def availability(self) -> PluginAvailability:
        """Check local dependencies without importing parser runtimes."""

    def create(self, configuration: Mapping[str, object]) -> DocumentParserBackend:
        """Create one configured parser without acquiring models."""


@runtime_checkable
class DocumentParserModelInstaller(Protocol):
    """Plan and explicitly install local models for a document parser."""

    def model_plan(self, profile_id: str, destination: Path) -> object:
        """Return a side-effect-free immutable installation plan."""

    def install_models(
        self,
        profile_id: str,
        destination: Path,
        *,
        confirmed: bool,
    ) -> object:
        """Install a planned profile only after explicit confirmation."""
