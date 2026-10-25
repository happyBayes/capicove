"""Side-effect-free Python entry-point discovery for Capicove plugins."""

from __future__ import annotations

from collections.abc import Iterable
from importlib import metadata
from typing import Protocol, cast

from capicove.contracts import DOCUMENT_PARSER_CONTRACT_VERSION
from capicove.contracts.retrieval import RETRIEVAL_CONTRACT_VERSION
from capicove.plugins.models import (
    DiscoveredPlugin,
    DocumentParserPlugin,
    ExecutionBackendPlugin,
    RetrieverPlugin,
)


_EXECUTION_BACKEND_GROUP = "capicove.execution_backends"
_EXECUTION_BACKEND_CONTRACT_VERSION = 1
_DOCUMENT_PARSER_GROUP = "capicove.document_parsers"
_RETRIEVER_GROUP = "capicove.retrievers"
_RETRIEVAL_CONTRACT_VERSION = 2


class _EntryPoint(Protocol):
    """Describe the importlib entry-point surface consumed here."""

    name: str
    value: str
    group: str

    def load(self) -> object:
        """Load the referenced object."""


def list_execution_backend_plugins(
    *,
    entry_points: Iterable[object] | None = None,
) -> tuple[DiscoveredPlugin, ...]:
    """List execution backends without loading their modules."""
    entries = _entries(entry_points, group=_EXECUTION_BACKEND_GROUP)
    discovered = tuple(
        sorted(
            (
                DiscoveredPlugin(
                    entry.name,
                    entry.value,
                )
                for entry in entries
            ),
            key=lambda item: item.name,
        )
    )
    names = tuple(item.name for item in discovered)
    if len(set(names)) != len(names):
        raise ValueError("duplicate execution backend entry-point name")
    return discovered


def load_execution_backend_plugin(
    name: str,
    *,
    entry_points: Iterable[object] | None = None,
) -> ExecutionBackendPlugin:
    """Explicitly load and validate one execution backend plugin."""
    matches = [
        entry
        for entry in _entries(entry_points, group=_EXECUTION_BACKEND_GROUP)
        if entry.name == name
    ]
    if not matches:
        raise LookupError(f"unknown execution backend plugin: {name}")
    if len(matches) != 1:
        raise ValueError("duplicate execution backend entry-point name")
    factory = matches[0].load()
    if not callable(factory):
        raise TypeError("execution backend entry point must load a callable factory")
    plugin = factory()
    if not isinstance(plugin, ExecutionBackendPlugin):
        raise TypeError("execution backend factory returned an incompatible plugin")
    plugin_metadata = plugin.metadata
    if plugin_metadata.capability != "execution_backend":
        raise TypeError("execution backend plugin metadata has the wrong capability")
    if not (
        plugin_metadata.contract_min
        <= _EXECUTION_BACKEND_CONTRACT_VERSION
        <= plugin_metadata.contract_max
    ):
        raise RuntimeError("execution backend plugin contract range is incompatible")
    return plugin


def list_document_parser_plugins(
    *, entry_points: Iterable[object] | None = None
) -> tuple[DiscoveredPlugin, ...]:
    """List document parsers without loading their modules."""
    entries = _entries(entry_points, group=_DOCUMENT_PARSER_GROUP)
    discovered = tuple(
        sorted(
            (DiscoveredPlugin(entry.name, entry.value) for entry in entries),
            key=lambda item: item.name,
        )
    )
    names = tuple(item.name for item in discovered)
    if len(set(names)) != len(names):
        raise ValueError("duplicate document parser entry-point name")
    return discovered


def load_document_parser_plugin(
    name: str,
    *,
    entry_points: Iterable[object] | None = None,
) -> DocumentParserPlugin:
    """Explicitly load and validate one document parser plugin."""
    matches = [
        entry
        for entry in _entries(entry_points, group=_DOCUMENT_PARSER_GROUP)
        if entry.name == name
    ]
    if not matches:
        raise LookupError(f"unknown document parser plugin: {name}")
    if len(matches) != 1:
        raise ValueError("duplicate document parser entry-point name")
    factory = matches[0].load()
    if not callable(factory):
        raise TypeError("document parser entry point must load a callable factory")
    plugin = factory()
    if not isinstance(plugin, DocumentParserPlugin):
        raise TypeError("document parser factory returned an incompatible plugin")
    plugin_metadata = plugin.metadata
    if plugin_metadata.capability != "document_parser":
        raise TypeError("document parser capability is required in plugin metadata")
    if not (
        plugin_metadata.contract_min
        <= DOCUMENT_PARSER_CONTRACT_VERSION
        <= plugin_metadata.contract_max
    ):
        raise RuntimeError("document parser plugin contract range is incompatible")
    return plugin


def list_retriever_plugins(
    *,
    entry_points: Iterable[object] | None = None,
) -> tuple[DiscoveredPlugin, ...]:
    """List retrievers without loading their modules."""
    entries = _entries(entry_points, group=_RETRIEVER_GROUP)
    discovered = tuple(
        sorted(
            (DiscoveredPlugin(entry.name, entry.value) for entry in entries),
            key=lambda item: item.name,
        )
    )
    names = tuple(item.name for item in discovered)
    if len(set(names)) != len(names):
        raise ValueError("duplicate retriever entry-point name")
    return discovered


def load_retriever_plugin(
    name: str,
    *,
    entry_points: Iterable[object] | None = None,
    contract_version: int = _RETRIEVAL_CONTRACT_VERSION,
) -> RetrieverPlugin:
    """Load for a startup-selected contract, preserving legacy v2 by default."""
    if (
        isinstance(contract_version, bool)
        or not isinstance(contract_version, int)
        or not 2 <= contract_version <= RETRIEVAL_CONTRACT_VERSION
    ):
        raise ValueError("unsupported retriever contract version")
    matches = [
        entry for entry in _entries(entry_points, group=_RETRIEVER_GROUP) if entry.name == name
    ]
    if not matches:
        raise LookupError(f"unknown retriever plugin: {name}")
    if len(matches) != 1:
        raise ValueError("duplicate retriever entry-point name")
    factory = matches[0].load()
    if not callable(factory):
        raise TypeError("retriever entry point must load a callable factory")
    plugin = factory()
    if not isinstance(plugin, RetrieverPlugin):
        raise TypeError("retriever factory returned an incompatible plugin")
    plugin_metadata = plugin.metadata
    if plugin_metadata.capability != "retriever":
        raise TypeError("retriever capability is required in plugin metadata")
    if not (plugin_metadata.contract_min <= contract_version <= plugin_metadata.contract_max):
        raise RuntimeError("retriever plugin contract range is incompatible")
    return plugin


def _entries(
    entry_points: Iterable[object] | None,
    *,
    group: str,
) -> tuple[_EntryPoint, ...]:
    if entry_points is not None:
        entries = tuple(entry_points)
    else:
        discovered = metadata.entry_points()
        entries = tuple(discovered.select(group=group))
    return tuple(
        cast(_EntryPoint, entry) for entry in entries if getattr(entry, "group", group) == group
    )
