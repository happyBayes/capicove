"""Dependency-light OpenSearch retriever plugin descriptor."""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from dataclasses import dataclass

from capicove.plugins import PluginAvailability, PluginMetadata
from capicove.retrieval import TextRetrieverBackend


@dataclass(frozen=True, slots=True)
class OpenSearchRetrieverPlugin:
    """Describe OpenSearch retrieval without constructing a client."""

    @property
    def metadata(self) -> PluginMetadata:
        """Return static retrieval-contract-v2 metadata."""
        return PluginMetadata(
            plugin_id="capicove.retriever.opensearch",
            plugin_version="0.1.0",
            capability="retriever",
            contract_min=2,
            contract_max=2,
            optional_packages=("opensearchpy",),
            external_binaries=(),
            network_access=True,
        )

    def availability(self) -> PluginAvailability:
        """Check SDK presence without importing it or accessing the network."""
        if importlib.util.find_spec("opensearchpy") is None:
            return PluginAvailability(False, "install capicove-retriever-opensearch")
        return PluginAvailability(True)

    def create(self, configuration: Mapping[str, object]) -> TextRetrieverBackend:
        """Create one lazy runtime from closed startup configuration."""
        from capicove_retriever_opensearch.adapter import OpenSearchTextRetriever
        from capicove_retriever_opensearch.config import OpenSearchConfiguration

        return OpenSearchTextRetriever(OpenSearchConfiguration.from_mapping(configuration))


def opensearch_retriever_plugin() -> OpenSearchRetrieverPlugin:
    """Return the dependency-light OpenSearch retriever descriptor."""
    return OpenSearchRetrieverPlugin()


__all__ = ["OpenSearchRetrieverPlugin", "opensearch_retriever_plugin"]
