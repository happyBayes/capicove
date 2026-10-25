"""Dependency-light Elasticsearch retriever plugin descriptor."""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from dataclasses import dataclass

from capicove.plugins import PluginAvailability, PluginMetadata
from capicove.retrieval import TextRetrieverBackend


@dataclass(frozen=True, slots=True)
class ElasticsearchRetrieverPlugin:
    """Describe Elasticsearch retrieval without constructing a client."""

    @property
    def metadata(self) -> PluginMetadata:
        """Return static retrieval-contract-v2 metadata."""
        return PluginMetadata(
            plugin_id="capicove.retriever.elasticsearch",
            plugin_version="0.1.0",
            capability="retriever",
            contract_min=2,
            contract_max=2,
            optional_packages=("elasticsearch",),
            external_binaries=(),
            network_access=True,
        )

    def availability(self) -> PluginAvailability:
        """Check SDK presence without importing it or accessing the network."""
        if importlib.util.find_spec("elasticsearch") is None:
            return PluginAvailability(
                False,
                "install capicove-retriever-elasticsearch with Elasticsearch 8 support",
            )
        return PluginAvailability(True)

    def create(self, configuration: Mapping[str, object]) -> TextRetrieverBackend:
        """Create one lazy runtime from closed startup configuration."""
        from capicove_retriever_elasticsearch.adapter import ElasticsearchTextRetriever
        from capicove_retriever_elasticsearch.config import ElasticsearchConfiguration

        return ElasticsearchTextRetriever(ElasticsearchConfiguration.from_mapping(configuration))


def elasticsearch_retriever_plugin() -> ElasticsearchRetrieverPlugin:
    """Return the dependency-light Elasticsearch retriever descriptor."""
    return ElasticsearchRetrieverPlugin()
