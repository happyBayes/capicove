"""Tests for dependency-light OpenSearch plugin metadata."""

import sys

from capicove_retriever_opensearch.plugin import opensearch_retriever_plugin

from capicove.plugins import ConfigurableRetrieverPlugin


def test_plugin_metadata_is_contract_v2_and_does_not_import_sdk() -> None:
    sys.modules.pop("opensearchpy", None)

    plugin = opensearch_retriever_plugin()

    assert isinstance(plugin, ConfigurableRetrieverPlugin)
    assert plugin.metadata.plugin_id == "capicove.retriever.opensearch"
    assert plugin.metadata.contract_min == plugin.metadata.contract_max == 2
    assert plugin.metadata.optional_packages == ("opensearchpy",)
    assert plugin.metadata.network_access is True
    assert "opensearchpy" not in sys.modules
