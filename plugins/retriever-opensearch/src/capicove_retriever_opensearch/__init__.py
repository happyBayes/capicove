"""OpenSearch retriever plugin for Capicove."""

from capicove_retriever_opensearch.adapter import (
    OpenSearchOperationError,
    OpenSearchTextRetriever,
)
from capicove_retriever_opensearch.config import OpenSearchConfiguration
from capicove_retriever_opensearch.plugin import (
    OpenSearchRetrieverPlugin,
    opensearch_retriever_plugin,
)


__all__ = [
    "OpenSearchConfiguration",
    "OpenSearchOperationError",
    "OpenSearchRetrieverPlugin",
    "OpenSearchTextRetriever",
    "opensearch_retriever_plugin",
]
