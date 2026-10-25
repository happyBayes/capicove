"""Elasticsearch retriever plugin for Capicove."""

from capicove_retriever_elasticsearch.adapter import (
    ElasticsearchOperationError,
    ElasticsearchTextRetriever,
)
from capicove_retriever_elasticsearch.config import ElasticsearchConfiguration
from capicove_retriever_elasticsearch.plugin import (
    ElasticsearchRetrieverPlugin,
    elasticsearch_retriever_plugin,
)


__all__ = [
    "ElasticsearchConfiguration",
    "ElasticsearchOperationError",
    "ElasticsearchRetrieverPlugin",
    "ElasticsearchTextRetriever",
    "elasticsearch_retriever_plugin",
]
