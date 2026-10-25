"""Plugin metadata, discovery, compatibility, and diagnostics."""

from capicove.plugins.discovery import (
    list_document_parser_plugins,
    list_execution_backend_plugins,
    list_retriever_plugins,
    load_document_parser_plugin,
    load_execution_backend_plugin,
    load_retriever_plugin,
)
from capicove.plugins.models import (
    ConfigurableRetrieverPlugin,
    DiscoveredPlugin,
    DocumentParserModelInstaller,
    DocumentParserPlugin,
    ExecutionBackendPlugin,
    PluginAvailability,
    PluginMetadata,
    RetrieverPlugin,
)


__all__ = [
    "ConfigurableRetrieverPlugin",
    "DiscoveredPlugin",
    "DocumentParserModelInstaller",
    "DocumentParserPlugin",
    "ExecutionBackendPlugin",
    "PluginAvailability",
    "PluginMetadata",
    "RetrieverPlugin",
    "list_document_parser_plugins",
    "list_execution_backend_plugins",
    "list_retriever_plugins",
    "load_document_parser_plugin",
    "load_execution_backend_plugin",
    "load_retriever_plugin",
]
