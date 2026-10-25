"""Dependency-light entry point for the Capicove FAISS retriever plugin."""

from capicove_retriever_faiss.plugin import (
    FaissRetrieverPlugin,
    faiss_retriever_plugin,
)


__all__ = ["FaissRetrieverPlugin", "faiss_retriever_plugin"]
