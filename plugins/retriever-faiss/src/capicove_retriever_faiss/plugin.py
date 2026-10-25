"""Dependency-light FAISS retriever plugin descriptor."""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from capicove.plugins import PluginAvailability, PluginMetadata


if TYPE_CHECKING:
    from capicove.retrieval.video_ports import VisualRetrieverBackend


@dataclass(frozen=True, slots=True)
class FaissRetrieverPlugin:
    """Describe local FAISS retrieval without importing its native module."""

    @property
    def metadata(self) -> PluginMetadata:
        """Return static retrieval-contract-v3 metadata."""
        return PluginMetadata(
            plugin_id="capicove.retriever.faiss",
            plugin_version="0.1.0",
            capability="retriever",
            contract_min=3,
            contract_max=3,
            optional_packages=("faiss",),
            external_binaries=(),
            network_access=False,
        )

    def availability(self) -> PluginAvailability:
        """Probe module metadata without importing FAISS or opening an index."""
        if importlib.util.find_spec("faiss") is None:
            return PluginAvailability(
                False,
                "FAISS runtime is unavailable; install capicove-retriever-faiss",
            )
        return PluginAvailability(True)

    def create(self, configuration: Mapping[str, object]) -> VisualRetrieverBackend:
        """Create a local visual runtime only after an explicit availability check."""
        availability = self.availability()
        if not availability.available:
            raise RuntimeError(availability.detail)
        from capicove_retriever_faiss.adapter import FaissVisualRetriever
        from capicove_retriever_faiss.config import FaissConfiguration

        return cast(
            "VisualRetrieverBackend",
            FaissVisualRetriever(FaissConfiguration.from_mapping(configuration)),
        )


def faiss_retriever_plugin() -> FaissRetrieverPlugin:
    """Return the dependency-light FAISS retriever descriptor."""
    return FaissRetrieverPlugin()


__all__ = ["FaissRetrieverPlugin", "faiss_retriever_plugin"]
