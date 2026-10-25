"""Typed configured inference ports; model SDKs remain adapter-owned."""

from typing import Protocol

from capicove.contracts import EmbeddingSpaceRef
from capicove.retrieval.ports import TextBatchEncoder


class ConfiguredTextEncoder(TextBatchEncoder, Protocol):
    """Expose the exact embedding space required by index and graph callers."""

    @property
    def space(self) -> EmbeddingSpaceRef:
        """Return model, preprocessing, dimension, metric and normalization identity."""
