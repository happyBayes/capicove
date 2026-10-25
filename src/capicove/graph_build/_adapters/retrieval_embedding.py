"""Bridge graph's narrow embedding port to the public retrieval operation."""

from capicove.contracts import EmbeddingSpaceRef
from capicove.retrieval import TextEmbedding, TextEmbeddingService


class RetrievalBatchTextEmbeddingProvider:
    """Delegate ordered graph text batches to an explicitly injected service."""

    def __init__(self, service: TextEmbeddingService) -> None:
        self._service = service

    def embed(
        self, texts: tuple[str, ...], *, space: EmbeddingSpaceRef, role: str
    ) -> tuple[TextEmbedding, ...]:
        """Preserve inputs, role and space across the capability boundary."""
        return self._service.encode(texts, space=space, role=role)
