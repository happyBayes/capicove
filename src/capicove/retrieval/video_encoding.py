"""Reuse one explicitly provisioned public model service for video vectors."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING

from capicove.contracts import EmbeddingSpaceRef, ImageQuery, TextQuery


if TYPE_CHECKING:
    from capicove.retrieval.models import EmbeddingResult
    from capicove.retrieval.service import Siglip2RetrievalService


class VideoEmbeddingAdapter:
    """Bind offline previews and online queries to one verified model identity.

    Image resolution is startup-injected and must authorize each query source.
    No default path opening, model selection, download or fallback is provided.
    The composition root is responsible for matching the preprocessing profile
    to the declared embedding space, in addition to the model checks here.
    """

    def __init__(
        self,
        service: Siglip2RetrievalService,
        embedding_space: EmbeddingSpaceRef,
        *,
        model_id: str,
        resolve_image: Callable[[ImageQuery], bytes] | None = None,
    ) -> None:
        if service.model_id != model_id or service.model_revision != embedding_space.model_revision:
            raise ValueError("embedding model identity mismatch")
        self._service = service
        self._space = embedding_space
        self._resolve_image = resolve_image

    def encode(
        self,
        query: TextQuery | ImageQuery,
        *,
        embedding_space: EmbeddingSpaceRef,
        deadline: datetime,
    ) -> tuple[float, ...]:
        """Encode one query with an unchanged absolute deadline."""
        from capicove.processing.image.invocation import image_invocation_deadline

        if deadline.tzinfo is None:
            raise ValueError("deadline must be timezone aware")
        if embedding_space != self._space:
            raise ValueError("embedding space mismatch")
        with image_invocation_deadline(deadline):
            if isinstance(query, TextQuery):
                result = self._service.embed_texts((query.text,))
            elif isinstance(query, ImageQuery):
                if self._resolve_image is None:
                    raise PermissionError("query image requires an authorized resolver")
                result = self._embed_preview(self._resolve_image(query))
            else:
                raise ValueError("unsupported query")
            if len(result.embeddings) != 1 or result.dimension != self._space.dimension:
                raise ValueError("embedding output dimension or count mismatch")
            return result.embeddings[0]

    def encode_preview_batch(self, previews: tuple[bytes, ...]) -> tuple[tuple[float, ...], ...]:
        """Encode bounded persisted previews for the offline preparation port.

        The preparation pipeline owns its absolute deadline and batch bound.
        Decoded images are released one at a time rather than retained as a batch.
        """
        if not 1 <= len(previews) <= 16:
            raise ValueError("preview batch must contain between one and sixteen images")
        vectors = []
        for preview in previews:
            result = self._embed_preview(preview)
            if len(result.embeddings) != 1 or result.dimension != self._space.dimension:
                raise ValueError("embedding output dimension or count mismatch")
            vectors.append(result.embeddings[0])
        return tuple(vectors)

    def _embed_preview(self, payload: bytes) -> EmbeddingResult:
        from capicove.processing.image import ImageSource
        from capicove.processing.image.io import load_image

        if not isinstance(payload, bytes) or not 0 < len(payload) <= 8 * 1024 * 1024:
            raise ValueError("image payload must be nonempty and at most 8 MiB")
        loaded = load_image(ImageSource(data=payload))
        try:
            return self._service.embed_image(loaded.image)
        finally:
            loaded.image.close()
