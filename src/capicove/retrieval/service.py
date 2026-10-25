"""Transport-neutral multimodal retrieval operations."""

from __future__ import annotations

import math

from PIL import Image

from capicove.retrieval.models import (
    EmbeddingResult,
    RankedMatch,
    RankingResult,
)
from capicove.retrieval.ports import MultimodalEmbeddingProvider


class Siglip2RetrievalService:
    """Validate requests and project results from an injected local provider."""

    def __init__(
        self,
        provider: MultimodalEmbeddingProvider,
        *,
        max_texts: int = 128,
        max_text_length: int = 500,
    ) -> None:
        if max_texts <= 0 or max_text_length <= 0:
            raise ValueError("text limits must be positive")
        self._provider = provider
        self._max_texts = max_texts
        self._max_text_length = max_text_length

    @property
    def model_id(self) -> str:
        """Return the configured model repository identity."""
        return self._provider.model_id

    @property
    def model_revision(self) -> str:
        """Return the immutable configured model revision."""
        return self._provider.model_revision

    @property
    def device(self) -> str:
        """Return the provider execution device."""
        return self._provider.device

    def rank(self, image: Image.Image, texts: tuple[str, ...]) -> RankingResult:
        """Rank candidate texts for one image."""
        normalized = self._texts(texts)
        provider_matches = self._provider.rank(image, normalized)
        if len(provider_matches) != len(normalized):
            raise ValueError("provider must return one match per candidate text")
        logits = tuple(item.logit for item in provider_matches)
        probabilities = _softmax(logits)
        order = sorted(range(len(logits)), key=logits.__getitem__, reverse=True)
        ranks = {index: rank for rank, index in enumerate(order, start=1)}
        matches = tuple(
            RankedMatch(
                text=normalized[index],
                logit=provider_matches[index].logit,
                match_probability=provider_matches[index].match_probability,
                candidate_probability=probabilities[index],
                rank=ranks[index],
            )
            for index in order
        )
        return RankingResult(
            model_id=self.model_id,
            model_revision=self.model_revision,
            device=self.device,
            matches=matches,
        )

    def embed_image(self, image: Image.Image) -> EmbeddingResult:
        """Return one normalized image vector."""
        return self._embedding_result(self._provider.embed_images((image,)))

    def embed_texts(self, texts: tuple[str, ...]) -> EmbeddingResult:
        """Return normalized vectors for the supplied texts."""
        normalized = self._texts(texts)
        return self._embedding_result(self._provider.embed_texts(normalized))

    def _texts(self, texts: tuple[str, ...]) -> tuple[str, ...]:
        if not texts:
            raise ValueError("at least one text is required")
        if len(texts) > self._max_texts:
            raise ValueError(f"at most {self._max_texts} texts are allowed")
        normalized = tuple(text.strip() for text in texts)
        if any(not text for text in normalized):
            raise ValueError("text values must not be blank")
        if any(len(text) > self._max_text_length for text in normalized):
            raise ValueError(f"text values must not exceed {self._max_text_length} characters")
        return normalized

    def _embedding_result(
        self,
        vectors: tuple[tuple[float, ...], ...],
    ) -> EmbeddingResult:
        if not vectors or not vectors[0]:
            raise ValueError("provider must return non-empty embeddings")
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors):
            raise ValueError("provider embeddings must have matching dimensions")
        for vector in vectors:
            if any(not math.isfinite(value) for value in vector):
                raise ValueError("provider embeddings must contain finite values")
            if not math.isclose(sum(value * value for value in vector), 1.0, abs_tol=1e-5):
                raise ValueError("provider embeddings must be L2-normalized")
        return EmbeddingResult(
            model_id=self.model_id,
            model_revision=self.model_revision,
            device=self.device,
            dimension=dimension,
            embeddings=vectors,
        )


def _softmax(values: tuple[float, ...]) -> tuple[float, ...]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("provider logits must be finite")
    maximum = max(values)
    exponentials = tuple(math.exp(value - maximum) for value in values)
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)
