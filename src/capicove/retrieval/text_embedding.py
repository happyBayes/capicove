"""Validation and public operations for ordered dense text embedding batches."""

import math

from capicove.contracts import (
    DistanceMetric,
    EmbeddingSpaceRef,
    QueryModality,
    TargetType,
    VectorNormalization,
)
from capicove.retrieval.models import TextEmbedding
from capicove.retrieval.ports import TextBatchEncoder


def validate_vector(vector: tuple[float, ...], dimension: int) -> None:
    """Reject non-dense, incorrectly sized, or non-finite vectors."""
    if type(dimension) is not int or dimension <= 0:
        raise ValueError("invalid_embedding_dimension")
    if not isinstance(vector, tuple) or len(vector) != dimension:
        raise ValueError("embedding_dimension_mismatch")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in vector
    ):
        raise ValueError("invalid_embedding_values")


def _validate_request(
    texts: tuple[str, ...], *, space: EmbeddingSpaceRef, expected: EmbeddingSpaceRef, role: str
) -> None:
    if not isinstance(texts, tuple) or not texts:
        raise ValueError("invalid_text_batch")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("invalid_text_batch")
    if not isinstance(role, str) or role not in ("query", "document"):
        raise ValueError("invalid_embedding_role")
    if not isinstance(space, EmbeddingSpaceRef) or space != expected:
        raise ValueError("embedding_space_mismatch")


def _validate_embeddings(
    output: tuple[TextEmbedding, ...], *, count: int, space: EmbeddingSpaceRef
) -> None:
    if not isinstance(output, tuple) or len(output) != count:
        raise ValueError("embedding_batch_count_mismatch")
    for embedding in output:
        if not isinstance(embedding, TextEmbedding):
            raise ValueError("invalid_dense_embedding")
        if embedding.space != space:
            raise ValueError("embedding_space_mismatch")
        validate_vector(embedding.vector, space.dimension)
        norm = math.hypot(*embedding.vector)
        if space.normalization is VectorNormalization.L2 and not math.isclose(
            norm, 1.0, rel_tol=1e-5, abs_tol=1e-6
        ):
            raise ValueError("embedding_normalization_mismatch")
        if space.distance_metric is DistanceMetric.COSINE and norm == 0:
            raise ValueError("undefined_cosine_embedding")


class TextEmbeddingService:
    """Validate ordered dense batches from one startup-selected encoder/space.

    Empty batches and blank/non-string elements are errors. Text is forwarded
    unchanged; the provider owns its versioned preprocessing. This service
    verifies every observable output invariant, but relies on the port's order
    guarantee because TextEmbedding contains no per-input correlation identity.
    """

    def __init__(self, encoder: TextBatchEncoder, *, space: EmbeddingSpaceRef) -> None:
        if not isinstance(space, EmbeddingSpaceRef) or not space.accepts(
            QueryModality.TEXT, TargetType.TEXT_CHUNK
        ):
            raise ValueError("invalid_text_embedding_space")
        self._encoder = encoder
        self._space = space

    def encode(
        self, texts: tuple[str, ...], *, space: EmbeddingSpaceRef, role: str
    ) -> tuple[TextEmbedding, ...]:
        """Return one validated embedding per input, preserving tuple order."""
        _validate_request(texts, space=space, expected=self._space, role=role)
        output = self._encoder.encode(texts, space=space, role=role)
        _validate_embeddings(output, count=len(texts), space=space)
        return output
