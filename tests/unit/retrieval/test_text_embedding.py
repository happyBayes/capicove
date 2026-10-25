"""Tests for strict batch text embedding validation."""

from dataclasses import replace
from typing import Any

import pytest

from capicove.contracts import (
    DistanceMetric,
    EmbeddingSpaceRef,
    QueryModality,
    TargetType,
    VectorNormalization,
)
from capicove.retrieval import TextBatchEncoder, TextEmbedding, TextEmbeddingService
from capicove.retrieval.text_embedding import validate_vector


def test_embedding_rejects_nan_and_wrong_dimensions() -> None:
    with pytest.raises(ValueError):
        validate_vector((float("nan"), 1.0), 2)
    with pytest.raises(ValueError):
        validate_vector((1.0,), 2)


def text_space() -> EmbeddingSpaceRef:
    return EmbeddingSpaceRef(
        space_id="fixture-text-v1",
        provider_id="local-bge",
        model_revision="revision-1",
        preprocessing_version="preprocessing-1",
        dimension=2,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.COSINE,
        query_modalities=frozenset({QueryModality.TEXT}),
        target_modalities=frozenset({TargetType.TEXT_CHUNK}),
    )


class Encoder:
    def __init__(self, output: Any = None) -> None:
        self.output = output
        self.calls: list[tuple[tuple[str, ...], EmbeddingSpaceRef, str]] = []

    def encode(
        self, texts: tuple[str, ...], *, space: EmbeddingSpaceRef, role: str
    ) -> tuple[TextEmbedding, ...]:
        self.calls.append((texts, space, role))
        if self.output is not None:
            return self.output  # type: ignore[no-any-return]
        return tuple(
            TextEmbedding(space, (1.0, 0.0) if t == "second" else (0.0, 1.0)) for t in texts
        )


@pytest.mark.parametrize("role", ["query", "document"])
def test_service_preserves_batch_order_duplicates_and_role(role: str) -> None:
    encoder = Encoder()
    service = TextEmbeddingService(encoder, space=text_space())
    texts = ("second", "first", "second")
    result = service.encode(texts, space=text_space(), role=role)
    assert isinstance(encoder, TextBatchEncoder)
    assert encoder.calls == [(texts, text_space(), role)]
    assert tuple(item.vector for item in result) == ((1.0, 0.0), (0.0, 1.0), (1.0, 0.0))


@pytest.mark.parametrize("texts", [(), ("",), (" \n",), (123,), "text", ["text"]])
def test_service_rejects_invalid_batches_before_calling_encoder(texts: Any) -> None:
    encoder = Encoder()
    with pytest.raises(ValueError):
        TextEmbeddingService(encoder, space=text_space()).encode(
            texts, space=text_space(), role="query"
        )
    assert encoder.calls == []


@pytest.mark.parametrize("role", ["target", "passage", "QUERY", "", None, []])
def test_service_rejects_unconfigured_role_before_encoding(role: Any) -> None:
    encoder = Encoder()
    with pytest.raises(ValueError):
        TextEmbeddingService(encoder, space=text_space()).encode(
            ("text",), space=text_space(), role=role
        )
    assert encoder.calls == []


@pytest.mark.parametrize(
    "changes",
    [
        {"space_id": "other"},
        {"provider_id": "other"},
        {"model_revision": "other"},
        {"preprocessing_version": "other"},
        {"dimension": 3},
        {"normalization": VectorNormalization.NONE},
        {"distance_metric": DistanceMetric.INNER_PRODUCT},
        {"query_modalities": frozenset({QueryModality.TEXT, QueryModality.IMAGE})},
        {"target_modalities": frozenset({TargetType.TEXT_CHUNK, TargetType.IMAGE})},
        {"schema_version": 2},
    ],
)
def test_service_matches_every_space_field_before_encoding(changes: dict[str, Any]) -> None:
    encoder = Encoder()
    with pytest.raises(ValueError, match="space"):
        TextEmbeddingService(encoder, space=text_space()).encode(
            ("text",), space=replace(text_space(), **changes), role="query"
        )
    assert encoder.calls == []


@pytest.mark.parametrize("output", [(), ("dense",), {"dense": [1.0, 0.0]}])
def test_service_rejects_wrong_count_or_noncanonical_output(output: Any) -> None:
    with pytest.raises(ValueError):
        TextEmbeddingService(Encoder(output), space=text_space()).encode(
            ("one", "two"), space=text_space(), role="document"
        )


def test_service_rejects_encoder_space_mismatch_with_identical_dimensions() -> None:
    output = (TextEmbedding(replace(text_space(), model_revision="other"), (1.0, 0.0)),)
    with pytest.raises(ValueError, match="space"):
        TextEmbeddingService(Encoder(output), space=text_space()).encode(
            ("one",), space=text_space(), role="document"
        )


@pytest.mark.parametrize(
    "vector", [(2.0, 0.0), (0.0, 0.0), (float("inf"), 0.0), (float("nan"), 0.0), (1.0,)]
)
def test_service_revalidates_provider_vectors(vector: tuple[float, ...]) -> None:
    malformed = object.__new__(TextEmbedding)
    object.__setattr__(malformed, "space", text_space())
    object.__setattr__(malformed, "vector", vector)
    with pytest.raises(ValueError):
        TextEmbeddingService(Encoder((malformed,)), space=text_space()).encode(
            ("one",), space=text_space(), role="document"
        )


@pytest.mark.parametrize("metric", list(DistanceMetric))
def test_unnormalized_vectors_obey_metric_semantics(metric: DistanceMetric) -> None:
    space = replace(text_space(), normalization=VectorNormalization.NONE, distance_metric=metric)
    service = TextEmbeddingService(Encoder((TextEmbedding(space, (0.0, 0.0)),)), space=space)
    if metric is DistanceMetric.COSINE:
        with pytest.raises(ValueError):
            service.encode(("one",), space=space, role="query")
    else:
        assert service.encode(("one",), space=space, role="query")[0].vector == (0.0, 0.0)


@pytest.mark.parametrize("vector", [(float("inf"),), (float("-inf"),), (True,), ("1",), ((1.0,),)])
def test_validate_vector_rejects_nonfinite_or_nondense_values(vector: Any) -> None:
    with pytest.raises(ValueError):
        validate_vector(vector, 1)


@pytest.mark.parametrize("dimension", [0, -1, True, 1.0])
def test_validate_vector_rejects_invalid_dimensions(dimension: Any) -> None:
    with pytest.raises(ValueError):
        validate_vector((1.0,), dimension)
