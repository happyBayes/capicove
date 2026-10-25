"""Tests for transport-neutral multimodal retrieval operations."""

from __future__ import annotations

import math

import pytest
from PIL import Image

from capicove.retrieval import ProviderMatch, Siglip2RetrievalService


class DeterministicProvider:
    """Return hand-derived values without a model SDK dependency."""

    model_id = "google/siglip2-base-patch16-224"
    model_revision = "a" * 40
    device = "mps"

    def rank(self, image: Image.Image, texts: tuple[str, ...]) -> tuple[ProviderMatch, ...]:
        assert image.size == (2, 2)
        assert texts == ("a cat", "a dog")
        return (
            ProviderMatch(logit=2.0, match_probability=0.8),
            ProviderMatch(logit=1.0, match_probability=0.6),
        )

    def embed_images(self, images: tuple[Image.Image, ...]) -> tuple[tuple[float, ...], ...]:
        assert len(images) == 1
        return ((0.6, 0.8),)

    def embed_texts(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0) for _ in texts)


def test_rank_returns_siglip_matches_and_candidate_distribution() -> None:
    service = Siglip2RetrievalService(DeterministicProvider())

    result = service.rank(Image.new("RGB", (2, 2)), (" a cat ", "a dog"))

    assert result.model_id == "google/siglip2-base-patch16-224"
    assert result.device == "mps"
    assert [item.text for item in result.matches] == ["a cat", "a dog"]
    assert [item.rank for item in result.matches] == [1, 2]
    assert result.matches[0].match_probability == 0.8
    assert result.matches[0].candidate_probability == pytest.approx(
        math.exp(2.0) / (math.exp(2.0) + math.exp(1.0))
    )


def test_embeddings_preserve_normalized_provider_vectors() -> None:
    service = Siglip2RetrievalService(DeterministicProvider())

    image = service.embed_image(Image.new("RGB", (2, 2)))
    texts = service.embed_texts(("first", "second"))

    assert image.dimension == 2
    assert image.embeddings == ((0.6, 0.8),)
    assert texts.embeddings == ((1.0, 0.0), (1.0, 0.0))


@pytest.mark.parametrize("texts", [(), ("",), ("   ",)])
def test_text_operations_reject_missing_or_blank_text(texts: tuple[str, ...]) -> None:
    service = Siglip2RetrievalService(DeterministicProvider())

    with pytest.raises(ValueError, match="text"):
        service.embed_texts(texts)


def test_rank_rejects_candidate_count_above_limit() -> None:
    service = Siglip2RetrievalService(DeterministicProvider(), max_texts=1)

    with pytest.raises(ValueError, match="at most 1"):
        service.rank(Image.new("RGB", (2, 2)), ("a", "b"))
