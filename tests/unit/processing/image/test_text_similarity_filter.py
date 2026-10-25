"""Tests for CLIP-style image-text similarity filtering."""

from __future__ import annotations

import pytest

from capicove.processing.image.filters import ImageTextSimilarityFilter
from capicove.processing.image.models import Aggregation
from tests.unit.processing.image.authored_images import authored_source
from tests.unit.processing.image.fakes import FixedImageTextScorer


SOURCES = (authored_source("landscape"), authored_source("portrait"))


def test_text_similarity_filter_preserves_cosine_scores() -> None:
    decision = ImageTextSimilarityFilter(
        scorer=FixedImageTextScorer((0.7, -0.2)),
        min_score=0.5,
        aggregation=Aggregation.ANY,
    ).evaluate(SOURCES, "a studio scene")

    assert decision.keep is True
    assert decision.metrics[0].values == (0.7, -0.2)
    assert decision.metrics[0].name == "image_text_similarity"


def test_text_similarity_filter_all_requires_every_similarity() -> None:
    decision = ImageTextSimilarityFilter(
        scorer=FixedImageTextScorer((0.7, -0.2)),
        min_score=0.5,
        aggregation=Aggregation.ALL,
    ).evaluate(SOURCES, "a studio scene")

    assert decision.keep is False


def test_text_similarity_filter_rejects_provider_score_outside_cosine_range() -> None:
    with pytest.raises(ValueError, match="similarity"):
        ImageTextSimilarityFilter(
            scorer=FixedImageTextScorer((1.2, 0.2)),
        ).evaluate(SOURCES, "a studio scene")
