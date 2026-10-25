"""Tests for image-pair visual similarity filtering."""

from __future__ import annotations

import pytest

from capicove.processing.image.filters import ImagePairSimilarityFilter
from tests.unit.processing.image.authored_images import authored_source
from tests.unit.processing.image.fakes import FixedImagePairScorer


LEFT = authored_source("landscape")
RIGHT = authored_source("portrait")


def test_pair_similarity_filter_scores_exactly_two_distinct_images() -> None:
    decision = ImagePairSimilarityFilter(
        scorer=FixedImagePairScorer(0.65), min_score=0.5, max_score=0.7
    ).evaluate((LEFT, RIGHT))

    assert decision.keep is True
    assert decision.metrics[0].values == (0.65,)
    assert len(decision.source_hashes) == 2
    assert dict(decision.parameters) == {"max_score": 0.7, "min_score": 0.5}


def test_pair_similarity_filter_rejects_same_content_pair() -> None:
    with pytest.raises(ValueError, match="distinct"):
        ImagePairSimilarityFilter(FixedImagePairScorer(1.0)).evaluate((LEFT, LEFT))


def test_pair_similarity_filter_requires_two_images() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        ImagePairSimilarityFilter(FixedImagePairScorer(1.0)).evaluate((LEFT,))


def test_pair_similarity_filter_rejects_provider_score_outside_cosine_range() -> None:
    with pytest.raises(ValueError, match="similarity score"):
        ImagePairSimilarityFilter(FixedImagePairScorer(1.2)).evaluate((LEFT, RIGHT))
