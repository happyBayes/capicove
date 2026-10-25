"""Tests for image-text matching probability filtering."""

from __future__ import annotations

from capicove.processing.image.filters import ImageTextMatchingFilter
from capicove.processing.image.models import Aggregation
from tests.unit.processing.image.authored_images import authored_source
from tests.unit.processing.image.fakes import FixedImageTextScorer


SOURCES = (authored_source("landscape"), authored_source("portrait"))


def test_text_matching_filter_scores_images_against_nonblank_text() -> None:
    scorer = FixedImageTextScorer((0.9, 0.2))

    decision = ImageTextMatchingFilter(
        scorer=scorer,
        min_score=0.5,
        aggregation=Aggregation.ANY,
    ).evaluate(SOURCES, "a person working")

    assert decision.keep is True
    assert scorer.texts == ["a person working"]
    assert decision.metrics[0].name == "image_text_matching_probability"
    assert (
        dict(decision.parameters)["text_sha256"]
        == (
            "14965ccc04c09a5b18ab8ad5383f79509432833e16258765610726d4b5788d9d"  # pragma: allowlist secret
        )
    )


def test_text_matching_filter_all_rejects_one_mismatch() -> None:
    decision = ImageTextMatchingFilter(
        scorer=FixedImageTextScorer((0.9, 0.2)),
        min_score=0.5,
        aggregation=Aggregation.ALL,
    ).evaluate(SOURCES, "a person working")

    assert decision.keep is False
