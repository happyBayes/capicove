"""Tests for local-model watermark probability filtering."""

from __future__ import annotations

from capicove.processing.image.filters import ImageWatermarkFilter
from capicove.processing.image.models import Aggregation
from tests.unit.processing.image.authored_images import authored_source
from tests.unit.processing.image.fakes import FixedImageScorer


SOURCES = (authored_source("landscape"), authored_source("portrait"))


def test_watermark_filter_rejects_probability_at_or_above_threshold() -> None:
    decision = ImageWatermarkFilter(
        scorer=FixedImageScorer((0.2, 0.8)),
        probability_threshold=0.8,
        aggregation=Aggregation.ALL,
    ).evaluate(SOURCES)

    assert decision.keep is False
    assert decision.metrics[0].values == (0.2, 0.8)


def test_watermark_filter_any_retains_when_one_image_is_clean() -> None:
    decision = ImageWatermarkFilter(
        scorer=FixedImageScorer((0.2, 0.9)),
        probability_threshold=0.8,
        aggregation=Aggregation.ANY,
    ).evaluate(SOURCES)

    assert decision.keep is True
