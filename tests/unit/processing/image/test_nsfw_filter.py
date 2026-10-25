"""Tests for local-model NSFW probability filtering."""

from __future__ import annotations

import pytest

from capicove.processing.image.filters import ImageNsfwFilter
from capicove.processing.image.models import Aggregation
from tests.unit.processing.image.authored_images import authored_source
from tests.unit.processing.image.fakes import FixedImageScorer


SOURCES = (authored_source("landscape"), authored_source("portrait"))


def test_nsfw_filter_keeps_only_probabilities_below_configured_maximum() -> None:
    decision = ImageNsfwFilter(
        scorer=FixedImageScorer((0.1, 0.8)),
        max_score=0.5,
        aggregation=Aggregation.ALL,
    ).evaluate(SOURCES)

    assert decision.keep is False
    assert decision.metrics[0].name == "image_nsfw_probability"
    assert decision.operation_version == 1
    assert dict(decision.parameters) == {
        "aggregation": "all",
        "max_score": 0.5,
        "maximum_inclusive": True,
        "min_score": 0.0,
    }


def test_nsfw_filter_any_can_keep_a_mixed_batch() -> None:
    decision = ImageNsfwFilter(
        scorer=FixedImageScorer((0.1, 0.8)),
        max_score=0.5,
        aggregation=Aggregation.ANY,
    ).evaluate(SOURCES)

    assert decision.keep is True


def test_nsfw_filter_coerces_transport_aggregation_value() -> None:
    decision = ImageNsfwFilter(
        scorer=FixedImageScorer((0.1, 0.8)),
        max_score=0.5,
        aggregation="any",  # type: ignore[arg-type]
    ).evaluate(SOURCES)

    assert decision.keep is True
    assert decision.aggregation is Aggregation.ANY


def test_nsfw_filter_rejects_provider_probability_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        ImageNsfwFilter(FixedImageScorer((1.2, 0.1))).evaluate(SOURCES)
