"""Tests for image aspect-ratio filtering."""

from __future__ import annotations

from capicove.processing.image.filters import ImageAspectRatioFilter
from capicove.processing.image.models import Aggregation
from tests.unit.processing.image.authored_images import authored_source


LANDSCAPE = authored_source("landscape")
PORTRAIT = authored_source("portrait")


def test_aspect_ratio_filter_computes_width_over_height() -> None:
    decision = ImageAspectRatioFilter(min_ratio=1.49, max_ratio=1.50).evaluate((LANDSCAPE,))

    assert decision.keep is True
    assert decision.metrics[0].values == (640 / 427,)
    assert dict(decision.parameters)["min_ratio"] == 1.49


def test_aspect_ratio_filter_applies_any_and_all_aggregation() -> None:
    sources = (LANDSCAPE, PORTRAIT)

    assert ImageAspectRatioFilter(1.0, 2.0, Aggregation.ANY).evaluate(sources).keep is True
    assert ImageAspectRatioFilter(1.0, 2.0, Aggregation.ALL).evaluate(sources).keep is False


def test_aspect_ratio_filter_keeps_empty_input_as_not_applicable() -> None:
    decision = ImageAspectRatioFilter().evaluate(())

    assert decision.keep is True
    assert decision.applicable is False
