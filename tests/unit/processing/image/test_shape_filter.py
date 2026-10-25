"""Tests for image shape filtering."""

from __future__ import annotations

from capicove.processing.image.filters import ImageShapeFilter
from capicove.processing.image.models import Aggregation
from tests.unit.processing.image.authored_images import authored_source


SMALL = authored_source("landscape")
LARGE = authored_source("large")


def test_shape_filter_checks_width_and_height_together() -> None:
    decision = ImageShapeFilter(
        min_width=600, max_width=700, min_height=400, max_height=500
    ).evaluate((SMALL,))

    assert decision.keep is True
    assert decision.metrics[0].values == (640.0,)
    assert decision.metrics[1].values == (427.0,)
    assert dict(decision.parameters)["min_width"] == 600


def test_shape_filter_applies_all_aggregation() -> None:
    decision = ImageShapeFilter(
        min_width=600,
        max_width=700,
        min_height=400,
        max_height=1000,
        aggregation=Aggregation.ALL,
    ).evaluate((SMALL, LARGE))

    assert decision.keep is False
