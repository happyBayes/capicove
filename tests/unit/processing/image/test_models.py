"""Unit tests for immutable image-processing contracts."""

from __future__ import annotations

import pytest

from capicove.processing.image.models import (
    Aggregation,
    BoundingBox,
    FilterDecision,
    ImageMetric,
    ImageSource,
)


def test_image_source_requires_exactly_one_representation() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        ImageSource()

    with pytest.raises(ValueError, match="exactly one"):
        ImageSource(path="image.jpg", data=b"image")


def test_image_metric_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        ImageMetric(name="aesthetics", values=(float("nan"),), provider_id="local")


def test_bounding_box_validates_geometry_and_probability() -> None:
    with pytest.raises(ValueError, match="positive"):
        BoundingBox(x=0.0, y=0.0, width=0.0, height=1.0)

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0, confidence=1.1)


def test_filter_decision_preserves_immutable_lineage() -> None:
    decision = FilterDecision(
        filter_name="image_aspect_ratio_filter",
        keep=True,
        applicable=True,
        aggregation=Aggregation.ALL,
        source_hashes=("a" * 64,),
        metrics=(ImageMetric("aspect_ratio", (1.5,), "pillow"),),
    )

    assert decision.source_hashes == ("a" * 64,)
    assert decision.metrics[0].values == (1.5,)


def test_operation_lineage_rejects_invalid_version_and_duplicate_parameters() -> None:
    metric = ImageMetric("aspect_ratio", (1.5,), "pillow")
    with pytest.raises(ValueError, match="operation_version"):
        FilterDecision(
            "image_aspect_ratio_filter",
            True,
            True,
            Aggregation.ALL,
            ("a" * 64,),
            (metric,),
            operation_version="1",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="unique"):
        FilterDecision(
            "image_aspect_ratio_filter",
            True,
            True,
            Aggregation.ALL,
            ("a" * 64,),
            (metric,),
            parameters=(("threshold", 1), ("threshold", 2)),
        )
