"""Tests for model-backed image aesthetics filtering."""

from __future__ import annotations

from capicove.processing.image.filters import ImageAestheticsFilter
from capicove.processing.image.models import Aggregation
from tests.unit.processing.image.authored_images import authored_source
from tests.unit.processing.image.fakes import FixedImageScorer


SOURCES = (authored_source("landscape"), authored_source("portrait"))


def test_aesthetics_filter_records_model_lineage_and_any_decision() -> None:
    decision = ImageAestheticsFilter(
        scorer=FixedImageScorer((0.4, 0.8)),
        min_score=0.7,
        max_score=1.0,
        aggregation=Aggregation.ANY,
    ).evaluate(SOURCES)

    assert decision.keep is True
    assert decision.metrics[0].values == (0.4, 0.8)
    assert decision.metrics[0].model_id == "authored-test-model"
    assert decision.metrics[0].preprocess_revision == "authored-rgb-v1"
    assert decision.metrics[0].model_manifest_sha256 == "a" * 64
    assert decision.metrics[0].device == "cpu"


def test_aesthetics_filter_all_requires_every_image_to_pass() -> None:
    decision = ImageAestheticsFilter(
        scorer=FixedImageScorer((0.4, 0.8)),
        min_score=0.7,
        aggregation=Aggregation.ALL,
    ).evaluate(SOURCES)

    assert decision.keep is False
