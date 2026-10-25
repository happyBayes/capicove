"""Shared model-score orchestration for image filters."""

from __future__ import annotations

from capicove.processing.image.filters._rules import aggregate, in_range
from capicove.processing.image.invocation import check_image_deadline, consume_image_model_calls
from capicove.processing.image.io import load_image
from capicove.processing.image.models import (
    Aggregation,
    FilterDecision,
    ImageMetric,
    ImageSource,
)
from capicove.processing.image.ports import ImageScorer


def evaluate_image_scorer(
    *,
    filter_name: str,
    metric_name: str,
    scorer: ImageScorer,
    sources: tuple[ImageSource, ...],
    minimum: float,
    maximum: float,
    aggregation: Aggregation,
    maximum_inclusive: bool = True,
    probability: bool = False,
) -> FilterDecision:
    """Load sources once, score them, and apply an inclusive range."""
    if not sources:
        return FilterDecision(
            filter_name,
            True,
            False,
            aggregation,
            (),
            (),
            parameters=_parameters(minimum, maximum, aggregation, maximum_inclusive),
        )
    loaded = tuple(load_image(source) for source in sources)
    check_image_deadline()
    consume_image_model_calls()
    values = tuple(float(value) for value in scorer.score(tuple(item.image for item in loaded)))
    check_image_deadline()
    if len(values) != len(loaded):
        raise ValueError("image scorer must return one score per image")
    if probability and any(value < 0 or value > 1 for value in values):
        raise ValueError("image scorer probabilities must be within [0, 1]")
    metric = ImageMetric(
        name=metric_name,
        values=values,
        provider_id=scorer.provider_id,
        provider_version=scorer.provider_version,
        model_id=scorer.model_id,
        model_revision=scorer.model_revision,
        model_manifest_sha256=scorer.model_manifest_sha256,
        preprocess_revision=scorer.preprocess_revision,
        device=scorer.device,
    )
    decisions = tuple(
        in_range(value, minimum, maximum) if maximum_inclusive else minimum <= value < maximum
        for value in values
    )
    return FilterDecision(
        filter_name=filter_name,
        keep=aggregate(decisions, aggregation),
        applicable=True,
        aggregation=Aggregation(aggregation),
        source_hashes=tuple(item.sha256 for item in loaded),
        metrics=(metric,),
        parameters=_parameters(minimum, maximum, aggregation, maximum_inclusive),
    )


def _parameters(
    minimum: float,
    maximum: float,
    aggregation: Aggregation,
    maximum_inclusive: bool,
) -> tuple[tuple[str, str | float | bool], ...]:
    return (
        ("aggregation", Aggregation(aggregation).value),
        ("max_score", maximum),
        ("maximum_inclusive", maximum_inclusive),
        ("min_score", minimum),
    )
