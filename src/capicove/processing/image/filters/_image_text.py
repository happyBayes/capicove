"""Shared orchestration for image-text score filters."""

from __future__ import annotations

import hashlib

from capicove.processing.image._validation import require_non_blank
from capicove.processing.image.filters._rules import aggregate, in_range
from capicove.processing.image.invocation import check_image_deadline, consume_image_model_calls
from capicove.processing.image.io import load_image
from capicove.processing.image.models import (
    Aggregation,
    FilterDecision,
    ImageMetric,
    ImageSource,
)
from capicove.processing.image.ports import ImageTextScorer


def evaluate_image_text_scorer(
    *,
    filter_name: str,
    metric_name: str,
    scorer: ImageTextScorer,
    sources: tuple[ImageSource, ...],
    text: str,
    minimum: float,
    maximum: float,
    aggregation: Aggregation,
    probability: bool,
) -> FilterDecision:
    """Score ordered images against text and apply an inclusive range."""
    require_non_blank("text", text)
    parameters = (
        ("aggregation", Aggregation(aggregation).value),
        ("max_score", maximum),
        ("min_score", minimum),
        ("probability", probability),
        ("text_sha256", hashlib.sha256(text.encode("utf-8")).hexdigest()),
    )
    if not sources:
        return FilterDecision(
            filter_name,
            True,
            False,
            aggregation,
            (),
            (),
            parameters=parameters,
        )
    loaded = tuple(load_image(source) for source in sources)
    check_image_deadline()
    consume_image_model_calls()
    scores = tuple(
        float(value) for value in scorer.score(tuple(item.image for item in loaded), text)
    )
    check_image_deadline()
    if len(scores) != len(loaded):
        raise ValueError("image-text scorer must return one score per image")
    if probability and any(value < 0 or value > 1 for value in scores):
        raise ValueError("image-text matching probabilities must be within [0, 1]")
    if not probability and any(value < -1 or value > 1 for value in scores):
        raise ValueError("image-text similarity scores must be within [-1, 1]")
    metric = ImageMetric(
        name=metric_name,
        values=scores,
        provider_id=scorer.provider_id,
        provider_version=scorer.provider_version,
        model_id=scorer.model_id,
        model_revision=scorer.model_revision,
        model_manifest_sha256=scorer.model_manifest_sha256,
        preprocess_revision=scorer.preprocess_revision,
        device=scorer.device,
    )
    return FilterDecision(
        filter_name=filter_name,
        keep=aggregate(tuple(in_range(value, minimum, maximum) for value in scores), aggregation),
        applicable=True,
        aggregation=aggregation,
        source_hashes=tuple(item.sha256 for item in loaded),
        metrics=(metric,),
        parameters=parameters,
    )
