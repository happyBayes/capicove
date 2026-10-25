"""Filter images using local-model watermark probabilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

from capicove.processing.image.filters._model_scores import evaluate_image_scorer
from capicove.processing.image.models import Aggregation, FilterDecision, ImageSource
from capicove.processing.image.ports import ImageScorer


@dataclass(frozen=True, slots=True)
class ImageWatermarkFilter:
    """Keep images whose watermark probability is below a configured threshold."""

    scorer: ImageScorer
    probability_threshold: float = 0.8
    aggregation: Aggregation = Aggregation.ANY

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.probability_threshold)
            or not 0 <= self.probability_threshold <= 1
        ):
            raise ValueError("watermark probability threshold must be within [0, 1]")

    def evaluate(self, sources: tuple[ImageSource, ...]) -> FilterDecision:
        """Evaluate watermark probabilities with model lineage."""
        return evaluate_image_scorer(
            filter_name="image_watermark_filter",
            metric_name="image_watermark_probability",
            scorer=self.scorer,
            sources=sources,
            minimum=0.0,
            maximum=self.probability_threshold,
            aggregation=self.aggregation,
            maximum_inclusive=False,
            probability=True,
        )
