"""Filter images using a local aesthetics scorer."""

from __future__ import annotations

from dataclasses import dataclass

from capicove.processing.image.filters._model_scores import evaluate_image_scorer
from capicove.processing.image.filters._rules import validate_range
from capicove.processing.image.models import Aggregation, FilterDecision, ImageSource
from capicove.processing.image.ports import ImageScorer


@dataclass(frozen=True, slots=True)
class ImageAestheticsFilter:
    """Keep images whose local-model aesthetics scores pass an inclusive range."""

    scorer: ImageScorer
    min_score: float = 0.5
    max_score: float = 1.0
    aggregation: Aggregation = Aggregation.ANY

    def __post_init__(self) -> None:
        validate_range(self.min_score, self.max_score, "score")

    def evaluate(self, sources: tuple[ImageSource, ...]) -> FilterDecision:
        """Evaluate aesthetics scores with model lineage."""
        return evaluate_image_scorer(
            filter_name="image_aesthetics_filter",
            metric_name="image_aesthetics_score",
            scorer=self.scorer,
            sources=sources,
            minimum=self.min_score,
            maximum=self.max_score,
            aggregation=self.aggregation,
        )
