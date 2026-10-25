"""Filter images using local-model NSFW probabilities."""

from __future__ import annotations

from dataclasses import dataclass

from capicove.processing.image.filters._model_scores import evaluate_image_scorer
from capicove.processing.image.filters._rules import validate_range
from capicove.processing.image.models import Aggregation, FilterDecision, ImageSource
from capicove.processing.image.ports import ImageScorer


@dataclass(frozen=True, slots=True)
class ImageNsfwFilter:
    """Keep images whose NSFW probabilities are within an inclusive safe range."""

    scorer: ImageScorer
    min_score: float = 0.0
    max_score: float = 0.5
    aggregation: Aggregation = Aggregation.ANY

    def __post_init__(self) -> None:
        validate_range(self.min_score, self.max_score, "score")
        if self.min_score < 0 or self.max_score > 1:
            raise ValueError("NSFW score bounds must be within [0, 1]")

    def evaluate(self, sources: tuple[ImageSource, ...]) -> FilterDecision:
        """Evaluate NSFW probabilities with model lineage."""
        return evaluate_image_scorer(
            filter_name="image_nsfw_filter",
            metric_name="image_nsfw_probability",
            scorer=self.scorer,
            sources=sources,
            minimum=self.min_score,
            maximum=self.max_score,
            aggregation=self.aggregation,
            probability=True,
        )
