"""Filter images using image-text cosine similarity."""

from __future__ import annotations

from dataclasses import dataclass

from capicove.processing.image.filters._image_text import evaluate_image_text_scorer
from capicove.processing.image.filters._rules import validate_range
from capicove.processing.image.models import Aggregation, FilterDecision, ImageSource
from capicove.processing.image.ports import ImageTextScorer


@dataclass(frozen=True, slots=True)
class ImageTextSimilarityFilter:
    """Keep images whose local-model text cosine similarity passes a range."""

    scorer: ImageTextScorer
    min_score: float = 0.1
    max_score: float = 1.0
    aggregation: Aggregation = Aggregation.ANY

    def __post_init__(self) -> None:
        validate_range(self.min_score, self.max_score, "score")
        if self.min_score < -1 or self.max_score > 1:
            raise ValueError("similarity score bounds must be within [-1, 1]")

    def evaluate(self, sources: tuple[ImageSource, ...], text: str) -> FilterDecision:
        """Evaluate image-text cosine similarity with model lineage."""
        return evaluate_image_text_scorer(
            filter_name="image_text_similarity_filter",
            metric_name="image_text_similarity",
            scorer=self.scorer,
            sources=sources,
            text=text,
            minimum=self.min_score,
            maximum=self.max_score,
            aggregation=self.aggregation,
            probability=False,
        )
