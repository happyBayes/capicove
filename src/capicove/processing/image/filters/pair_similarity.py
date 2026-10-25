"""Filter one image pair by visual cosine similarity."""

from __future__ import annotations

import math
from dataclasses import dataclass

from capicove.processing.image.filters._rules import in_range, validate_range
from capicove.processing.image.invocation import check_image_deadline, consume_image_model_calls
from capicove.processing.image.io import load_image
from capicove.processing.image.models import (
    Aggregation,
    FilterDecision,
    ImageMetric,
    ImageSource,
)
from capicove.processing.image.ports import ImagePairScorer


@dataclass(frozen=True, slots=True)
class ImagePairSimilarityFilter:
    """Keep exactly two distinct images when their cosine similarity passes a range."""

    scorer: ImagePairScorer
    min_score: float = 0.1
    max_score: float = 1.0

    def __post_init__(self) -> None:
        validate_range(self.min_score, self.max_score, "score")
        if self.min_score < -1 or self.max_score > 1:
            raise ValueError("similarity score bounds must be within [-1, 1]")

    def evaluate(self, sources: tuple[ImageSource, ...]) -> FilterDecision:
        """Evaluate one distinct pair with model lineage."""
        if len(sources) != 2:
            raise ValueError("image pair similarity requires exactly two images")
        left, right = (load_image(source) for source in sources)
        if left.sha256 == right.sha256:
            raise ValueError("image pair similarity requires distinct image content")
        check_image_deadline()
        consume_image_model_calls()
        score = float(self.scorer.score(left.image, right.image))
        check_image_deadline()
        if not math.isfinite(score) or not -1 <= score <= 1:
            raise ValueError("image pair similarity score must be within [-1, 1]")
        metric = ImageMetric(
            name="image_pair_similarity",
            values=(score,),
            provider_id=self.scorer.provider_id,
            provider_version=self.scorer.provider_version,
            model_id=self.scorer.model_id,
            model_revision=self.scorer.model_revision,
            model_manifest_sha256=self.scorer.model_manifest_sha256,
            preprocess_revision=self.scorer.preprocess_revision,
            device=self.scorer.device,
        )
        return FilterDecision(
            filter_name="image_pair_similarity_filter",
            keep=in_range(score, self.min_score, self.max_score),
            applicable=True,
            aggregation=Aggregation.ALL,
            source_hashes=(left.sha256, right.sha256),
            metrics=(metric,),
            parameters=(("max_score", self.max_score), ("min_score", self.min_score)),
        )
