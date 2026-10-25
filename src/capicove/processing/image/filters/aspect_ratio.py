"""Filter images by their width-to-height ratio."""

from __future__ import annotations

from dataclasses import dataclass

from capicove.processing.image.filters._rules import aggregate, in_range, validate_range
from capicove.processing.image.io import load_image
from capicove.processing.image.models import (
    Aggregation,
    FilterDecision,
    ImageMetric,
    ImageSource,
)


@dataclass(frozen=True, slots=True)
class ImageAspectRatioFilter:
    """Keep images whose width divided by height is within an inclusive range."""

    min_ratio: float = 0.333
    max_ratio: float = 3.0
    aggregation: Aggregation = Aggregation.ANY

    def __post_init__(self) -> None:
        validate_range(self.min_ratio, self.max_ratio, "ratio")
        if self.min_ratio < 0:
            raise ValueError("minimum ratio must be non-negative")

    def evaluate(self, sources: tuple[ImageSource, ...]) -> FilterDecision:
        """Evaluate ordered image sources without mutating them."""
        parameters = (
            ("aggregation", Aggregation(self.aggregation).value),
            ("max_ratio", self.max_ratio),
            ("min_ratio", self.min_ratio),
        )
        if not sources:
            return FilterDecision(
                filter_name="image_aspect_ratio_filter",
                keep=True,
                applicable=False,
                aggregation=self.aggregation,
                source_hashes=(),
                metrics=(),
                parameters=parameters,
            )
        loaded = tuple(load_image(source) for source in sources)
        ratios = tuple(image.width / image.height for image in loaded)
        return FilterDecision(
            filter_name="image_aspect_ratio_filter",
            keep=aggregate(
                tuple(in_range(value, self.min_ratio, self.max_ratio) for value in ratios),
                self.aggregation,
            ),
            applicable=True,
            aggregation=self.aggregation,
            source_hashes=tuple(image.sha256 for image in loaded),
            metrics=(ImageMetric("aspect_ratio", ratios, "pillow"),),
            parameters=parameters,
        )
