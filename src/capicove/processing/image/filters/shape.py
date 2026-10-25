"""Filter images by decoded pixel dimensions."""

from __future__ import annotations

import sys
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
class ImageShapeFilter:
    """Keep images whose width and height are both within inclusive ranges."""

    min_width: int = 1
    max_width: int = sys.maxsize
    min_height: int = 1
    max_height: int = sys.maxsize
    aggregation: Aggregation = Aggregation.ANY

    def __post_init__(self) -> None:
        validate_range(self.min_width, self.max_width, "width")
        validate_range(self.min_height, self.max_height, "height")
        if self.min_width < 1 or self.min_height < 1:
            raise ValueError("minimum width and height must be positive")

    def evaluate(self, sources: tuple[ImageSource, ...]) -> FilterDecision:
        """Evaluate ordered image sources without mutating them."""
        parameters = (
            ("aggregation", Aggregation(self.aggregation).value),
            ("max_height", self.max_height),
            ("max_width", self.max_width),
            ("min_height", self.min_height),
            ("min_width", self.min_width),
        )
        if not sources:
            return FilterDecision(
                filter_name="image_shape_filter",
                keep=True,
                applicable=False,
                aggregation=self.aggregation,
                source_hashes=(),
                metrics=(),
                parameters=parameters,
            )
        loaded = tuple(load_image(source) for source in sources)
        widths = tuple(float(image.width) for image in loaded)
        heights = tuple(float(image.height) for image in loaded)
        per_image = tuple(
            in_range(width, self.min_width, self.max_width)
            and in_range(height, self.min_height, self.max_height)
            for width, height in zip(widths, heights, strict=True)
        )
        return FilterDecision(
            filter_name="image_shape_filter",
            keep=aggregate(per_image, self.aggregation),
            applicable=True,
            aggregation=self.aggregation,
            source_hashes=tuple(image.sha256 for image in loaded),
            metrics=(
                ImageMetric("image_width", widths, "pillow"),
                ImageMetric("image_height", heights, "pillow"),
            ),
            parameters=parameters,
        )
