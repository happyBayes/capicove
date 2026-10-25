"""Filter images by encoded source byte size."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from capicove.processing.image.filters._rules import aggregate, in_range
from capicove.processing.image.invocation import check_image_deadline
from capicove.processing.image.models import (
    Aggregation,
    FilterDecision,
    ImageMetric,
    ImageSource,
)


_SIZE_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB|KIB|MIB|GIB|TIB)?$", re.I)
_MULTIPLIERS = {
    "B": 1,
    "KB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
    "KIB": 1024,
    "MIB": 1024**2,
    "GIB": 1024**3,
    "TIB": 1024**4,
}


def parse_size(value: int | str) -> int:
    """Parse a non-negative byte size or a decimal/binary unit string."""
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0:
            raise ValueError("size must be non-negative")
        return value
    if not isinstance(value, str):
        raise ValueError("size must be an integer or unit string")
    match = _SIZE_PATTERN.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"invalid size value: {value}")
    number = float(match.group(1))
    unit = (match.group(2) or "B").upper()
    return int(number * _MULTIPLIERS[unit])


@dataclass(frozen=True, slots=True)
class ImageSizeFilter:
    """Keep images whose encoded source size is within an inclusive byte range."""

    min_size: int | str = 0
    max_size: int | str = "1TiB"
    aggregation: Aggregation = Aggregation.ANY
    _min_bytes: int = field(init=False, repr=False)
    _max_bytes: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        minimum = parse_size(self.min_size)
        maximum = parse_size(self.max_size)
        if minimum > maximum:
            raise ValueError("minimum size must not exceed maximum size")
        object.__setattr__(self, "_min_bytes", minimum)
        object.__setattr__(self, "_max_bytes", maximum)

    def evaluate(self, sources: tuple[ImageSource, ...]) -> FilterDecision:
        """Measure source bytes without decoding image pixels."""
        contents = tuple(_read_source(source) for source in sources)
        return self._build_decision(sources, tuple(len(content) for content in contents), contents)

    def evaluate_sizes(
        self,
        sources: tuple[ImageSource, ...],
        sizes: tuple[int, ...],
    ) -> FilterDecision:
        """Evaluate already-known sizes while retaining source-hash evidence."""
        if len(sources) != len(sizes):
            raise ValueError("sources and sizes must have the same length")
        contents = tuple(_read_source(source) for source in sources)
        return self._build_decision(sources, sizes, contents)

    def _build_decision(
        self,
        sources: tuple[ImageSource, ...],
        sizes: tuple[int, ...],
        contents: tuple[bytes, ...],
    ) -> FilterDecision:
        parameters = (
            ("aggregation", Aggregation(self.aggregation).value),
            ("max_size_bytes", self._max_bytes),
            ("min_size_bytes", self._min_bytes),
        )
        if not sources:
            return FilterDecision(
                filter_name="image_size_filter",
                keep=True,
                applicable=False,
                aggregation=self.aggregation,
                source_hashes=(),
                metrics=(),
                parameters=parameters,
            )
        values = tuple(float(size) for size in sizes)
        decisions = tuple(in_range(value, self._min_bytes, self._max_bytes) for value in values)
        return FilterDecision(
            filter_name="image_size_filter",
            keep=aggregate(decisions, self.aggregation),
            applicable=True,
            aggregation=self.aggregation,
            source_hashes=tuple(hashlib.sha256(content).hexdigest() for content in contents),
            metrics=(ImageMetric("image_size_bytes", values, "filesystem"),),
            parameters=parameters,
        )


def _read_source(source: ImageSource) -> bytes:
    check_image_deadline()
    if source.data is not None:
        return source.data
    if source.path is None:  # pragma: no cover - protected by ImageSource validation
        raise ValueError("image source has no readable representation")
    content = Path(source.path).read_bytes()
    check_image_deadline()
    return content
