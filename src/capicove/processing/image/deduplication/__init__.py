"""Perceptual image deduplication for training-dataset curation."""

from capicove.processing.image._adapters.ray_dedup import RayImagePerceptualDeduplicator
from capicove.processing.image.deduplication.lsh import (
    HashRecord,
    ImageLshSearch,
    LshMatch,
    LshSearchResult,
)
from capicove.processing.image.deduplication.perceptual import (
    DuplicateGroup,
    ImagePerceptualDeduplicator,
    ImagePhash,
    ImagePhashResult,
    ImageRecord,
    PerceptualDeduplicationResult,
    PhashRecord,
    PreparedImageRecord,
)


__all__ = [
    "DuplicateGroup",
    "HashRecord",
    "ImageLshSearch",
    "ImagePerceptualDeduplicator",
    "ImagePhash",
    "ImagePhashResult",
    "ImageRecord",
    "LshMatch",
    "LshSearchResult",
    "PerceptualDeduplicationResult",
    "PhashRecord",
    "PreparedImageRecord",
    "RayImagePerceptualDeduplicator",
]
