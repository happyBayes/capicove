"""Public image mapper operations."""

from capicove.processing.image.mappers.difference_area import (
    ImgDiffDifferenceAreaGeneratorMapper,
)
from capicove.processing.image.mappers.difference_caption import (
    ImgDiffDifferenceCaptionGeneratorMapper,
)
from capicove.processing.image.mappers.remove_background import (
    BackgroundRemovalOptions,
    ImageRemoveBackgroundMapper,
)


__all__ = [
    "BackgroundRemovalOptions",
    "ImageRemoveBackgroundMapper",
    "ImgDiffDifferenceAreaGeneratorMapper",
    "ImgDiffDifferenceCaptionGeneratorMapper",
]
