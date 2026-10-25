"""Tests for deliberate public image-processing exports."""

from __future__ import annotations

import capicove.processing.image as image_processing


def test_public_image_processing_api_exports_all_capabilities() -> None:
    expected = {
        "ImageRemoveBackgroundMapper",
        "ImgDiffDifferenceAreaGeneratorMapper",
        "ImgDiffDifferenceCaptionGeneratorMapper",
        "ImageAestheticsFilter",
        "ImageAspectRatioFilter",
        "ImageNsfwFilter",
        "ImageShapeFilter",
        "ImageSizeFilter",
        "ImageSubplotFilter",
        "ImageTextMatchingFilter",
        "ImageTextSimilarityFilter",
        "ImageWatermarkFilter",
        "ImagePairSimilarityFilter",
        "ImageLshSearch",
        "ImagePhash",
        "ImagePhashResult",
        "ImagePerceptualDeduplicator",
        "RayImagePerceptualDeduplicator",
        "ImageOcr",
        "ImageOcrResult",
        "OcrWord",
        "DEFAULT_OCR_LANGUAGES",
        "binary_mask_iou",
        "bounding_box_iou",
        "coefficient_of_variation",
        "information_gain",
    }

    assert expected.issubset(set(image_processing.__all__))
    assert all(hasattr(image_processing, name) for name in expected)
