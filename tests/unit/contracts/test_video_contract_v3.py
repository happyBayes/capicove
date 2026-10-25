"""Regression checks for additive video contract compatibility."""

import inspect

import pytest

from capicove.contracts.retrieval import (
    RETRIEVAL_CONTRACT_VERSION,
    CharacterRange,
    CompactHit,
    FullHit,
    TargetType,
    TemporalRange,
    TextChunkLocator,
    TextOrigin,
)


def test_temporal_ocr_and_document_ocr_are_exclusive() -> None:
    locator = TextChunkLocator(TextOrigin.OCR, CharacterRange(0, 5), time_range=TemporalRange(1, 2))
    assert locator.page_number is None
    # Version 4 (RFC-0032) is additive: every version-3 discriminator is unchanged.
    assert RETRIEVAL_CONTRACT_VERSION == 4
    assert TextChunkLocator(TextOrigin.OCR, CharacterRange(0, 5), page_number=1).time_range is None
    for page, time in ((None, None), (1, TemporalRange(1, 2))):
        with pytest.raises(ValueError):
            TextChunkLocator(
                TextOrigin.OCR, CharacterRange(0, 5), page_number=page, time_range=time
            )


def test_preview_is_keyword_only_and_preserves_full_hit_positions() -> None:
    legacy = FullHit("k", TargetType.KEYFRAME, 0.5, "e", 10, None, None, None, "artifact")
    assert legacy.artifact_ref == "artifact"
    assert legacy.preview_resource_ref is None
    parameter = inspect.signature(FullHit).parameters["preview_resource_ref"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    hit = CompactHit(
        "k", TargetType.KEYFRAME, 0.5, "e", 10, preview_resource_ref="resource:preview:k"
    )
    assert hit.preview_resource_ref == "resource:preview:k"
    with pytest.raises(ValueError):
        CompactHit("k", TargetType.KEYFRAME, 0.5, "e", 10, preview_resource_ref=" ")
