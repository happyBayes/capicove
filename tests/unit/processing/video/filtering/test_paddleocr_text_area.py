"""Paddle video OCR composition preserves frame channels and area semantics."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from capicove.processing.image import OcrWord
from capicove.processing.image.ocr import OcrProviderOutput
from capicove.processing.image.ocr_configuration import ImageOcrConfiguration
from capicove.processing.video.filtering import (
    BackendUnavailableFilterError,
    OcrAreaCriterion,
    SampledFrame,
)
from capicove.processing.video.filtering._adapters.paddleocr_text_area import (
    PaddleOcrTextAreaEvaluator,
)
from tests.unit.processing.video.filtering.test_easyocr_text_area import (
    _deadline,
    _item,
    _metadata,
    _source,
)


def test_bgr_frames_are_converted_and_overlapping_lines_count_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def recognize(image: Image.Image, languages: tuple[str, ...]) -> OcrProviderOutput:
        assert image.getpixel((0, 0)) == (255, 0, 0)
        assert languages == ("ch_sim", "en")
        return OcrProviderOutput(
            (
                OcrWord("one", 0, 0, 5, 5, 1.0, 1, 1, 1, 1, 1),
                OcrWord("two", 3, 0, 5, 5, 1.0, 1, 1, 1, 2, 1),
            )
        )

    monkeypatch.setattr(
        ImageOcrConfiguration, "create_provider", lambda self: SimpleNamespace(recognize=recognize)
    )
    pixels = np.full((10, 10, 3), [0, 0, 255], dtype=np.uint8)
    result = PaddleOcrTextAreaEvaluator(ImageOcrConfiguration(model_root=tmp_path)).evaluate(
        _item(),
        _source(),
        _metadata(),
        (SampledFrame(0, 0, 10, 10, pixels),),
        OcrAreaCriterion("ocr"),
        _deadline(),
    )
    assert result.score == pytest.approx(0.4)
    assert result.details["backend"] == "paddleocr-line-boxes"


def test_missing_root_reports_unavailable_without_easyocr_fallback() -> None:
    with pytest.raises(BackendUnavailableFilterError, match="Paddle"):
        PaddleOcrTextAreaEvaluator().evaluate(
            _item(), _source(), _metadata(), (), OcrAreaCriterion("ocr"), _deadline()
        )
