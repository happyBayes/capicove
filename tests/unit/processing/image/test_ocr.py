"""Tests for structured image OCR contracts and orchestration."""

from __future__ import annotations

import dataclasses
from io import BytesIO

import pytest
from PIL import Image

from capicove.processing.image.models import ImageOcrResult, ImageSource, OcrWord
from capicove.processing.image.ocr import ImageOcr, OcrProcessingError, OcrProviderOutput


class _FixedProvider:
    provider_id = "fixed-ocr"
    provider_version = "1"

    def __init__(self, outputs: tuple[OcrProviderOutput, ...]) -> None:
        self._outputs = list(outputs)
        self.languages: list[tuple[str, ...]] = []

    def recognize(
        self,
        image: Image.Image,
        languages: tuple[str, ...],
    ) -> OcrProviderOutput:
        del image
        self.languages.append(languages)
        return self._outputs.pop(0)


def test_ocr_word_requires_valid_geometry_confidence_and_hierarchy() -> None:
    word = OcrWord("你好", 1, 2, 20, 10, 0.91, 1, 1, 1, 1, 1)

    assert word.text == "你好"
    with pytest.raises(ValueError, match="confidence"):
        dataclasses.replace(word, confidence=1.01)
    with pytest.raises(ValueError, match="positive"):
        dataclasses.replace(word, width=0)
    with pytest.raises(ValueError, match="hierarchy"):
        dataclasses.replace(word, line=0)
    with pytest.raises(ValueError, match="integers"):
        dataclasses.replace(word, x=0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="integers"):
        dataclasses.replace(word, page=True)


def test_image_ocr_result_validates_source_and_provider_lineage() -> None:
    result = ImageOcrResult(
        source_hash="a" * 64,
        full_text="hello",
        words=(OcrWord("hello", 0, 0, 10, 10, 0.9, 1, 1, 1, 1, 1),),
        provider_id="tesseract-cli",
        provider_version="5.5.0",
        languages=("chi_sim", "eng"),
        parameters=(("languages", "chi_sim+eng"),),
    )

    assert result.languages == ("chi_sim", "eng")
    with pytest.raises(ValueError, match="unique"):
        dataclasses.replace(result, languages=("eng", "eng"))


def test_image_ocr_defaults_to_chinese_and_english_and_builds_lines() -> None:
    words = (
        OcrWord("你好", 0, 0, 10, 10, 0.95, 1, 1, 1, 1, 1),
        OcrWord("world", 12, 0, 10, 10, 0.90, 1, 1, 1, 1, 2),
        OcrWord("next", 0, 20, 10, 10, 0.85, 1, 1, 1, 2, 1),
    )
    provider = _FixedProvider((OcrProviderOutput(words),))

    result = ImageOcr(provider).recognize((ImageSource(data=_png_bytes(size=(30, 40))),))[0]

    assert provider.languages == [("chi_sim", "eng")]
    assert result.full_text == "你好 world\nnext"
    assert result.parameters == (("languages", "chi_sim+eng"),)


def test_image_ocr_preserves_source_order_and_hashes() -> None:
    first = _png_bytes("white")
    second = _png_bytes("black")
    provider = _FixedProvider((OcrProviderOutput(()), OcrProviderOutput(())))

    results = ImageOcr(provider).recognize(
        (ImageSource(data=first), ImageSource(data=second)),
        ("eng",),
    )

    assert [result.source_hash for result in results] != [results[1].source_hash] * 2
    assert [result.languages for result in results] == [("eng",), ("eng",)]


@pytest.mark.parametrize(
    "languages",
    [(), ("eng", "eng"), ("-psm",), tuple("a" for _ in range(9))],
)
def test_image_ocr_rejects_unsafe_languages(languages: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="languages"):
        ImageOcr(_FixedProvider(())).recognize((), languages)


def test_image_ocr_empty_sources_do_not_call_provider() -> None:
    provider = _FixedProvider(())

    assert ImageOcr(provider).recognize(()) == ()
    assert provider.languages == []


def test_image_ocr_rejects_provider_boxes_outside_the_image() -> None:
    word = OcrWord("outside", 15, 0, 10, 10, 0.9, 1, 1, 1, 1, 1)
    provider = _FixedProvider((OcrProviderOutput((word,)),))

    with pytest.raises(OcrProcessingError, match="bounds"):
        ImageOcr(provider).recognize((ImageSource(data=_png_bytes(size=(20, 20))),))


def _png_bytes(color: str = "white", size: tuple[int, int] = (20, 20)) -> bytes:
    stream = BytesIO()
    Image.new("RGB", size, color).save(stream, format="PNG")
    return stream.getvalue()


def test_upload_recognition_obeys_cumulative_invocation_pixel_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import UTC, datetime, timedelta

    from capicove.processing.image import invocation

    monkeypatch.setattr(invocation, "_MAX_INVOCATION_DECODED_PIXELS", 10)
    provider = _FixedProvider((OcrProviderOutput(()),))
    with invocation.image_invocation_deadline(datetime.now(UTC) + timedelta(seconds=10)):
        with pytest.raises(ValueError, match="cumulative decoded-pixel"):
            ImageOcr(provider).recognize_upload(
                _png_bytes(size=(4, 4)), media_type="image/png", max_pixels=100
            )
    assert not provider.languages
