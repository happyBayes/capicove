"""Structured image OCR orchestration independent from provider implementation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from capicove.processing.image.invocation import (
    account_decoded_image,
    check_image_deadline,
    consume_image_model_calls,
)
from capicove.processing.image.io import load_image
from capicove.processing.image.models import ImageOcrResult, ImageSource, OcrWord
from capicove.processing.image.ports import ImageOcrProvider
from capicove.processing.image.preparation import ImageFormatError, prepare_image


if TYPE_CHECKING:
    from PIL import Image


OCR_MEDIA_TYPES = {
    "image/png": "PNG",
    "image/jpeg": "JPEG",
    "image/webp": "WEBP",
    "image/bmp": "BMP",
}


DEFAULT_OCR_LANGUAGES = ("chi_sim", "eng")
_LANGUAGE_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
_MAX_LANGUAGES = 8
_MAX_WORDS = 100_000


class OcrProviderUnavailableError(RuntimeError):
    """Raised when the configured OCR provider cannot run locally."""


class OcrProcessingError(RuntimeError):
    """Raised when an OCR provider returns invalid or failed output."""


@dataclass(frozen=True, slots=True)
class OcrProviderOutput:
    """Carry provider-recognized words without source lineage."""

    words: tuple[OcrWord, ...]


def validate_ocr_languages(languages: tuple[str, ...]) -> tuple[str, ...]:
    """Validate and preserve ordered Tesseract language identifiers."""
    if not 1 <= len(languages) <= _MAX_LANGUAGES:
        raise ValueError("OCR languages must contain between one and eight values")
    if len(set(languages)) != len(languages):
        raise ValueError("OCR languages must be unique")
    if any(_LANGUAGE_PATTERN.fullmatch(language) is None for language in languages):
        raise ValueError("OCR languages contain an invalid identifier")
    return languages


class ImageOcr:
    """Recognize ordered image sources through an injected local provider."""

    def __init__(self, provider: ImageOcrProvider) -> None:
        self._provider = provider

    def recognize(
        self,
        sources: tuple[ImageSource, ...],
        languages: tuple[str, ...] | None = None,
    ) -> tuple[ImageOcrResult, ...]:
        """Return full text and structured words for every source in order."""
        effective_languages = validate_ocr_languages(
            DEFAULT_OCR_LANGUAGES if languages is None else languages
        )
        results: list[ImageOcrResult] = []
        for source in sources:
            check_image_deadline()
            loaded = load_image(source)
            results.append(self._recognize_image(loaded.image, loaded.sha256, effective_languages))
        return tuple(results)

    def recognize_upload(
        self,
        payload: bytes,
        *,
        media_type: str,
        max_pixels: int,
        languages: tuple[str, ...] = DEFAULT_OCR_LANGUAGES,
    ) -> ImageOcrResult:
        """Verify and decode a single upload without altering its source bytes."""
        effective = validate_ocr_languages(languages)
        if media_type not in OCR_MEDIA_TYPES:
            raise ImageFormatError("unsupported image media type")
        check_image_deadline()
        prepared = prepare_image(
            payload,
            target_bytes=max(1, len(payload)),
            max_input_bytes=max(1, len(payload)),
            max_image_pixels=max_pixels,
            allowed_formats=(OCR_MEDIA_TYPES[media_type],),
            single_frame=True,
        )
        account_decoded_image(prepared.width, prepared.height, len(prepared.image.getbands()))
        return self._recognize_image(prepared.image, prepared.original_sha256, effective)

    def _recognize_image(
        self, image: Image.Image, source_hash: str, languages: tuple[str, ...]
    ) -> ImageOcrResult:
        check_image_deadline()
        consume_image_model_calls()
        output = self._provider.recognize(image, languages)
        check_image_deadline()
        if not isinstance(output, OcrProviderOutput):
            raise OcrProcessingError("OCR provider returned an incompatible result")
        if len(output.words) > _MAX_WORDS:
            raise OcrProcessingError("OCR provider returned too many words")
        _validate_word_bounds(output.words, image.width, image.height)
        return ImageOcrResult(
            source_hash=source_hash,
            full_text=_build_full_text(output.words),
            words=output.words,
            provider_id=self._provider.provider_id,
            provider_version=self._provider.provider_version,
            languages=languages,
            parameters=(("languages", "+".join(languages)),),
        )


def _validate_word_bounds(words: tuple[OcrWord, ...], width: int, height: int) -> None:
    for word in words:
        if word.x + word.width > width or word.y + word.height > height:
            raise OcrProcessingError("OCR word bounds exceed the decoded image")


def _build_full_text(words: tuple[OcrWord, ...]) -> str:
    lines: list[str] = []
    current_key: tuple[int, int, int, int] | None = None
    current_words: list[str] = []
    for word in words:
        line_key = word.page, word.block, word.paragraph, word.line
        if current_key is not None and line_key != current_key:
            lines.append(" ".join(current_words))
            current_words = []
        current_key = line_key
        current_words.append(word.text)
    if current_words:
        lines.append(" ".join(current_words))
    return "\n".join(lines)
