"""Measure recognized Paddle line-box unions through the public image port."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from capicove.processing.video.filtering._adapters.easyocr_text_area import EasyOcrTextAreaEvaluator
from capicove.processing.video.filtering.video_filter_contracts import OcrAreaCriterion
from capicove.processing.video.filtering.video_filter_ports import BackendUnavailableFilterError


if TYPE_CHECKING:
    from capicove.model import ModelFactory
    from capicove.processing.image.ocr_configuration import ImageOcrConfiguration


class PaddleOcrTextAreaEvaluator:
    """Optional CPU OCR-area evaluator over an explicitly provisioned root."""

    def __init__(
        self,
        configuration: ImageOcrConfiguration | None = None,
        *,
        model_factory: ModelFactory | None = None,
        binding: str = "video.ocr",
    ) -> None:
        self._configuration = configuration
        self._model_factory = model_factory
        self._binding = binding
        self._provider: Any = None

    def evaluate(
        self, item: Any, source: Any, metadata: Any, frames: Any, criterion: Any, deadline: datetime
    ) -> Any:
        """Count each recognized line's axis-aligned box at most once."""
        if not isinstance(criterion, OcrAreaCriterion):
            raise ValueError("Paddle OCR area requires an OCR-area criterion")
        if self._model_factory is not None:
            try:
                with self._model_factory.image_ocr(self._binding) as provider:
                    return self._evaluate(
                        provider, item, source, metadata, frames, criterion, deadline
                    )
            except (ValueError, RuntimeError, ImportError) as error:
                raise BackendUnavailableFilterError("configure local Paddle OCR assets") from error
        if self._provider is None:
            try:
                from capicove.processing.image.ocr_configuration import ImageOcrConfiguration

                configuration = self._configuration or ImageOcrConfiguration()
                self._provider = configuration.create_provider()
                if self._provider is None:
                    raise ValueError("Paddle OCR is disabled")
            except (ValueError, RuntimeError) as error:
                raise BackendUnavailableFilterError("configure local Paddle OCR assets") from error

        return self._evaluate(self._provider, item, source, metadata, frames, criterion, deadline)

    def _evaluate(
        self,
        provider: Any,
        item: Any,
        source: Any,
        metadata: Any,
        frames: Any,
        criterion: Any,
        deadline: datetime,
    ) -> Any:
        def detect(payload: Any) -> Any:
            from PIL import Image

            # The filtering sampler emits OpenCV-compatible BGR arrays.
            image = (
                payload
                if isinstance(payload, Image.Image)
                else Image.fromarray(payload[:, :, ::-1])
            )
            output = provider.recognize(image, criterion.languages)
            return tuple((w.x, w.x + w.width, w.y, w.y + w.height) for w in output.words), ()

        return EasyOcrTextAreaEvaluator(
            detector=detect, backend_label="paddleocr-line-boxes"
        ).evaluate(item, source, metadata, frames, criterion, deadline)
