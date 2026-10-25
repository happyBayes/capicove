"""Optional image OCR adapter over the pinned Paddle book model inventory."""

from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import islice
from pathlib import Path
from threading import Lock
from typing import Any

from PIL import Image

from capicove.contracts import ProcessingFingerprint
from capicove.processing.image import OcrWord
from capicove.processing.image.ocr import (
    OcrProcessingError,
    OcrProviderOutput,
    OcrProviderUnavailableError,
)
from capicove_processor_paddleocr.models import load_book_balanced_manifest, verify_installed_models
from capicove_processor_paddleocr.runtime import (
    enforce_offline_runtime_policy,
    import_verified_runtime_module,
    resolve_runtime_identity,
    verify_loaded_runtime_modules,
)


class PaddleImageOcrProvider:
    """Recognize one image at a time without downloading additional models.

    Each recognized line is one OcrWord with line-level confidence and bounds.
    No word segmentation, orientation correction, or geometry is invented.
    """

    provider_id = "paddleocr"
    provider_version = "book-balanced-v2-image-v1"

    def __init__(
        self, model_root: Path, *, device: str = "cpu", enable_mkldnn: bool = False
    ) -> None:
        if not model_root.is_absolute() or model_root.is_symlink():
            raise ValueError("absolute unlinked Paddle model root is required")
        if device != "cpu":
            raise ValueError("Paddle image OCR currently requires explicit CPU execution")
        self._root = model_root
        if type(enable_mkldnn) is not bool:
            raise ValueError("enable_mkldnn must be a boolean")
        self._enable_mkldnn = enable_mkldnn
        self._pipeline: Any = None
        self._lock = Lock()

    def close(self) -> None:
        """Release the cached local pipeline under its inference lock."""
        with self._lock:
            self._pipeline = None

    @property
    def fingerprint(self) -> ProcessingFingerprint:
        """Bind the exact model profile and fixed inference policy."""
        manifest = load_book_balanced_manifest()
        return ProcessingFingerprint.from_manifest(
            {
                "adapter": self.provider_version,
                "models": manifest.source_sha256,
                "runtime": dict(resolve_runtime_identity(manifest.inference_engine)),
                "device": "cpu",
                "det_limit": 736,
                "det_limit_type": "max",
                "rec_batch": 1,
                "threads": 2,
                "enable_mkldnn": self._enable_mkldnn,
                "orientation": False,
                "geometry": "line-axis-aligned-v1",
            }
        )

    def recognize(self, image: Image.Image, languages: tuple[str, ...]) -> OcrProviderOutput:
        """Return exact lines and reject unavailable languages or invalid output."""
        if not languages or set(languages) - {"chi_sim", "ch_sim", "eng", "jpn", "zh", "en", "ja"}:
            raise ValueError("unsupported Paddle OCR languages")
        with self._lock:
            try:
                manifest = load_book_balanced_manifest()
                verify_installed_models(manifest, self._root)
                enforce_offline_runtime_policy()
                resolve_runtime_identity(manifest.inference_engine)
                if self._pipeline is None:
                    runtime = import_verified_runtime_module("paddleocr")
                    self._pipeline = runtime.PaddleOCR(
                        text_detection_model_name="PP-OCRv5_server_det",
                        text_detection_model_dir=str(self._root / "pp-ocrv5-server-det"),
                        text_recognition_model_name="PP-OCRv5_server_rec",
                        text_recognition_model_dir=str(self._root / "pp-ocrv5-server-rec"),
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                        device="cpu",
                        cpu_threads=2,
                        enable_mkldnn=self._enable_mkldnn,
                        text_det_limit_side_len=736,
                        text_det_limit_type="max",
                        text_recognition_batch_size=1,
                    )
                    verify_loaded_runtime_modules()
            except (ValueError, RuntimeError, ImportError, OSError) as error:
                raise OcrProviderUnavailableError(
                    "Paddle OCR assets or runtime are unavailable"
                ) from error
            import numpy as np

            # Paddle's ndarray input follows OpenCV's BGR channel ordering.
            pixels = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
            predictions = tuple(islice(self._pipeline.predict(pixels), 2))
            verify_loaded_runtime_modules()
            if len(predictions) != 1:
                raise OcrProcessingError("Paddle OCR must return exactly one image result")
            raw = predictions[0]
            if not isinstance(raw, Mapping):
                raw = raw.json
            if not isinstance(raw, Mapping):
                raise OcrProcessingError("Paddle OCR returned invalid image evidence")
            raw = raw.get("res", raw)
            return _words(raw, image.size)


def _words(raw: Any, size: tuple[int, int]) -> OcrProviderOutput:
    """Translate recognized lines without discarding malformed evidence."""
    try:
        texts, scores, boxes = raw["rec_texts"], raw["rec_scores"], raw["rec_boxes"]
        if len(texts) > 100_000 or not len(texts) == len(scores) == len(boxes):
            raise ValueError("invalid line count")
        words = []
        for index, (text, score, box) in enumerate(zip(texts, scores, boxes, strict=True), 1):
            if not isinstance(text, str) or not text.strip():
                continue
            coordinates = [float(value) for value in box]
            if len(coordinates) != 4 or not all(math.isfinite(value) for value in coordinates):
                raise ValueError("invalid geometry")
            left, top, right, bottom = coordinates
            x, y = max(0, math.floor(left)), max(0, math.floor(top))
            right, bottom = min(size[0], math.ceil(right)), min(size[1], math.ceil(bottom))
            words.append(
                OcrWord(text, x, y, right - x, bottom - y, float(score), 1, 1, 1, index, 1)
            )
        return OcrProviderOutput(tuple(words))
    except (KeyError, TypeError, ValueError, OverflowError):
        raise OcrProcessingError("Paddle OCR returned invalid line evidence") from None
