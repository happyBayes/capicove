"""Startup-selected local OCR composition without runtime model acquisition."""

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from capicove.contracts import ProcessingFingerprint
from capicove.processing.image.ports import ImageOcrProvider


class ConfiguredImageOcrProvider(ImageOcrProvider, Protocol):
    """An OCR provider with immutable startup processing identity."""

    @property
    def fingerprint(self) -> ProcessingFingerprint:
        """Identify the selected model bytes and inference policy."""
        ...


@dataclass(frozen=True, slots=True)
class ImageOcrConfiguration:
    """Select PaddleOCR by default; provision its root explicitly before use."""

    provider: str = "paddleocr"
    model_root: Path | None = None
    device: str = "cpu"
    enable_mkldnn: bool = False

    def __post_init__(self) -> None:
        if self.provider not in {"paddleocr", "none"}:
            raise ValueError("unsupported OCR provider")
        if self.device != "cpu":
            raise ValueError("OCR configuration currently requires CPU")
        if type(self.enable_mkldnn) is not bool:
            raise ValueError("enable_mkldnn must be a boolean")
        if self.model_root is not None and (
            not self.model_root.is_absolute() or self.model_root.is_symlink()
        ):
            raise ValueError("absolute unlinked OCR model_root is required")

    def create_provider(self) -> ConfiguredImageOcrProvider | None:
        """Compose the optional provider without loading an inference engine."""
        if self.provider == "none":
            return None
        if self.model_root is None:
            raise ValueError("Paddle OCR requires an explicit model_root")
        try:
            module = importlib.import_module("capicove_processor_paddleocr")
        except ImportError:
            raise RuntimeError("install capicove-processor-paddleocr to use Paddle OCR") from None
        return cast(
            ConfiguredImageOcrProvider,
            module.PaddleImageOcrProvider(
                self.model_root, device=self.device, enable_mkldnn=self.enable_mkldnn
            ),
        )
