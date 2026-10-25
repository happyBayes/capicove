"""Lazy imagededup pHash adapter for visual training-set curation."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
from PIL import Image


class ImageDedupPhashProvider:
    """Compute pHash through imagededup without eager optional imports."""

    provider_id = "imagededup-phash"
    provider_version: str | None = None
    hash_version = "imagededup-phash-64-v1"

    def __init__(self) -> None:
        self._hasher: Any | None = None

    def phash(self, image: Image.Image) -> int:
        """Return imagededup's 64-bit perceptual hash as an integer."""
        if self._hasher is None:
            try:
                package = importlib.import_module("imagededup")
                methods = importlib.import_module("imagededup.methods")
            except ModuleNotFoundError as error:
                raise RuntimeError(
                    "imagededup is unavailable; install capicove[image-dedup]"
                ) from error
            self._hasher = methods.PHash()
            version = getattr(package, "__version__", None)
            if isinstance(version, str) and version.strip():
                self.provider_version = version
        encoded = self._hasher.encode_image(image_array=np.asarray(image.convert("RGB")))
        if not isinstance(encoded, str):
            raise TypeError("imagededup pHash must be a hexadecimal string")
        try:
            value = int(encoded, 16)
        except ValueError as error:
            raise ValueError("imagededup returned an invalid hexadecimal pHash") from error
        if value < 0 or value >= 2**64:
            raise ValueError("imagededup pHash must fit in 64 bits")
        return value
