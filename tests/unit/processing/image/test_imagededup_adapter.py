"""Tests for lazy imagededup pHash integration."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from capicove.processing.image._adapters.imagededup import ImageDedupPhashProvider


def test_imagededup_provider_imports_and_constructs_hasher_on_first_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class PHash:
        def __init__(self) -> None:
            events.append("constructed")

        def encode_image(self, image_array: np.ndarray) -> str:
            assert image_array.shape == (2, 2, 3)
            return "000000000000000f"

    def import_module(name: str) -> object:
        events.append(name)
        if name == "imagededup":
            return SimpleNamespace(__version__="0.3.2")
        return SimpleNamespace(PHash=PHash)

    monkeypatch.setattr(
        "capicove.processing.image._adapters.imagededup.importlib.import_module",
        import_module,
    )
    provider = ImageDedupPhashProvider()

    assert events == []
    assert provider.phash(Image.new("RGB", (2, 2))) == 15
    assert events == ["imagededup", "imagededup.methods", "constructed"]
    assert provider.provider_version == "0.3.2"
