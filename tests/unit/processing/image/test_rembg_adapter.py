"""Tests for lazy rembg background removal."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from capicove.processing.image._adapters.rembg import RembgBackgroundRemovalProvider
from capicove.processing.image.mappers import BackgroundRemovalOptions


def test_rembg_adapter_imports_provider_only_when_remove_is_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def remove(image: Image.Image, **kwargs: object) -> Image.Image:
        calls.append(kwargs)
        return image.convert("RGBA")

    monkeypatch.setattr(
        "capicove.processing.image._adapters.rembg.importlib.import_module",
        lambda name: SimpleNamespace(remove=remove, __version__="2.0") if name == "rembg" else None,
    )
    session = object()
    provider = RembgBackgroundRemovalProvider(
        session=session,
        model_id="/models/u2net.onnx",
        model_revision="sha256:test-model",
    )

    output = provider.remove(
        Image.new("RGB", (2, 2)),
        BackgroundRemovalOptions(alpha_matting=True, background_color=(1, 2, 3, 4)),
    )

    assert output.mode == "RGBA"
    assert calls == [
        {
            "alpha_matting": True,
            "alpha_matting_foreground_threshold": 240,
            "alpha_matting_background_threshold": 10,
            "alpha_matting_erode_size": 10,
            "bgcolor": (1, 2, 3, 4),
            "session": session,
        }
    ]
    assert provider.model_id == "/models/u2net.onnx"
    assert provider.model_revision == "sha256:test-model"


def test_rembg_adapter_requires_a_preprovisioned_session() -> None:
    """Prevent the adapter from falling back to rembg's downloading session."""
    with pytest.raises(ValueError, match="session"):
        RembgBackgroundRemovalProvider(
            session=None,
            model_id="/models/u2net.onnx",
        )
