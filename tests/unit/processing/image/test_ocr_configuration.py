"""Default OCR configuration stays optional and never downloads assets."""

from pathlib import Path

import pytest
from PIL import Image

from capicove.processing.image.ocr_configuration import ImageOcrConfiguration


def test_paddle_is_default_and_missing_root_is_explicit() -> None:
    config = ImageOcrConfiguration()
    assert config.provider == "paddleocr"
    with pytest.raises(ValueError, match="model_root"):
        config.create_provider()


def test_disabled_ocr_requires_no_optional_import() -> None:
    assert ImageOcrConfiguration(provider="none").create_provider() is None


def test_provider_receives_exact_model_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    from types import SimpleNamespace

    class Provider:
        def __init__(self, root: Path, *, device: str, enable_mkldnn: bool) -> None:
            assert root == tmp_path
            assert device == "cpu"
            assert enable_mkldnn is False

        def recognize(self, image: Image.Image, languages: tuple[str, ...]) -> None:
            raise AssertionError("configuration must not run inference")

    monkeypatch.setitem(
        sys.modules,
        "capicove_processor_paddleocr",
        SimpleNamespace(PaddleImageOcrProvider=Provider),
    )
    assert isinstance(ImageOcrConfiguration(model_root=tmp_path).create_provider(), Provider)
