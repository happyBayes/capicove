"""Paddle image OCR uses only verified local models and preserves geometry."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from capicove_processor_paddleocr import PaddleImageOcrProvider
from PIL import Image

from capicove.processing.image.ocr import OcrProviderUnavailableError


def test_local_configuration_and_line_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import capicove_processor_paddleocr.image_ocr as module

    configurations = []
    monkeypatch.setattr(module, "verify_installed_models", lambda manifest, root: None)
    monkeypatch.setattr(module, "resolve_runtime_identity", lambda requirements: ())
    monkeypatch.setattr(module, "verify_loaded_runtime_modules", lambda: None)
    monkeypatch.setattr(module, "enforce_offline_runtime_policy", lambda: None)

    def engine(**kwargs: object) -> SimpleNamespace:
        configurations.append(kwargs)
        return SimpleNamespace(
            predict=lambda pixels: iter(
                [
                    {
                        "rec_texts": ["English", "学习"],
                        "rec_scores": [0.98, 0.95],
                        "rec_boxes": np.asarray([[2, 3, 80, 20], [2, 30, 80, 48]]),
                    }
                ]
            )
        )

    monkeypatch.setattr(
        module, "import_verified_runtime_module", lambda name: SimpleNamespace(PaddleOCR=engine)
    )
    provider = PaddleImageOcrProvider(tmp_path)
    output = provider.recognize(Image.new("RGB", (100, 60)), ("chi_sim", "eng"))
    assert [word.text for word in output.words] == ["English", "学习"]
    assert [word.line for word in output.words] == [1, 2]
    assert output.words[0].width == 78
    assert configurations[0]["use_doc_orientation_classify"] is False
    assert configurations[0]["use_textline_orientation"] is False
    assert configurations[0]["enable_mkldnn"] is False
    assert configurations[0]["text_det_limit_type"] == "max"
    assert configurations[0]["text_recognition_model_dir"] == str(tmp_path / "pp-ocrv5-server-rec")


def test_missing_assets_fail_before_runtime_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import capicove_processor_paddleocr.image_ocr as module

    monkeypatch.setattr(
        module, "import_verified_runtime_module", lambda name: pytest.fail("runtime loaded")
    )
    with pytest.raises(OcrProviderUnavailableError) as caught:
        PaddleImageOcrProvider(tmp_path).recognize(Image.new("RGB", (100, 60)), ("eng",))
    assert isinstance(caught.value.__cause__, ValueError)
    assert "receipt" in str(caught.value.__cause__)
