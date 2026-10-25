"""Business operations consume leased public ports rather than escaped models."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from capicove.cli import main
from capicove.contracts import ProcessingFingerprint
from capicove.contracts.graph_build import GraphScope
from capicove.graph_build.document_preparation import DocumentGraphConfiguration
from capicove.model import (
    ModelCapability,
    ModelConfiguration,
    ModelFactory,
    ModelProfile,
    ProviderDefinition,
    ProviderRegistry,
)
from capicove.processing.image import ImageSource, OcrWord
from capicove.processing.image.ocr import OcrProviderOutput
from capicove.processing.video.filtering import OcrAreaCriterion, SampledFrame
from capicove.processing.video.filtering._adapters.paddleocr_text_area import (
    PaddleOcrTextAreaEvaluator,
)
from tests.unit.cli.test_book import _Parser
from tests.unit.processing.video.filtering.test_easyocr_text_area import (
    _deadline,
    _item,
    _metadata,
    _source,
)


def ocr_factory(root: Path) -> ModelFactory:
    provider = SimpleNamespace(
        fingerprint=ProcessingFingerprint("a" * 64),
        provider_id="paddleocr",
        provider_version="fake",
        recognize=lambda image, languages: OcrProviderOutput(
            (OcrWord("Exact text", 0, 0, 5, 5, 1.0, 1, 1, 1, 1, 1),)
        ),
    )
    registry = ProviderRegistry()
    registry.register(
        ProviderDefinition("paddleocr", "1", ModelCapability.IMAGE_OCR, lambda p: provider)
    )
    profile = ModelProfile("ocr", "paddleocr", ModelCapability.IMAGE_OCR, root, "a" * 64)
    return ModelFactory(
        ModelConfiguration({"ocr": profile}, {"graph.ocr": "ocr", "video.ocr": "ocr"}),
        registry=registry,
    )


def test_graph_and_video_share_a_configured_ocr_port(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    Image.new("RGB", (10, 10)).save(path)
    with ocr_factory(tmp_path) as factory:
        graph = DocumentGraphConfiguration(model_factory=factory)
        rep = graph.prepare_image(
            ImageSource(path=str(path)),
            document_ref="image",
            scope=GraphScope("workspace", "private"),
        )
        assert rep.text == "Exact text"
        assert rep.issue_codes == ("image-ocr:" + "a" * 64,)
        result = PaddleOcrTextAreaEvaluator(model_factory=factory).evaluate(
            _item(),
            _source(),
            _metadata(),
            (SampledFrame(0, 0, 10, 10, np.zeros((10, 10, 3), dtype=np.uint8)),),
            OcrAreaCriterion("ocr"),
            _deadline(),
        )
        assert result.score == pytest.approx(0.25)


def test_book_cli_uses_configured_parser_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import capicove.model.providers

    registry = ProviderRegistry()
    registry.register(
        ProviderDefinition(
            "paddle-document", "1", ModelCapability.DOCUMENT_PARSER, lambda p: _Parser()
        )
    )
    monkeypatch.setattr(capicove.model.providers, "builtin_registry", lambda: registry)
    config = tmp_path / "models.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profiles": {
                    "parser": {
                        "provider": "paddle-document",
                        "capability": "document_parser",
                        "root": str(tmp_path),
                        "manifest_sha256": "a" * 64,
                    }
                },
                "bindings": {"book.parser": "parser"},
            }
        )
    )
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-1.7\nbook")
    assert (
        main(
            [
                "book",
                "prepare",
                str(tmp_path / "books"),
                str(source),
                "--model-config",
                str(config),
                "--evidence-only",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["summary_tasks"] == 0
