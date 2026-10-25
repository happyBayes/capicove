"""Graph input composition preserves public document evidence."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from capicove.contracts import (
    DocumentPage,
    DocumentPageStatus,
    DocumentParseRequest,
    DocumentParseResult,
    LocalPathSource,
    ProcessingFingerprint,
)
from capicove.contracts.graph_build import GraphScope
from capicove.graph_build.document_preparation import (
    DocumentGraphConfiguration,
    parsed_document_representation,
)
from capicove.processing.image import ImageSource, OcrWord
from capicove.processing.image.ocr import OcrProviderOutput
from capicove.processing.image.ocr_configuration import ImageOcrConfiguration
from tests.unit.contracts.test_documents import _block, _receipt


def test_paddle_is_default() -> None:
    assert DocumentGraphConfiguration().parser_id == "paddleocr"


def test_exact_blocks_pages_and_parser_identity_are_preserved() -> None:
    result = DocumentParseResult(
        "d" * 64,
        "e" * 64,
        (DocumentPage(0, "1", 1200, 1600, DocumentPageStatus.SUCCEEDED, (_block(),)),),
        _receipt(),
    )
    rep = parsed_document_representation("document", result, GraphScope("workspace", "private"))
    assert rep.source_sha256 == result.source_sha256
    assert rep.text == "\n\n".join(
        block.exact_text for page in result.pages for block in page.blocks
    )
    assert rep.blocks[0].page_index == result.pages[0].page_index
    assert result.representation_sha256 in rep.issue_codes[0]
    changed = replace(result, representation_sha256="f" * 64)
    assert (
        parsed_document_representation("document", changed, rep.scope).representation_ref
        != rep.representation_ref
    )


def test_pdf_composition_rejects_a_parser_with_different_source_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = DocumentParseRequest(
        LocalPathSource(str(tmp_path / "source.pdf")),
        "a" * 64,
        "application/pdf",
        "book-balanced",
        1,
    )
    result = DocumentParseResult(
        "b" * 64,
        "e" * 64,
        (DocumentPage(0, "1", 100, 200, DocumentPageStatus.SUCCEEDED, (_block(),)),),
        _receipt(),
    )
    monkeypatch.setattr(
        "capicove.plugins.load_document_parser_plugin",
        lambda name: SimpleNamespace(
            create=lambda config: SimpleNamespace(parse=lambda request: result)
        ),
    )
    configuration = DocumentGraphConfiguration(model_root=tmp_path)
    with pytest.raises(ValueError, match="source identity"):
        configuration.prepare_pdf(
            request, document_ref="doc", scope=GraphScope("workspace", "private")
        )
    result = replace(result, source_sha256=request.source_sha256)
    parsed, rep = configuration.prepare_pdf(
        request, document_ref="doc", scope=GraphScope("workspace", "private")
    )
    assert parsed is result
    assert rep.source_sha256 == request.source_sha256


def test_image_composition_binds_original_bytes_and_model_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "source.png"
    Image.new("RGB", (100, 60)).save(path)
    provider = SimpleNamespace(
        provider_id="paddleocr",
        provider_version="authored",
        fingerprint=ProcessingFingerprint("a" * 64),
        recognize=lambda image, languages: OcrProviderOutput(
            (OcrWord("Exact text", 0, 0, 10, 10, 1.0, 1, 1, 1, 1, 1),)
        ),
    )
    monkeypatch.setattr(ImageOcrConfiguration, "create_provider", lambda self: provider)
    rep = DocumentGraphConfiguration(model_root=tmp_path).prepare_image(
        ImageSource(path=str(path)), document_ref="image", scope=GraphScope("workspace", "private")
    )
    assert rep.text == "Exact text"
    assert rep.issue_codes == ("image-ocr:" + "a" * 64,)
