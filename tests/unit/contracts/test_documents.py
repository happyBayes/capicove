from __future__ import annotations

import math
from typing import cast

import pytest

from capicove.contracts import (
    DOCUMENT_PARSER_CONTRACT_VERSION,
    DocumentBlock,
    DocumentBlockType,
    DocumentOutlineEntry,
    DocumentPage,
    DocumentPageRange,
    DocumentPageStatus,
    DocumentParseRequest,
    DocumentParseResult,
    DocumentParserReceipt,
    LocalPathSource,
    MediaSource,
    NormalizedBox,
)


def _block(*, block_id: str = "block:TITLE", page_index: int = 0) -> DocumentBlock:
    return DocumentBlock(
        block_id=block_id,
        page_index=page_index,
        block_type=DocumentBlockType.DOCUMENT_TITLE,
        exact_text=" Chapter 1 ",
        normalized_text="Chapter 1",
        box=NormalizedBox(0.1, 0.1, 0.9, 0.2),
        reading_order=0,
        confidence=0.95,
    )


def _receipt() -> DocumentParserReceipt:
    return DocumentParserReceipt(
        plugin_id="capicove.processor.paddleocr",
        plugin_version="1.0.0",
        profile_id="book-balanced",
        profile_version=1,
        model_manifest_sha256="a" * 64,
        configuration_sha256="b" * 64,
        processing_sha256="c" * 64,
    )


def test_document_parser_contract_version_and_closed_values() -> None:
    assert DOCUMENT_PARSER_CONTRACT_VERSION == 1
    assert tuple(item.value for item in DocumentPageStatus) == ("succeeded", "failed")
    assert {item.value for item in DocumentBlockType} == {
        "document_title",
        "paragraph_title",
        "text",
        "list",
        "table",
        "image",
        "formula",
        "header",
        "footer",
        "page_number",
        "other",
    }


@pytest.mark.parametrize("value", [-0.1, 1.1, math.inf, -math.inf, math.nan])
def test_normalized_box_rejects_non_finite_or_out_of_range_values(value: float) -> None:
    with pytest.raises(ValueError):
        NormalizedBox(value, 0.1, 0.9, 0.9)


def test_normalized_box_and_half_open_page_range_validate_order() -> None:
    assert NormalizedBox(0.1, 0.2, 0.8, 0.9).right == 0.8
    assert DocumentPageRange(0, 10).end_page_index == 10
    with pytest.raises(ValueError, match="right"):
        NormalizedBox(0.8, 0.2, 0.8, 0.9)
    with pytest.raises(ValueError, match="end_page_index"):
        DocumentPageRange(4, 4)


def test_document_page_requires_matching_unique_ordered_blocks() -> None:
    page = DocumentPage(
        page_index=0,
        printed_page_label="i",
        width=1200,
        height=1600,
        status=DocumentPageStatus.SUCCEEDED,
        blocks=(_block(),),
    )
    assert page.blocks[0].exact_text == " Chapter 1 "
    with pytest.raises(ValueError, match="page_index"):
        DocumentPage(
            0,
            None,
            1200,
            1600,
            DocumentPageStatus.SUCCEEDED,
            (_block(page_index=1),),
        )
    with pytest.raises(ValueError, match="block_id"):
        DocumentPage(
            0,
            None,
            1200,
            1600,
            DocumentPageStatus.SUCCEEDED,
            (_block(), _block()),
        )


def test_failed_page_requires_error_and_forbids_blocks() -> None:
    with pytest.raises(ValueError, match="error"):
        DocumentPage(0, None, 1200, 1600, DocumentPageStatus.FAILED, ())
    failed = DocumentPage(
        0,
        None,
        1200,
        1600,
        DocumentPageStatus.FAILED,
        (),
        "decode failed",
    )
    assert failed.error == "decode failed"


def test_parse_request_and_result_preserve_source_and_representation_identity() -> None:
    request = DocumentParseRequest(
        source=LocalPathSource("/authorized/book.pdf"),
        source_sha256="d" * 64,
        media_type="application/pdf",
        profile_id="book-balanced",
        profile_version=1,
        language_hints=("zh-Hans", "en"),
        page_range=DocumentPageRange(0, 1),
    )
    page = DocumentPage(
        0,
        "1",
        1200,
        1600,
        DocumentPageStatus.SUCCEEDED,
        (_block(),),
    )
    result = DocumentParseResult(
        source_sha256=request.source_sha256,
        representation_sha256="e" * 64,
        pages=(page,),
        receipt=_receipt(),
    )
    assert result.pages == (page,)
    with pytest.raises(ValueError, match="strictly increasing"):
        DocumentParseResult(
            request.source_sha256,
            "e" * 64,
            (page, page),
            _receipt(),
        )


@pytest.mark.parametrize("confidence", [-0.01, 1.01, math.nan])
def test_document_block_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        DocumentBlock(
            "block:TITLE",
            0,
            DocumentBlockType.DOCUMENT_TITLE,
            "Title",
            "Title",
            NormalizedBox(0.1, 0.1, 0.9, 0.2),
            0,
            confidence,
        )


def test_parse_request_rejects_non_pdf_and_duplicate_languages() -> None:
    source = LocalPathSource("/authorized/book.pdf")
    with pytest.raises(ValueError, match="application/pdf"):
        DocumentParseRequest(
            source,
            "d" * 64,
            "text/plain",
            "book-balanced",
            1,
        )
    with pytest.raises(ValueError, match="unique"):
        DocumentParseRequest(
            source,
            "d" * 64,
            "application/pdf",
            "book-balanced",
            1,
            ("en", "en"),
        )


def test_parse_request_rejects_unknown_source_shape() -> None:
    with pytest.raises(TypeError, match="source"):
        DocumentParseRequest(
            cast(MediaSource, object()),
            "d" * 64,
            "application/pdf",
            "book-balanced",
            1,
        )


@pytest.mark.parametrize("value", [True, False])
def test_document_indices_and_versions_reject_booleans(value: bool) -> None:
    with pytest.raises(TypeError, match="start_page_index"):
        DocumentPageRange(value, 1)
    with pytest.raises(TypeError, match="profile_version"):
        DocumentParserReceipt(
            "capicove.processor.paddleocr",
            "1.0.0",
            "book-balanced",
            value,
            "a" * 64,
            "b" * 64,
            "c" * 64,
        )


def test_parse_result_requires_pages_and_globally_unique_block_ids() -> None:
    with pytest.raises(ValueError, match="at least one"):
        DocumentParseResult("d" * 64, "e" * 64, (), _receipt())
    first = DocumentPage(
        0,
        None,
        1200,
        1600,
        DocumentPageStatus.SUCCEEDED,
        (_block(),),
    )
    second = DocumentPage(
        1,
        None,
        1200,
        1600,
        DocumentPageStatus.SUCCEEDED,
        (_block(page_index=1),),
    )
    with pytest.raises(ValueError, match="block_id"):
        DocumentParseResult("d" * 64, "e" * 64, (first, second), _receipt())


def test_document_contracts_reject_invalid_hash_and_enum_shapes() -> None:
    with pytest.raises(ValueError, match="source_sha256"):
        DocumentParseRequest(
            LocalPathSource("/authorized/book.pdf"),
            "invalid",
            "application/pdf",
            "book-balanced",
            1,
        )
    with pytest.raises(TypeError, match="block_type"):
        DocumentBlock(
            "block:TITLE",
            0,
            cast(DocumentBlockType, "text"),
            "Title",
            "Title",
            NormalizedBox(0.1, 0.1, 0.9, 0.2),
            0,
            0.9,
        )


def test_outline_entries_validate_level_page_and_children() -> None:
    child = DocumentOutlineEntry("1.1 Scope", 2, 1)
    root = DocumentOutlineEntry("Chapter 1", 1, 0, (child,))

    assert root.children == (child,)
    with pytest.raises(ValueError, match="level"):
        DocumentOutlineEntry("Broken", 0, 0)
    with pytest.raises(ValueError, match="deeper"):
        DocumentOutlineEntry("Chapter 1", 1, 0, (DocumentOutlineEntry("Peer", 1, 1),))


def test_parse_result_validates_outline_pages_and_bounded_warnings() -> None:
    page = DocumentPage(0, "1", 1200, 1600, DocumentPageStatus.SUCCEEDED, (_block(),))
    result = DocumentParseResult(
        "d" * 64,
        "e" * 64,
        (page,),
        _receipt(),
        (DocumentOutlineEntry("Chapter 1", 1, 0),),
        ("low resolution",),
    )

    assert result.outline[0].page_index == 0
    with pytest.raises(ValueError, match="outline"):
        DocumentParseResult(
            "d" * 64,
            "e" * 64,
            (page,),
            _receipt(),
            (DocumentOutlineEntry("Missing", 1, 2),),
        )
    with pytest.raises(ValueError, match="warnings"):
        DocumentParseResult(
            "d" * 64,
            "e" * 64,
            (page,),
            _receipt(),
            warnings=("x" * 501,),
        )
