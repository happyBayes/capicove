from __future__ import annotations

import pytest

from capicove.contracts import (
    DocumentBlock,
    DocumentBlockType,
    DocumentOutlineEntry,
    DocumentPage,
    DocumentPageStatus,
    DocumentParseResult,
    DocumentParserReceipt,
    NormalizedBox,
)
from capicove.documents import (
    document_parse_result_from_json,
    document_parse_result_to_json,
)


def _result() -> DocumentParseResult:
    block = DocumentBlock(
        "block:chapter-1",
        0,
        DocumentBlockType.DOCUMENT_TITLE,
        " Chapter 1 ",
        "Chapter 1",
        NormalizedBox(0.1, 0.1, 0.9, 0.2),
        0,
        0.98,
    )
    page = DocumentPage(0, "1", 1200, 1600, DocumentPageStatus.SUCCEEDED, (block,))
    receipt = DocumentParserReceipt(
        "capicove.processor.paddleocr",
        "0.1.0.dev0",
        "book-balanced",
        1,
        "a" * 64,
        "b" * 64,
        "c" * 64,
    )
    return DocumentParseResult(
        "d" * 64,
        "e" * 64,
        (page,),
        receipt,
        (DocumentOutlineEntry("Chapter 1", 1, 0),),
        ("synthetic fixture",),
    )


def test_document_result_json_codec_round_trips_exact_values() -> None:
    result = _result()

    encoded = document_parse_result_to_json(result)

    assert encoded["source_sha256"] == "d" * 64
    pages = encoded["pages"]
    assert isinstance(pages, list)
    page = pages[0]
    assert isinstance(page, dict)
    blocks = page["blocks"]
    assert isinstance(blocks, list)
    block = blocks[0]
    assert isinstance(block, dict)
    assert block["exact_text"] == " Chapter 1 "
    assert document_parse_result_from_json(encoded) == result


def test_document_result_json_codec_rejects_unknown_fields() -> None:
    encoded = document_parse_result_to_json(_result())
    encoded["unexpected"] = True

    with pytest.raises(ValueError, match="unknown"):
        document_parse_result_from_json(encoded)
