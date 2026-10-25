from __future__ import annotations

import pytest
from capicove_processor_paddleocr.translation import translate_pages

from capicove.books import StructureConfidence, build_book_structure
from capicove.contracts import DocumentBlockType, DocumentParserReceipt


def test_ppstructure_pages_translate_to_normalized_ordered_evidence() -> None:
    raw_pages = (
        {
            "res": {
                "page_index": 0,
                "width": 1000,
                "height": 2000,
                "parsing_res_list": [
                    {
                        "block_bbox": [100, 100, 900, 220],
                        "block_label": "doc_title",
                        "block_content": "  Chapter 1  ",
                        "block_id": 7,
                        "block_order": 0,
                    },
                    {
                        "block_bbox": [100, 300, 900, 700],
                        "block_label": "text",
                        "block_content": "Overview\ntext",
                        "block_id": 8,
                        "block_order": 1,
                    },
                ],
                "layout_det_res": {
                    "boxes": [
                        {"label": "doc_title", "coordinate": [100, 100, 900, 220], "score": 0.97},
                        {"label": "text", "coordinate": [100, 300, 900, 700], "score": 0.91},
                    ]
                },
                "overall_ocr_res": {
                    "rec_texts": ["  Chapter 1  ", "Overview\ntext"],
                    "rec_boxes": [[100, 100, 900, 220], [100, 300, 900, 700]],
                    "rec_scores": [0.99, 0.93],
                },
            }
        },
    )
    receipt = DocumentParserReceipt(
        "capicove.processor.paddleocr",
        "0.1.0.dev0",
        "book-balanced",
        1,
        "a" * 64,
        "b" * 64,
        "c" * 64,
    )

    result = translate_pages(raw_pages, source_sha256="d" * 64, receipt=receipt)

    assert result.pages[0].blocks[0].block_id == "p0000:b000007"
    assert result.pages[0].blocks[0].block_type is DocumentBlockType.DOCUMENT_TITLE
    assert result.pages[0].blocks[0].exact_text == "  Chapter 1  "
    assert result.pages[0].blocks[0].normalized_text == "Chapter 1"
    assert result.pages[0].blocks[0].box.right == 0.9
    assert result.pages[0].blocks[1].confidence == 0.91


def test_translation_preserves_printed_page_labels() -> None:
    raw_pages = (
        {
            "res": {
                "page_index": 4,
                "width": 100,
                "height": 200,
                "parsing_res_list": [
                    {
                        "block_id": 0,
                        "block_label": "text",
                        "block_content": "Preface",
                        "block_bbox": [10, 10, 90, 30],
                    }
                ],
            }
        },
    )
    receipt = DocumentParserReceipt(
        "capicove.processor.paddleocr",
        "0.1.0.dev0",
        "book-balanced",
        1,
        "a" * 64,
        "b" * 64,
        "c" * 64,
    )

    result = translate_pages(
        raw_pages,
        source_sha256="d" * 64,
        receipt=receipt,
        page_labels={4: "iv"},
    )

    assert result.pages[0].printed_page_label == "iv"

    without_label = translate_pages(
        raw_pages,
        source_sha256="d" * 64,
        receipt=receipt,
    )
    assert without_label.representation_sha256 != result.representation_sha256


@pytest.mark.parametrize("orders", [(None, None, 1, 2), (4, 1, 1, None)])
def test_native_optional_ranks_do_not_collide_with_canonical_positions(
    orders: tuple[int | None, ...],
) -> None:
    raw = {
        "page_index": 4,
        "width": 100,
        "height": 200,
        "parsing_res_list": [
            {
                "block_id": index + 10,
                "block_order": order,
                "block_label": "text",
                "block_content": f"Line {index}\nNext line",
                "block_bbox": [10, 10 + index * 20, 90, 25 + index * 20],
            }
            for index, order in enumerate(orders)
        ],
    }
    receipt = DocumentParserReceipt("parser", "1", "profile", 1, "a" * 64, "b" * 64, "c" * 64)

    result = translate_pages((raw,), source_sha256="d" * 64, receipt=receipt)

    assert [block.reading_order for block in result.pages[0].blocks] == [0, 1, 2, 3]
    assert [block.block_id for block in result.pages[0].blocks] == [
        "p0004:b000010",
        "p0004:b000011",
        "p0004:b000012",
        "p0004:b000013",
    ]
    assert result.pages[0].blocks[0].exact_text == "Line 0\nNext line"
    assert result.pages[0].blocks[0].normalized_text == "Line 0\nNext line"


def test_layout_confidence_is_matched_by_box_not_list_position() -> None:
    raw = {
        "page_index": 0,
        "width": 100,
        "height": 200,
        "parsing_res_list": [
            {
                "block_id": 0,
                "block_order": 1,
                "block_label": "paragraph_title",
                "block_content": "Chapter 1",
                "block_bbox": [10, 10, 90, 30],
            }
        ],
        "layout_det_res": {
            "boxes": [
                {"label": "text", "coordinate": [10, 50, 90, 90], "score": 0.99},
                {"label": "paragraph_title", "coordinate": [10, 10, 90, 30], "score": 0.2},
            ]
        },
        "overall_ocr_res": {
            "rec_texts": ["Chapter 1"],
            "rec_boxes": [[10, 10, 90, 30]],
            "rec_scores": [0.9],
        },
    }
    receipt = DocumentParserReceipt("parser", "1", "profile", 1, "a" * 64, "b" * 64, "c" * 64)
    result = translate_pages((raw,), source_sha256="d" * 64, receipt=receipt)
    assert result.pages[0].blocks[0].confidence == 0.2


def test_empty_table_uses_aligned_overall_ocr_lines_without_duplicating_furniture() -> None:
    raw = {
        "page_index": 27,
        "width": 100,
        "height": 200,
        "parsing_res_list": [
            {
                "block_id": 4,
                "block_label": "table",
                "block_content": "",
                "block_bbox": [0, 0, 100, 180],
            },
            {
                "block_id": 5,
                "block_label": "number",
                "block_content": "-28-",
                "block_bbox": [40, 180, 60, 200],
            },
        ],
        "layout_det_res": {
            "boxes": [
                {"label": "table", "coordinate": [0, 0, 100, 180], "score": 0.95},
                {"label": "number", "coordinate": [40, 180, 60, 200], "score": 0.99},
            ]
        },
        "overall_ocr_res": {
            "rec_texts": ["Fully Self Training", "技術願景", "-28-"],
            "rec_boxes": [[10, 10, 90, 30], [10, 50, 90, 70], [45, 185, 55, 195]],
            "rec_scores": [0.91, 0.83, 0.98],
        },
    }
    receipt = DocumentParserReceipt("parser", "1", "profile", 1, "a" * 64, "b" * 64, "c" * 64)

    result = translate_pages((raw,), source_sha256="d" * 64, receipt=receipt)

    table, number = result.pages[0].blocks
    assert table.block_id == "p0027:b000004"
    assert table.block_type is DocumentBlockType.TABLE
    assert table.exact_text == "Fully Self Training\n技術願景"
    assert table.reading_order == 0
    assert table.confidence == 0.83
    assert number.exact_text == "-28-"
    assert "-28-" not in table.exact_text


@pytest.mark.parametrize(
    ("layout_score", "ocr_score", "expected"),
    ((0.97, 0.21, 0.21), (0.20, 0.99, 0.20)),
)
def test_text_confidence_is_bounded_by_ocr_and_layout_confidence(
    layout_score: float,
    ocr_score: float,
    expected: float,
) -> None:
    raw = {
        "page_index": 0,
        "width": 100,
        "height": 200,
        "parsing_res_list": [
            {
                "block_id": 0,
                "block_label": "text",
                "block_content": "recognized text",
                "block_bbox": [10, 10, 90, 30],
            }
        ],
        "layout_det_res": {
            "boxes": [
                {
                    "label": "text",
                    "coordinate": [10, 10, 90, 30],
                    "score": layout_score,
                }
            ]
        },
        "overall_ocr_res": {
            "rec_texts": ["recognized text"],
            "rec_boxes": [[10, 10, 90, 30]],
            "rec_scores": [ocr_score],
        },
    }
    receipt = DocumentParserReceipt("parser", "1", "profile", 1, "a" * 64, "b" * 64, "c" * 64)

    result = translate_pages((raw,), source_sha256="d" * 64, receipt=receipt)

    assert result.pages[0].blocks[0].confidence == expected


@pytest.mark.parametrize(
    ("exact_text", "ocr_text"),
    (
        ("Chapter 999 invented", "Completely different source text"),
        ("Chapter 999 invented", "Chapter 999"),
        ("1.2 Scope", "12 Scope"),
        ("1 1 Scope", "11 Scope"),
        ("Chapter 1 0", "Chapter 10"),
        ("Chapter 1", "Chapter 10"),
        ("1 Scope", "11 Scope"),
    ),
)
def test_heading_confidence_requires_complete_matching_ocr_evidence(
    exact_text: str, ocr_text: str
) -> None:
    raw = {
        "page_index": 0,
        "width": 100,
        "height": 200,
        "parsing_res_list": [
            {
                "block_id": 0,
                "block_label": "paragraph_title",
                "block_content": exact_text,
                "block_bbox": [10, 10, 90, 30],
            }
        ],
        "layout_det_res": {
            "boxes": [
                {
                    "label": "paragraph_title",
                    "coordinate": [10, 10, 90, 30],
                    "score": 0.99,
                }
            ]
        },
        "overall_ocr_res": {
            "rec_texts": [ocr_text],
            "rec_boxes": [[10, 10, 90, 30]],
            "rec_scores": [0.99],
        },
    }
    receipt = DocumentParserReceipt("parser", "1", "profile", 1, "a" * 64, "b" * 64, "c" * 64)

    result = translate_pages((raw,), source_sha256="d" * 64, receipt=receipt)
    structure = build_book_structure(result, profile_version=2)

    assert result.pages[0].blocks[0].confidence == 0.0
    assert structure.sections[0].confidence is StructureConfidence.LOW


@pytest.mark.parametrize("score", (float("nan"), float("inf"), float("-inf")))
def test_nonfinite_ocr_confidence_is_rejected(score: float) -> None:
    raw = {
        "page_index": 0,
        "width": 100,
        "height": 200,
        "parsing_res_list": [],
        "overall_ocr_res": {
            "rec_texts": ["text"],
            "rec_boxes": [[10, 10, 90, 30]],
            "rec_scores": [score],
        },
    }
    receipt = DocumentParserReceipt("parser", "1", "profile", 1, "a" * 64, "b" * 64, "c" * 64)

    with pytest.raises(ValueError, match="finite"):
        translate_pages((raw,), source_sha256="d" * 64, receipt=receipt)


def test_unrecoverable_empty_page_and_table_are_rejected() -> None:
    receipt = DocumentParserReceipt("parser", "1", "profile", 1, "a" * 64, "b" * 64, "c" * 64)
    blank = {"page_index": 0, "width": 100, "height": 200, "parsing_res_list": []}
    broken = {
        "page_index": 0,
        "width": 100,
        "height": 200,
        "parsing_res_list": [
            {
                "block_id": 0,
                "block_label": "table",
                "block_content": "",
                "block_bbox": [10, 10, 90, 190],
            }
        ],
    }

    with pytest.raises(ValueError, match="no usable evidence"):
        translate_pages((blank,), source_sha256="d" * 64, receipt=receipt)
    with pytest.raises(ValueError, match="no usable evidence"):
        translate_pages((broken,), source_sha256="d" * 64, receipt=receipt)


def test_nonblank_ocr_line_outside_layout_blocks_is_rejected() -> None:
    receipt = DocumentParserReceipt("parser", "1", "profile", 1, "a" * 64, "b" * 64, "c" * 64)
    raw = {
        "page_index": 0,
        "width": 100,
        "height": 200,
        "parsing_res_list": [
            {
                "block_id": 0,
                "block_label": "text",
                "block_content": "covered",
                "block_bbox": [10, 10, 90, 30],
            }
        ],
        "overall_ocr_res": {
            "rec_texts": ["covered", "lost evidence"],
            "rec_boxes": [[10, 10, 90, 30], [10, 100, 90, 120]],
            "rec_scores": [0.9, 0.9],
        },
    }

    with pytest.raises(ValueError, match="not covered"):
        translate_pages((raw,), source_sha256="d" * 64, receipt=receipt)


def test_misaligned_overall_ocr_arrays_are_rejected() -> None:
    raw = {
        "page_index": 0,
        "width": 100,
        "height": 200,
        "parsing_res_list": [],
        "overall_ocr_res": {
            "rec_texts": ["first", "second"],
            "rec_boxes": [[10, 10, 90, 30]],
            "rec_scores": [0.9, 0.8],
        },
    }
    receipt = DocumentParserReceipt("parser", "1", "profile", 1, "a" * 64, "b" * 64, "c" * 64)

    with pytest.raises(ValueError, match="aligned"):
        translate_pages((raw,), source_sha256="d" * 64, receipt=receipt)
