"""Literal Unicode offsets and independent representation checks."""

import hashlib
import json
from dataclasses import replace

import pytest

from capicove.wiki.maintenance.evidence import markdown_span, quote_interval, resolve_markdown
from capicove.wiki.maintenance.representations import markdown_representation


def test_representation_binds_source_transform_and_non_nfc_text() -> None:
    raw = "\ufeff甲😀e\u0301\r\n乙\r".encode()
    value = markdown_representation(raw)
    assert value.text == "甲😀e\u0301\n乙\n"
    expected = {
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "transform": "wiki-markdown-v1",
        "text": "甲😀e\u0301\n乙\n",
    }
    encoded = json.dumps(
        expected, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    assert value.encoded == encoded
    assert value.representation_sha256 == hashlib.sha256(encoded).hexdigest()
    span = markdown_span(value, 1, 4)
    assert span.exact_sha256 == hashlib.sha256("😀e\u0301".encode()).hexdigest()
    assert resolve_markdown(span, raw) == "😀e\u0301"
    assert len("😀e\u0301".encode()) != 3


@pytest.mark.parametrize("start,end", [(0, 0), (-1, 2), (0, 99), (0, 4097), (True, 2)])
def test_invalid_intervals(start: int, end: int) -> None:
    value = markdown_representation(b"evidence")
    with pytest.raises((ValueError, TypeError)):
        markdown_span(value, start, end)


def test_wrong_source_representation_and_slice_are_rejected() -> None:
    raw = b"Evidence"
    span = markdown_span(markdown_representation(raw), 0, 3)
    for invalid in (
        replace(span, exact_sha256="0" * 64),
        replace(span, representation_sha256="0" * 64),
    ):
        with pytest.raises(ValueError):
            resolve_markdown(invalid, raw)
    with pytest.raises(ValueError):
        resolve_markdown(span, b"Different")


def test_quote_resolution_rejects_ambiguous_missing_and_overlapping() -> None:
    assert quote_interval("甲😀乙", "😀") == (1, 2)
    for text, quote in (("x x", "x"), ("aaa", "aa"), ("a", "b"), ("a", "")):
        with pytest.raises(ValueError):
            quote_interval(text, quote)


def test_pdf_resolves_only_exact_verified_block_snapshot() -> None:
    from capicove.contracts import (
        DocumentBlock,
        DocumentBlockType,
        DocumentPage,
        DocumentPageStatus,
        DocumentParseResult,
        DocumentParserReceipt,
        NormalizedBox,
    )
    from capicove.contracts.wiki_maintenance import EvidenceSpan
    from capicove.documents import document_parse_result_to_json
    from capicove.wiki.maintenance.evidence import resolve_pdf

    raw = b"%PDF-synthetic"
    source = hashlib.sha256(raw).hexdigest()
    block = DocumentBlock(
        "b1", 2, DocumentBlockType.TEXT, "甲😀e\u0301", "different", NormalizedBox(0, 0, 1, 1), 0, 1
    )
    page = DocumentPage(2, "iii", 100, 100, DocumentPageStatus.SUCCEEDED, (block,))
    receipt = DocumentParserReceipt("fixture", "1", "fixture", 1, "a" * 64, "b" * 64, "c" * 64)
    result = DocumentParseResult(source, "d" * 64, (page,), receipt)
    encoded = json.dumps(
        document_parse_result_to_json(result),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    commitment = hashlib.sha256(encoded).hexdigest()
    span = EvidenceSpan(
        kind="pdf_block_span",
        source_id="source:sha256:" + source,
        source_sha256=source,
        representation_sha256="d" * 64,
        start_char=1,
        end_char=4,
        exact_sha256=hashlib.sha256("😀e\u0301".encode()).hexdigest(),
        page_index=2,
        block_id="b1",
    )
    assert resolve_pdf(span, result, raw, snapshot_sha256=commitment) == "😀e\u0301"
    for invalid in (
        replace(span, page_index=0),
        replace(span, block_id="missing"),
        replace(span, end_char=5),
        replace(span, exact_sha256="0" * 64),
    ):
        with pytest.raises(ValueError):
            resolve_pdf(invalid, result, raw, snapshot_sha256=commitment)
    with pytest.raises(ValueError):
        resolve_pdf(span, result, raw, snapshot_sha256="0" * 64)
    with pytest.raises(ValueError):
        resolve_pdf(span, result, b"different", snapshot_sha256=commitment)
