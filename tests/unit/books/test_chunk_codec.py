import json
from dataclasses import replace

import pytest

from capicove.books import build_book_structure
from capicove.books.chunk_codec import decode_run, encode_run
from capicove.books.chunk_models import SourceSpan
from capicove.books.chunking import build_chunk_run
from capicove.contracts import DocumentBlockType
from tests.unit.books.test_chunking import _config
from tests.unit.books.test_structure import _block, _result


def _encoded() -> bytes:
    result = _result({0: (_block(0, 0, "Definition: source", DocumentBlockType.TEXT),)})
    return encode_run(build_chunk_run(result, build_book_structure(result), _config()))


def test_closed_codec_roundtrip() -> None:
    raw = _encoded()
    assert encode_run(decode_run(raw)) == raw


@pytest.mark.parametrize(
    "field,value",
    [
        ("core_tokens", True),
        ("rendered_tokens", -1),
        ("chunk_id", "bad"),
        ("parent_id", ""),
        ("heading_path", "not an array"),
        ("content_types", ["invented"]),
        ("core_spans", []),
    ],
)
def test_reject_invalid_chunk_fields(field: str, value: object) -> None:
    raw = json.loads(_encoded())
    raw["run"]["chunks"][0][field] = value
    with pytest.raises(ValueError):
        decode_run(json.dumps(raw).encode())


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", "bad"),
        ("coverage_valid", "true"),
        ("source_sha256", "bad"),
        ("token_count_mode", "unknown"),
        ("tokenizer_fingerprint", ""),
        ("exclusions", [["id", "text", "extra"]]),
    ],
)
def test_reject_invalid_run_fields(field: str, value: object) -> None:
    raw = json.loads(_encoded())
    raw["run"][field] = value
    with pytest.raises(ValueError):
        decode_run(json.dumps(raw).encode())


@pytest.mark.parametrize(
    "raw",
    [
        b"invalid",
        b"[]",
        b'{"schema_version": 2, "run": {}}',
        b'{"schema_version": true, "run": {}}',
        b'{"schema_version": 1, "run": {"extra": 1}}',
    ],
)
def test_reject_invalid_envelopes(raw: bytes) -> None:
    with pytest.raises(ValueError):
        decode_run(raw)


def test_span_guards_enforce_source_location_and_hash() -> None:
    block = _block(0, 0, "source", DocumentBlockType.TEXT)
    result = _result({0: (block,)})
    span = SourceSpan.from_block(block)
    for changes in ({"block_id": ""}, {"start_char": -1}, {"end_char": 0}, {"text_sha256": "bad"}):
        with pytest.raises(ValueError):
            replace(span, **changes)
    with pytest.raises(ValueError, match="bounds"):
        SourceSpan.from_block(block, 0, 100)
    with pytest.raises(ValueError, match="resolve"):
        replace(span, page_index=1).extract(block)
    with pytest.raises(ValueError, match="unavailable"):
        replace(span, block_id="absent").text(result)
