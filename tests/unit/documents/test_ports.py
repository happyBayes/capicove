from __future__ import annotations

from capicove.contracts import DocumentParseRequest, DocumentParseResult
from capicove.documents import DocumentParserBackend


class _Backend:
    parser_id = "paddleocr"

    def parse(self, request: DocumentParseRequest, /) -> DocumentParseResult:
        raise NotImplementedError


class _MissingParserId:
    def parse(self, request: DocumentParseRequest, /) -> DocumentParseResult:
        raise NotImplementedError


def test_document_parser_backend_is_a_minimal_runtime_protocol() -> None:
    assert isinstance(_Backend(), DocumentParserBackend)
    assert not isinstance(_MissingParserId(), DocumentParserBackend)
