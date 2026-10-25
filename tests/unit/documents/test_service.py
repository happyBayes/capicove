"""Capability-level PDF policy and typed failures without HTTP dependencies."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import pytest
from pypdf import PdfWriter

from capicove.contracts import DocumentParseRequest, DocumentParseResult
from capicove.documents import (
    DocumentInputError,
    DocumentLimits,
    DocumentParserBinding,
    DocumentParsingService,
    DocumentProcessingError,
    DocumentRejectedError,
)


class Backend:
    parser_id = "test"

    def __init__(self, failure: Exception) -> None:
        self.failure = failure

    def parse(self, request: DocumentParseRequest, /) -> DocumentParseResult:
        raise self.failure


def test_missing_staged_file_is_a_server_failure(tmp_path: Path) -> None:
    binding = DocumentParserBinding(Backend(AssertionError("must not parse")), "test", 1)
    service = DocumentParsingService(binding, DocumentLimits())
    with pytest.raises(DocumentProcessingError) as caught:
        service.parse(tmp_path / "missing.pdf", "a" * 64)
    assert isinstance(caught.value.__cause__, OSError)


def test_parser_rejections_are_explicit_and_unknown_value_errors_are_not_input_errors(
    tmp_path: Path,
) -> None:
    output = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.write(output)
    source = tmp_path / "source.pdf"
    source.write_bytes(output.getvalue())
    digest = hashlib.sha256(output.getvalue()).hexdigest()
    backend = Backend(DocumentRejectedError("unsupported document"))
    service = DocumentParsingService(DocumentParserBinding(backend, "test", 1), DocumentLimits())
    with pytest.raises(DocumentRejectedError):
        service.parse(source, digest)
    backend.failure = ValueError("invalid installed asset")
    with pytest.raises(DocumentProcessingError) as caught:
        service.parse(source, digest)
    assert isinstance(caught.value.__cause__, ValueError)
    assert not isinstance(caught.value, DocumentInputError)


def test_core_document_import_does_not_load_optional_pdf_or_http_runtimes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import capicove.documents; "
            "assert not {'fastapi','pypdf','paddleocr','paddle'} & sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
