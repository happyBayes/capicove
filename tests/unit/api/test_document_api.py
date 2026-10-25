"""PDF HTTP translation through public document parser contracts."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from capicove.contracts import DocumentParseRequest, DocumentParseResult, LocalPathSource
from capicove.documents import DocumentRejectedError, document_parse_result_from_json
from tests.unit.api.test_ocr_api import Provider, picture
from tests.unit.documents.test_codec import _result


def pdf(pages: int = 1, *, width: int = 612, encrypted: bool = False) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=width, height=792)
    if encrypted:
        writer.encrypt("private-test-password")
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


class Parser:
    parser_id = "test-parser"

    def __init__(self) -> None:
        self.requests: list[DocumentParseRequest] = []
        self.path: Path | None = None
        self.fail: Exception | None = None
        self.wrong_hash = False

    def parse(self, request: DocumentParseRequest, /) -> DocumentParseResult:
        self.requests.append(request)
        assert isinstance(request.source, LocalPathSource)
        self.path = Path(request.source.path)
        assert self.path.parent.stat().st_mode & 0o077 == 0
        data = self.path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == request.source_sha256
        if self.fail is not None:
            raise self.fail
        assert request.page_range is not None
        first = _result()
        pages = tuple(
            replace(first.pages[0], page_index=i, blocks=())
            for i in range(request.page_range.start_page_index, request.page_range.end_page_index)
        )
        return replace(
            first,
            source_sha256="a" * 64 if self.wrong_hash else request.source_sha256,
            pages=pages,
            outline=(),
            receipt=replace(
                first.receipt,
                profile_id=request.profile_id,
                profile_version=request.profile_version,
            ),
        )


def app(parser: Parser, **kwargs: Any) -> Any:
    from capicove.api.documents import DocumentParserBinding
    from capicove.api.ocr import create_app

    return create_app(
        lambda: Provider(),
        document_parser=DocumentParserBinding(parser, "test-profile", 3),
        **kwargs,
    )


def test_pdf_upload_preserves_canonical_evidence_and_cleans_temp_files(tmp_path: Path) -> None:
    parser = Parser()
    data = pdf(3)
    with TestClient(app(parser, document_temp_directory=tmp_path)) as client:
        response = client.post(
            "/v1/documents/parse?start_page_index=1&end_page_index=3",
            content=data,
            headers={"Content-Type": "application/pdf"},
        )
        assert response.status_code == 200, response.text
        result = document_parse_result_from_json(response.json()["result"])
        assert result.source_sha256 == hashlib.sha256(data).hexdigest()
        assert [page.page_index for page in result.pages] == [1, 2]
        assert result.receipt.profile_id == "test-profile"
        assert parser.path is not None and not parser.path.exists()
        assert list(tmp_path.iterdir()) == []
        assert client.get("/health").json()["document_parser"] == "test-parser"
        schema = client.get("/openapi.json").json()["paths"]["/v1/documents/parse"]["post"]
        assert "application/pdf" in schema["requestBody"]["content"]


@pytest.mark.parametrize(
    "query",
    [
        "start_page_index=-1",
        "end_page_index=0",
        "start_page_index=2&end_page_index=1",
        "end_page_index=2",
        "start_page_index=x",
        "path=/etc/passwd",
        "start_page_index=0&start_page_index=1",
    ],
)
def test_invalid_ranges_and_unrecognized_parameters_do_not_invoke_parser(query: str) -> None:
    parser = Parser()
    with TestClient(app(parser)) as client:
        response = client.post(
            "/v1/documents/parse?" + query,
            content=pdf(),
            headers={"Content-Type": "application/pdf"},
        )
        assert response.status_code == 400
        assert not parser.requests


@pytest.mark.parametrize(
    ("body", "media", "status"),
    [
        (b"", "application/pdf", 400),
        (b"%PDF-invalid", "application/pdf", 400),
        (b'{"path":"/etc/passwd"}', "application/json", 415),
        (pdf(encrypted=True), "application/pdf", 400),
    ],
)
def test_invalid_or_encrypted_uploads_are_rejected(body: bytes, media: str, status: int) -> None:
    parser = Parser()
    with TestClient(app(parser)) as client:
        assert (
            client.post(
                "/v1/documents/parse", content=body, headers={"Content-Type": media}
            ).status_code
            == status
        )
        assert not parser.requests


def test_page_byte_and_geometry_limits_precede_inference(tmp_path: Path) -> None:
    parser = Parser()
    with TestClient(
        app(parser, max_document_pages=1, max_document_bytes=2000, document_temp_directory=tmp_path)
    ) as client:
        headers = {"Content-Type": "application/pdf"}
        assert (
            client.post("/v1/documents/parse", content=pdf(2), headers=headers).status_code == 413
        )
        assert (
            client.post("/v1/documents/parse", content=pdf(width=3000), headers=headers).status_code
            == 413
        )
        assert (
            client.post(
                "/v1/documents/parse", content=iter([b"x" * 1001, b"x" * 1001]), headers=headers
            ).status_code
            == 413
        )
        assert not parser.requests
        assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("failure", "status"),
    [
        (DocumentRejectedError("private parser detail"), 422),
        (RuntimeError("private runtime detail"), 500),
    ],
)
def test_failure_responses_are_safe_and_upload_is_removed(
    failure: Exception, status: int, tmp_path: Path
) -> None:
    parser = Parser()
    parser.fail = failure
    with TestClient(app(parser, document_temp_directory=tmp_path)) as client:
        response = client.post(
            "/v1/documents/parse", content=pdf(), headers={"Content-Type": "application/pdf"}
        )
        assert response.status_code == status
        assert "private" not in response.text
        assert not list(tmp_path.iterdir())


def test_document_auth_and_unconfigured_route() -> None:
    from capicove.api.ocr import create_app

    # Synthetic credential confined to this authentication test.
    with TestClient(
        app(Parser(), api_key="test-only-key")  # pragma: allowlist secret
    ) as client:
        assert client.post("/v1/documents/parse", content=pdf()).status_code == 401
        response = client.post(
            "/v1/documents/parse",
            content=pdf(),
            headers={"Content-Type": "application/pdf", "Authorization": "Bearer test-only-key"},
        )
        assert response.status_code == 200
    with TestClient(create_app(Provider)) as client:
        assert client.post("/v1/documents/parse", content=pdf()).status_code == 404


def test_parser_cannot_return_evidence_for_another_source() -> None:
    parser = Parser()
    parser.wrong_hash = True
    with TestClient(app(parser)) as client:
        response = client.post(
            "/v1/documents/parse", content=pdf(), headers={"Content-Type": "application/pdf"}
        )
        assert response.status_code == 500


def test_image_and_document_inference_share_admission() -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    started, release = Event(), Event()

    class BlockingParser(Parser):
        def parse(self, request: DocumentParseRequest, /) -> DocumentParseResult:
            started.set()
            assert release.wait(10)
            return super().parse(request)

    with TestClient(app(BlockingParser())) as client, ThreadPoolExecutor() as pool:
        pending = pool.submit(
            client.post,
            "/v1/documents/parse",
            content=pdf(),
            headers={"Content-Type": "application/pdf"},
        )
        try:
            assert started.wait(5)
            assert client.get("/health").status_code == 200
            assert (
                client.post(
                    "/v1/ocr", content=picture(), headers={"Content-Type": "image/png"}
                ).status_code
                == 429
            )
        finally:
            release.set()
        assert pending.result().status_code == 200


def test_staging_failure_returns_safe_json_and_releases_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tempfile

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise OSError("private staging path")

    with TestClient(app(Parser())) as client:
        monkeypatch.setattr(tempfile, "TemporaryDirectory", fail)
        response = client.post(
            "/v1/documents/parse", content=pdf(), headers={"Content-Type": "application/pdf"}
        )
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "document_upload_failed"
        assert "private" not in response.text
        assert (
            client.post(
                "/v1/ocr", content=picture(), headers={"Content-Type": "image/png"}
            ).status_code
            == 200
        )


@pytest.mark.parametrize(
    ("width", "height", "unit", "status"),
    [
        (10000, 10000, 0.01, 413),
        (10000, 792, 0.01, 413),
        (612, 10000, 0.01, 413),
        (612, 792, 10, 413),
        (612, 792, 0.01, 200),
        (612, 792, 2, 200),
    ],
)
@pytest.mark.parametrize("box", ["/MediaBox", "/CropBox"])
def test_pdf_geometry_bounds_raw_and_user_unit_scaled_dimensions(
    width: int, height: int, unit: float, status: int, box: str, tmp_path: Path
) -> None:
    from pypdf.generic import FloatObject, NameObject, RectangleObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    page[NameObject(box)] = RectangleObject([0, 0, width, height])
    page[NameObject("/UserUnit")] = FloatObject(unit)
    output = BytesIO()
    writer.write(output)
    parser = Parser()
    with TestClient(app(parser, document_temp_directory=tmp_path)) as client:
        response = client.post(
            "/v1/documents/parse",
            content=output.getvalue(),
            headers={"Content-Type": "application/pdf"},
        )
        assert response.status_code == status
        if status == 413:
            assert response.json()["error"]["code"] == "page_too_large"
            assert not parser.requests
        else:
            assert len(parser.requests) == 1
        assert not list(tmp_path.iterdir())
