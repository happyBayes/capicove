"""Resource ownership, failure classification and cancellation contracts."""

from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Event
from typing import Any, cast

import anyio
import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from capicove.contracts import ProcessingFingerprint
from capicove.processing.image.ocr import OcrProviderOutput
from tests.unit.api.test_document_api import Parser, app, pdf
from tests.unit.api.test_ocr_api import Provider, picture


def test_unknown_parser_value_error_is_a_server_failure() -> None:
    parser = Parser()
    parser.fail = ValueError("private model receipt path")
    with TestClient(app(parser)) as client:
        response = client.post(
            "/v1/documents/parse", content=pdf(), headers={"Content-Type": "application/pdf"}
        )
        assert response.status_code == 500
        assert "private" not in response.text
        assert response.headers["x-request-id"]


def test_missing_model_assets_are_not_reported_as_document_rejection() -> None:
    from capicove.documents import DocumentBackendUnavailableError

    parser = Parser()
    parser.fail = DocumentBackendUnavailableError("private receipt path")
    with TestClient(app(parser)) as client:
        response = client.post(
            "/v1/documents/parse", content=pdf(), headers={"Content-Type": "application/pdf"}
        )
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "backend_unavailable"
        assert "private" not in response.text


def test_document_success_and_errors_have_structured_openapi_schemas() -> None:
    with TestClient(app(Parser())) as client:
        schema = client.get("/openapi.json").json()
        operation = schema["paths"]["/v1/documents/parse"]["post"]
        for status in ("200", "400", "500"):
            assert "$ref" in operation["responses"][status]["content"]["application/json"]["schema"]
        components = schema["components"]["schemas"]
        assert components["DocumentBlockResponse"]["properties"]["box"]["type"] == "array"


def test_unknown_routes_use_the_same_json_error_contract() -> None:
    with TestClient(app(Parser())) as client:
        response = client.get("/missing")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "http_error"


def test_cli_and_lifespan_close_underlying_provider_exactly_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import uvicorn

    from capicove.cli.commands import ocr_serve

    class CountedProvider(Provider):
        closes = 0

        def close(self) -> None:
            self.closes += 1

    provider = CountedProvider()
    monkeypatch.setattr(ocr_serve, "build_provider", lambda *args, **kwargs: provider)
    monkeypatch.setattr(ocr_serve, "build_document_parser", lambda *args, **kwargs: None)

    def serve(application: Any, **kwargs: Any) -> None:
        with TestClient(application) as client:
            assert client.get("/health").status_code == 200

    monkeypatch.setattr(uvicorn, "run", serve)
    ocr_serve.serve(
        model_directory=tmp_path,
        device="cpu",
        enable_mkldnn=False,
        host="127.0.0.1",
        port=0,
        api_key=None,
        max_body_bytes=1024,
        max_pixels=100,
    )
    assert provider.closes == 1


def test_cancellation_keeps_upload_and_admission_until_native_work_finishes(tmp_path: Path) -> None:
    started, release = Event(), Event()

    class BlockingParser(Parser):
        def parse(self, request: Any, /) -> Any:
            self.path = Path(request.source.path)
            started.set()
            assert release.wait(10)
            assert self.path.exists()
            return super().parse(request)

    async def exercise() -> None:
        parser = BlockingParser()
        application = app(parser, document_temp_directory=tmp_path)
        async with application.router.lifespan_context(application):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=application), base_url="http://test"
            ) as client:
                pending = asyncio.create_task(
                    client.post(
                        "/v1/documents/parse",
                        content=pdf(),
                        headers={"Content-Type": "application/pdf"},
                    )
                )
                try:
                    assert await anyio.to_thread.run_sync(started.wait, 5)
                    pending.cancel()
                    await asyncio.sleep(0.02)
                    assert parser.path is not None and parser.path.exists()
                    response = await client.post(
                        "/v1/ocr", content=picture(), headers={"Content-Type": "image/png"}
                    )
                    assert response.status_code == 429
                finally:
                    release.set()
                with pytest.raises(asyncio.CancelledError):
                    await pending
                assert not list(tmp_path.iterdir())
                assert (
                    await client.post(
                        "/v1/ocr", content=picture(), headers={"Content-Type": "image/png"}
                    )
                ).status_code == 200

    anyio.run(exercise)


@pytest.mark.parametrize(("field", "value"), [("width", 0), ("unexpected", "private extra field")])
def test_invalid_wire_evidence_is_rejected_by_response_validation(
    field: str, value: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    from capicove.documents import document_parse_result_to_json

    def corrupt(result: Any) -> dict[str, object]:
        encoded = document_parse_result_to_json(result)
        pages = cast(list[dict[str, object]], encoded["pages"])
        pages[0][field] = value
        return encoded

    monkeypatch.setattr("capicove.api.documents.document_parse_result_to_json", corrupt)
    with TestClient(app(Parser())) as client:
        response = client.post(
            "/v1/documents/parse", content=pdf(), headers={"Content-Type": "application/pdf"}
        )
        assert response.status_code == 500
        assert "private" not in response.text


def test_internal_diagnostics_correlate_without_logging_private_exception_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    parser = Parser()
    parser.fail = RuntimeError("PRIVATE_DOCUMENT_TEXT /private/model/file")
    with TestClient(app(parser)) as client:
        response = client.post(
            "/v1/documents/parse", content=pdf(), headers={"Content-Type": "application/pdf"}
        )
        assert response.status_code == 500
        assert response.headers["x-request-id"] in caplog.text
        assert "RuntimeError" in caplog.text
        assert "PRIVATE_DOCUMENT_TEXT" not in caplog.text
        assert "/private/model/file" not in caplog.text


def test_fingerprint_startup_failure_still_releases_provider_once() -> None:
    from capicove.api.ocr import create_app

    class FailingProvider:
        provider_id = "broken"
        provider_version = "test"
        closes = 0

        @property
        def fingerprint(self) -> ProcessingFingerprint:
            raise RuntimeError("fingerprint unavailable")

        def recognize(self, image: Image.Image, languages: tuple[str, ...]) -> OcrProviderOutput:
            raise AssertionError("startup failure must not reach inference")

        def close(self) -> None:
            self.closes += 1

    provider = FailingProvider()
    with (
        pytest.raises(RuntimeError, match="fingerprint unavailable"),
        TestClient(create_app(lambda: provider)),
    ):
        pass
    assert provider.closes == 1


def test_disconnect_during_pdf_upload_cleans_staging_and_releases_slot(tmp_path: Path) -> None:
    from starlette.requests import ClientDisconnect

    async def broken_upload() -> Any:
        yield pdf()[:30]
        raise ClientDisconnect()

    async def exercise() -> None:
        parser = Parser()
        application = app(parser, document_temp_directory=tmp_path)
        async with application.router.lifespan_context(application):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=application), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/v1/documents/parse",
                    content=broken_upload(),
                    headers={"Content-Type": "application/pdf"},
                )
                assert response.status_code == 400
                assert response.json()["error"]["code"] == "client_disconnected"
                assert not parser.requests
                assert not list(tmp_path.iterdir())
                assert (
                    await client.post(
                        "/v1/ocr", content=picture(), headers={"Content-Type": "image/png"}
                    )
                ).status_code == 200

    anyio.run(exercise)


def test_settings_cannot_silently_override_legacy_limit_arguments() -> None:
    from capicove.api.ocr import OcrHttpSettings, create_app

    with pytest.raises(ValueError, match="settings"):
        create_app(Provider, settings=OcrHttpSettings(max_pixels=100), max_pixels=50)


def test_image_asset_failure_is_backend_unavailable() -> None:
    from capicove.api.ocr import create_app
    from capicove.processing.image.ocr import OcrProviderUnavailableError

    class UnavailableProvider(Provider):
        def recognize(self, image: Image.Image, languages: tuple[str, ...]) -> OcrProviderOutput:
            raise OcrProviderUnavailableError("private model receipt")

    with TestClient(create_app(UnavailableProvider)) as client:
        response = client.post("/v1/ocr", content=picture(), headers={"Content-Type": "image/png"})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "backend_unavailable"
        assert "private" not in response.text
