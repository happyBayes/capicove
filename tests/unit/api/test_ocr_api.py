"""HTTP contract tests using an injected OCR provider without model downloads."""

from __future__ import annotations

import hashlib
from io import BytesIO
from threading import Event
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from capicove.contracts import ProcessingFingerprint
from capicove.processing.image import OcrWord
from capicove.processing.image.ocr import OcrProviderOutput


class Provider:
    provider_id = "test-ocr"
    provider_version = "1"
    fingerprint = ProcessingFingerprint("a" * 64)

    def __init__(self) -> None:
        self.closed = False
        self.calls = 0
        self.fail = False

    def recognize(self, image: Image.Image, languages: tuple[str, ...]) -> OcrProviderOutput:
        self.calls += 1
        if self.fail:
            raise RuntimeError("private runtime path and input")
        return OcrProviderOutput((OcrWord("hello", 0, 0, 4, 4, 0.9, 1, 1, 1, 1, 1),))

    def close(self) -> None:
        self.closed = True


def picture(size: tuple[int, int] = (8, 8)) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, "white").save(output, format="PNG")
    return output.getvalue()


def test_ocr_upload_preserves_text_geometry_identity_and_lifecycle() -> None:
    from capicove.api.ocr import create_app

    provider = Provider()
    data = picture()
    with TestClient(create_app(lambda: provider)) as client:
        assert client.get("/health").json()["provider_id"] == "test-ocr"
        response = client.post("/v1/ocr", content=data, headers={"Content-Type": "image/png"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["processing_fingerprint"] == "a" * 64
        assert payload["result"]["source_hash"] == hashlib.sha256(data).hexdigest()
        assert payload["result"]["full_text"] == "hello"
        assert payload["result"]["words"][0]["confidence"] == 0.9
        assert payload["result"]["provider_id"] == "test-ocr"
        schema = client.get("/openapi.json").json()
        assert "requestBody" in schema["paths"]["/v1/ocr"]["post"]
    assert provider.closed
    assert provider.calls == 1


@pytest.mark.parametrize(
    ("body", "content_type", "status"),
    [
        (b"", "image/png", 400),
        (b"invalid", "image/png", 400),
        (b'{"path":"/etc/passwd"}', "application/json", 415),
        (b"https://example.com/image.png", "text/plain", 415),
    ],
)
def test_invalid_upload_never_reaches_provider(body: bytes, content_type: str, status: int) -> None:
    from capicove.api.ocr import create_app

    provider = Provider()
    with TestClient(create_app(lambda: provider)) as client:
        response = client.post("/v1/ocr", content=body, headers={"Content-Type": content_type})
        assert response.status_code == status
        assert provider.calls == 0


def test_authentication_limits_and_safe_failures() -> None:
    from capicove.api.ocr import create_app

    provider = Provider()
    with TestClient(
        create_app(
            lambda: provider,
            # Synthetic credential confined to this authentication test.
            api_key="test-only-key",  # pragma: allowlist secret
            max_body_bytes=1000,
            max_pixels=100,
        )
    ) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/docs").status_code == 401
        assert client.post("/v1/ocr", content=picture()).status_code == 401
        headers = {"Authorization": "Bearer test-only-key", "Content-Type": "image/png"}
        assert client.post("/v1/ocr", content=b"x" * 1001, headers=headers).status_code == 413
        assert client.post("/v1/ocr", content=picture((11, 10)), headers=headers).status_code == 413
        provider.fail = True
        response = client.post("/v1/ocr", content=picture(), headers=headers)
        assert response.status_code == 500
        assert "private" not in response.text
        assert provider.calls == 1


def test_busy_service_rejects_second_upload_and_health_remains_responsive() -> None:
    from concurrent.futures import ThreadPoolExecutor

    from capicove.api.ocr import create_app

    started, release = Event(), Event()

    class BlockingProvider(Provider):
        def recognize(self, image: Image.Image, languages: tuple[str, ...]) -> OcrProviderOutput:
            started.set()
            assert release.wait(10)
            return super().recognize(image, languages)

    with TestClient(create_app(BlockingProvider)) as client, ThreadPoolExecutor() as pool:
        pending = pool.submit(
            client.post, "/v1/ocr", content=picture(), headers={"Content-Type": "image/png"}
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


def test_startup_failure_is_propagated() -> None:
    from capicove.api.ocr import create_app

    def fail() -> Any:
        raise RuntimeError("startup failed")

    with pytest.raises(RuntimeError, match="startup failed"), TestClient(create_app(fail)):
        pass


def test_truncated_jpeg_is_a_client_error() -> None:
    from capicove.api.ocr import create_app

    data = BytesIO()
    Image.new("RGB", (64, 64), "white").save(data, format="JPEG")
    provider = Provider()
    with TestClient(create_app(lambda: provider)) as client:
        response = client.post(
            "/v1/ocr", content=data.getvalue()[:-20], headers={"Content-Type": "image/jpeg"}
        )
        assert response.status_code == 400
        assert provider.calls == 0


def test_chunked_upload_limit_is_enforced_without_content_length() -> None:
    from capicove.api.ocr import create_app

    provider = Provider()
    with TestClient(create_app(lambda: provider, max_body_bytes=10)) as client:
        response = client.post(
            "/v1/ocr", content=iter([b"x" * 6, b"x" * 6]), headers={"Content-Type": "image/png"}
        )
        assert response.status_code == 413
        assert provider.calls == 0


def test_animated_and_mismatched_images_are_rejected() -> None:
    from capicove.api.ocr import create_app

    data = BytesIO()
    Image.new("RGB", (8, 8), "white").save(
        data,
        format="PNG",
        save_all=True,
        append_images=[Image.new("RGB", (8, 8), "black")],
        duration=100,
        loop=0,
    )
    provider = Provider()
    with TestClient(create_app(lambda: provider)) as client:
        assert (
            client.post(
                "/v1/ocr", content=data.getvalue(), headers={"Content-Type": "image/png"}
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/v1/ocr", content=picture(), headers={"Content-Type": "image/jpeg"}
            ).status_code
            == 400
        )
        assert provider.calls == 0


def test_png_with_invalid_chunk_checksum_returns_json_client_error() -> None:
    from capicove.api.ocr import create_app

    data = bytearray(picture())
    offset = 8
    while data[offset + 4 : offset + 8] != b"IDAT":
        offset += 12 + int.from_bytes(data[offset : offset + 4], "big")
    checksum = offset + 8 + int.from_bytes(data[offset : offset + 4], "big")
    data[checksum] ^= 1
    corrupted = bytes(data)
    # Opening succeeds; checksum verification raises Pillow's SyntaxError.
    with Image.open(BytesIO(corrupted)) as image, pytest.raises(SyntaxError):
        image.verify()
    provider = Provider()
    with TestClient(create_app(lambda: provider), raise_server_exceptions=False) as client:
        response = client.post("/v1/ocr", content=corrupted, headers={"Content-Type": "image/png"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_image"
        assert provider.calls == 0
        assert (
            client.post(
                "/v1/ocr", content=picture(), headers={"Content-Type": "image/png"}
            ).status_code
            == 200
        )
        assert provider.calls == 1
