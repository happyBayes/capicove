"""Tests for the optional SigLIP2 HTTP boundary."""

from __future__ import annotations

import asyncio
import io
from typing import Any

import httpx
from PIL import Image

from capicove.api.siglip2 import create_app
from capicove.processing.image.preparation import prepare_image
from capicove.retrieval import ProviderMatch, Siglip2RetrievalService


class ApiProvider:
    """Provide deterministic values to the real HTTP application."""

    model_id = "google/siglip2-base-patch16-224"
    model_revision = "a" * 40
    device = "mps"

    def __init__(self) -> None:
        self.received_images: list[Image.Image] = []

    def rank(self, image: Image.Image, texts: tuple[str, ...]) -> tuple[ProviderMatch, ...]:
        del image
        return tuple(
            ProviderMatch(logit=float(len(texts) - index), match_probability=0.75)
            for index, _ in enumerate(texts)
        )

    def embed_images(self, images: tuple[Image.Image, ...]) -> tuple[tuple[float, ...], ...]:
        self.received_images.extend(image.copy() for image in images)
        return tuple((0.6, 0.8) for _ in images)

    def embed_texts(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0) for _ in texts)


def _png() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(stream, format="PNG")
    return stream.getvalue()


def _large_bmp() -> bytes:
    tile_size = 16
    pixels = bytes(
        channel
        for y in range(tile_size)
        for x in range(tile_size)
        for channel in ((x * 17) % 256, (y * 17) % 256, ((x + y) * 11) % 256)
    )
    tile = Image.frombytes("RGB", (tile_size, tile_size), pixels)
    source = tile.resize((1_200, 1_200), Image.Resampling.NEAREST)
    stream = io.BytesIO()
    source.save(stream, format="BMP")
    return stream.getvalue()


def _request(app: object, method: str, path: str, **kwargs: Any) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def test_api_exposes_health_ranking_and_embedding_contracts() -> None:
    app = create_app(Siglip2RetrievalService(ApiProvider()))

    health = _request(app, "GET", "/health")
    ranking = _request(
        app,
        "POST",
        "/v1/similarity",
        files={"image": ("test.png", _png(), "image/png")},
        data={"texts": '["a cat", "a dog"]'},
    )
    image_embedding = _request(
        app,
        "POST",
        "/v1/embeddings/image",
        files={"image": ("test.png", _png(), "image/png")},
    )
    text_embedding = _request(
        app,
        "POST",
        "/v1/embeddings/text",
        json={"texts": ["a cat", "a dog"]},
    )

    assert health.json() == {
        "status": "ok",
        "model_id": "google/siglip2-base-patch16-224",
        "model_revision": "a" * 40,
        "device": "mps",
    }
    assert ranking.status_code == 200
    assert [item["text"] for item in ranking.json()["matches"]] == ["a cat", "a dog"]
    assert image_embedding.json()["embeddings"] == [[0.6, 0.8]]
    assert text_embedding.json()["embeddings"] == [[1.0, 0.0], [1.0, 0.0]]


def test_api_rejects_oversized_image_before_decoding() -> None:
    app = create_app(
        Siglip2RetrievalService(ApiProvider()),
        max_upload_bytes=4,
    )

    response = _request(
        app,
        "POST",
        "/v1/embeddings/image",
        files={"image": ("test.png", _png(), "image/png")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "image exceeds the 4-byte limit"


def test_api_compresses_synthetic_image_above_four_mib_before_inference() -> None:
    payload = _large_bmp()
    assert len(payload) > 4 * 1024 * 1024
    provider = ApiProvider()
    app = create_app(
        Siglip2RetrievalService(provider),
        max_upload_bytes=8 * 1024 * 1024,
    )

    response = _request(
        app,
        "POST",
        "/v1/embeddings/image",
        files={"image": ("synthetic.bmp", payload, "image/bmp")},
    )

    assert response.status_code == 200
    assert response.json()["embeddings"] == [[0.6, 0.8]]
    prepared = prepare_image(payload)
    assert prepared.was_compressed is True
    with Image.open(io.BytesIO(prepared.encoded_bytes)) as encoded:
        expected = encoded.convert("RGB").copy()
    with Image.open(io.BytesIO(payload)) as original:
        original_rgb = original.convert("RGB").copy()
    assert expected.tobytes() != original_rgb.tobytes()
    assert len(provider.received_images) == 1
    assert provider.received_images[0].size == expected.size
    assert provider.received_images[0].tobytes() == expected.tobytes()


def test_api_maps_invalid_candidate_json_to_validation_error() -> None:
    app = create_app(Siglip2RetrievalService(ApiProvider()))

    response = _request(
        app,
        "POST",
        "/v1/similarity",
        files={"image": ("test.png", _png(), "image/png")},
        data={"texts": "not-json"},
    )

    assert response.status_code == 422
    assert "JSON array" in response.json()["detail"]
