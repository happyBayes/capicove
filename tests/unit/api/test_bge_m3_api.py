"""Compatibility and bounded admission for local BGE-M3 HTTP inference."""

from __future__ import annotations

import base64
import struct
import threading
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from capicove.api.bge_m3 import create_app
from capicove.contracts import EmbeddingSpaceRef
from capicove.retrieval import TextEmbedding
from tests.unit.retrieval.test_text_embedding import text_space


class Encoder:
    """Return authored normalized vectors and actual test token counts."""

    space = replace(text_space(), dimension=1024)

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.closed = False
        self.failure: Exception | None = None
        self.entered = threading.Event()
        self.release: threading.Event | None = None

    def encode_with_token_counts(
        self,
        texts: tuple[str, ...],
        *,
        space: EmbeddingSpaceRef,
        role: str,
        max_total_tokens: int | None = None,
    ) -> tuple[tuple[TextEmbedding, ...], tuple[int, ...]]:
        assert space == self.space and role == "document"
        assert max_total_tokens == 300_000
        self.calls.append(texts)
        self.entered.set()
        if self.release is not None:
            assert self.release.wait(5)
        if self.failure:
            raise self.failure
        vectors = tuple(
            TextEmbedding(space, (float(index % 2), float(1 - index % 2), *((0.0,) * 1022)))
            for index in range(len(texts))
        )
        return vectors, tuple(len(text) + 2 for text in texts)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def client() -> Iterator[tuple[TestClient, Encoder]]:
    encoder = Encoder()
    with TestClient(create_app(encoder)) as client:
        yield client, encoder
    assert encoder.closed


def test_single_and_batch_embeddings_preserve_order_and_count_tokens(
    client: tuple[TestClient, Encoder],
) -> None:
    http, encoder = client
    single = http.post("/v1/embeddings", json={"model": "bge-m3", "input": "中文"})
    assert single.status_code == 200
    assert single.json()["usage"] == {"prompt_tokens": 4, "total_tokens": 4}
    response = http.post(
        "/v1/embeddings",
        json={"model": "BAAI/bge-m3", "input": ["second", "first", "second"], "user": "local"},
    )
    data = response.json()
    assert response.status_code == 200
    assert data["object"] == "list" and data["model"] == "BAAI/bge-m3"
    assert data["usage"] == {"prompt_tokens": 23, "total_tokens": 23}
    assert [r["index"] for r in data["data"]] == [0, 1, 2]
    assert [r["object"] for r in data["data"]] == ["embedding"] * 3
    assert [r["embedding"][:2] for r in data["data"]] == [[0, 1], [1, 0], [0, 1]]
    assert encoder.calls == [("中文",), ("second", "first", "second")]


def test_base64_is_little_endian_float32(client: tuple[TestClient, Encoder]) -> None:
    http, _ = client
    response = http.post(
        "/v1/embeddings",
        json={"model": "bge-m3", "input": "hello", "encoding_format": "base64", "dimensions": 1024},
    )
    vector = struct.unpack("<1024f", base64.b64decode(response.json()["data"][0]["embedding"]))
    assert vector == (0.0, 1.0, *((0.0,) * 1022))


@pytest.mark.parametrize(
    "change",
    [
        {"input": ""},
        {"input": "  "},
        {"input": []},
        {"input": ["ok", ""]},
        {"input": [1, 2]},
        {"input": [[1, 2]]},
        {"input": ["a", 1]},
        {"input": ["a"] * 129},
        {"input": "\ud800"},
        {"model": "different"},
        {"dimensions": 512},
        {"dimensions": True},
        {"encoding_format": "bytes"},
        {"user": 1},
        {"extra": "ignored?"},
    ],
)
def test_invalid_requests_never_invoke_encoder(
    client: tuple[TestClient, Encoder], change: dict[str, Any]
) -> None:
    http, encoder = client
    body = {"model": "bge-m3", "input": "valid", **change}
    # JSON's escaped representation can carry invalid Unicode without client encoding failure.
    import json

    response = http.post("/v1/embeddings", content=json.dumps(body))
    assert response.status_code in {400, 404}
    error = response.json()["error"]
    assert {"message", "type", "param", "code"} == error.keys()
    assert encoder.calls == []


def test_malformed_and_oversized_json_are_bounded(client: tuple[TestClient, Encoder]) -> None:
    http, encoder = client
    for payload in (b"{", b"[]", b'{"model":"bge-m3","input":"x","input":"y"}'):
        assert http.post("/v1/embeddings", content=payload).status_code == 400
    assert http.post("/v1/embeddings", content=b"x" * (2 * 1024 * 1024 + 1)).status_code == 413
    assert encoder.calls == []


def test_missing_required_fields_use_standard_client_errors(
    client: tuple[TestClient, Encoder],
) -> None:
    http, encoder = client
    for payload in ({}, {"model": "bge-m3"}, {"input": "hello"}):
        assert http.post("/v1/embeddings", json=payload).status_code == 400
    assert encoder.calls == []


def test_streamed_body_is_bounded_before_inference(client: tuple[TestClient, Encoder]) -> None:
    http, encoder = client

    def chunks() -> Iterator[bytes]:
        for _ in range(33):
            yield b"x" * 65536

    assert http.post("/v1/embeddings", content=chunks()).status_code == 413
    assert encoder.calls == []


def test_builder_closes_encoder_after_failed_warmup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    from capicove.api.bge_m3 import build_encoder
    from capicove.retrieval import bge

    encoder = Encoder()

    def fail(*args: Any, **kwargs: Any) -> None:
        raise ValueError("runtime unavailable")

    wrapped = SimpleNamespace(space=encoder.space, encode=fail, close=encoder.close)
    monkeypatch.setattr(bge, "load_bge_m3_profile", lambda *args, **kwargs: object())
    monkeypatch.setattr(bge, "BgeTextBatchEncoder", lambda *args, **kwargs: wrapped)
    with pytest.raises(ValueError, match="runtime unavailable"):
        build_encoder(tmp_path, expected_model_manifest_sha256="a" * 64)
    assert encoder.closed


def test_health_model_discovery_and_bearer_authentication() -> None:
    encoder = Encoder()
    # Synthetic credential used only by the in-process test application.
    api_key = "test-only-key"  # pragma: allowlist secret
    with TestClient(create_app(encoder, api_key=api_key)) as http:
        assert http.get("/health").json()["status"] == "ok"
        assert http.get("/v1/models").status_code == 401
        assert http.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
        headers = {"Authorization": "Bearer test-only-key"}
        assert http.get("/v1/models", headers=headers).json()["data"][0]["id"] == "BAAI/bge-m3"
        assert http.get("/v1/models/BAAI/bge-m3", headers=headers).status_code == 200
        assert http.get("/v1/models/unknown", headers=headers).status_code == 404
        assert http.post("/v1/embeddings", content=b"{", headers={}).status_code == 401
        assert (
            http.post(
                "/v1/embeddings", json={"model": "bge-m3", "input": "ok"}, headers=headers
            ).status_code
            == 200
        )


@pytest.mark.parametrize(
    "failure,status",
    [
        (ValueError("embedding_input_too_long"), 400),
        (ValueError("embedding_batch_too_long"), 400),
        (RuntimeError("private/path/secret"), 500),
    ],
)
def test_runtime_errors_are_safe(
    client: tuple[TestClient, Encoder], failure: Exception, status: int
) -> None:
    http, encoder = client
    encoder.failure = failure
    response = http.post("/v1/embeddings", json={"model": "bge-m3", "input": "ok"})
    assert response.status_code == status
    assert "private" not in response.text and "secret" not in response.text
    encoder.failure = None
    assert http.post("/v1/embeddings", json={"model": "bge-m3", "input": "ok"}).status_code == 200


def test_busy_inference_does_not_block_health(client: tuple[TestClient, Encoder]) -> None:
    http, encoder = client
    encoder.release = threading.Event()
    responses: list[int] = []
    thread = threading.Thread(
        target=lambda: responses.append(
            http.post("/v1/embeddings", json={"model": "bge-m3", "input": "ok"}).status_code
        )
    )
    thread.start()
    try:
        assert encoder.entered.wait(5)
        assert http.get("/health").status_code == 200
        assert (
            http.post("/v1/embeddings", json={"model": "bge-m3", "input": "ok"}).status_code == 429
        )
    finally:
        encoder.release.set()
        thread.join(5)
    assert responses[0] == 200


def test_openai_sdk_default_and_explicit_encoding(client: tuple[TestClient, Encoder]) -> None:
    openai = pytest.importorskip("openai")
    http, _ = client
    # Dummy SDK credential; the transport never contacts an external service.
    api_key = "local"  # pragma: allowlist secret
    with openai.OpenAI(api_key=api_key, base_url="http://testserver/v1", http_client=http) as sdk:
        for options in ({}, {"encoding_format": "float"}):
            result = sdk.embeddings.create(model="bge-m3", input=["one", "two"], **options)
            assert result.data[0].embedding == [0.0, 1.0, *([0.0] * 1022)]
            assert result.usage.total_tokens == 10
