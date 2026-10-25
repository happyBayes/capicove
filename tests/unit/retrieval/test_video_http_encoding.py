"""Local model process transport validates identity and never follows redirects."""

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from capicove.contracts import ImageQuery, LocalPathSource, TextQuery
from capicove.retrieval.video_http_encoding import LocalSiglip2QueryEncoder
from tests.unit.retrieval.test_video_models import visual_space


def encoder(monkeypatch: pytest.MonkeyPatch) -> tuple[LocalSiglip2QueryEncoder, Mock]:
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.read.return_value = json.dumps(
        {
            "model_id": "public-model",
            "model_revision": visual_space().model_revision,
            "dimension": 2,
            "embeddings": [[1.0, 0.0]],
        }
    ).encode()
    opener = Mock()
    opener.open.return_value = response
    monkeypatch.setattr("urllib.request.build_opener", lambda *args: opener)
    return LocalSiglip2QueryEncoder(
        endpoint="http://127.0.0.1:19422",
        model_id="public-model",
        embedding_space=visual_space(),
    ), opener


def test_local_model_process_returns_only_matching_finite_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, opener = encoder(monkeypatch)
    result = adapter.encode(
        TextQuery("scene"),
        embedding_space=visual_space(),
        deadline=datetime.now(UTC) + timedelta(seconds=10),
    )
    assert result == (1.0, 0.0)
    request = opener.open.call_args.args[0]
    assert request.full_url == "http://127.0.0.1:19422/v1/embeddings/text"
    assert json.loads(request.data) == {"texts": ["scene"]}
    assert 0 < opener.open.call_args.kwargs["timeout"] <= 10


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://example.com",
        "http://localhost:19422",
        "http://user@127.0.0.1:19422",
        "http://127.0.0.1:19422/path",
        "http://127.0.0.1:19422?secret=1",
    ],
)
def test_model_transport_forbids_remote_or_ambiguous_endpoints(endpoint: str) -> None:
    with pytest.raises(ValueError):
        LocalSiglip2QueryEncoder(endpoint=endpoint, model_id="m", embedding_space=visual_space())


@pytest.mark.parametrize(
    "payload",
    [
        {"model_id": "wrong"},
        {"model_id": "public-model", "model_revision": "wrong"},
        {
            "model_id": "public-model",
            "model_revision": visual_space().model_revision,
            "dimension": 2,
            "embeddings": [[0.0, 0.0]],
        },
    ],
)
def test_model_transport_rejects_malformed_or_changed_model(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
) -> None:
    adapter, opener = encoder(monkeypatch)
    opener.open.return_value.read.return_value = json.dumps(payload).encode()
    with pytest.raises(ValueError):
        adapter.encode(
            TextQuery("x"),
            embedding_space=visual_space(),
            deadline=datetime.now(UTC) + timedelta(seconds=2),
        )


def test_expired_query_never_contacts_model(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, opener = encoder(monkeypatch)
    with pytest.raises(TimeoutError):
        adapter.encode(
            TextQuery("x"),
            embedding_space=visual_space(),
            deadline=datetime.now(UTC) - timedelta(seconds=1),
        )
    opener.open.assert_not_called()


def test_image_query_requires_explicit_authorized_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, opener = encoder(monkeypatch)
    with pytest.raises(PermissionError):
        adapter.encode(
            ImageQuery(LocalPathSource("/not-read.png")),
            embedding_space=visual_space(),
            deadline=datetime.now(UTC) + timedelta(seconds=2),
        )
    opener.open.assert_not_called()


def test_preview_batch_reuses_public_image_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, opener = encoder(monkeypatch)
    assert adapter.encode_preview_batch((b"authorized-preview",)) == ((1.0, 0.0),)
    request = opener.open.call_args.args[0]
    assert request.full_url.endswith("/v1/embeddings/image")
    assert b'filename="preview"' in request.data
    assert b"authorized-preview" in request.data


@pytest.mark.parametrize("previews", [(), (b"x",) * 17, (b"",), (b"x" * (8 * 1024**2 + 1),)])
def test_preview_capacity_rejected_before_transport(
    monkeypatch: pytest.MonkeyPatch, previews: tuple[bytes, ...]
) -> None:
    adapter, opener = encoder(monkeypatch)
    with pytest.raises(ValueError):
        adapter.encode_preview_batch(previews)
    opener.open.assert_not_called()


@pytest.mark.parametrize("body", [b"x" * (1024**2 + 1), b"[]", b"not-json"])
def test_response_capacity_and_shape_fail_closed(
    monkeypatch: pytest.MonkeyPatch, body: bytes
) -> None:
    adapter, opener = encoder(monkeypatch)
    opener.open.return_value.read.return_value = body
    with pytest.raises(ValueError):
        adapter.encode(
            TextQuery("x"),
            embedding_space=visual_space(),
            deadline=datetime.now(UTC) + timedelta(seconds=2),
        )


def test_redirect_handler_refuses_redirect() -> None:
    from capicove.retrieval.video_http_encoding import _NoRedirect

    with pytest.raises(ValueError, match="redirects"):
        _NoRedirect().redirect_request()
