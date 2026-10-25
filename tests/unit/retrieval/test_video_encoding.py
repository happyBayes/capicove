"""One provisioned public embedding service serves offline and online vectors."""

from datetime import UTC, datetime, timedelta
from io import BytesIO

import pytest
from PIL import Image

from capicove.contracts import ImageQuery, LocalPathSource, TextQuery
from capicove.retrieval.models import ProviderMatch
from capicove.retrieval.service import Siglip2RetrievalService
from capicove.retrieval.video_encoding import VideoEmbeddingAdapter
from tests.unit.retrieval.test_video_models import visual_space


class Provider:
    model_id = "local-model"
    model_revision = "revision-1"
    device = "cpu"

    def rank(self, image: Image.Image, texts: tuple[str, ...]) -> tuple[ProviderMatch, ...]:
        return ()

    def embed_images(self, images: tuple[Image.Image, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0) for _ in images)

    def embed_texts(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0) for _ in texts)


def png() -> bytes:
    output = BytesIO()
    Image.new("RGB", (8, 8)).save(output, format="PNG")
    return output.getvalue()


def test_offline_and_online_share_declared_space() -> None:
    adapter = VideoEmbeddingAdapter(
        Siglip2RetrievalService(Provider()),
        visual_space(),
        model_id="local-model",
        resolve_image=lambda query: png(),
    )
    deadline = datetime.now(UTC) + timedelta(seconds=5)
    assert adapter.encode(
        TextQuery("slide"), embedding_space=visual_space(), deadline=deadline
    ) == (1.0, 0.0)
    assert adapter.encode(
        ImageQuery(LocalPathSource("/authorized/image.png")),
        embedding_space=visual_space(),
        deadline=deadline,
    ) == (1.0, 0.0)
    assert adapter.encode_preview_batch((png(),)) == ((1.0, 0.0),)


def test_image_requires_explicit_authorized_resolver() -> None:
    adapter = VideoEmbeddingAdapter(
        Siglip2RetrievalService(Provider()), visual_space(), model_id="local-model"
    )
    with pytest.raises(PermissionError):
        adapter.encode(
            ImageQuery(LocalPathSource("/not-authorized")),
            embedding_space=visual_space(),
            deadline=datetime.now(UTC) + timedelta(seconds=5),
        )


def test_wrong_model_identity_rejected_before_inference() -> None:
    with pytest.raises(ValueError, match="identity"):
        VideoEmbeddingAdapter(Siglip2RetrievalService(Provider()), visual_space(), model_id="wrong")
