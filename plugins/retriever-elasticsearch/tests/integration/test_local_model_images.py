"""Opt-in real image embedding through the existing isolated local model API."""

import io
import math
import os
from datetime import UTC, datetime, timedelta

import pytest


@pytest.mark.skipif(
    not os.environ.get("CAPICOVE_TEST_VIDEO_MODEL_ENDPOINT"),
    reason="explicit verified local model endpoint required",
)
def test_authored_image_query_reuses_existing_model_image_api() -> None:
    from PIL import Image

    from capicove.contracts import (
        DistanceMetric,
        EmbeddingSpaceRef,
        ImageQuery,
        LocalPathSource,
        QueryModality,
        TargetType,
        VectorNormalization,
    )
    from capicove.model import load_packaged_manifest
    from capicove.retrieval.video_http_encoding import LocalSiglip2QueryEncoder

    model = next(
        item for item in load_packaged_manifest().models if item.key == "siglip2-base-patch16-224"
    )
    space = EmbeddingSpaceRef(
        "siglip2-local-test",
        model.repository,
        model.revision,
        model.preprocess_revision,
        768,
        VectorNormalization.L2,
        DistanceMetric.INNER_PRODUCT,
        frozenset({QueryModality.TEXT, QueryModality.IMAGE}),
        frozenset({TargetType.KEYFRAME, TargetType.VIDEO_SEGMENT}),
    )
    buffer = io.BytesIO()
    Image.new("RGB", (224, 224), (32, 96, 192)).save(buffer, format="PNG")
    payload = buffer.getvalue()
    queries: list[ImageQuery] = []

    def resolve(query: ImageQuery) -> bytes:
        queries.append(query)
        return payload

    encoder = LocalSiglip2QueryEncoder(
        endpoint=os.environ["CAPICOVE_TEST_VIDEO_MODEL_ENDPOINT"],
        model_id=model.repository,
        embedding_space=space,
        resolve_image=resolve,
    )
    query = ImageQuery(LocalPathSource("/authored-fixture-not-read.png"))
    actual = encoder.encode(
        query, embedding_space=space, deadline=datetime.now(UTC) + timedelta(seconds=30)
    )
    prepared = encoder.encode_preview_batch((payload,))[0]
    assert queries == [query]
    assert len(actual) == 768
    assert math.isclose(sum(value * value for value in actual), 1.0, abs_tol=2e-4)
    assert actual == pytest.approx(prepared, abs=1e-6)
