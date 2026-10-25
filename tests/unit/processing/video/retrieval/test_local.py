"""Public local composition retains model and source authorization boundaries."""

from pathlib import Path

import pytest

from capicove.contracts.retrieval import (
    DistanceMetric,
    EmbeddingSpaceRef,
    QueryModality,
    TargetType,
    VectorNormalization,
)
from capicove.processing.video.retrieval.local import create_local_video_preparation_service
from capicove.processing.video.retrieval.pipeline import VideoRetrievalPreparationService


def space() -> EmbeddingSpaceRef:
    return EmbeddingSpaceRef(
        "local",
        "fixture",
        "1",
        "1",
        2,
        VectorNormalization.L2,
        DistanceMetric.INNER_PRODUCT,
        frozenset({QueryModality.TEXT, QueryModality.IMAGE}),
        frozenset({TargetType.KEYFRAME, TargetType.VIDEO_SEGMENT}),
    )


def test_public_helper_has_no_model_calls_or_filesystem_mutation(tmp_path: Path) -> None:
    calls = []

    def encode(payloads: tuple[bytes, ...]) -> tuple[tuple[float, ...], ...]:
        calls.append(payloads)
        return tuple((1.0, 0.0) for _ in payloads)

    service = create_local_video_preparation_service(
        tmp_path / "workspace", embedding_space=space(), encode_batch=encode
    )
    assert isinstance(service, VideoRetrievalPreparationService)
    assert not calls
    assert not (tmp_path / "workspace").exists()


def test_public_helper_rejects_incomplete_visual_space(tmp_path: Path) -> None:
    from dataclasses import replace

    with pytest.raises(ValueError, match="text and image"):
        create_local_video_preparation_service(
            tmp_path,
            embedding_space=replace(space(), query_modalities=frozenset({QueryModality.IMAGE})),
            encode_batch=lambda payloads: (),
        )
