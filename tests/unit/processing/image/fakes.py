"""Deterministic provider fakes for image-processing tests."""

from __future__ import annotations

from PIL import Image


class FixedImageScorer:
    """Return configured scores for ordered images."""

    provider_id = "fixed-image-scorer"
    provider_version = "1"
    model_id = "authored-test-model"
    model_revision = "1"
    model_manifest_sha256 = "a" * 64
    preprocess_revision = "authored-rgb-v1"
    device = "cpu"

    def __init__(self, scores: tuple[float, ...]) -> None:
        self._scores = scores

    def score(self, images: tuple[Image.Image, ...]) -> tuple[float, ...]:
        assert len(images) == len(self._scores)
        return self._scores


class FixedImageTextScorer:
    """Return configured image-text scores."""

    provider_id = "fixed-image-text-scorer"
    provider_version = "1"
    model_id = "authored-test-model"
    model_revision = "1"
    model_manifest_sha256 = "a" * 64
    preprocess_revision = "authored-rgb-v1"
    device = "cpu"

    def __init__(self, scores: tuple[float, ...]) -> None:
        self._scores = scores
        self.texts: list[str] = []

    def score(self, images: tuple[Image.Image, ...], text: str) -> tuple[float, ...]:
        assert len(images) == len(self._scores)
        self.texts.append(text)
        return self._scores


class FixedImagePairScorer:
    """Return one configured image-pair score."""

    provider_id = "fixed-image-pair-scorer"
    provider_version = "1"
    model_id = "authored-test-model"
    model_revision = "1"
    model_manifest_sha256 = "a" * 64
    preprocess_revision = "authored-rgb-v1"
    device = "cpu"

    def __init__(self, score: float) -> None:
        self._score = score

    def score(self, left: Image.Image, right: Image.Image) -> float:
        return self._score
