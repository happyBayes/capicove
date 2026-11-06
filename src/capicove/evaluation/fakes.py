"""Deterministic stand-ins for models used by offline evaluation baselines.

Nothing here is a model. Every fake carries an identity beginning with
``capicove/deterministic-fake`` and every run record that uses one says so.
Numbers produced with these fakes are engineering baselines for ranking
mechanics; real-model quality validation has not been run.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from capicove.retrieval.reranking import RerankResult, rank_scores, validate_rerank_request


FAKE_RERANKER_MODEL_ID = "capicove/deterministic-fake-reranker"
FAKE_REVISION = "fake-1"
_TOKEN = re.compile(r"[a-z0-9]+|[㐀-鿿]")


def tokens(text: str) -> list[str]:
    """Return casefolded Latin words and CJK characters."""
    return _TOKEN.findall(text.casefold())


class DeterministicFakeReranker:
    """Score pairs by distinct query-token coverage, then by shorter documents.

    It satisfies :class:`capicove.retrieval.reranking.TextReranker`.
    """

    model_id = FAKE_RERANKER_MODEL_ID
    model_revision = FAKE_REVISION

    def rerank(
        self, query: str, documents: tuple[str, ...], *, top_n: int | None = None
    ) -> RerankResult:
        """Return stable descending scores for every document."""
        count = validate_rerank_request(query, documents, top_n)
        wanted = set(tokens(query))
        scores = []
        for document in documents:
            present = set(tokens(document))
            scores.append(len(wanted & present) + 1.0 / (1.0 + len(document) / 1000.0))
        return RerankResult(
            self.model_id,
            self.model_revision,
            rank_scores(tuple(scores), top_n=count),
            sum(len(tokens(text)) for text in (query, *documents)),
        )


def hashed_bag_of_words(text: str, dimension: int = 64) -> tuple[float, ...]:
    """Return an L2-normalized feature-hashed token vector (never all zero)."""
    values = [0.0] * dimension
    for token in tokens(text) or ["\x00empty"]:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        values[index] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        values[0], norm = 1.0, 1.0
    return tuple(value / norm for value in values)


def fake_text_encoder(texts: Sequence[str], role: str) -> tuple[str, list[list[float]]]:
    """Encode texts for the knowledge-base dense lane with hashed bags of words."""
    space = "capicove/deterministic-fake-hashed-bow-64"
    return space, [list(hashed_bag_of_words(text)) for text in texts]


COLOR_WORDS = {
    "red": (220, 30, 30),
    "green": (30, 170, 60),
    "blue": (30, 60, 220),
    "yellow": (230, 210, 40),
    "purple": (140, 40, 170),
    "orange": (240, 140, 20),
    "black": (15, 15, 15),
    "white": (245, 245, 245),
}


def dominant_color_word(pixels: Sequence[tuple[int, int, int]]) -> str:
    """Name the authored palette color nearest to the mean pixel."""
    count = max(1, len(pixels))
    mean = tuple(sum(pixel[channel] for pixel in pixels) / count for channel in range(3))
    return min(
        COLOR_WORDS,
        key=lambda name: (
            sum((mean[channel] - COLOR_WORDS[name][channel]) ** 2 for channel in range(3)),
            name,
        ),
    )


class FakeSharedSpaceEncoder:
    """Shared-space fake: texts by hashed words, images by their mean palette color.

    Text documents and queries share the hashed bag-of-words space. An image is
    encoded as the words ``image <color>``, so ``red image`` finds red images.
    Audio clips hash their PCM bytes (no semantic signal).
    """

    provider_id = "capicove-deterministic-fake"

    def __init__(self, space: Any) -> None:
        self.space = space

    def encode_text_queries(
        self, texts: Sequence[str], *, deadline: datetime
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(hashed_bag_of_words(text, self.space.dimension) for text in texts)

    def encode_text_documents(
        self, documents: Sequence[Any], *, deadline: datetime
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(
            hashed_bag_of_words(document.text, self.space.dimension) for document in documents
        )

    def encode_images(
        self, images: Sequence[bytes], *, deadline: datetime
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(
            hashed_bag_of_words("image " + _image_color(image), self.space.dimension)
            for image in images
        )

    def encode_audio(
        self, clips: Sequence[Any], *, deadline: datetime
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(
            hashed_bag_of_words(hashlib.sha256(clip.pcm).hexdigest(), self.space.dimension)
            for clip in clips
        )

    def close(self) -> None:
        return None


def _image_color(payload: bytes) -> str:
    import io

    from PIL import Image

    with Image.open(io.BytesIO(payload)) as image:
        small = image.convert("RGB").resize((8, 8))
        pixels = [small.getpixel((x, y)) for y in range(8) for x in range(8)]
    return dominant_color_word(pixels)  # type: ignore[arg-type]


__all__ = [
    "COLOR_WORDS",
    "FAKE_RERANKER_MODEL_ID",
    "FAKE_REVISION",
    "DeterministicFakeReranker",
    "FakeSharedSpaceEncoder",
    "dominant_color_word",
    "fake_text_encoder",
    "hashed_bag_of_words",
    "tokens",
]
