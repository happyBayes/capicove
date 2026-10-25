"""Typed shared-space multimodal embedding port (RFC-0032).

One provider encodes text queries, text documents, images and PCM audio into a
single declared space. Every output is validated for count, dimension, finite
values and unit norm before it reaches retrieval. This module imports no
numerical, model or native-runtime SDK.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from capicove.contracts.retrieval import EmbeddingSpaceRef


MAX_BATCH_ITEMS = 64
MAX_TEXT_CHARACTERS = 16_384
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_AUDIO_FRAMES = 330 * 48_000
UNIT_NORM_TOLERANCE = 1e-3

Vector = tuple[float, ...]


class MultimodalEmbeddingError(RuntimeError):
    """Report a stable, payload-free encoder failure.

    ``code`` is one of ``invalid_input``, ``unavailable``, ``deadline_exceeded``
    or ``invalid_output``; malformed outputs are never repaired or retried.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True, slots=True)
class TextDocument:
    """Carry one document body and an optional real title."""

    text: str
    title: str | None = None

    def __post_init__(self) -> None:
        _bounded_text("text", self.text)
        if self.title is not None:
            _bounded_text("title", self.title)


@dataclass(frozen=True, slots=True)
class PcmAudio:
    """Carry prepared mono PCM16LE frames at an explicit sample rate."""

    pcm: bytes
    sample_rate_hz: int

    def __post_init__(self) -> None:
        if not isinstance(self.pcm, bytes) or not self.pcm or len(self.pcm) % 2:
            raise ValueError("pcm must contain whole non-empty 16-bit frames")
        if (
            isinstance(self.sample_rate_hz, bool)
            or not isinstance(self.sample_rate_hz, int)
            or not 8_000 <= self.sample_rate_hz <= 48_000
        ):
            raise ValueError("sample_rate_hz must be from 8000 to 48000")
        if len(self.pcm) // 2 > MAX_AUDIO_FRAMES:
            raise ValueError("pcm exceeds the provider input ceiling")

    @property
    def frames(self) -> int:
        """Return the number of mono frames."""
        return len(self.pcm) // 2


class SharedSpaceEncoder(Protocol):
    """Encode every supported modality into one complete embedding space."""

    provider_id: str
    space: EmbeddingSpaceRef

    def encode_text_queries(
        self, texts: Sequence[str], *, deadline: datetime
    ) -> tuple[Vector, ...]:
        """Encode search queries with the provider's fixed query prompt."""
        ...

    def encode_text_documents(
        self, documents: Sequence[TextDocument], *, deadline: datetime
    ) -> tuple[Vector, ...]:
        """Encode corpus text with the provider's fixed document prompt."""
        ...

    def encode_images(self, images: Sequence[bytes], *, deadline: datetime) -> tuple[Vector, ...]:
        """Encode encoded image bytes without any text prefix."""
        ...

    def encode_audio(self, clips: Sequence[PcmAudio], *, deadline: datetime) -> tuple[Vector, ...]:
        """Encode prepared waveform content directly, without ASR."""
        ...

    def close(self) -> None:
        """Release native resources and wait for owned processes."""
        ...


def require_batch(items: Sequence[object], limit: int = MAX_BATCH_ITEMS) -> int:
    """Return a bounded non-empty batch size; oversize batches are rejected."""
    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
        raise MultimodalEmbeddingError("invalid_input", "batch must be a sequence")
    if not 1 <= len(items) <= limit:
        raise MultimodalEmbeddingError("invalid_input", f"batch must contain 1 to {limit} items")
    return len(items)


def validate_vectors(raw: object, *, count: int, space: EmbeddingSpaceRef) -> tuple[Vector, ...]:
    """Reject wrong count/dimension, non-finite, zero or non-normalized output."""
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != count:
        raise MultimodalEmbeddingError("invalid_output", "embedding count mismatch")
    vectors: list[Vector] = []
    for row in raw:
        if (
            not isinstance(row, Sequence)
            or isinstance(row, (str, bytes))
            or len(row) != space.dimension
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in row)
        ):
            raise MultimodalEmbeddingError("invalid_output", "embedding dimension mismatch")
        vector = tuple(float(value) for value in row)
        if not all(math.isfinite(value) for value in vector):
            raise MultimodalEmbeddingError("invalid_output", "embedding contains non-finite values")
        norm = math.sqrt(math.fsum(value * value for value in vector))
        if norm == 0.0:
            raise MultimodalEmbeddingError("invalid_output", "embedding has zero norm")
        if abs(norm - 1.0) > UNIT_NORM_TOLERANCE:
            raise MultimodalEmbeddingError("invalid_output", "embedding is not unit normalized")
        vectors.append(vector)
    return tuple(vectors)


def _bounded_text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_TEXT_CHARACTERS:
        raise ValueError(
            f"{name} must be non-blank text of at most {MAX_TEXT_CHARACTERS} characters"
        )


__all__ = [
    "MAX_BATCH_ITEMS",
    "MultimodalEmbeddingError",
    "PcmAudio",
    "SharedSpaceEncoder",
    "TextDocument",
    "Vector",
    "require_batch",
    "validate_vectors",
]
