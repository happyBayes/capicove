"""Retrieval-owned OpenCV adapter for the internal query pHash lane."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import cv2
import numpy as np
from PIL import Image

from capicove.contracts.retrieval import HashSpaceRef, TargetType
from capicove.retrieval.search_models import PerceptualHashQuery


MAX_RESEARCH_IMAGE_PIXELS = 40_000_000


@dataclass(frozen=True, slots=True)
class ResearchPhashSpaceIdentity:
    """Bind compatible research hashes to one complete algorithm identity."""

    space_id: str
    provider_id: str
    provider_version: str
    preprocessing_version: str
    algorithm: str
    bit_width: int
    hash_version: str

    def __post_init__(self) -> None:
        for field_name in (
            "space_id",
            "provider_id",
            "provider_version",
            "preprocessing_version",
            "algorithm",
            "hash_version",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-blank string")
        if (
            isinstance(self.bit_width, bool)
            or not isinstance(self.bit_width, int)
            or self.bit_width != 64
        ):
            raise ValueError("research pHash identity must have a 64-bit width")


OPENCV_PHASH_64_V1_SPACE = ResearchPhashSpaceIdentity(
    space_id="opencv-phash-64-v1",
    provider_id="opencv-phash",
    provider_version=cv2.__version__,
    preprocessing_version="pillow-luma-opencv-area32-dct8-median-excluding-dc-v1",
    algorithm="phash",
    bit_width=64,
    hash_version="phash-64-v1",
)


OPENCV_PHASH_HASH_SPACE = HashSpaceRef(
    space_id=OPENCV_PHASH_64_V1_SPACE.space_id,
    provider_id=OPENCV_PHASH_64_V1_SPACE.provider_id,
    provider_version=OPENCV_PHASH_64_V1_SPACE.provider_version,
    preprocessing_version=OPENCV_PHASH_64_V1_SPACE.preprocessing_version,
    algorithm=OPENCV_PHASH_64_V1_SPACE.algorithm,
    bit_width=OPENCV_PHASH_64_V1_SPACE.bit_width,
    hash_version=OPENCV_PHASH_64_V1_SPACE.hash_version,
    target_modalities=frozenset({TargetType.IMAGE}),
)


@dataclass(frozen=True, slots=True)
class ResearchPhashValue:
    """Carry one pHash value together with the space that gives it meaning."""

    space: ResearchPhashSpaceIdentity
    value: int

    def __post_init__(self) -> None:
        if not isinstance(self.space, ResearchPhashSpaceIdentity):
            raise TypeError("space must be a ResearchPhashSpaceIdentity")
        if (
            isinstance(self.value, bool)
            or not isinstance(self.value, int)
            or not 0 <= self.value < 2**64
        ):
            raise ValueError("pHash value must be an unsigned 64-bit integer")


class OpenCvQueryPhashAdapter:
    """Compute the retrieval query-side ``phash-64-v1`` identity."""

    space = OPENCV_PHASH_64_V1_SPACE
    provider_id = space.provider_id
    provider_version = space.provider_version
    hash_version = space.hash_version

    def phash(self, image: Image.Image) -> ResearchPhashValue:
        """Compute one unsigned 64-bit pHash from an already resolved image."""
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image")
        if image.width <= 0 or image.height <= 0:
            raise ValueError("research image dimensions must be positive")
        if image.width * image.height > MAX_RESEARCH_IMAGE_PIXELS:
            raise ValueError(
                f"image exceeds the {MAX_RESEARCH_IMAGE_PIXELS:,}-pixel research limit"
            )
        grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
        resized = cv2.resize(grayscale, (32, 32), interpolation=cv2.INTER_AREA)
        coefficients = cv2.dct(np.asarray(resized, dtype=np.float32))[:8, :8]
        median = float(np.median(coefficients.reshape(-1)[1:]))
        value = 0
        for coefficient in coefficients.reshape(-1):
            value = (value << 1) | int(float(coefficient) > median)
        return ResearchPhashValue(space=self.space, value=value)


class OpenCvPerceptualQueryEncoder:
    """Compute the production ``phash-64-v1`` query representation."""

    def __init__(self, hash_space: HashSpaceRef = OPENCV_PHASH_HASH_SPACE) -> None:
        if not isinstance(hash_space, HashSpaceRef):
            raise TypeError("hash_space must be a HashSpaceRef")
        if hash_space != OPENCV_PHASH_HASH_SPACE:
            raise ValueError("hash_space must match the complete OpenCV pHash identity")
        self.hash_space = hash_space
        self._adapter = OpenCvQueryPhashAdapter()

    def encode(
        self,
        image: Image.Image,
        source_sha256: str,
        max_hamming_distance: int,
        deadline: datetime,
    ) -> PerceptualHashQuery:
        """Hash one already authorized image under an absolute deadline."""
        _require_future_deadline(deadline)
        value = self._adapter.phash(image).value
        _require_future_deadline(deadline)
        return PerceptualHashQuery(
            source_sha256=source_sha256,
            space=self.hash_space,
            value=value,
            max_hamming_distance=max_hamming_distance,
        )


def _require_future_deadline(deadline: datetime) -> None:
    if (
        not isinstance(deadline, datetime)
        or deadline.tzinfo is None
        or deadline.utcoffset() is None
    ):
        raise ValueError("deadline must be timezone-aware")
    if datetime.now(UTC) >= deadline:
        raise TimeoutError("reverse-image query deadline exceeded")
