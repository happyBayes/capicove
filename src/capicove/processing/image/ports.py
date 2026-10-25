"""Capability-owned provider ports for image processing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from PIL import Image

from capicove.contracts.retrieval import EmbeddingSpaceRef


if TYPE_CHECKING:
    from capicove.processing.image.mappers.remove_background import BackgroundRemovalOptions
    from capicove.processing.image.ocr import OcrProviderOutput


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ImageSemanticLane(StrEnum):
    """Identify one build-side semantic image-retrieval lane."""

    INSTANCE = "instance"
    CONCEPT = "concept"


@dataclass(frozen=True, slots=True)
class SemanticTargetIdentity:
    """Bind a target encoder to one lane, vector space, and verified bundle."""

    lane: ImageSemanticLane
    space: EmbeddingSpaceRef
    bundle_manifest_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.lane, ImageSemanticLane):
            raise TypeError("lane must be an ImageSemanticLane")
        if not isinstance(self.space, EmbeddingSpaceRef):
            raise TypeError("space must be an EmbeddingSpaceRef")
        if _SHA256_PATTERN.fullmatch(self.bundle_manifest_sha256) is None:
            raise ValueError("bundle_manifest_sha256 must be a lowercase SHA-256 digest")


@runtime_checkable
class ImageTargetEncoder(Protocol):
    """Encode ordered image batches into one immutable semantic target space."""

    identity: SemanticTargetIdentity

    def verify_ready(self, *, deadline: datetime) -> None:
        """Verify the complete offline bundle before any build-side source I/O."""
        ...

    def encode_images(
        self,
        images: tuple[Image.Image, ...],
        space: EmbeddingSpaceRef,
        deadline: datetime,
    ) -> tuple[tuple[float, ...], ...]:
        """Encode a bounded ordered target batch in the declared space."""
        ...


class BackgroundRemovalProvider(Protocol):
    """Remove image backgrounds without owning artifact persistence."""

    @property
    def provider_id(self) -> str:
        """Return stable provider identity."""
        ...

    @property
    def provider_version(self) -> str | None:
        """Return the provider version when available."""
        ...

    @property
    def model_id(self) -> str | None:
        """Return the explicitly provisioned local model identity."""
        ...

    @property
    def model_revision(self) -> str | None:
        """Return the immutable local model revision when available."""
        ...

    @property
    def model_manifest_sha256(self) -> str | None:
        """Return the trusted digest of the installed model file manifest."""
        ...

    @property
    def preprocess_revision(self) -> str | None: ...

    @property
    def device(self) -> str | None: ...

    def remove(self, image: Image.Image, options: BackgroundRemovalOptions) -> Image.Image:
        """Return a foreground image, normally with an alpha channel."""
        ...


class ImageScorer(Protocol):
    """Score an ordered batch of decoded images."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str | None: ...

    @property
    def model_id(self) -> str | None: ...

    @property
    def model_revision(self) -> str | None: ...

    @property
    def model_manifest_sha256(self) -> str | None: ...

    @property
    def preprocess_revision(self) -> str | None: ...

    @property
    def device(self) -> str | None: ...

    def score(self, images: tuple[Image.Image, ...]) -> tuple[float, ...]:
        """Return one finite score per image."""
        ...


class ImageTextScorer(Protocol):
    """Score ordered images against one text value."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str | None: ...

    @property
    def model_id(self) -> str | None: ...

    @property
    def model_revision(self) -> str | None: ...

    @property
    def model_manifest_sha256(self) -> str | None: ...

    @property
    def preprocess_revision(self) -> str | None: ...

    @property
    def device(self) -> str | None: ...

    def score(self, images: tuple[Image.Image, ...], text: str) -> tuple[float, ...]:
        """Return one finite score per image."""
        ...


class ImagePairScorer(Protocol):
    """Score the visual similarity of exactly two images."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str | None: ...

    @property
    def model_id(self) -> str | None: ...

    @property
    def model_revision(self) -> str | None: ...

    @property
    def model_manifest_sha256(self) -> str | None: ...

    @property
    def preprocess_revision(self) -> str | None: ...

    @property
    def device(self) -> str | None: ...

    def score(self, left: Image.Image, right: Image.Image) -> float:
        """Return one finite image-pair score."""
        ...


class DifferenceCaptionProvider(Protocol):
    """Generate structured region and paired-difference captions."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str | None: ...

    @property
    def model_id(self) -> str | None: ...

    @property
    def model_revision(self) -> str | None: ...

    @property
    def model_manifest_sha256(self) -> str | None: ...

    @property
    def preprocess_revision(self) -> str | None: ...

    @property
    def device(self) -> str | None: ...

    def caption(self, image: Image.Image) -> str:
        """Caption one cropped region."""
        ...

    def describe_difference(
        self,
        left: Image.Image,
        right: Image.Image,
        left_caption: str,
        right_caption: str,
    ) -> str:
        """Describe the visible change between paired crops."""
        ...


class ImageOcrProvider(Protocol):
    """Recognize structured text in one decoded image."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str | None: ...

    def recognize(
        self,
        image: Image.Image,
        languages: tuple[str, ...],
    ) -> OcrProviderOutput:
        """Return ordered recognized words for one image."""
        ...
