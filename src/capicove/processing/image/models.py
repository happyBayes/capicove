"""Immutable contracts shared by image-processing capabilities."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum

from capicove.processing.image._validation import require_non_blank, require_sha256


ParameterValue = str | int | float | bool | None
OperationParameters = tuple[tuple[str, ParameterValue], ...]
_OCR_LANGUAGE_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


class Aggregation(StrEnum):
    """Combine decisions across an ordered collection of images."""

    ALL = "all"
    ANY = "any"


@dataclass(frozen=True, slots=True)
class ImageSource:
    """Identify one image using a local path or in-process bytes."""

    path: str | None = None
    data: bytes | None = None
    media_type: str | None = None

    def __post_init__(self) -> None:
        if (self.path is None) == (self.data is None):
            raise ValueError("image source requires exactly one representation")
        if self.path is not None:
            require_non_blank("path", self.path)
        if self.data is not None and not self.data:
            raise ValueError("image data must not be empty")
        if self.media_type is not None:
            require_non_blank("media_type", self.media_type)


@dataclass(frozen=True, slots=True)
class ImageMetric:
    """Record ordered metric values with provider and model lineage."""

    name: str
    values: tuple[float, ...]
    provider_id: str
    provider_version: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    preprocess_revision: str | None = None
    device: str | None = None
    model_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        require_non_blank("name", self.name)
        require_non_blank("provider_id", self.provider_id)
        for value in self.values:
            if not math.isfinite(value):
                raise ValueError("metric values must be finite")
        for field_name in (
            "provider_version",
            "model_id",
            "model_revision",
            "preprocess_revision",
            "device",
        ):
            value = getattr(self, field_name)
            if value is not None:
                require_non_blank(field_name, value)
        if self.model_manifest_sha256 is not None:
            require_sha256("model_manifest_sha256", self.model_manifest_sha256)


@dataclass(frozen=True, slots=True)
class FilterDecision:
    """Describe an auditable image-filter keep decision."""

    filter_name: str
    keep: bool
    applicable: bool
    aggregation: Aggregation
    source_hashes: tuple[str, ...]
    metrics: tuple[ImageMetric, ...]
    operation_version: int = 1
    parameters: OperationParameters = ()

    def __post_init__(self) -> None:
        require_non_blank("filter_name", self.filter_name)
        object.__setattr__(self, "aggregation", Aggregation(self.aggregation))
        _validate_operation_metadata(self.operation_version, self.parameters)
        for source_hash in self.source_hashes:
            require_sha256("source_hash", source_hash)
        if not self.applicable and (self.source_hashes or self.metrics or not self.keep):
            raise ValueError("a not-applicable decision must retain the sample without metrics")


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Describe one axis-aligned pixel-space image region."""

    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height, self.confidence)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("bounding-box values must be finite")
        if self.x < 0 or self.y < 0:
            raise ValueError("bounding-box coordinates must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("bounding-box width and height must be positive")
        if not 0 <= self.confidence <= 1:
            raise ValueError("bounding-box confidence must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class OcrWord:
    """Describe one recognized word and its Tesseract layout hierarchy."""

    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    page: int
    block: int
    paragraph: int
    line: int
    word: int

    def __post_init__(self) -> None:
        require_non_blank("text", self.text)
        integer_values = (
            self.x,
            self.y,
            self.width,
            self.height,
            self.page,
            self.block,
            self.paragraph,
            self.line,
            self.word,
        )
        if any(type(value) is not int for value in integer_values):
            raise ValueError("OCR word geometry and hierarchy values must be integers")
        if self.x < 0 or self.y < 0:
            raise ValueError("OCR word coordinates must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("OCR word dimensions must be positive")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("OCR word confidence must be within [0, 1]")
        if any(value <= 0 for value in self.hierarchy):
            raise ValueError("OCR word hierarchy values must be positive")

    @property
    def hierarchy(self) -> tuple[int, int, int, int, int]:
        """Return page-to-word hierarchy identifiers in reading order."""
        return self.page, self.block, self.paragraph, self.line, self.word


@dataclass(frozen=True, slots=True)
class ImageOcrResult:
    """Record structured OCR text with source and provider lineage."""

    source_hash: str
    full_text: str
    words: tuple[OcrWord, ...]
    provider_id: str
    provider_version: str | None
    languages: tuple[str, ...]
    operation_version: int = 1
    parameters: OperationParameters = ()

    def __post_init__(self) -> None:
        require_sha256("source_hash", self.source_hash)
        require_non_blank("provider_id", self.provider_id)
        if self.provider_version is not None:
            require_non_blank("provider_version", self.provider_version)
        if not 1 <= len(self.languages) <= 8:
            raise ValueError("OCR languages must contain between one and eight values")
        if len(set(self.languages)) != len(self.languages):
            raise ValueError("OCR languages must be unique")
        for language in self.languages:
            if _OCR_LANGUAGE_PATTERN.fullmatch(language) is None:
                raise ValueError("OCR languages contain an invalid identifier")
        _validate_operation_metadata(self.operation_version, self.parameters)


@dataclass(frozen=True, slots=True)
class DifferenceStageEvidence:
    """Record one deterministic difference-pipeline stage outcome."""

    stage: str
    score: float
    retained: bool
    reason: str

    def __post_init__(self) -> None:
        require_non_blank("stage", self.stage)
        require_non_blank("reason", self.reason)
        if not math.isfinite(self.score) or self.score < 0:
            raise ValueError("difference stage score must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class DerivedImage:
    """Describe one immutable image produced from source evidence."""

    path: str
    sha256: str
    media_type: str
    width: int
    height: int
    byte_size: int
    source_hashes: tuple[str, ...]
    operation: str
    provider_id: str
    provider_version: str | None = None
    operation_version: int = 1
    parameters: OperationParameters = ()
    model_id: str | None = None
    model_revision: str | None = None
    preprocess_revision: str | None = None
    device: str | None = None
    model_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        require_non_blank("path", self.path)
        require_sha256("sha256", self.sha256)
        require_non_blank("media_type", self.media_type)
        require_non_blank("operation", self.operation)
        require_non_blank("provider_id", self.provider_id)
        _validate_operation_metadata(self.operation_version, self.parameters)
        if self.width <= 0 or self.height <= 0 or self.byte_size <= 0:
            raise ValueError("derived image dimensions and byte size must be positive")
        for source_hash in self.source_hashes:
            require_sha256("source_hash", source_hash)
        for field_name in (
            "provider_version",
            "model_id",
            "model_revision",
            "preprocess_revision",
            "device",
        ):
            value = getattr(self, field_name)
            if value is not None:
                require_non_blank(field_name, value)
        if self.model_manifest_sha256 is not None:
            require_sha256("model_manifest_sha256", self.model_manifest_sha256)


@dataclass(frozen=True, slots=True)
class DifferenceAreaResult:
    """Record deterministic difference regions for an ordered image pair."""

    source_hashes: tuple[str, str]
    boxes: tuple[BoundingBox, ...]
    provider_id: str
    provider_version: str | None = None
    operation_version: int = 1
    parameters: OperationParameters = ()
    stage_evidence: tuple[DifferenceStageEvidence, ...] = ()

    def __post_init__(self) -> None:
        for source_hash in self.source_hashes:
            require_sha256("source_hash", source_hash)
        require_non_blank("provider_id", self.provider_id)
        _validate_operation_metadata(self.operation_version, self.parameters)
        if self.provider_version is not None:
            require_non_blank("provider_version", self.provider_version)


@dataclass(frozen=True, slots=True)
class DifferenceCaptionRegion:
    """Record structured captions for one paired difference region."""

    box: BoundingBox
    left_caption: str
    right_caption: str
    difference_caption: str
    left_match_score: float
    right_match_score: float
    caption_difference_score: float

    def __post_init__(self) -> None:
        require_non_blank("left_caption", self.left_caption)
        require_non_blank("right_caption", self.right_caption)
        require_non_blank("difference_caption", self.difference_caption)
        if not -1 <= self.left_match_score <= 1:
            raise ValueError("left_match_score must be within [-1, 1]")
        if not -1 <= self.right_match_score <= 1:
            raise ValueError("right_match_score must be within [-1, 1]")
        if not 0 <= self.caption_difference_score <= 1:
            raise ValueError("caption_difference_score must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class DifferenceCaptionResult:
    """Record structured difference captions and provider lineage."""

    source_hashes: tuple[str, str]
    regions: tuple[DifferenceCaptionRegion, ...]
    provider_id: str
    provider_version: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    preprocess_revision: str | None = None
    device: str | None = None
    model_manifest_sha256: str | None = None
    operation_version: int = 1
    parameters: OperationParameters = ()

    def __post_init__(self) -> None:
        for source_hash in self.source_hashes:
            require_sha256("source_hash", source_hash)
        require_non_blank("provider_id", self.provider_id)
        _validate_operation_metadata(self.operation_version, self.parameters)
        for field_name in (
            "provider_version",
            "model_id",
            "model_revision",
            "preprocess_revision",
            "device",
        ):
            value = getattr(self, field_name)
            if value is not None:
                require_non_blank(field_name, value)
        if self.model_manifest_sha256 is not None:
            require_sha256("model_manifest_sha256", self.model_manifest_sha256)


def _validate_operation_metadata(
    operation_version: int,
    parameters: OperationParameters,
) -> None:
    if (
        not isinstance(operation_version, int)
        or isinstance(operation_version, bool)
        or operation_version <= 0
    ):
        raise ValueError("operation_version must be a positive integer")
    keys: set[str] = set()
    for key, value in parameters:
        require_non_blank("parameter name", key)
        if key in keys:
            raise ValueError("operation parameter names must be unique")
        keys.add(key)
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise ValueError("operation parameter values must be scalar")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("operation parameter values must be finite")
