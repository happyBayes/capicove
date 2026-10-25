"""Deterministic pHash grouping for visually similar training images."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from PIL import Image

from capicove.processing.image._validation import require_non_blank
from capicove.processing.image.deduplication.lsh import HashRecord, candidate_pairs
from capicove.processing.image.invocation import check_image_deadline
from capicove.processing.image.io import load_image
from capicove.processing.image.models import ImageSource


class PerceptualHashProvider(Protocol):
    """Produce a versioned 64-bit perceptual image hash."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str | None: ...

    @property
    def hash_version(self) -> str: ...

    def phash(self, image: Image.Image) -> int:
        """Return a non-negative 64-bit perceptual hash."""
        ...


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """Identify one training record and its ordered image collection."""

    record_id: str
    images: tuple[ImageSource, ...]
    text: str | None = None

    def __post_init__(self) -> None:
        require_non_blank("record_id", self.record_id)
        if self.text is not None:
            require_non_blank("text", self.text)


@dataclass(frozen=True, slots=True)
class DuplicateGroup:
    """Describe one visual near-duplicate component and deterministic winner."""

    kept_record_id: str
    duplicate_record_ids: tuple[str, ...]
    distances_from_kept: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class PerceptualDeduplicationResult:
    """Return non-destructive perceptual duplicate groups and retained IDs."""

    groups: tuple[DuplicateGroup, ...]
    retained_record_ids: tuple[str, ...]
    provider_id: str
    provider_version: str | None
    hash_version: str
    max_hamming_distance: int
    record_source_hashes: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True, slots=True)
class PhashRecord:
    """Expose one record's normalized pHashes and immutable source lineage."""

    record_id: str
    hashes: tuple[str, ...]
    source_hashes: tuple[str, ...]
    text_hash: str | None


@dataclass(frozen=True, slots=True)
class ImagePhashResult:
    """Return batch pHashes, duplicate groups, and candidate statistics."""

    records: tuple[PhashRecord, ...]
    groups: tuple[DuplicateGroup, ...]
    retained_record_ids: tuple[str, ...]
    provider_id: str
    provider_version: str | None
    hash_version: str
    bit_width: int
    max_hamming_distance: int
    candidate_strategy: str
    candidate_count: int
    comparison_count: int


class OpenCvPhashProvider:
    """Compute a standard 64-bit pHash with direct OpenCV primitives."""

    provider_id = "opencv-phash"
    provider_version = cv2.__version__
    hash_version = "phash-64-v1"

    def phash(self, image: Image.Image) -> int:
        """Compute pHash from the low-frequency DCT coefficients."""
        grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
        resized = cv2.resize(grayscale, (32, 32), interpolation=cv2.INTER_AREA)
        coefficients = cv2.dct(np.asarray(resized, dtype=np.float32))[:8, :8]
        median = float(np.median(coefficients.reshape(-1)[1:]))
        value = 0
        for coefficient in coefficients.reshape(-1):
            value = (value << 1) | int(float(coefficient) > median)
        return value


@dataclass(frozen=True, slots=True)
class PreparedImageRecord:
    """Carry one record's immutable pHash preparation result."""

    record_id: str
    image_hashes: tuple[int, ...]
    source_hashes: tuple[str, ...]
    text_hash: str | None


@dataclass(frozen=True, slots=True)
class _GroupingOutcome:
    groups: tuple[DuplicateGroup, ...]
    retained_record_ids: tuple[str, ...]
    candidate_strategy: str
    candidate_count: int
    comparison_count: int


class ImagePhash:
    """Compute public pHash records and group a bounded batch."""

    def __init__(
        self,
        max_hamming_distance: int = 0,
        hasher: PerceptualHashProvider | None = None,
        *,
        enable_lsh: bool = False,
    ) -> None:
        if not isinstance(enable_lsh, bool):
            raise ValueError("enable_lsh must be a boolean")
        self._deduplicator = ImagePerceptualDeduplicator(max_hamming_distance, hasher)
        self._max_hamming_distance = max_hamming_distance
        self._enable_lsh = enable_lsh

    def process(
        self,
        records: tuple[ImageRecord, ...],
        *,
        max_hamming_distance: int | None = None,
        enable_lsh: bool | None = None,
    ) -> ImagePhashResult:
        """Compute every pHash once and optionally accelerate batch grouping."""
        effective_distance = (
            self._max_hamming_distance if max_hamming_distance is None else max_hamming_distance
        )
        if (
            not isinstance(effective_distance, int)
            or isinstance(effective_distance, bool)
            or effective_distance < 0
            or effective_distance > 64
        ):
            raise ValueError("max_hamming_distance must be within [0, 64]")
        effective_lsh = self._enable_lsh if enable_lsh is None else enable_lsh
        if not isinstance(effective_lsh, bool):
            raise ValueError("enable_lsh must be a boolean")
        record_ids = tuple(record.record_id for record in records)
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("record_id values must be unique")
        prepared = tuple(self._deduplicator.prepare(record) for record in records)
        outcome = _group_prepared(
            prepared,
            effective_distance,
            enable_lsh=effective_lsh,
        )
        hasher = self._deduplicator.hasher
        return ImagePhashResult(
            records=tuple(
                PhashRecord(
                    record_id=record.record_id,
                    hashes=tuple(f"{value:016x}" for value in record.image_hashes),
                    source_hashes=record.source_hashes,
                    text_hash=record.text_hash,
                )
                for record in prepared
            ),
            groups=outcome.groups,
            retained_record_ids=outcome.retained_record_ids,
            provider_id=hasher.provider_id,
            provider_version=hasher.provider_version,
            hash_version=hasher.hash_version,
            bit_width=64,
            max_hamming_distance=effective_distance,
            candidate_strategy=outcome.candidate_strategy,
            candidate_count=outcome.candidate_count,
            comparison_count=outcome.comparison_count,
        )


class ImagePerceptualDeduplicator:
    """Group visually similar images when curating image training datasets.

    This capability uses pHash and Hamming distance. It intentionally detects
    visual near-duplicates and does not establish byte-exact image identity.
    """

    def __init__(
        self,
        max_hamming_distance: int = 0,
        hasher: PerceptualHashProvider | None = None,
    ) -> None:
        if (
            not isinstance(max_hamming_distance, int)
            or isinstance(max_hamming_distance, bool)
            or max_hamming_distance < 0
            or max_hamming_distance > 64
        ):
            raise ValueError("max_hamming_distance must be within [0, 64]")
        self._max_hamming_distance = max_hamming_distance
        self._hasher = hasher or OpenCvPhashProvider()

    def group(self, records: tuple[ImageRecord, ...]) -> PerceptualDeduplicationResult:
        """Group records with connected-component semantics and stable winners."""
        record_ids = tuple(record.record_id for record in records)
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("record_id values must be unique")
        return self.group_prepared(tuple(self.prepare(record) for record in records))

    @property
    def hasher(self) -> PerceptualHashProvider:
        """Return the configured provider for execution-backend parity."""
        return self._hasher

    def prepare(self, record: ImageRecord) -> PreparedImageRecord:
        """Decode and hash one record for local or distributed grouping."""
        check_image_deadline()
        loaded = tuple(load_image(source) for source in record.images)
        image_hashes = tuple(self._hasher.phash(item.image) for item in loaded)
        check_image_deadline()
        if any(value < 0 or value >= 2**64 for value in image_hashes):
            raise ValueError("perceptual hash provider must return unsigned 64-bit values")
        text_hash = (
            hashlib.sha256(record.text.encode("utf-8")).hexdigest()
            if record.text is not None
            else None
        )
        return PreparedImageRecord(
            record_id=record.record_id,
            image_hashes=image_hashes,
            source_hashes=tuple(item.sha256 for item in loaded),
            text_hash=text_hash,
        )

    def group_prepared(
        self,
        prepared: tuple[PreparedImageRecord, ...],
    ) -> PerceptualDeduplicationResult:
        """Group already-hashed records with identical local and Ray semantics."""
        record_ids = tuple(record.record_id for record in prepared)
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("record_id values must be unique")
        outcome = _group_prepared(
            prepared,
            self._max_hamming_distance,
            enable_lsh=False,
        )
        return PerceptualDeduplicationResult(
            groups=outcome.groups,
            retained_record_ids=outcome.retained_record_ids,
            provider_id=self._hasher.provider_id,
            provider_version=self._hasher.provider_version,
            hash_version=self._hasher.hash_version,
            max_hamming_distance=self._max_hamming_distance,
            record_source_hashes=tuple(
                sorted((record.record_id, record.source_hashes) for record in prepared)
            ),
        )


def _distance(left: PreparedImageRecord, right: PreparedImageRecord) -> int:
    if len(left.image_hashes) != len(right.image_hashes) or left.text_hash != right.text_hash:
        return 65
    return sum(
        (left_hash ^ right_hash).bit_count()
        for left_hash, right_hash in zip(left.image_hashes, right.image_hashes, strict=True)
    )


def _group_prepared(
    prepared: tuple[PreparedImageRecord, ...],
    max_hamming_distance: int,
    *,
    enable_lsh: bool,
) -> _GroupingOutcome:
    if enable_lsh:
        pair_result = candidate_pairs(
            tuple(
                HashRecord(record.record_id, record.image_hashes, record.text_hash)
                for record in prepared
            ),
            max_hamming_distance,
        )
        verified_pairs = pair_result.pairs
        strategy = "lsh" if prepared else pair_result.strategy
        candidate_count = pair_result.candidate_count
        comparison_count = pair_result.comparison_count
    else:
        verified_pairs, candidate_count = _exhaustive_pairs(
            prepared,
            max_hamming_distance,
        )
        strategy = "exhaustive"
        comparison_count = candidate_count
    groups, retained = _components(prepared, verified_pairs)
    return _GroupingOutcome(
        groups=groups,
        retained_record_ids=retained,
        candidate_strategy=strategy,
        candidate_count=candidate_count,
        comparison_count=comparison_count,
    )


def _exhaustive_pairs(
    prepared: tuple[PreparedImageRecord, ...],
    max_hamming_distance: int,
) -> tuple[tuple[tuple[str, str], ...], int]:
    verified: list[tuple[str, str]] = []
    comparison_count = 0
    for left_index, left in enumerate(prepared):
        check_image_deadline()
        if not left.image_hashes:
            continue
        for right in prepared[left_index + 1 :]:
            if (
                len(left.image_hashes) != len(right.image_hashes)
                or left.text_hash != right.text_hash
            ):
                continue
            comparison_count += 1
            if _distance(left, right) <= max_hamming_distance:
                ordered_ids = sorted((left.record_id, right.record_id))
                verified.append((ordered_ids[0], ordered_ids[1]))
    return tuple(sorted(verified)), comparison_count


def _components(
    prepared: tuple[PreparedImageRecord, ...],
    verified_pairs: tuple[tuple[str, str], ...],
) -> tuple[tuple[DuplicateGroup, ...], tuple[str, ...]]:
    parents = list(range(len(prepared)))
    indexes_by_id = {record.record_id: index for index, record in enumerate(prepared)}
    for left_id, right_id in verified_pairs:
        _union(parents, indexes_by_id[left_id], indexes_by_id[right_id])

    components: dict[int, list[int]] = {}
    for index in range(len(prepared)):
        root = _find(parents, index)
        components.setdefault(root, []).append(index)

    groups: list[DuplicateGroup] = []
    duplicate_ids: set[str] = set()
    for indexes in components.values():
        if len(indexes) <= 1 or not prepared[indexes[0]].image_hashes:
            continue
        ordered = sorted(indexes, key=lambda index: prepared[index].record_id)
        kept_index = ordered[0]
        duplicate_indexes = ordered[1:]
        duplicate_ids.update(prepared[index].record_id for index in duplicate_indexes)
        groups.append(
            DuplicateGroup(
                kept_record_id=prepared[kept_index].record_id,
                duplicate_record_ids=tuple(
                    prepared[index].record_id for index in duplicate_indexes
                ),
                distances_from_kept=tuple(
                    (
                        prepared[index].record_id,
                        _distance(prepared[kept_index], prepared[index]),
                    )
                    for index in duplicate_indexes
                ),
            )
        )
    groups.sort(key=lambda group: group.kept_record_id)
    retained = tuple(
        sorted(record.record_id for record in prepared if record.record_id not in duplicate_ids)
    )
    return tuple(groups), retained


def _find(parents: list[int], index: int) -> int:
    while parents[index] != index:
        parents[index] = parents[parents[index]]
        index = parents[index]
    return index


def _union(parents: list[int], left: int, right: int) -> None:
    left_root = _find(parents, left)
    right_root = _find(parents, right)
    if left_root != right_root:
        parents[max(left_root, right_root)] = min(left_root, right_root)
