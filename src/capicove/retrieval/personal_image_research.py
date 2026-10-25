"""Internal-only personal reverse-image research vertical.

This module is deliberately not exported from :mod:`capicove.retrieval`. Its exact
module path remains importable from the wheel, but it is an unsupported research
surface rather than a public contract or production capability claim.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from numbers import Real
from typing import Protocol

from PIL import Image

from capicove.contracts.retrieval import (
    DistanceMetric,
    EmbeddingSpaceRef,
    QueryModality,
    TargetType,
    VectorNormalization,
)
from capicove.retrieval._adapters.opencv_phash import (
    MAX_RESEARCH_IMAGE_PIXELS,
    OPENCV_PHASH_64_V1_SPACE,
    OpenCvQueryPhashAdapter,
    ResearchPhashSpaceIdentity,
    ResearchPhashValue,
)


RESEARCH_CAPABILITY_STATUS = "internal-research-only"
_MAX_PERSONAL_UNITS = 10_000
_MAX_RESEARCH_VECTOR_DIMENSION = 16_384
_MAX_PHASH_DISTANCE = 2
_RRF_K = 60
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_OPAQUE_UNIT_ID_PATTERN = re.compile(r"^unit:[0-9a-f]{64}$")


class ResearchSearchIntent(StrEnum):
    """Carry caller-declared routing for the internal research-only search path."""

    INSTANCE = "instance"
    CONCEPT = "concept"
    BALANCED = "balanced"


class ResearchSearchDeadlineExceeded(TimeoutError):
    """Raised when an internal research search reaches its absolute deadline."""


@dataclass(frozen=True, slots=True)
class ResearchSemanticVector:
    """Bind one immutable L2 vector to its complete embedding-space identity."""

    space: EmbeddingSpaceRef
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.space, EmbeddingSpaceRef):
            raise TypeError("space must be an EmbeddingSpaceRef")
        _require_research_space(self.space)
        values = tuple(self.values)
        if len(values) != self.space.dimension:
            raise ValueError("vector dimension must match its complete space identity")
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
            raise ValueError("vector values must be finite numbers")
        try:
            normalized = tuple(float(value) for value in values)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError("vector values must be finite numbers") from error
        if any(not math.isfinite(value) for value in normalized):
            raise ValueError("vector values must be finite numbers")
        norm_squared = sum(value * value for value in normalized)
        if not math.isclose(norm_squared, 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError("vector must be L2-normalized")
        object.__setattr__(self, "values", normalized)


@dataclass(frozen=True, slots=True)
class PersonalImageResearchUnit:
    """Hold one immutable target used only by the personal research index."""

    unit_id: str
    source_sha256: str
    phash_space: ResearchPhashSpaceIdentity
    phash: int
    instance_vector: ResearchSemanticVector
    concept_vector: ResearchSemanticVector

    def __post_init__(self) -> None:
        _require_opaque_unit_id(self.unit_id)
        _require_sha256("source_sha256", self.source_sha256)
        if not isinstance(self.phash_space, ResearchPhashSpaceIdentity):
            raise TypeError("phash_space must be a ResearchPhashSpaceIdentity")
        _require_uint64("phash", self.phash)
        if not isinstance(self.instance_vector, ResearchSemanticVector):
            raise TypeError("instance_vector must be a ResearchSemanticVector")
        if not isinstance(self.concept_vector, ResearchSemanticVector):
            raise TypeError("concept_vector must be a ResearchSemanticVector")


@dataclass(frozen=True, slots=True)
class PersonalImageResearchHit:
    """Project bounded evidence without source paths or raw vectors.

    ``exact_content`` is the primary ordering class. ``rrf_score`` orders hits only
    within the non-exact class and is not a complete sortable relevance score.
    """

    unit_id: str
    exact_content: bool
    phash_distance: int | None
    semantic_lanes: tuple[ResearchSearchIntent, ...]
    rrf_score: float


@dataclass(frozen=True, slots=True)
class PersonalImageResearchResult:
    """Return an explicitly non-public internal research result."""

    intent: ResearchSearchIntent
    hits: tuple[PersonalImageResearchHit, ...]
    capability_status: str = field(init=False, default=RESEARCH_CAPABILITY_STATUS)
    research_only: bool = field(init=False, default=True)


class ResearchImageEncoder(Protocol):
    """Encode a resolved image and cooperatively honor the absolute deadline."""

    space: EmbeddingSpaceRef

    def encode(
        self,
        image: Image.Image,
        *,
        deadline: datetime,
    ) -> ResearchSemanticVector: ...


class ResearchQueryPhashProvider(Protocol):
    """Produce the retrieval-owned query-side ``phash-64-v1`` value."""

    space: ResearchPhashSpaceIdentity

    def phash(self, image: Image.Image) -> ResearchPhashValue: ...


class PersonalImageResearchIndex:
    """Search at most 10,000 immutable targets for internal research only."""

    def __init__(
        self,
        units: tuple[PersonalImageResearchUnit, ...],
        *,
        phash_space: ResearchPhashSpaceIdentity = OPENCV_PHASH_64_V1_SPACE,
        instance_space: EmbeddingSpaceRef,
        concept_space: EmbeddingSpaceRef,
        instance_encoder: ResearchImageEncoder,
        concept_encoder: ResearchImageEncoder,
        phash_provider: ResearchQueryPhashProvider | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        materialized = tuple(units)
        if len(materialized) > _MAX_PERSONAL_UNITS:
            raise ValueError("personal research index supports at most 10,000 units")
        if any(not isinstance(unit, PersonalImageResearchUnit) for unit in materialized):
            raise TypeError("units must contain PersonalImageResearchUnit values")
        unit_ids = tuple(unit.unit_id for unit in materialized)
        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError("unit_id values must be unique")

        if not isinstance(phash_space, ResearchPhashSpaceIdentity):
            raise TypeError("phash_space must be a ResearchPhashSpaceIdentity")
        _require_research_space(instance_space)
        _require_research_space(concept_space)
        if instance_space == concept_space:
            raise ValueError("instance and concept spaces must be distinct complete identities")
        _require_encoder_space("instance", instance_encoder, instance_space)
        _require_encoder_space("concept", concept_encoder, concept_space)

        for unit in materialized:
            if unit.phash_space != phash_space:
                raise ValueError("unit target phash space must match the index phash space")
            if unit.instance_vector.space != instance_space:
                raise ValueError("unit instance space must match the index instance space")
            if unit.concept_vector.space != concept_space:
                raise ValueError("unit concept space must match the index concept space")

        resolved_phash = OpenCvQueryPhashAdapter() if phash_provider is None else phash_provider
        if getattr(resolved_phash, "space", None) != phash_space:
            raise ValueError("query phash space must match the index phash space")
        self._units = tuple(sorted(materialized, key=lambda unit: unit.unit_id))
        self._phash_space = phash_space
        self._instance_space = instance_space
        self._concept_space = concept_space
        self._instance_encoder = instance_encoder
        self._concept_encoder = concept_encoder
        self._phash_provider = resolved_phash
        self._clock = _utc_now if clock is None else clock

    def search_image(
        self,
        image: Image.Image,
        *,
        source_sha256: str,
        intent: ResearchSearchIntent,
        deadline: datetime,
        top_k: int = 20,
    ) -> PersonalImageResearchResult:
        """Search one resolved image without exposing model or index controls.

        ``source_sha256`` is immutable evidence produced by the trusted caller that
        resolved ``image``. This bounded research index does not accept paths,
        URLs, model identifiers, or provider parameters.
        """
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image")
        if image.width <= 0 or image.height <= 0:
            raise ValueError("research image dimensions must be positive")
        if image.width * image.height > MAX_RESEARCH_IMAGE_PIXELS:
            raise ValueError(
                f"image exceeds the {MAX_RESEARCH_IMAGE_PIXELS:,}-pixel research limit"
            )
        _require_sha256("source_sha256", source_sha256)
        if not isinstance(intent, ResearchSearchIntent):
            raise TypeError("intent must be a ResearchSearchIntent")
        _require_aware_datetime("deadline", deadline)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise ValueError("top_k must be an integer from 1 to 100")

        self._check_deadline(deadline)
        exact_ids = {unit.unit_id for unit in self._units if unit.source_sha256 == source_sha256}
        self._check_deadline(deadline)

        self._check_deadline(deadline)
        query_phash = self._phash_provider.phash(image)
        if not isinstance(query_phash, ResearchPhashValue):
            raise TypeError("query phash provider must return a ResearchPhashValue")
        if query_phash.space != self._phash_space:
            raise ValueError("query phash space must match the index phash space")
        query_phash_value = query_phash.value
        phash_observations = sorted(
            (
                ((query_phash_value ^ unit.phash).bit_count(), unit.unit_id)
                for unit in self._units
                if (query_phash_value ^ unit.phash).bit_count() <= _MAX_PHASH_DISTANCE
            ),
            key=lambda item: (item[0], item[1]),
        )
        phash_distances = {unit_id: distance for distance, unit_id in phash_observations}
        phash_ranks = {
            unit_id: rank for rank, (_distance, unit_id) in enumerate(phash_observations, start=1)
        }
        self._check_deadline(deadline)

        selected = self._selected_semantic_lanes(intent)
        semantic_ranks: dict[ResearchSearchIntent, dict[str, int]] = {}
        for lane, encoder, expected_space, vector_field in selected:
            self._check_deadline(deadline)
            query_vector = encoder.encode(image, deadline=deadline)
            if not isinstance(query_vector, ResearchSemanticVector):
                raise TypeError("encoder output must be a ResearchSemanticVector")
            if query_vector.space != expected_space:
                raise ValueError("encoder output space must match its selected index space")
            ranked = sorted(
                self._units,
                key=lambda unit: (
                    -_dot(query_vector.values, getattr(unit, vector_field).values),
                    unit.unit_id,
                ),
            )
            semantic_ranks[lane] = {unit.unit_id: rank for rank, unit in enumerate(ranked, start=1)}
            self._check_deadline(deadline)

        rrf_scores = {unit.unit_id: 0.0 for unit in self._units}
        for unit_id, rank in phash_ranks.items():
            rrf_scores[unit_id] += 2.0 / (_RRF_K + rank)
        for ranks in semantic_ranks.values():
            for unit_id, rank in ranks.items():
                rrf_scores[unit_id] += 1.0 / (_RRF_K + rank)

        ranked_units = sorted(
            self._units,
            key=lambda unit: _final_sort_key(
                unit.unit_id,
                exact_ids=exact_ids,
                rrf_scores=rrf_scores,
            ),
        )
        self._check_deadline(deadline)
        selected_lanes = tuple(lane for lane, *_rest in selected)
        hits = tuple(
            PersonalImageResearchHit(
                unit_id=unit.unit_id,
                exact_content=unit.unit_id in exact_ids,
                phash_distance=phash_distances.get(unit.unit_id),
                semantic_lanes=selected_lanes,
                rrf_score=rrf_scores[unit.unit_id],
            )
            for unit in ranked_units[:top_k]
        )
        self._check_deadline(deadline)
        return PersonalImageResearchResult(intent=intent, hits=hits)

    def _selected_semantic_lanes(
        self,
        intent: ResearchSearchIntent,
    ) -> tuple[
        tuple[ResearchSearchIntent, ResearchImageEncoder, EmbeddingSpaceRef, str],
        ...,
    ]:
        instance = (
            ResearchSearchIntent.INSTANCE,
            self._instance_encoder,
            self._instance_space,
            "instance_vector",
        )
        concept = (
            ResearchSearchIntent.CONCEPT,
            self._concept_encoder,
            self._concept_space,
            "concept_vector",
        )
        if intent is ResearchSearchIntent.INSTANCE:
            return (instance,)
        if intent is ResearchSearchIntent.CONCEPT:
            return (concept,)
        if intent is ResearchSearchIntent.BALANCED:
            return (instance, concept)
        raise ValueError("unsupported research search intent")

    def _check_deadline(self, deadline: datetime) -> None:
        now = self._clock()
        _require_aware_datetime("clock result", now)
        if now >= deadline:
            raise ResearchSearchDeadlineExceeded("personal image research deadline exceeded")


def _require_research_space(space: EmbeddingSpaceRef) -> None:
    if not isinstance(space, EmbeddingSpaceRef):
        raise TypeError("research space must be an EmbeddingSpaceRef")
    if space.normalization is not VectorNormalization.L2:
        raise ValueError("research space normalization must be L2")
    if space.dimension > _MAX_RESEARCH_VECTOR_DIMENSION:
        raise ValueError("research space dimension must be at most 16,384")
    if space.distance_metric not in (DistanceMetric.INNER_PRODUCT, DistanceMetric.COSINE):
        raise ValueError("research space metric must be inner product or cosine")
    if not space.accepts(QueryModality.IMAGE, TargetType.IMAGE):
        raise ValueError("research space must support image queries and image targets")


def _require_encoder_space(
    lane: str,
    encoder: ResearchImageEncoder,
    expected: EmbeddingSpaceRef,
) -> None:
    if getattr(encoder, "space", None) != expected:
        raise ValueError(f"{lane} encoder space must match the complete index space")


def _require_non_blank(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")


def _require_opaque_unit_id(value: object) -> None:
    if not isinstance(value, str) or _OPAQUE_UNIT_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("unit_id must be an opaque unit identifier in unit:<sha256> form")


def _require_sha256(field_name: str, value: object) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hash")


def _require_uint64(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**64:
        raise ValueError(f"{field_name} must be an unsigned 64-bit integer")


def _require_aware_datetime(field_name: str, value: object) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _final_sort_key(
    unit_id: str,
    *,
    exact_ids: set[str],
    rrf_scores: dict[str, float],
) -> tuple[int, float, str]:
    if unit_id in exact_ids:
        return (0, 0.0, unit_id)
    return (1, -rrf_scores[unit_id], unit_id)


def _utc_now() -> datetime:
    return datetime.now(UTC)
