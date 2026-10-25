"""Retrieval-owned immutable resource profiles and aliases."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from capicove.contracts.retrieval import SearchFailureCode
from capicove.retrieval.search_models import RetrievalLane, SearchFailure


@dataclass(frozen=True, slots=True)
class RetrievalProfile:
    """Describe inspectable, request-immutable retrieval resource policy."""

    profile_ref: str
    max_units: int
    semantic_formats: tuple[tuple[RetrievalLane, str], ...]
    phash_format: str
    phash_radius: int
    fusion_policy_id: str
    rrf_k: int
    lane_weights: tuple[tuple[RetrievalLane, float], ...]
    partial_results: bool
    candidate_depth_ceiling: int
    sparse_ordinal_limit: int
    nprobe_ceiling: int | None = None

    def __post_init__(self) -> None:
        _non_blank("profile_ref", self.profile_ref)
        _positive_int("max_units", self.max_units)
        formats = tuple(self.semantic_formats)
        if {lane for lane, _ in formats} != {RetrievalLane.INSTANCE, RetrievalLane.CONCEPT}:
            raise ValueError("semantic_formats must declare instance and concept exactly once")
        for _, format_name in formats:
            _non_blank("semantic_format", format_name)
        object.__setattr__(self, "semantic_formats", formats)
        _non_blank("phash_format", self.phash_format)
        if self.phash_radius != 2:
            raise ValueError("phash_radius must be 2")
        _non_blank("fusion_policy_id", self.fusion_policy_id)
        _positive_int("rrf_k", self.rrf_k)
        weights = tuple(self.lane_weights)
        if {lane for lane, _ in weights} != set(RetrievalLane):
            raise ValueError("lane_weights must declare every retrieval lane exactly once")
        if any(weight <= 0 for _, weight in weights):
            raise ValueError("lane weights must be positive")
        object.__setattr__(self, "lane_weights", weights)
        if not isinstance(self.partial_results, bool):
            raise TypeError("partial_results must be a boolean")
        _positive_int("candidate_depth_ceiling", self.candidate_depth_ceiling)
        _positive_int("sparse_ordinal_limit", self.sparse_ordinal_limit)
        if self.nprobe_ceiling is not None:
            _positive_int("nprobe_ceiling", self.nprobe_ceiling)

    def validate_unit_count(self, unit_count: int) -> None:
        """Fail without silently switching to a different runtime profile."""
        if isinstance(unit_count, bool) or not isinstance(unit_count, int):
            raise SearchFailure(SearchFailureCode.CAPACITY_EXCEEDED)
        if not 0 < unit_count <= self.max_units:
            raise SearchFailure(SearchFailureCode.CAPACITY_EXCEEDED)

    def lane_weight(self, lane: RetrievalLane) -> float:
        """Return the runtime-owned fusion weight for one lane."""
        try:
            return dict(self.lane_weights)[lane]
        except KeyError as error:  # pragma: no cover - validated closed set
            raise SearchFailure(SearchFailureCode.PROFILE_UNAVAILABLE) from error


class RetrievalProfileRegistry:
    """Resolve a closed profile or startup-defined immutable alias."""

    def __init__(
        self,
        profiles: tuple[RetrievalProfile, ...],
        *,
        aliases: tuple[tuple[str, str], ...] = (),
    ) -> None:
        materialized = {profile.profile_ref: profile for profile in profiles}
        if len(materialized) != len(profiles) or not materialized:
            raise ValueError("profile refs must be non-empty and unique")
        alias_map = dict(aliases)
        if len(alias_map) != len(aliases):
            raise ValueError("profile aliases must be unique")
        if set(alias_map) & set(materialized):
            raise ValueError("profile aliases must not shadow canonical refs")
        if any(target not in materialized for target in alias_map.values()):
            raise ValueError("profile aliases must target a registered profile")
        self._profiles = MappingProxyType(materialized)
        self._aliases = MappingProxyType(alias_map)

    def resolve(self, profile_ref: str) -> RetrievalProfile:
        """Resolve without importing or probing an optional backend."""
        if not isinstance(profile_ref, str):
            raise SearchFailure(SearchFailureCode.PROFILE_UNAVAILABLE)
        canonical = self._aliases.get(profile_ref, profile_ref)
        try:
            return self._profiles[canonical]
        except KeyError as error:
            raise SearchFailure(SearchFailureCode.PROFILE_UNAVAILABLE) from error


def default_retrieval_profiles() -> RetrievalProfileRegistry:
    """Return only the implemented Stage 2 personal policy."""
    common_weights = (
        (RetrievalLane.CONTENT, 1.0),
        (RetrievalLane.PERCEPTUAL, 2.0),
        (RetrievalLane.INSTANCE, 1.0),
        (RetrievalLane.CONCEPT, 1.0),
    )
    personal = RetrievalProfile(
        profile_ref="personal-lite",
        max_units=10_000,
        semantic_formats=(
            (RetrievalLane.INSTANCE, "numpy-flat-f32-v1"),
            (RetrievalLane.CONCEPT, "numpy-flat-f32-v1"),
        ),
        phash_format="phash-flat-u64-v1",
        phash_radius=2,
        fusion_policy_id="weighted-rrf-v1",
        rrf_k=60,
        lane_weights=common_weights,
        partial_results=True,
        candidate_depth_ceiling=10_000,
        sparse_ordinal_limit=10_000,
    )
    return RetrievalProfileRegistry(
        (personal,),
        aliases=(("local-default", "personal-lite"),),
    )


def _non_blank(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")


def _positive_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


__all__ = [
    "RetrievalProfile",
    "RetrievalProfileRegistry",
    "default_retrieval_profiles",
]
