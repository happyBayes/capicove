"""Tests for the internal-only personal reverse-image research vertical."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from PIL import Image

from capicove.contracts.retrieval import (
    DistanceMetric,
    EmbeddingSpaceRef,
    QueryModality,
    TargetType,
    VectorNormalization,
)
from capicove.retrieval._adapters.opencv_phash import (
    OPENCV_PHASH_64_V1_SPACE,
    ResearchPhashSpaceIdentity,
    ResearchPhashValue,
)
from capicove.retrieval.personal_image_research import (
    RESEARCH_CAPABILITY_STATUS,
    PersonalImageResearchIndex,
    PersonalImageResearchUnit,
    ResearchImageEncoder,
    ResearchSearchDeadlineExceeded,
    ResearchSearchIntent,
    ResearchSemanticVector,
)


FUTURE = datetime.now(UTC) + timedelta(hours=1)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _opaque_id(label: str) -> str:
    return f"unit:{_sha(label)}"


def _space(name: str, *, dimension: int = 2) -> EmbeddingSpaceRef:
    return EmbeddingSpaceRef(
        space_id=f"research-{name}-v1",
        provider_id=f"research-{name}-provider",
        model_revision=_sha(f"{name}-model"),
        preprocessing_version="research-rgb-v1",
        dimension=dimension,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.INNER_PRODUCT,
        query_modalities=frozenset({QueryModality.IMAGE}),
        target_modalities=frozenset({TargetType.IMAGE}),
    )


class FixedPhash:
    """Return one deterministic query hash and record calls."""

    def __init__(
        self,
        value: int,
        *,
        space: ResearchPhashSpaceIdentity = OPENCV_PHASH_64_V1_SPACE,
    ) -> None:
        self.value = value
        self.space = space
        self.calls = 0

    def phash(self, image: Image.Image) -> ResearchPhashValue:
        assert image.size == (8, 8)
        self.calls += 1
        return ResearchPhashValue(space=self.space, value=self.value)


class FixedEncoder:
    """Return one deterministic vector and record selected-lane calls."""

    def __init__(
        self,
        space: EmbeddingSpaceRef,
        values: tuple[float, ...],
        *,
        output_space: EmbeddingSpaceRef | None = None,
        on_encode: Callable[[], None] | None = None,
    ) -> None:
        self.space = space
        self._output = ResearchSemanticVector(output_space or space, values)
        self._on_encode = on_encode
        self.calls = 0

    def encode(
        self,
        image: Image.Image,
        *,
        deadline: datetime,
    ) -> ResearchSemanticVector:
        assert image.size == (8, 8)
        assert deadline.tzinfo is not None
        self.calls += 1
        if self._on_encode is not None:
            self._on_encode()
        return self._output


def _unit(
    unit_id: str,
    *,
    source: str,
    phash: int,
    instance_space: EmbeddingSpaceRef,
    concept_space: EmbeddingSpaceRef,
    instance: tuple[float, ...] = (1.0, 0.0),
    concept: tuple[float, ...] = (1.0, 0.0),
) -> PersonalImageResearchUnit:
    return PersonalImageResearchUnit(
        unit_id=_opaque_id(unit_id),
        source_sha256=_sha(source),
        phash_space=OPENCV_PHASH_64_V1_SPACE,
        phash=phash,
        instance_vector=ResearchSemanticVector(instance_space, instance),
        concept_vector=ResearchSemanticVector(concept_space, concept),
    )


def _index(
    units: tuple[PersonalImageResearchUnit, ...],
    *,
    instance_encoder: ResearchImageEncoder | None = None,
    concept_encoder: ResearchImageEncoder | None = None,
    phash: int = 0,
    phash_provider: FixedPhash | None = None,
    query_phash_space: ResearchPhashSpaceIdentity = OPENCV_PHASH_64_V1_SPACE,
    clock: Callable[[], datetime] | None = None,
) -> tuple[PersonalImageResearchIndex, FixedEncoder, FixedEncoder, FixedPhash]:
    instance_space = _space("instance")
    concept_space = _space("concept")
    resolved_instance = instance_encoder or FixedEncoder(instance_space, (1.0, 0.0))
    resolved_concept = concept_encoder or FixedEncoder(concept_space, (1.0, 0.0))
    resolved_phash = (
        FixedPhash(phash, space=query_phash_space) if phash_provider is None else phash_provider
    )
    index = PersonalImageResearchIndex(
        units,
        phash_space=OPENCV_PHASH_64_V1_SPACE,
        instance_space=instance_space,
        concept_space=concept_space,
        instance_encoder=resolved_instance,
        concept_encoder=resolved_concept,
        phash_provider=resolved_phash,
        clock=clock,
    )
    assert isinstance(resolved_instance, FixedEncoder)
    assert isinstance(resolved_concept, FixedEncoder)
    return index, resolved_instance, resolved_concept, resolved_phash


def test_research_vertical_is_not_exported_as_public_retrieval_api() -> None:
    import capicove.retrieval as public_retrieval

    assert not hasattr(public_retrieval, "PersonalImageResearchIndex")
    assert not hasattr(public_retrieval, "ResearchSearchIntent")


@pytest.mark.parametrize(
    ("intent", "instance_calls", "concept_calls"),
    [
        (ResearchSearchIntent.INSTANCE, 1, 0),
        (ResearchSearchIntent.CONCEPT, 0, 1),
        (ResearchSearchIntent.BALANCED, 1, 1),
    ],
)
def test_search_calls_only_caller_declared_semantic_encoders(
    intent: ResearchSearchIntent,
    instance_calls: int,
    concept_calls: int,
) -> None:
    instance_space = _space("instance")
    concept_space = _space("concept")
    units = (
        _unit(
            "unit-a",
            source="a",
            phash=0,
            instance_space=instance_space,
            concept_space=concept_space,
        ),
    )
    index, instance, concept, phash = _index(units)

    result = index.search_image(
        Image.new("RGB", (8, 8), "white"),
        source_sha256=_sha("query"),
        intent=intent,
        deadline=FUTURE,
    )

    assert result.capability_status == RESEARCH_CAPABILITY_STATUS
    assert result.research_only is True
    assert instance.calls == instance_calls
    assert concept.calls == concept_calls
    assert phash.calls == 1


def test_exact_content_always_ranks_before_higher_rrf_candidate() -> None:
    instance_space = _space("instance")
    concept_space = _space("concept")
    units = (
        _unit(
            "better-rrf",
            source="other",
            phash=0,
            instance_space=instance_space,
            concept_space=concept_space,
            instance=(1.0, 0.0),
        ),
        _unit(
            "exact-but-distant",
            source="query",
            phash=0b111,
            instance_space=instance_space,
            concept_space=concept_space,
            instance=(0.0, 1.0),
        ),
    )
    index, _, _, _ = _index(units)

    result = index.search_image(
        Image.new("RGB", (8, 8), "white"),
        source_sha256=_sha("query"),
        intent=ResearchSearchIntent.INSTANCE,
        deadline=FUTURE,
    )

    assert [hit.unit_id for hit in result.hits] == [
        _opaque_id("exact-but-distant"),
        _opaque_id("better-rrf"),
    ]
    assert result.hits[0].exact_content is True
    assert result.hits[0].phash_distance is None
    assert result.hits[1].semantic_lanes == (ResearchSearchIntent.INSTANCE,)


def test_exact_content_still_executes_phash_and_declared_semantic_lane() -> None:
    instance_space = _space("instance")
    concept_space = _space("concept")
    units = (
        _unit(
            "exact",
            source="query",
            phash=0,
            instance_space=instance_space,
            concept_space=concept_space,
        ),
    )
    index, instance, concept, phash = _index(units)

    result = index.search_image(
        Image.new("RGB", (8, 8), "white"),
        source_sha256=_sha("query"),
        intent=ResearchSearchIntent.INSTANCE,
        deadline=FUTURE,
    )

    assert result.hits[0].exact_content is True
    assert phash.calls == 1
    assert instance.calls == 1
    assert concept.calls == 0


def test_phash_radius_includes_distance_two_and_excludes_distance_three() -> None:
    instance_space = _space("instance")
    concept_space = _space("concept")
    units = (
        _unit(
            "distance-two",
            source="distance-two",
            phash=0b11,
            instance_space=instance_space,
            concept_space=concept_space,
        ),
        _unit(
            "distance-three",
            source="distance-three",
            phash=0b111,
            instance_space=instance_space,
            concept_space=concept_space,
        ),
    )
    index, _, _, _ = _index(units, phash=0)

    result = index.search_image(
        Image.new("RGB", (8, 8), "white"),
        source_sha256=_sha("query"),
        intent=ResearchSearchIntent.INSTANCE,
        deadline=FUTURE,
    )
    hits = {hit.unit_id: hit for hit in result.hits}

    assert hits[_opaque_id("distance-two")].phash_distance == 2
    assert hits[_opaque_id("distance-three")].phash_distance is None


def test_weighted_rrf_uses_phash_weight_two_and_stable_unit_id_ties() -> None:
    instance_space = _space("instance")
    concept_space = _space("concept")
    weighted_units = (
        _unit(
            "semantic-only",
            source="semantic",
            phash=0b111,
            instance_space=instance_space,
            concept_space=concept_space,
            instance=(1.0, 0.0),
        ),
        _unit(
            "phash-and-semantic",
            source="phash",
            phash=0,
            instance_space=instance_space,
            concept_space=concept_space,
            instance=(0.8, 0.6),
        ),
    )
    weighted, _, _, _ = _index(weighted_units)

    weighted_result = weighted.search_image(
        Image.new("RGB", (8, 8), "white"),
        source_sha256=_sha("query"),
        intent=ResearchSearchIntent.INSTANCE,
        deadline=FUTURE,
    )

    assert [hit.unit_id for hit in weighted_result.hits] == [
        _opaque_id("phash-and-semantic"),
        _opaque_id("semantic-only"),
    ]
    assert weighted_result.hits[0].rrf_score > weighted_result.hits[1].rrf_score

    tied_units = (
        _unit(
            "unit-b",
            source="b",
            phash=0b111,
            instance_space=instance_space,
            concept_space=concept_space,
            instance=(1.0, 0.0),
            concept=(0.0, 1.0),
        ),
        _unit(
            "unit-a",
            source="a",
            phash=0b111,
            instance_space=instance_space,
            concept_space=concept_space,
            instance=(0.0, 1.0),
            concept=(1.0, 0.0),
        ),
    )
    instance = FixedEncoder(instance_space, (1.0, 0.0))
    concept = FixedEncoder(concept_space, (1.0, 0.0))
    tied, _, _, _ = _index(
        tied_units,
        instance_encoder=instance,
        concept_encoder=concept,
    )

    tied_result = tied.search_image(
        Image.new("RGB", (8, 8), "white"),
        source_sha256=_sha("query"),
        intent=ResearchSearchIntent.BALANCED,
        deadline=FUTURE,
    )

    assert tied_result.hits[0].rrf_score == pytest.approx(tied_result.hits[1].rrf_score)
    assert [hit.unit_id for hit in tied_result.hits] == sorted(
        (_opaque_id("unit-a"), _opaque_id("unit-b"))
    )


def test_result_projection_contains_no_vectors_or_paths() -> None:
    instance_space = _space("instance")
    concept_space = _space("concept")
    units = (
        _unit(
            "unit-a",
            source="a",
            phash=0,
            instance_space=instance_space,
            concept_space=concept_space,
        ),
    )
    index, _, _, _ = _index(units)

    result = index.search_image(
        Image.new("RGB", (8, 8), "white"),
        source_sha256=_sha("query"),
        intent=ResearchSearchIntent.INSTANCE,
        deadline=FUTURE,
    )

    assert set(result.hits[0].__dataclass_fields__) == {
        "unit_id",
        "exact_content",
        "phash_distance",
        "semantic_lanes",
        "rrf_score",
    }
    assert "vector" not in repr(result)
    assert "path" not in repr(result)


@pytest.mark.parametrize("top_k", [0, 101, True, 1.5])
def test_search_rejects_invalid_top_k(top_k: object) -> None:
    index, _, _, _ = _index(())

    with pytest.raises(ValueError, match="top_k"):
        index.search_image(
            Image.new("RGB", (8, 8), "white"),
            source_sha256=_sha("query"),
            intent=ResearchSearchIntent.INSTANCE,
            deadline=FUTURE,
            top_k=top_k,  # type: ignore[arg-type]
        )


def test_empty_index_returns_zero_hits_after_required_lanes_run() -> None:
    index, instance, concept, phash = _index(())

    result = index.search_image(
        Image.new("RGB", (8, 8), "white"),
        source_sha256=_sha("query"),
        intent=ResearchSearchIntent.INSTANCE,
        deadline=FUTURE,
    )

    assert result.hits == ()
    assert phash.calls == 1
    assert instance.calls == 1
    assert concept.calls == 0


def test_search_rejects_malformed_source_sha_before_any_lane() -> None:
    index, instance, concept, phash = _index(())

    with pytest.raises(ValueError, match="source_sha256"):
        index.search_image(
            Image.new("RGB", (8, 8), "white"),
            source_sha256="not-a-sha",
            intent=ResearchSearchIntent.INSTANCE,
            deadline=FUTURE,
        )

    assert phash.calls == 0
    assert instance.calls == 0
    assert concept.calls == 0


def test_index_preserves_falsey_injected_phash_provider() -> None:
    class FalseyPhash(FixedPhash):
        def __bool__(self) -> bool:
            return False

    provider = FalseyPhash(0)
    index, instance, concept, _ = _index((), phash_provider=provider)

    result = index.search_image(
        Image.new("RGB", (8, 8), "white"),
        source_sha256=_sha("query"),
        intent=ResearchSearchIntent.INSTANCE,
        deadline=FUTURE,
    )

    assert result.hits == ()
    assert provider.calls == 1
    assert instance.calls == 1
    assert concept.calls == 0


def test_index_preserves_falsey_injected_clock() -> None:
    class FalseyClock:
        def __init__(self) -> None:
            self.calls = 0

        def __bool__(self) -> bool:
            return False

        def __call__(self) -> datetime:
            self.calls += 1
            return FUTURE - timedelta(minutes=1)

    clock = FalseyClock()
    index, _, _, _ = _index((), clock=clock)

    result = index.search_image(
        Image.new("RGB", (8, 8), "white"),
        source_sha256=_sha("query"),
        intent=ResearchSearchIntent.INSTANCE,
        deadline=FUTURE,
    )

    assert result.hits == ()
    assert clock.calls > 0


def test_index_rejects_duplicate_ids_and_capacity_above_personal_limit() -> None:
    instance_space = _space("instance")
    concept_space = _space("concept")
    shared = _unit(
        "duplicate",
        source="a",
        phash=0,
        instance_space=instance_space,
        concept_space=concept_space,
    )

    with pytest.raises(ValueError, match="unit_id"):
        _index((shared, shared))

    vector_instance = ResearchSemanticVector(instance_space, (1.0, 0.0))
    vector_concept = ResearchSemanticVector(concept_space, (1.0, 0.0))
    too_many = tuple(
        PersonalImageResearchUnit(
            unit_id=_opaque_id(str(index)),
            source_sha256=_sha(str(index)),
            phash_space=OPENCV_PHASH_64_V1_SPACE,
            phash=0,
            instance_vector=vector_instance,
            concept_vector=vector_concept,
        )
        for index in range(10_001)
    )
    _index(too_many[:10_000])
    with pytest.raises(ValueError, match="10,000"):
        _index(too_many)


def test_units_reject_hash_space_dimension_and_vector_contract_violations() -> None:
    instance_space = _space("instance")
    concept_space = _space("concept")
    wrong_space = _space("wrong")

    for path_shaped_id in ("/private/library/image.jpg", "photo.jpg", "C:private.jpg"):
        with pytest.raises(ValueError, match="opaque unit identifier"):
            PersonalImageResearchUnit(
                path_shaped_id,
                _sha("path-shaped-id"),
                OPENCV_PHASH_64_V1_SPACE,
                0,
                ResearchSemanticVector(instance_space, (1.0, 0.0)),
                ResearchSemanticVector(concept_space, (1.0, 0.0)),
            )
    with pytest.raises(ValueError, match="source_sha256"):
        PersonalImageResearchUnit(
            _opaque_id("bad-sha"),
            "not-a-sha",
            OPENCV_PHASH_64_V1_SPACE,
            0,
            ResearchSemanticVector(instance_space, (1.0, 0.0)),
            ResearchSemanticVector(concept_space, (1.0, 0.0)),
        )
    for invalid_hash in (-1, 2**64, True):
        with pytest.raises(ValueError, match="phash"):
            PersonalImageResearchUnit(
                _opaque_id("bad-hash"),
                _sha("bad-hash"),
                OPENCV_PHASH_64_V1_SPACE,
                invalid_hash,
                ResearchSemanticVector(instance_space, (1.0, 0.0)),
                ResearchSemanticVector(concept_space, (1.0, 0.0)),
            )

    with pytest.raises(ValueError, match="dimension"):
        ResearchSemanticVector(instance_space, (1.0,))
    with pytest.raises(ValueError, match="finite"):
        ResearchSemanticVector(instance_space, (float("nan"), 0.0))
    with pytest.raises(ValueError, match="finite"):
        ResearchSemanticVector(instance_space, (float("inf"), 0.0))
    with pytest.raises(ValueError, match="finite"):
        ResearchSemanticVector(instance_space, (10**400, 0.0))
    with pytest.raises(ValueError, match="L2"):
        ResearchSemanticVector(instance_space, (1.0, 1.0))

    mismatched = _unit(
        "wrong-space",
        source="wrong",
        phash=0,
        instance_space=wrong_space,
        concept_space=concept_space,
    )
    with pytest.raises(ValueError, match="instance space"):
        _index((mismatched,))


def test_semantic_vectors_accept_finite_numpy_float32_values() -> None:
    vector = ResearchSemanticVector(
        _space("float32"),
        (np.float32(1.0), np.float32(0.0)),  # type: ignore[arg-type]
    )

    assert vector.values == (1.0, 0.0)
    assert all(type(value) is float for value in vector.values)


def test_research_space_dimension_is_bounded_before_vector_materialization() -> None:
    maximum = _space("maximum-dimension", dimension=16_384)
    vector = ResearchSemanticVector(maximum, (1.0,) + (0.0,) * 16_383)

    assert len(vector.values) == 16_384

    with pytest.raises(ValueError, match="at most 16,384"):
        _space("excessive-dimension", dimension=16_385)


def test_search_rejects_encoder_output_space_mismatch() -> None:
    instance_space = _space("instance")
    concept_space = _space("concept")
    wrong_space = _space("wrong")
    units = (
        _unit(
            "unit-a",
            source="a",
            phash=0,
            instance_space=instance_space,
            concept_space=concept_space,
        ),
    )
    mismatched_encoder = FixedEncoder(
        instance_space,
        (1.0, 0.0),
        output_space=wrong_space,
    )
    index, _, _, _ = _index(units, instance_encoder=mismatched_encoder)

    with pytest.raises(ValueError, match="encoder output space"):
        index.search_image(
            Image.new("RGB", (8, 8), "white"),
            source_sha256=_sha("query"),
            intent=ResearchSearchIntent.INSTANCE,
            deadline=FUTURE,
        )


def test_search_requires_absolute_aware_deadline() -> None:
    index, instance, concept, phash = _index(())

    with pytest.raises(ValueError, match="timezone-aware"):
        index.search_image(
            Image.new("RGB", (8, 8), "white"),
            source_sha256=_sha("query"),
            intent=ResearchSearchIntent.INSTANCE,
            deadline=datetime.now(),
        )
    with pytest.raises(ResearchSearchDeadlineExceeded):
        index.search_image(
            Image.new("RGB", (8, 8), "white"),
            source_sha256=_sha("query"),
            intent=ResearchSearchIntent.INSTANCE,
            deadline=datetime.now(UTC) - timedelta(seconds=1),
        )

    assert instance.calls == 0
    assert concept.calls == 0
    assert phash.calls == 0


def test_search_rejects_string_intent_without_running_any_lane() -> None:
    index, instance, concept, phash = _index(())

    with pytest.raises(TypeError, match="ResearchSearchIntent"):
        index.search_image(
            Image.new("RGB", (8, 8), "white"),
            source_sha256=_sha("query"),
            intent="instance",  # type: ignore[arg-type]
            deadline=FUTURE,
        )

    assert instance.calls == 0
    assert concept.calls == 0
    assert phash.calls == 0


def test_query_phash_must_be_unsigned_uint64() -> None:
    index, instance, concept, phash = _index((), phash=-1)

    with pytest.raises(ValueError, match="unsigned 64-bit"):
        index.search_image(
            Image.new("RGB", (8, 8), "white"),
            source_sha256=_sha("query"),
            intent=ResearchSearchIntent.INSTANCE,
            deadline=FUTURE,
        )

    assert phash.calls == 1
    assert instance.calls == 0
    assert concept.calls == 0


def test_index_rejects_mixed_target_or_query_phash_spaces() -> None:
    instance_space = _space("instance")
    concept_space = _space("concept")
    incompatible = replace(
        OPENCV_PHASH_64_V1_SPACE,
        preprocessing_version="different-preprocessing-v1",
    )
    mismatched_target = PersonalImageResearchUnit(
        unit_id=_opaque_id("target"),
        source_sha256=_sha("target"),
        phash_space=incompatible,
        phash=0,
        instance_vector=ResearchSemanticVector(instance_space, (1.0, 0.0)),
        concept_vector=ResearchSemanticVector(concept_space, (1.0, 0.0)),
    )

    with pytest.raises(ValueError, match="target phash space"):
        _index((mismatched_target,))

    compatible_target = _unit(
        "target",
        source="target",
        phash=0,
        instance_space=instance_space,
        concept_space=concept_space,
    )
    with pytest.raises(ValueError, match="query phash space"):
        _index((compatible_target,), query_phash_space=incompatible)


def test_search_rejects_oversized_resolved_image_before_any_lane() -> None:
    index, instance, concept, phash = _index(())
    oversized = Image.new("1", (40_000_001, 1))

    with pytest.raises(ValueError, match="40,000,000-pixel"):
        index.search_image(
            oversized,
            source_sha256=_sha("query"),
            intent=ResearchSearchIntent.INSTANCE,
            deadline=FUTURE,
        )

    assert phash.calls == 0
    assert instance.calls == 0
    assert concept.calls == 0


def test_semantic_lane_selection_fails_closed_for_unknown_future_intent() -> None:
    index, _, _, _ = _index(())

    with pytest.raises(ValueError, match="unsupported research search intent"):
        index._selected_semantic_lanes(object())  # type: ignore[arg-type]


def test_search_rejects_query_provider_space_changed_after_index_construction() -> None:
    index, instance, concept, phash = _index(())
    phash.space = replace(
        OPENCV_PHASH_64_V1_SPACE,
        preprocessing_version="changed-after-construction-v1",
    )

    with pytest.raises(ValueError, match="query phash space"):
        index.search_image(
            Image.new("RGB", (8, 8), "white"),
            source_sha256=_sha("query"),
            intent=ResearchSearchIntent.INSTANCE,
            deadline=FUTURE,
        )

    assert phash.calls == 1
    assert instance.calls == 0
    assert concept.calls == 0


def test_deadline_is_checked_after_phash_before_semantic_lane() -> None:
    deadline = datetime(2030, 1, 1, tzinfo=UTC)
    before = deadline - timedelta(seconds=1)
    moments = iter((before, before, before, deadline))
    index, instance, concept, phash = _index((), clock=lambda: next(moments))

    with pytest.raises(ResearchSearchDeadlineExceeded):
        index.search_image(
            Image.new("RGB", (8, 8), "white"),
            source_sha256=_sha("query"),
            intent=ResearchSearchIntent.INSTANCE,
            deadline=deadline,
        )

    assert phash.calls == 1
    assert instance.calls == 0
    assert concept.calls == 0


def test_deadline_is_checked_after_selected_semantic_lane() -> None:
    deadline = datetime(2030, 1, 1, tzinfo=UTC)
    before = deadline - timedelta(seconds=1)
    moments = iter((before, before, before, before, before, deadline))
    index, instance, concept, phash = _index((), clock=lambda: next(moments))

    with pytest.raises(ResearchSearchDeadlineExceeded):
        index.search_image(
            Image.new("RGB", (8, 8), "white"),
            source_sha256=_sha("query"),
            intent=ResearchSearchIntent.BALANCED,
            deadline=deadline,
        )

    assert phash.calls == 1
    assert instance.calls == 1
    assert concept.calls == 0


def test_deadline_expiring_during_fusion_prevents_result_return() -> None:
    deadline = datetime(2030, 1, 1, tzinfo=UTC)
    before = deadline - timedelta(seconds=1)
    moments = iter((*((before,) * 6), deadline))
    index, instance, concept, phash = _index((), clock=lambda: next(moments))

    with pytest.raises(ResearchSearchDeadlineExceeded):
        index.search_image(
            Image.new("RGB", (8, 8), "white"),
            source_sha256=_sha("query"),
            intent=ResearchSearchIntent.INSTANCE,
            deadline=deadline,
        )

    assert phash.calls == 1
    assert instance.calls == 1
    assert concept.calls == 0


def test_deadline_expiring_during_projection_prevents_result_return() -> None:
    deadline = datetime(2030, 1, 1, tzinfo=UTC)
    before = deadline - timedelta(seconds=1)
    instance_space = _space("instance")
    concept_space = _space("concept")
    unit = _unit(
        "projected",
        source="target",
        phash=0,
        instance_space=instance_space,
        concept_space=concept_space,
    )
    moments = iter((*((before,) * 7), deadline))
    index, instance, concept, phash = _index((unit,), clock=lambda: next(moments))

    with pytest.raises(ResearchSearchDeadlineExceeded):
        index.search_image(
            Image.new("RGB", (8, 8), "white"),
            source_sha256=_sha("query"),
            intent=ResearchSearchIntent.INSTANCE,
            deadline=deadline,
        )

    assert phash.calls == 1
    assert instance.calls == 1
    assert concept.calls == 0
