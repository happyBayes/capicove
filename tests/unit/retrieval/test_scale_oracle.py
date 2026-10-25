"""Tests for the provisional personal exact-oracle producer."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

import pytest

from capicove.contracts.retrieval import (
    DistanceMetric,
    EmbeddingSpaceRef,
    HashSpaceRef,
    ProcessingFingerprint,
    QueryModality,
    TargetType,
    VectorNormalization,
)
from capicove.processing.image.ports import ImageSemanticLane, SemanticTargetIdentity
from capicove.processing.image.representations import (
    EncodedImageVector,
    ImageTargetRepresentation,
)
from capicove.retrieval.scale_oracle import (
    PERSONAL_EXACT_ORACLE_ALGORITHM_ID,
    PERSONAL_EXACT_ORACLE_MAX_QUERIES,
    PersonalExactOracleEvidence,
    PersonalExactOracleQuery,
    PersonalExactOracleSubject,
    PersonalExactQueryVector,
    PersonalExactSemanticIdentity,
    personal_exact_target_selection_sha256,
    produce_personal_exact_oracle,
)


FUTURE = datetime.now(UTC) + timedelta(hours=1)
FINGERPRINT = ProcessingFingerprint("d" * 64)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _space(lane: ImageSemanticLane, *, dimension: int = 2) -> EmbeddingSpaceRef:
    return EmbeddingSpaceRef(
        space_id=f"{lane.value}-space-v1",
        provider_id=f"{lane.value}-provider",
        model_revision="revision-1",
        preprocessing_version="rgb-v1",
        dimension=dimension,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.INNER_PRODUCT,
        query_modalities=frozenset({QueryModality.IMAGE}),
        target_modalities=frozenset({TargetType.IMAGE}),
    )


def _target_identity(lane: ImageSemanticLane) -> SemanticTargetIdentity:
    return SemanticTargetIdentity(
        lane=lane,
        space=_space(lane),
        bundle_manifest_sha256=_sha(f"{lane.value}-target-bundle"),
    )


INSTANCE_TARGET = _target_identity(ImageSemanticLane.INSTANCE)
CONCEPT_TARGET = _target_identity(ImageSemanticLane.CONCEPT)


def _semantic_identity(target: SemanticTargetIdentity) -> PersonalExactSemanticIdentity:
    return PersonalExactSemanticIdentity(
        target_identity=target,
        query_bundle_manifest_sha256=_sha(f"{target.lane.value}-query-bundle"),
    )


INSTANCE_SEMANTIC = _semantic_identity(INSTANCE_TARGET)
CONCEPT_SEMANTIC = _semantic_identity(CONCEPT_TARGET)


def _hash_space() -> HashSpaceRef:
    return HashSpaceRef(
        space_id="phash-space-v1",
        provider_id="fake-phash",
        provider_version="1",
        preprocessing_version="rgb-v1",
        algorithm="phash",
        bit_width=64,
        hash_version="phash-64-v1",
        target_modalities=frozenset({TargetType.IMAGE}),
    )


HASH_SPACE = _hash_space()


@dataclass(frozen=True, slots=True)
class _SourceIdentity:
    unit_manifest_sha256: str
    target_representation_manifest_sha256: str
    unit_count: int
    semantic_targets: tuple[SemanticTargetIdentity, ...]


def _subject() -> PersonalExactOracleSubject:
    unit_manifest_sha256 = _sha("units")
    target_unit_ids = tuple(f"unit-{ordinal:05d}" for ordinal in range(10_000))
    return PersonalExactOracleSubject(
        evaluation_id="personal-evaluation-1",
        definition_sha256=_sha("definition"),
        unit_manifest_sha256=unit_manifest_sha256,
        target_representation_manifest_sha256=_sha("representations"),
        target_selection_sha256=personal_exact_target_selection_sha256(
            unit_manifest_sha256,
            target_unit_ids,
        ),
        generation_manifest_sha256=_sha("generation"),
        semantic_identities=(INSTANCE_SEMANTIC, CONCEPT_SEMANTIC),
        target_unit_ids=target_unit_ids,
    )


def _query(index: int = 0) -> PersonalExactOracleQuery:
    query_ref = f"query-{index}"
    query_sha256 = _sha(f"query-image-{index}")
    return PersonalExactOracleQuery(
        query_ref=query_ref,
        query_sha256=query_sha256,
        vectors=(
            PersonalExactQueryVector.create(
                query_ref=query_ref,
                query_sha256=query_sha256,
                semantic_identity=INSTANCE_SEMANTIC,
                values=(1.0, 0.0),
            ),
            PersonalExactQueryVector.create(
                query_ref=query_ref,
                query_sha256=query_sha256,
                semantic_identity=CONCEPT_SEMANTIC,
                values=(0.0, 1.0),
            ),
        ),
        instance_relevance=(("unit-00000", 3), ("unit-00003", 1)),
        concept_relevance=(("unit-00001", 3), ("unit-00004", 1)),
    )


def _row(
    ordinal: int,
    *,
    second_pass: bool = False,
    mutate_low_ranked_target: bool = False,
    late_instance_winner: bool = False,
    near_cutoff: bool = False,
    wrong_instance_identity: bool = False,
    unit_id_override: str | None = None,
    ordinal_override: int | None = None,
) -> ImageTargetRepresentation:
    instance_values = (1.0, 0.0)
    if late_instance_winner:
        instance_values = (1.0, 0.0) if ordinal == 9_999 else (0.0, 1.0)
    if near_cutoff:
        if ordinal <= 96:
            x_value = 0.9
        elif ordinal == 500:
            x_value = 0.8
        elif ordinal == 97:
            x_value = 0.500002
        elif ordinal == 98:
            x_value = 0.500001
        elif ordinal == 99:
            x_value = 0.5
        else:
            x_value = 0.1
        instance_values = (x_value, math.sqrt(1.0 - (x_value * x_value)))
    if second_pass and mutate_low_ranked_target and ordinal == 9_999:
        instance_values = (-1.0, 0.0)
    instance_identity = INSTANCE_TARGET
    if wrong_instance_identity:
        instance_identity = replace(
            INSTANCE_TARGET,
            bundle_manifest_sha256=_sha("wrong-instance-target-bundle"),
        )
    actual_ordinal = ordinal if ordinal_override is None else ordinal_override
    return ImageTargetRepresentation(
        ordinal=actual_ordinal,
        unit_id=unit_id_override or f"unit-{ordinal:05d}",
        content_sha256=_sha(f"content-{ordinal}"),
        media_type="image/png",
        perceptual_hash=ordinal,
        hash_space=HASH_SPACE,
        embeddings=(
            EncodedImageVector(instance_identity, instance_values),
            EncodedImageVector(CONCEPT_TARGET, (0.0, 1.0)),
        ),
        processing_fingerprint=FINGERPRINT,
    )


class _LazySource:
    def __init__(
        self,
        subject: PersonalExactOracleSubject,
        *,
        target_count: int = 10_000,
        mutate_second_pass: bool = False,
        drift_identity: bool = False,
        ordinal_drift_at: int | None = None,
        oversized_batch: bool = False,
        late_instance_winner: bool = False,
        near_cutoff: bool = False,
        wrong_instance_identity: bool = False,
        unit_id_drift_at: int | None = None,
        close_error: bool = False,
    ) -> None:
        self._subject = subject
        self._target_count = target_count
        self._mutate_second_pass = mutate_second_pass
        self._drift_identity = drift_identity
        self._ordinal_drift_at = ordinal_drift_at
        self._oversized_batch = oversized_batch
        self._late_instance_winner = late_instance_winner
        self._near_cutoff = near_cutoff
        self._wrong_instance_identity = wrong_instance_identity
        self._unit_id_drift_at = unit_id_drift_at
        self._close_error = close_error
        self.pass_count = 0
        self.closed_passes = 0
        self.requested_batch_sizes: list[int] = []

    @property
    def identity(self) -> _SourceIdentity:
        target_manifest = self._subject.target_representation_manifest_sha256
        if self._drift_identity and self.pass_count >= 1:
            target_manifest = _sha("changed-representations")
        return _SourceIdentity(
            unit_manifest_sha256=self._subject.unit_manifest_sha256,
            target_representation_manifest_sha256=target_manifest,
            unit_count=10_000,
            semantic_targets=(INSTANCE_TARGET, CONCEPT_TARGET),
        )

    def iter_batches(
        self,
        *,
        start_ordinal: int,
        maximum_batch_size: int,
        deadline: datetime,
    ) -> Iterator[tuple[ImageTargetRepresentation, ...]]:
        assert start_ordinal == 0
        assert deadline == FUTURE
        self.pass_count += 1
        if self.pass_count > 2:
            raise AssertionError("the producer requested a forbidden third pass")
        self.requested_batch_sizes.append(maximum_batch_size)
        emitted_batch_size = maximum_batch_size + 1 if self._oversized_batch else maximum_batch_size
        try:
            for start in range(0, self._target_count, emitted_batch_size):
                stop = min(self._target_count, start + emitted_batch_size)
                yield tuple(
                    _row(
                        ordinal,
                        second_pass=self.pass_count == 2,
                        mutate_low_ranked_target=self._mutate_second_pass,
                        late_instance_winner=self._late_instance_winner,
                        near_cutoff=self._near_cutoff,
                        wrong_instance_identity=(self._wrong_instance_identity and ordinal == 0),
                        unit_id_override=(
                            "unit-drifted" if ordinal == self._unit_id_drift_at else None
                        ),
                        ordinal_override=(
                            ordinal + 1 if ordinal == self._ordinal_drift_at else None
                        ),
                    )
                    for ordinal in range(start, stop)
                )
        finally:
            self.closed_passes += 1
            if self._close_error:
                raise RuntimeError("source close failed")


def _produce(
    source: _LazySource,
    queries: object | None = None,
    *,
    maximum_target_batch_size: int = 256,
    deadline: datetime = FUTURE,
) -> PersonalExactOracleEvidence:
    return produce_personal_exact_oracle(
        source,
        _subject(),
        (_query(0), _query(1)) if queries is None else queries,  # type: ignore[arg-type]
        maximum_target_batch_size=maximum_target_batch_size,
        deadline=deadline,
    )


@pytest.mark.slow
def test_exact_oracle_scans_all_targets_twice_and_seals_replayable_evidence() -> None:
    subject = _subject()
    source = _LazySource(subject)

    evidence = produce_personal_exact_oracle(
        source,
        subject,
        (_query(0), _query(1)),
        maximum_target_batch_size=257,
        deadline=FUTURE,
    )

    assert source.pass_count == 2
    assert source.closed_passes == 2
    assert source.requested_batch_sizes == [257, 257]
    assert evidence.receipt.algorithm_id == PERSONAL_EXACT_ORACLE_ALGORITHM_ID
    assert not hasattr(evidence.receipt, "candidate_provider_call_count")
    assert evidence.receipt.exact_target_count == 10_000
    assert len(evidence.receipt.receipt_sha256) == 64
    assert len(evidence.lane_oracles) == 4
    for oracle in evidence.lane_oracles:
        assert len(oracle.neighbors) == 100
        assert tuple(item.ordinal for item in oracle.neighbors) == tuple(range(100))
        assert {item.score_bits for item in oracle.neighbors} == {"3f800000"}

    manifest = evidence.to_provisional_evaluation_manifest()
    assert manifest.evaluation_id == subject.evaluation_id
    assert manifest.query_count == 2
    assert manifest.target_selection_sha256 == subject.target_selection_sha256
    assert manifest.generation_manifest_sha256 == subject.generation_manifest_sha256
    assert manifest.queries[0].instance_relevance == _query(0).instance_relevance
    assert manifest.queries[0].concept_relevance == _query(0).concept_relevance
    assert manifest.queries[0].instance_exact_unit_ids == tuple(
        f"unit-{ordinal:05d}" for ordinal in range(100)
    )


@pytest.mark.slow
def test_exact_oracle_rejects_any_second_pass_representation_change() -> None:
    subject = _subject()
    source = _LazySource(subject, mutate_second_pass=True)

    with pytest.raises(ValueError, match="second pass"):
        produce_personal_exact_oracle(
            source,
            subject,
            (_query(0), _query(1)),
            maximum_target_batch_size=1_000,
            deadline=FUTURE,
        )

    assert source.pass_count == 2


@pytest.mark.slow
def test_exact_oracle_orders_score_before_ordinal_and_unit_id() -> None:
    subject = _subject()
    evidence = produce_personal_exact_oracle(
        _LazySource(subject, late_instance_winner=True),
        subject,
        (_query(0), _query(1)),
        maximum_target_batch_size=1_000,
        deadline=FUTURE,
    )

    instance = evidence.lane_oracles[0]
    assert instance.neighbors[0].ordinal == 9_999
    assert instance.neighbors[0].score_bits == "3f800000"
    assert tuple(item.ordinal for item in instance.neighbors[1:]) == tuple(range(99))
    assert {item.score_bits for item in instance.neighbors[1:]} == {"00000000"}


@pytest.mark.slow
def test_exact_oracle_is_invariant_to_source_batch_size_at_near_cutoff() -> None:
    subject = _subject()

    large_batch = produce_personal_exact_oracle(
        _LazySource(subject, near_cutoff=True),
        subject,
        (_query(0), _query(1)),
        maximum_target_batch_size=1_000,
        deadline=FUTURE,
    )
    small_batch = produce_personal_exact_oracle(
        _LazySource(subject, near_cutoff=True),
        subject,
        (_query(0), _query(1)),
        maximum_target_batch_size=7,
        deadline=FUTURE,
    )

    assert small_batch == large_batch
    assert small_batch.receipt.receipt_sha256 == large_batch.receipt.receipt_sha256
    assert tuple(item.ordinal for item in small_batch.lane_oracles[0].neighbors[-2:]) == (
        97,
        98,
    )
    assert 99 not in {item.ordinal for item in small_batch.lane_oracles[0].neighbors}


@pytest.mark.slow
@pytest.mark.parametrize(
    ("source_kwargs", "message"),
    [
        ({"target_count": 9_999}, "10,000"),
        ({"target_count": 10_001}, "10,000"),
        ({"ordinal_drift_at": 8_000}, "ordinal"),
        ({"oversized_batch": True}, "batch"),
        ({"drift_identity": True}, "identity"),
        ({"wrong_instance_identity": True}, "semantic identity"),
        ({"unit_id_drift_at": 5_000}, "target selection"),
    ],
)
def test_exact_oracle_fails_closed_on_source_drift(
    source_kwargs: dict[str, object],
    message: str,
) -> None:
    subject = _subject()
    source = _LazySource(subject, **source_kwargs)  # type: ignore[arg-type]

    with pytest.raises((TypeError, ValueError), match=message):
        produce_personal_exact_oracle(
            source,
            subject,
            (_query(0), _query(1)),
            maximum_target_batch_size=256,
            deadline=FUTURE,
        )
    assert source.closed_passes == 1


def test_exact_oracle_rejects_query_iterable_overflow_before_source_access() -> None:
    subject = _subject()
    source = _LazySource(subject)

    def too_many_queries() -> Iterator[PersonalExactOracleQuery]:
        for index in range(PERSONAL_EXACT_ORACLE_MAX_QUERIES + 1):
            yield _query(index)

    with pytest.raises(ValueError, match="at most"):
        produce_personal_exact_oracle(
            source,
            subject,
            too_many_queries(),
            maximum_target_batch_size=256,
            deadline=FUTURE,
        )

    assert source.pass_count == 0


class _ClosingOverflowQueries:
    def __init__(self) -> None:
        self._index = 0
        self.closed = False

    def __iter__(self) -> _ClosingOverflowQueries:
        return self

    def __next__(self) -> PersonalExactOracleQuery:
        if self._index > PERSONAL_EXACT_ORACLE_MAX_QUERIES:
            raise StopIteration
        query = _query(self._index)
        self._index += 1
        return query

    def close(self) -> None:
        self.closed = True
        raise RuntimeError("query close failed")


def test_query_close_failure_does_not_mask_overflow() -> None:
    subject = _subject()
    source = _LazySource(subject)
    queries = _ClosingOverflowQueries()

    with pytest.raises(ValueError, match="at most") as caught:
        produce_personal_exact_oracle(
            source,
            subject,
            queries,
            maximum_target_batch_size=256,
            deadline=FUTURE,
        )

    assert queries.closed is True
    assert source.pass_count == 0
    assert getattr(caught.value, "__notes__", ()) == [
        "secondary iterator close failure: RuntimeError: query close failed"
    ]


def test_source_close_failure_does_not_mask_batch_violation() -> None:
    subject = _subject()
    source = _LazySource(subject, oversized_batch=True, close_error=True)

    with pytest.raises(ValueError, match="batch") as caught:
        produce_personal_exact_oracle(
            source,
            subject,
            (_query(0), _query(1)),
            maximum_target_batch_size=256,
            deadline=FUTURE,
        )

    assert source.closed_passes == 1
    assert getattr(caught.value, "__notes__", ()) == [
        "secondary iterator close failure: RuntimeError: source close failed"
    ]


def test_exact_oracle_rejects_duplicate_or_missing_query_cells() -> None:
    subject = _subject()
    source = _LazySource(subject)
    first = _query(0)

    with pytest.raises(ValueError, match="query refs and hashes must be unique"):
        produce_personal_exact_oracle(
            source,
            subject,
            (first, first),
            maximum_target_batch_size=256,
            deadline=FUTURE,
        )
    with pytest.raises(ValueError, match="at least 2"):
        produce_personal_exact_oracle(
            source,
            subject,
            (first,),
            maximum_target_batch_size=256,
            deadline=FUTURE,
        )

    with pytest.raises(ValueError, match="instance and concept"):
        replace(first, vectors=(first.vectors[0],))
    assert source.pass_count == 0


def test_query_vector_is_float32_content_addressed_and_normalized() -> None:
    query_ref = "query"
    query_sha256 = _sha("query")
    vector = PersonalExactQueryVector.create(
        query_ref=query_ref,
        query_sha256=query_sha256,
        semantic_identity=INSTANCE_SEMANTIC,
        values=(1.0, 0.0),
    )

    assert len(vector.query_vector_sha256) == 64
    with pytest.raises(ValueError, match="digest"):
        replace(vector, query_vector_sha256="0" * 64)
    with pytest.raises(ValueError, match="dimension"):
        PersonalExactQueryVector.create(
            query_ref=query_ref,
            query_sha256=query_sha256,
            semantic_identity=INSTANCE_SEMANTIC,
            values=(1.0,),
        )
    with pytest.raises(ValueError, match="normalized"):
        PersonalExactQueryVector.create(
            query_ref=query_ref,
            query_sha256=query_sha256,
            semantic_identity=INSTANCE_SEMANTIC,
            values=(1.0, 1.0),
        )
    with pytest.raises(ValueError, match="finite"):
        PersonalExactQueryVector.create(
            query_ref=query_ref,
            query_sha256=query_sha256,
            semantic_identity=INSTANCE_SEMANTIC,
            values=(float("nan"), 0.0),
        )


def test_subject_rejects_non_10k_count_and_semantic_lane_drift() -> None:
    subject = _subject()

    with pytest.raises(ValueError, match="all 10,000"):
        replace(subject, exact_target_count=9_999)
    with pytest.raises(ValueError, match="instance and concept"):
        replace(
            subject,
            semantic_identities=(CONCEPT_SEMANTIC, INSTANCE_SEMANTIC),
        )
    with pytest.raises(ValueError, match="selection digest"):
        replace(subject, target_selection_sha256=_sha("drifted-selection"))
    swapped = (
        subject.target_unit_ids[1],
        subject.target_unit_ids[0],
        *subject.target_unit_ids[2:],
    )
    with pytest.raises(ValueError, match="selection digest"):
        replace(subject, target_unit_ids=swapped)


def test_target_selection_rejects_10001_and_overlong_unit_ids() -> None:
    unit_manifest_sha256 = _sha("units")
    with pytest.raises(ValueError, match="exactly 10,000"):
        personal_exact_target_selection_sha256(
            unit_manifest_sha256,
            tuple(f"unit-{ordinal}" for ordinal in range(10_001)),
        )
    unit_ids = [f"unit-{ordinal:05d}" for ordinal in range(10_000)]
    unit_ids[0] = "u" * 257
    with pytest.raises(ValueError, match="UTF-8 bytes"):
        personal_exact_target_selection_sha256(
            unit_manifest_sha256,
            tuple(unit_ids),
        )


class _TextTransformationBomb(str):
    def strip(self, chars: str | None = None, /) -> str:
        raise AssertionError("strip must not be called for obviously oversized text")

    def encode(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
        raise AssertionError("encode must not be called for obviously oversized text")


def test_query_and_relevance_canonical_bounds_fail_closed() -> None:
    query = _query(0)
    with pytest.raises(ValueError, match="UTF-8 bytes"):
        replace(query, query_ref="q" * 513)
    oversized = _TextTransformationBomb("中" * 2_000_000)
    with pytest.raises(ValueError, match="UTF-8 bytes"):
        replace(query, query_ref=oversized)
    with pytest.raises(ValueError, match="UTF-8 bytes"):
        replace(query, instance_relevance=(("u" * 257, 3),))
    with pytest.raises(ValueError, match="at most 100"):
        replace(
            query,
            instance_relevance=tuple((f"unit-{ordinal:05d}", 1) for ordinal in range(101)),
        )


def test_total_relevance_bound_is_checked_before_source_access() -> None:
    subject = _subject()
    source = _LazySource(subject)
    relevance = tuple((f"unit-{ordinal:05d}", 1) for ordinal in range(100))
    queries = tuple(
        replace(
            _query(index),
            instance_relevance=relevance,
            concept_relevance=relevance,
        )
        for index in range(51)
    )

    with pytest.raises(ValueError, match="total relevance"):
        produce_personal_exact_oracle(
            source,
            subject,
            queries,
            maximum_target_batch_size=256,
            deadline=FUTURE,
        )
    assert source.pass_count == 0


def test_exact_oracle_rejects_expired_deadline_without_scanning() -> None:
    subject = _subject()
    source = _LazySource(subject)

    with pytest.raises(TimeoutError, match="deadline"):
        produce_personal_exact_oracle(
            source,
            subject,
            (_query(0), _query(1)),
            maximum_target_batch_size=256,
            deadline=datetime.now(UTC) - timedelta(seconds=1),
        )

    assert source.pass_count == 0


@pytest.mark.slow
def test_evidence_rejects_receipt_tampering() -> None:
    subject = _subject()
    evidence = produce_personal_exact_oracle(
        _LazySource(subject),
        subject,
        (_query(0), _query(1)),
        maximum_target_batch_size=1_000,
        deadline=FUTURE,
    )

    with pytest.raises(ValueError, match="receipt"):
        replace(
            evidence,
            receipt=replace(evidence.receipt, evidence_sha256="0" * 64),
        )


def test_evidence_exposes_no_public_seal_or_formal_projection() -> None:
    assert not hasattr(PersonalExactOracleEvidence, "seal")
    assert not hasattr(PersonalExactOracleEvidence, "to_evaluation_manifest")
