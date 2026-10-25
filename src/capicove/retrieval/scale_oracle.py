"""Provisional internal personal exact-oracle evidence production.

This module has no filesystem, candidate-index, activation, or qualification
authority. It only computes bounded in-memory evidence from a separately staged,
two-pass representation source. Its output is not formal scale qualification.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
import struct
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from numbers import Real
from typing import TYPE_CHECKING, Protocol, cast

from capicove.contracts.retrieval import (
    DistanceMetric,
    EmbeddingSpaceRef,
    QueryModality,
    TargetType,
    VectorNormalization,
)
from capicove.processing.image.ports import ImageSemanticLane, SemanticTargetIdentity
from capicove.processing.image.representations import ImageTargetRepresentation
from capicove.retrieval.scale import (
    EvaluationQueryEvidence,
    PersonalEvaluationManifest,
)


if TYPE_CHECKING:
    import numpy as np


PERSONAL_EXACT_ORACLE_ALGORITHM_ID = "numpy-streaming-inner-product-f32-v1"
PERSONAL_EXACT_ORACLE_TARGET_COUNT = 10_000
PERSONAL_EXACT_ORACLE_TOP_K = 100
PERSONAL_EXACT_ORACLE_MAX_QUERIES = 1_000
PERSONAL_EXACT_ORACLE_MAX_TARGET_BATCH_SIZE = 1_000

_SCHEMA_VERSION = 1
_COMPUTE_BLOCK_SIZE = 1_000
_MAX_REF_UTF8_BYTES = 512
_MAX_UNIT_ID_UTF8_BYTES = 256
_MAX_RELEVANCE_PER_LANE = 100
_MAX_TOTAL_RELEVANCE = 10_000
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SCORE_BITS_PATTERN = re.compile(r"^[0-9a-f]{8}$")
_QUERY_VECTOR_DOMAIN = b"capicove-personal-query-vector-f32-v1\x00"
_TARGET_SELECTION_DOMAIN = b"capicove-personal-exact-target-selection-v1\x00"
_TARGET_STREAM_DOMAIN = b"capicove-personal-exact-target-stream-v1\x00"
_EVIDENCE_DOMAIN = b"capicove-personal-exact-oracle-evidence-v1\x00"
_RECEIPT_DOMAIN = b"capicove-personal-exact-oracle-receipt-v1\x00"


class StagedRepresentationOracleIdentity(Protocol):
    """Expose the staged identity fields consumed by the personal oracle."""

    @property
    def unit_manifest_sha256(self) -> str: ...

    @property
    def target_representation_manifest_sha256(self) -> str: ...

    @property
    def unit_count(self) -> int: ...

    @property
    def semantic_targets(self) -> tuple[SemanticTargetIdentity, ...]: ...


class StagedRepresentationOracleSource(Protocol):
    """Provide exactly two complete, sequential target traversals."""

    @property
    def identity(self) -> StagedRepresentationOracleIdentity:
        """Return the immutable identity checked before and after both passes."""
        ...

    def iter_batches(
        self,
        *,
        start_ordinal: int,
        maximum_batch_size: int,
        deadline: datetime,
    ) -> Iterator[tuple[ImageTargetRepresentation, ...]]:
        """Yield the next of exactly two full ordinal-zero traversals."""
        ...


@dataclass(frozen=True, slots=True)
class PersonalExactSemanticIdentity:
    """Bind one query bundle to its complete target lane identity."""

    target_identity: SemanticTargetIdentity
    query_bundle_manifest_sha256: str
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.target_identity, SemanticTargetIdentity):
            raise TypeError("target_identity must be a SemanticTargetIdentity")
        _require_sha256(
            "query_bundle_manifest_sha256",
            self.query_bundle_manifest_sha256,
        )
        _require_schema_version(self.schema_version)
        _require_exact_image_space(self.target_identity.space)


@dataclass(frozen=True, slots=True)
class PersonalExactOracleSubject:
    """Freeze the personal-only manifests and semantic lanes being evaluated."""

    evaluation_id: str
    definition_sha256: str
    unit_manifest_sha256: str
    target_representation_manifest_sha256: str
    target_selection_sha256: str
    generation_manifest_sha256: str
    semantic_identities: tuple[PersonalExactSemanticIdentity, ...]
    target_unit_ids: tuple[str, ...]
    exact_target_count: int = PERSONAL_EXACT_ORACLE_TARGET_COUNT
    algorithm_id: str = PERSONAL_EXACT_ORACLE_ALGORITHM_ID
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_ref("evaluation_id", self.evaluation_id)
        for name in (
            "definition_sha256",
            "unit_manifest_sha256",
            "target_representation_manifest_sha256",
            "target_selection_sha256",
            "generation_manifest_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        identities = _semantic_identity_tuple(self.semantic_identities)
        if tuple(item.target_identity.lane for item in identities) != (
            ImageSemanticLane.INSTANCE,
            ImageSemanticLane.CONCEPT,
        ):
            raise ValueError(
                "semantic_identities must contain instance and concept in canonical order"
            )
        spaces = tuple(item.target_identity.space.space_id for item in identities)
        if len(set(spaces)) != len(spaces):
            raise ValueError("semantic lanes must use distinct embedding spaces")
        target_unit_ids = _target_unit_id_tuple(self.target_unit_ids)
        if self.exact_target_count != PERSONAL_EXACT_ORACLE_TARGET_COUNT:
            raise ValueError("exact_target_count must contain all 10,000 personal targets")
        expected_selection_sha256 = personal_exact_target_selection_sha256(
            self.unit_manifest_sha256,
            target_unit_ids,
        )
        if self.target_selection_sha256 != expected_selection_sha256:
            raise ValueError("target_selection_sha256 does not match the ordered selection digest")
        if self.algorithm_id != PERSONAL_EXACT_ORACLE_ALGORITHM_ID:
            raise ValueError(f"algorithm_id must be {PERSONAL_EXACT_ORACLE_ALGORITHM_ID}")
        _require_schema_version(self.schema_version)


@dataclass(frozen=True, slots=True)
class PersonalExactQueryVector:
    """Carry one immutable, content-addressed float32 query vector."""

    query_ref: str
    query_sha256: str
    semantic_identity: PersonalExactSemanticIdentity
    query_vector_sha256: str
    values: tuple[float, ...]
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_ref("query_ref", self.query_ref)
        _require_sha256("query_sha256", self.query_sha256)
        if not isinstance(self.semantic_identity, PersonalExactSemanticIdentity):
            raise TypeError("semantic_identity must be a PersonalExactSemanticIdentity")
        _require_sha256("query_vector_sha256", self.query_vector_sha256)
        _require_schema_version(self.schema_version)
        normalized, payload = _normalized_float32_vector(
            self.values,
            self.semantic_identity.target_identity.space,
            name="query vector",
        )
        expected_sha256 = hashlib.sha256(_QUERY_VECTOR_DOMAIN + payload).hexdigest()
        if self.query_vector_sha256 != expected_sha256:
            raise ValueError("query vector digest does not match its float32 values")
        object.__setattr__(self, "values", normalized)

    @classmethod
    def create(
        cls,
        *,
        query_ref: str,
        query_sha256: str,
        semantic_identity: PersonalExactSemanticIdentity,
        values: tuple[float, ...],
    ) -> PersonalExactQueryVector:
        """Create a vector with the canonical little-endian float32 digest."""
        if not isinstance(semantic_identity, PersonalExactSemanticIdentity):
            raise TypeError("semantic_identity must be a PersonalExactSemanticIdentity")
        normalized, payload = _normalized_float32_vector(
            values,
            semantic_identity.target_identity.space,
            name="query vector",
        )
        return cls(
            query_ref=query_ref,
            query_sha256=query_sha256,
            semantic_identity=semantic_identity,
            query_vector_sha256=hashlib.sha256(_QUERY_VECTOR_DOMAIN + payload).hexdigest(),
            values=normalized,
        )


@dataclass(frozen=True, slots=True)
class PersonalExactOracleQuery:
    """Bind one query image, both vector lanes, and authored relevance."""

    query_ref: str
    query_sha256: str
    vectors: tuple[PersonalExactQueryVector, ...]
    instance_relevance: tuple[tuple[str, int], ...]
    concept_relevance: tuple[tuple[str, int], ...]
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_ref("query_ref", self.query_ref)
        _require_sha256("query_sha256", self.query_sha256)
        if not isinstance(self.vectors, tuple) or any(
            not isinstance(item, PersonalExactQueryVector) for item in self.vectors
        ):
            raise TypeError("vectors must contain PersonalExactQueryVector values")
        if tuple(item.semantic_identity.target_identity.lane for item in self.vectors) != (
            ImageSemanticLane.INSTANCE,
            ImageSemanticLane.CONCEPT,
        ):
            raise ValueError("vectors must contain instance and concept in canonical order")
        if any(
            item.query_ref != self.query_ref or item.query_sha256 != self.query_sha256
            for item in self.vectors
        ):
            raise ValueError("query vector identity does not match its query image")
        _require_relevance("instance_relevance", self.instance_relevance)
        _require_relevance("concept_relevance", self.concept_relevance)
        _require_schema_version(self.schema_version)


@dataclass(frozen=True, slots=True)
class PersonalExactNeighbor:
    """Record one exact rank with canonical big-endian IEEE-754 score bits."""

    unit_id: str
    ordinal: int
    score_bits: str
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_unit_id("unit_id", self.unit_id)
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or not 0 <= self.ordinal < PERSONAL_EXACT_ORACLE_TARGET_COUNT
        ):
            raise ValueError("ordinal must be from 0 through 9,999")
        if (
            not isinstance(self.score_bits, str)
            or _SCORE_BITS_PATTERN.fullmatch(self.score_bits) is None
        ):
            raise ValueError("score_bits must be eight lowercase IEEE-754 hex digits")
        score = struct.unpack(">f", bytes.fromhex(self.score_bits))[0]
        if not math.isfinite(score):
            raise ValueError("score_bits must encode a finite float32 score")
        _require_schema_version(self.schema_version)


@dataclass(frozen=True, slots=True)
class PersonalExactLaneOracle:
    """Seal one query-lane exact top-100 against all personal target identities."""

    query_ref: str
    query_sha256: str
    query_vector_sha256: str
    semantic_identity: PersonalExactSemanticIdentity
    unit_manifest_sha256: str
    target_representation_manifest_sha256: str
    target_selection_sha256: str
    generation_manifest_sha256: str
    exact_target_count: int
    algorithm_id: str
    neighbors: tuple[PersonalExactNeighbor, ...]
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_ref("query_ref", self.query_ref)
        for name in (
            "query_sha256",
            "query_vector_sha256",
            "unit_manifest_sha256",
            "target_representation_manifest_sha256",
            "target_selection_sha256",
            "generation_manifest_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if not isinstance(self.semantic_identity, PersonalExactSemanticIdentity):
            raise TypeError("semantic_identity must be a PersonalExactSemanticIdentity")
        if self.exact_target_count != PERSONAL_EXACT_ORACLE_TARGET_COUNT:
            raise ValueError("exact_target_count must contain all 10,000 personal targets")
        if self.algorithm_id != PERSONAL_EXACT_ORACLE_ALGORITHM_ID:
            raise ValueError(f"algorithm_id must be {PERSONAL_EXACT_ORACLE_ALGORITHM_ID}")
        if not isinstance(self.neighbors, tuple) or any(
            not isinstance(item, PersonalExactNeighbor) for item in self.neighbors
        ):
            raise TypeError("neighbors must contain PersonalExactNeighbor values")
        if len(self.neighbors) != PERSONAL_EXACT_ORACLE_TOP_K:
            raise ValueError("neighbors must contain exactly 100 exact targets")
        if len({item.unit_id for item in self.neighbors}) != len(self.neighbors):
            raise ValueError("neighbors must not repeat a unit ID")
        if len({item.ordinal for item in self.neighbors}) != len(self.neighbors):
            raise ValueError("neighbors must not repeat an ordinal")
        ordered = tuple(sorted(self.neighbors, key=_neighbor_rank_key))
        if self.neighbors != ordered:
            raise ValueError("neighbors must use exact score, ordinal, and unit ordering")
        _require_schema_version(self.schema_version)


@dataclass(frozen=True, slots=True)
class PersonalExactOracleReceipt:
    """Content-address replay-matched oracle content, not its provenance."""

    evaluation_id: str
    definition_sha256: str
    unit_manifest_sha256: str
    target_representation_manifest_sha256: str
    target_selection_sha256: str
    generation_manifest_sha256: str
    semantic_identities: tuple[PersonalExactSemanticIdentity, ...]
    exact_target_count: int
    query_count: int
    algorithm_id: str
    target_stream_sha256: str
    evidence_sha256: str
    receipt_sha256: str
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_ref("evaluation_id", self.evaluation_id)
        for name in (
            "definition_sha256",
            "unit_manifest_sha256",
            "target_representation_manifest_sha256",
            "target_selection_sha256",
            "generation_manifest_sha256",
            "target_stream_sha256",
            "evidence_sha256",
            "receipt_sha256",
        ):
            _require_sha256(f"receipt {name}", getattr(self, name))
        identities = _semantic_identity_tuple(self.semantic_identities)
        if tuple(item.target_identity.lane for item in identities) != (
            ImageSemanticLane.INSTANCE,
            ImageSemanticLane.CONCEPT,
        ):
            raise ValueError("receipt semantic identities must contain instance and concept")
        if self.exact_target_count != PERSONAL_EXACT_ORACLE_TARGET_COUNT:
            raise ValueError("receipt must cover all 10,000 personal targets")
        if (
            isinstance(self.query_count, bool)
            or not isinstance(self.query_count, int)
            or not 2 <= self.query_count <= PERSONAL_EXACT_ORACLE_MAX_QUERIES
        ):
            raise ValueError("receipt query_count must be from 2 through 1,000")
        if self.algorithm_id != PERSONAL_EXACT_ORACLE_ALGORITHM_ID:
            raise ValueError(f"algorithm_id must be {PERSONAL_EXACT_ORACLE_ALGORITHM_ID}")
        _require_schema_version(self.schema_version)
        expected = hashlib.sha256(
            _RECEIPT_DOMAIN + _canonical_json(_receipt_document(self, include_digest=False))
        ).hexdigest()
        if self.receipt_sha256 != expected:
            raise ValueError("receipt digest does not match its canonical body")

    @classmethod
    def create(
        cls,
        subject: PersonalExactOracleSubject,
        *,
        query_count: int,
        target_stream_sha256: str,
        evidence_sha256: str,
    ) -> PersonalExactOracleReceipt:
        """Create a self-digesting receipt for one replay-matched payload."""
        document = {
            "algorithm_id": subject.algorithm_id,
            "definition_sha256": subject.definition_sha256,
            "evaluation_id": subject.evaluation_id,
            "evidence_sha256": evidence_sha256,
            "exact_target_count": subject.exact_target_count,
            "generation_manifest_sha256": subject.generation_manifest_sha256,
            "query_count": query_count,
            "schema_version": _SCHEMA_VERSION,
            "semantic_identities": [
                _exact_semantic_identity_document(item) for item in subject.semantic_identities
            ],
            "target_representation_manifest_sha256": (
                subject.target_representation_manifest_sha256
            ),
            "target_selection_sha256": subject.target_selection_sha256,
            "target_stream_sha256": target_stream_sha256,
            "unit_manifest_sha256": subject.unit_manifest_sha256,
        }
        digest = hashlib.sha256(_RECEIPT_DOMAIN + _canonical_json(document)).hexdigest()
        return cls(
            evaluation_id=subject.evaluation_id,
            definition_sha256=subject.definition_sha256,
            unit_manifest_sha256=subject.unit_manifest_sha256,
            target_representation_manifest_sha256=(subject.target_representation_manifest_sha256),
            target_selection_sha256=subject.target_selection_sha256,
            generation_manifest_sha256=subject.generation_manifest_sha256,
            semantic_identities=subject.semantic_identities,
            exact_target_count=subject.exact_target_count,
            query_count=query_count,
            algorithm_id=subject.algorithm_id,
            target_stream_sha256=target_stream_sha256,
            evidence_sha256=evidence_sha256,
            receipt_sha256=digest,
        )


@dataclass(frozen=True, slots=True)
class PersonalExactOracleEvidence:
    """Hold replay-matched content without proving provenance or qualification."""

    subject: PersonalExactOracleSubject
    queries: tuple[PersonalExactOracleQuery, ...]
    lane_oracles: tuple[PersonalExactLaneOracle, ...]
    target_stream_sha256: str
    receipt: PersonalExactOracleReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.subject, PersonalExactOracleSubject):
            raise TypeError("subject must be a PersonalExactOracleSubject")
        queries = _query_tuple(self.queries)
        _validate_query_set(queries, self.subject)
        if not isinstance(self.lane_oracles, tuple) or any(
            not isinstance(item, PersonalExactLaneOracle) for item in self.lane_oracles
        ):
            raise TypeError("lane_oracles must contain PersonalExactLaneOracle values")
        _require_sha256("target_stream_sha256", self.target_stream_sha256)
        _validate_lane_oracles(self.lane_oracles, queries, self.subject)
        if not isinstance(self.receipt, PersonalExactOracleReceipt):
            raise TypeError("receipt must be a PersonalExactOracleReceipt")
        evidence_sha256 = _evidence_sha256(
            self.subject,
            queries,
            self.lane_oracles,
            self.target_stream_sha256,
        )
        expected_receipt = PersonalExactOracleReceipt.create(
            self.subject,
            query_count=len(queries),
            target_stream_sha256=self.target_stream_sha256,
            evidence_sha256=evidence_sha256,
        )
        if self.receipt != expected_receipt:
            raise ValueError("receipt does not match the exact evidence content")

    def to_provisional_evaluation_manifest(self) -> PersonalEvaluationManifest:
        """Project lossy structural input for the provisional pure evaluator.

        This discards exact score bits and receipt content. It cannot create a
        formal or sealed evaluation manifest and conveys no provenance.
        """
        by_cell = {
            (item.query_ref, item.semantic_identity.target_identity.lane): item
            for item in self.lane_oracles
        }
        query_evidence = tuple(
            EvaluationQueryEvidence(
                query_ref=query.query_ref,
                query_sha256=query.query_sha256,
                instance_relevance=query.instance_relevance,
                concept_relevance=query.concept_relevance,
                instance_exact_unit_ids=tuple(
                    item.unit_id
                    for item in by_cell[(query.query_ref, ImageSemanticLane.INSTANCE)].neighbors
                ),
                concept_exact_unit_ids=tuple(
                    item.unit_id
                    for item in by_cell[(query.query_ref, ImageSemanticLane.CONCEPT)].neighbors
                ),
            )
            for query in self.queries
        )
        return PersonalEvaluationManifest(
            evaluation_id=self.subject.evaluation_id,
            definition_sha256=self.subject.definition_sha256,
            unit_manifest_sha256=self.subject.unit_manifest_sha256,
            target_representation_manifest_sha256=(
                self.subject.target_representation_manifest_sha256
            ),
            target_selection_sha256=self.subject.target_selection_sha256,
            exact_target_count=self.subject.exact_target_count,
            generation_manifest_sha256=self.subject.generation_manifest_sha256,
            query_count=len(query_evidence),
            queries=query_evidence,
        )


def _create_personal_exact_oracle_evidence(
    subject: PersonalExactOracleSubject,
    queries: tuple[PersonalExactOracleQuery, ...],
    lane_oracles: tuple[PersonalExactLaneOracle, ...],
    *,
    target_stream_sha256: str,
) -> PersonalExactOracleEvidence:
    evidence_sha256 = _evidence_sha256(
        subject,
        queries,
        lane_oracles,
        target_stream_sha256,
    )
    receipt = PersonalExactOracleReceipt.create(
        subject,
        query_count=len(queries),
        target_stream_sha256=target_stream_sha256,
        evidence_sha256=evidence_sha256,
    )
    return PersonalExactOracleEvidence(
        subject=subject,
        queries=queries,
        lane_oracles=lane_oracles,
        target_stream_sha256=target_stream_sha256,
        receipt=receipt,
    )


@dataclass(frozen=True, slots=True)
class _SourceIdentitySnapshot:
    unit_manifest_sha256: str
    target_representation_manifest_sha256: str
    unit_count: int
    semantic_targets: tuple[SemanticTargetIdentity, ...]


@dataclass(frozen=True, slots=True)
class _ScoredTarget:
    score: float
    ordinal: int
    unit_id: str


@dataclass(frozen=True, slots=True)
class _ScanResult:
    lane_oracles: tuple[PersonalExactLaneOracle, ...]
    target_stream_sha256: str
    unit_ids: frozenset[str]


def personal_exact_target_selection_sha256(
    unit_manifest_sha256: str,
    target_unit_ids: tuple[str, ...],
) -> str:
    """Hash the frozen all-10,000 target order for one unit manifest."""
    _require_sha256("unit_manifest_sha256", unit_manifest_sha256)
    normalized = _target_unit_id_tuple(target_unit_ids)
    document = {
        "ordered_target_unit_ids": list(normalized),
        "schema_version": _SCHEMA_VERSION,
        "selection_kind": "all-personal-units-by-manifest-ordinal",
        "unit_count": PERSONAL_EXACT_ORACLE_TARGET_COUNT,
        "unit_manifest_sha256": unit_manifest_sha256,
    }
    return hashlib.sha256(_TARGET_SELECTION_DOMAIN + _canonical_json(document)).hexdigest()


def produce_personal_exact_oracle(
    source: StagedRepresentationOracleSource,
    subject: PersonalExactOracleSubject,
    queries: Iterable[PersonalExactOracleQuery],
    *,
    maximum_target_batch_size: int,
    deadline: datetime,
) -> PersonalExactOracleEvidence:
    """Compute and replay provisional all-target top-100 content."""
    if not isinstance(subject, PersonalExactOracleSubject):
        raise TypeError("subject must be a PersonalExactOracleSubject")
    _check_deadline(deadline)
    _require_target_batch_size(maximum_target_batch_size)
    materialized_queries = _consume_queries(queries, deadline=deadline)
    _validate_query_set(materialized_queries, subject)
    query_snapshot = _query_snapshot_sha256(materialized_queries)
    subject_snapshot = _subject_sha256(subject)
    source_snapshot = _read_source_identity(source)
    _require_source_matches_subject(source_snapshot, subject)

    query_matrices = _prepare_query_matrices(materialized_queries)
    first = _scan_exact_targets(
        source,
        source_snapshot,
        subject,
        materialized_queries,
        query_matrices,
        maximum_target_batch_size=maximum_target_batch_size,
        deadline=deadline,
    )
    _require_inputs_unchanged(
        source,
        source_snapshot,
        subject,
        subject_snapshot,
        materialized_queries,
        query_snapshot,
    )
    _require_relevance_within_targets(materialized_queries, first.unit_ids)

    second = _scan_exact_targets(
        source,
        source_snapshot,
        subject,
        materialized_queries,
        query_matrices,
        maximum_target_batch_size=maximum_target_batch_size,
        deadline=deadline,
    )
    _require_inputs_unchanged(
        source,
        source_snapshot,
        subject,
        subject_snapshot,
        materialized_queries,
        query_snapshot,
    )
    if (
        second.target_stream_sha256 != first.target_stream_sha256
        or second.unit_ids != first.unit_ids
        or second.lane_oracles != first.lane_oracles
    ):
        raise ValueError("second pass does not exactly match first-pass evidence")
    _check_deadline(deadline)
    evidence = _create_personal_exact_oracle_evidence(
        subject,
        materialized_queries,
        first.lane_oracles,
        target_stream_sha256=first.target_stream_sha256,
    )
    _check_deadline(deadline)
    return evidence


def _scan_exact_targets(
    source: StagedRepresentationOracleSource,
    source_snapshot: _SourceIdentitySnapshot,
    subject: PersonalExactOracleSubject,
    queries: tuple[PersonalExactOracleQuery, ...],
    query_matrices: tuple[np.ndarray, np.ndarray],
    *,
    maximum_target_batch_size: int,
    deadline: datetime,
) -> _ScanResult:
    _check_deadline(deadline)
    if _read_source_identity(source) != source_snapshot:
        raise ValueError("staged representation source identity drifted")
    winners: tuple[list[list[_ScoredTarget]], list[list[_ScoredTarget]]] = (
        [[] for _ in queries],
        [[] for _ in queries],
    )
    unit_ids, target_stream_sha256 = _consume_target_stream(
        source,
        subject,
        queries,
        query_matrices,
        winners,
        maximum_target_batch_size=maximum_target_batch_size,
        deadline=deadline,
    )
    if _read_source_identity(source) != source_snapshot:
        raise ValueError("staged representation source identity drifted during traversal")
    oracles: list[PersonalExactLaneOracle] = []
    for query_index, query in enumerate(queries):
        _check_deadline(deadline)
        for lane_index in range(2):
            vector = query.vectors[lane_index]
            neighbors = tuple(
                PersonalExactNeighbor(
                    unit_id=item.unit_id,
                    ordinal=item.ordinal,
                    score_bits=struct.pack(">f", item.score).hex(),
                )
                for item in winners[lane_index][query_index]
            )
            oracles.append(
                PersonalExactLaneOracle(
                    query_ref=query.query_ref,
                    query_sha256=query.query_sha256,
                    query_vector_sha256=vector.query_vector_sha256,
                    semantic_identity=vector.semantic_identity,
                    unit_manifest_sha256=subject.unit_manifest_sha256,
                    target_representation_manifest_sha256=(
                        subject.target_representation_manifest_sha256
                    ),
                    target_selection_sha256=subject.target_selection_sha256,
                    generation_manifest_sha256=subject.generation_manifest_sha256,
                    exact_target_count=subject.exact_target_count,
                    algorithm_id=subject.algorithm_id,
                    neighbors=neighbors,
                )
            )
    return _ScanResult(tuple(oracles), target_stream_sha256, unit_ids)


def _consume_target_stream(
    source: StagedRepresentationOracleSource,
    subject: PersonalExactOracleSubject,
    queries: tuple[PersonalExactOracleQuery, ...],
    query_matrices: tuple[np.ndarray, np.ndarray],
    winners: tuple[list[list[_ScoredTarget]], list[list[_ScoredTarget]]],
    *,
    maximum_target_batch_size: int,
    deadline: datetime,
) -> tuple[frozenset[str], str]:
    batches = source.iter_batches(
        start_ordinal=0,
        maximum_batch_size=maximum_target_batch_size,
        deadline=deadline,
    )
    try:
        iterator = iter(batches)
    except TypeError as error:
        raise TypeError("staged representation batches must be iterable") from error
    expected_ordinal = 0
    unit_ids: set[str] = set()
    stream_digest = hashlib.sha256(_TARGET_STREAM_DOMAIN)
    compute_rows: list[ImageTargetRepresentation] = []
    primary_error: BaseException | None = None
    try:
        for batch in iterator:
            _check_deadline(deadline)
            if not isinstance(batch, tuple) or not batch:
                raise TypeError("each staged representation batch must be a non-empty tuple")
            if len(batch) > maximum_target_batch_size:
                raise ValueError("staged representation batch exceeds the requested batch bound")
            if expected_ordinal + len(batch) > PERSONAL_EXACT_ORACLE_TARGET_COUNT:
                raise ValueError("staged representation source exceeds all 10,000 targets")
            for row in batch:
                _check_deadline(deadline)
                if not isinstance(row, ImageTargetRepresentation):
                    raise TypeError(
                        "staged representation batches must contain ImageTargetRepresentation"
                    )
                if row.ordinal != expected_ordinal:
                    raise ValueError("target ordinals must be contiguous from zero")
                _require_unit_id("target unit_id", row.unit_id)
                if row.unit_id != subject.target_unit_ids[expected_ordinal]:
                    raise ValueError("target unit ID does not match the frozen target selection")
                if row.unit_id in unit_ids:
                    raise ValueError("target unit IDs must be unique")
                if tuple(item.identity for item in row.embeddings) != (
                    subject.semantic_identities[0].target_identity,
                    subject.semantic_identities[1].target_identity,
                ):
                    raise ValueError("target semantic identity does not match oracle subject")
                unit_ids.add(row.unit_id)
                _update_target_stream_digest(stream_digest, row)
                compute_rows.append(row)
                expected_ordinal += 1
                if len(compute_rows) == _COMPUTE_BLOCK_SIZE:
                    _score_compute_block(
                        tuple(compute_rows),
                        subject,
                        queries,
                        query_matrices,
                        winners,
                        deadline=deadline,
                    )
                    compute_rows.clear()
        if expected_ordinal != PERSONAL_EXACT_ORACLE_TARGET_COUNT:
            raise ValueError("staged representation source must contain exactly 10,000 targets")
        if compute_rows:
            raise ValueError("internal compute blocks do not cover all 10,000 targets")
        if len(unit_ids) != PERSONAL_EXACT_ORACLE_TARGET_COUNT:
            raise ValueError("staged representation source must contain 10,000 unique unit IDs")
    except BaseException as error:
        primary_error = error
        raise
    finally:
        _close_iterator(iterator, primary_error=primary_error)
    return frozenset(unit_ids), stream_digest.hexdigest()


def _score_compute_block(
    rows: tuple[ImageTargetRepresentation, ...],
    subject: PersonalExactOracleSubject,
    queries: tuple[PersonalExactOracleQuery, ...],
    query_matrices: tuple[np.ndarray, np.ndarray],
    winners: tuple[list[list[_ScoredTarget]], list[list[_ScoredTarget]]],
    *,
    deadline: datetime,
) -> None:
    import numpy as np

    if len(rows) != _COMPUTE_BLOCK_SIZE:
        raise ValueError("exact-oracle compute blocks must contain exactly 1,000 targets")
    batch_targets = tuple(_ScoredTarget(0.0, row.ordinal, row.unit_id) for row in rows)
    for lane_index in range(2):
        _check_deadline(deadline)
        target_matrix = np.asarray(
            [row.embeddings[lane_index].values for row in rows],
            dtype="<f4",
        )
        expected_dimension = subject.semantic_identities[lane_index].target_identity.space.dimension
        if target_matrix.shape != (_COMPUTE_BLOCK_SIZE, expected_dimension):
            raise ValueError("target vector dimension does not match semantic identity")
        if not bool(np.isfinite(target_matrix).all()):
            raise ValueError("target vectors must remain finite after float32 conversion")
        norms = np.linalg.norm(target_matrix, axis=1)
        if not bool(np.allclose(norms, np.float32(1.0), rtol=1e-6, atol=1e-6)):
            raise ValueError("target vectors must remain L2 normalized as float32")
        scores = target_matrix @ query_matrices[lane_index].T
        _check_deadline(deadline)
        if scores.dtype != np.dtype(np.float32) or not bool(np.isfinite(scores).all()):
            raise ValueError("exact inner-product scores must be finite float32 values")
        for query_index in range(len(queries)):
            _check_deadline(deadline)
            candidates = winners[lane_index][query_index]
            candidates.extend(
                _ScoredTarget(
                    float(scores[row_index, query_index]),
                    target.ordinal,
                    target.unit_id,
                )
                for row_index, target in enumerate(batch_targets)
            )
            winners[lane_index][query_index] = heapq.nsmallest(
                PERSONAL_EXACT_ORACLE_TOP_K,
                candidates,
                key=_scored_target_rank_key,
            )


def _prepare_query_matrices(
    queries: tuple[PersonalExactOracleQuery, ...],
) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    matrices: list[np.ndarray] = []
    for lane_index in range(2):
        matrix = np.asarray(
            [query.vectors[lane_index].values for query in queries],
            dtype="<f4",
        )
        if matrix.dtype != np.dtype(np.float32) or not bool(np.isfinite(matrix).all()):
            raise ValueError("query vectors must be finite float32 values")
        matrices.append(np.ascontiguousarray(matrix, dtype=np.float32))
    return cast(tuple[np.ndarray, np.ndarray], tuple(matrices))


def _consume_queries(
    queries: Iterable[PersonalExactOracleQuery],
    *,
    deadline: datetime,
) -> tuple[PersonalExactOracleQuery, ...]:
    try:
        iterator = iter(queries)
    except TypeError as error:
        raise TypeError("queries must be an iterable of PersonalExactOracleQuery") from error
    primary_error: BaseException | None = None
    try:
        values: list[PersonalExactOracleQuery] = []
        for _ in range(PERSONAL_EXACT_ORACLE_MAX_QUERIES + 1):
            _check_deadline(deadline)
            try:
                item = next(iterator)
            except StopIteration:
                break
            if not isinstance(item, PersonalExactOracleQuery):
                raise TypeError("queries must contain PersonalExactOracleQuery values")
            values.append(item)
        if len(values) > PERSONAL_EXACT_ORACLE_MAX_QUERIES:
            raise ValueError("queries must contain at most 1,000 values")
        if len(values) < 2:
            raise ValueError("queries must contain at least 2 values")
        return tuple(values)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        _close_iterator(iterator, primary_error=primary_error)


def _validate_query_set(
    queries: tuple[PersonalExactOracleQuery, ...],
    subject: PersonalExactOracleSubject,
) -> None:
    if not 2 <= len(queries) <= PERSONAL_EXACT_ORACLE_MAX_QUERIES:
        raise ValueError("queries must contain from 2 through 1,000 values")
    refs = tuple(item.query_ref for item in queries)
    hashes = tuple(item.query_sha256 for item in queries)
    if len(set(refs)) != len(refs) or len(set(hashes)) != len(hashes):
        raise ValueError("query refs and hashes must be unique")
    total_relevance = sum(
        len(query.instance_relevance) + len(query.concept_relevance) for query in queries
    )
    if total_relevance > _MAX_TOTAL_RELEVANCE:
        raise ValueError(f"total relevance must contain at most {_MAX_TOTAL_RELEVANCE:,} entries")
    for query in queries:
        if tuple(item.semantic_identity for item in query.vectors) != subject.semantic_identities:
            raise ValueError("query semantic identities do not match oracle subject")
        for vector in query.vectors:
            _validate_query_vector(vector)


def _close_iterator(
    iterator: object,
    *,
    primary_error: BaseException | None,
) -> None:
    close = getattr(iterator, "close", None)
    if not callable(close):
        return
    try:
        close()
    except BaseException as close_error:
        if primary_error is None:
            raise
        primary_error.add_note(
            f"secondary iterator close failure: {type(close_error).__name__}: {close_error}"
        )


def _validate_query_vector(vector: PersonalExactQueryVector) -> None:
    normalized, payload = _normalized_float32_vector(
        vector.values,
        vector.semantic_identity.target_identity.space,
        name="query vector",
    )
    expected = hashlib.sha256(_QUERY_VECTOR_DOMAIN + payload).hexdigest()
    if vector.values != normalized or vector.query_vector_sha256 != expected:
        raise ValueError("query vector digest or float32 values drifted")


def _validate_lane_oracles(
    oracles: tuple[PersonalExactLaneOracle, ...],
    queries: tuple[PersonalExactOracleQuery, ...],
    subject: PersonalExactOracleSubject,
) -> None:
    expected_cells = tuple(
        (query.query_ref, identity.target_identity.lane)
        for query in queries
        for identity in subject.semantic_identities
    )
    actual_cells = tuple(
        (item.query_ref, item.semantic_identity.target_identity.lane) for item in oracles
    )
    if actual_cells != expected_cells:
        raise ValueError("lane_oracles must contain every query lane in canonical order")
    by_ref = {item.query_ref: item for item in queries}
    for oracle in oracles:
        query = by_ref[oracle.query_ref]
        lane_index = (
            0 if oracle.semantic_identity.target_identity.lane is ImageSemanticLane.INSTANCE else 1
        )
        vector = query.vectors[lane_index]
        if (
            oracle.query_sha256 != query.query_sha256
            or oracle.query_vector_sha256 != vector.query_vector_sha256
            or oracle.semantic_identity != vector.semantic_identity
        ):
            raise ValueError("lane oracle query identity does not match bound query")
        if (
            oracle.unit_manifest_sha256 != subject.unit_manifest_sha256
            or oracle.target_representation_manifest_sha256
            != subject.target_representation_manifest_sha256
            or oracle.target_selection_sha256 != subject.target_selection_sha256
            or oracle.generation_manifest_sha256 != subject.generation_manifest_sha256
            or oracle.exact_target_count != subject.exact_target_count
            or oracle.algorithm_id != subject.algorithm_id
        ):
            raise ValueError("lane oracle subject identity does not match bound subject")


def _read_source_identity(
    source: StagedRepresentationOracleSource,
) -> _SourceIdentitySnapshot:
    try:
        identity = source.identity
        unit_manifest_sha256 = identity.unit_manifest_sha256
        target_manifest_sha256 = identity.target_representation_manifest_sha256
        unit_count = identity.unit_count
        semantic_targets = identity.semantic_targets
    except AttributeError as error:
        raise TypeError("source identity is missing a required staged field") from error
    _require_sha256("source unit_manifest_sha256", unit_manifest_sha256)
    _require_sha256(
        "source target_representation_manifest_sha256",
        target_manifest_sha256,
    )
    if isinstance(unit_count, bool) or not isinstance(unit_count, int):
        raise TypeError("source unit_count must be an integer")
    if not isinstance(semantic_targets, tuple) or any(
        not isinstance(item, SemanticTargetIdentity) for item in semantic_targets
    ):
        raise TypeError("source semantic_targets must contain SemanticTargetIdentity values")
    return _SourceIdentitySnapshot(
        unit_manifest_sha256,
        target_manifest_sha256,
        unit_count,
        semantic_targets,
    )


def _require_source_matches_subject(
    source: _SourceIdentitySnapshot,
    subject: PersonalExactOracleSubject,
) -> None:
    if source.unit_count != PERSONAL_EXACT_ORACLE_TARGET_COUNT:
        raise ValueError("source identity must declare exactly 10,000 targets")
    if source.unit_manifest_sha256 != subject.unit_manifest_sha256:
        raise ValueError("source unit manifest does not match oracle subject")
    if (
        source.target_representation_manifest_sha256
        != subject.target_representation_manifest_sha256
    ):
        raise ValueError("source target representation manifest does not match oracle subject")
    if source.semantic_targets != tuple(
        item.target_identity for item in subject.semantic_identities
    ):
        raise ValueError("source semantic target identities do not match oracle subject")


def _require_inputs_unchanged(
    source: StagedRepresentationOracleSource,
    source_snapshot: _SourceIdentitySnapshot,
    subject: PersonalExactOracleSubject,
    subject_snapshot: str,
    queries: tuple[PersonalExactOracleQuery, ...],
    query_snapshot: str,
) -> None:
    if _read_source_identity(source) != source_snapshot:
        raise ValueError("staged representation source identity drifted")
    if _subject_sha256(subject) != subject_snapshot:
        raise ValueError("oracle manifest, selection, or generation identity drifted")
    if _query_snapshot_sha256(queries) != query_snapshot:
        raise ValueError("oracle query identity or vector drifted")


def _require_relevance_within_targets(
    queries: tuple[PersonalExactOracleQuery, ...],
    unit_ids: frozenset[str],
) -> None:
    for query in queries:
        relevance_ids = {
            unit_id
            for relevance in (query.instance_relevance, query.concept_relevance)
            for unit_id, _ in relevance
        }
        if not relevance_ids.issubset(unit_ids):
            raise ValueError("authored relevance contains a unit outside the all-target selection")


def _normalized_float32_vector(
    values: object,
    space: EmbeddingSpaceRef,
    *,
    name: str,
) -> tuple[tuple[float, ...], bytes]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} values must be an immutable tuple")
    if len(values) != space.dimension:
        raise ValueError(f"{name} dimension must match its semantic identity")
    normalized: list[float] = []
    payload = bytearray()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} values must be real numbers")
        as_float = float(value)
        if not math.isfinite(as_float):
            raise ValueError(f"{name} values must be finite")
        try:
            packed = struct.pack("<f", as_float)
        except OverflowError as error:
            raise ValueError(f"{name} values must remain finite as float32") from error
        quantized = struct.unpack("<f", packed)[0]
        if not math.isfinite(quantized):
            raise ValueError(f"{name} values must remain finite as float32")
        normalized.append(quantized)
        payload.extend(packed)
    norm = math.sqrt(math.fsum(value * value for value in normalized))
    if not math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"{name} must be L2 normalized")
    return tuple(normalized), bytes(payload)


def _require_exact_image_space(space: EmbeddingSpaceRef) -> None:
    if not isinstance(space, EmbeddingSpaceRef):
        raise TypeError("semantic target space must be an EmbeddingSpaceRef")
    if space.normalization is not VectorNormalization.L2:
        raise ValueError("exact oracle spaces must use L2 normalization")
    if space.distance_metric is not DistanceMetric.INNER_PRODUCT:
        raise ValueError("exact oracle spaces must use inner product")
    if not space.accepts(QueryModality.IMAGE, TargetType.IMAGE):
        raise ValueError("exact oracle spaces must support image-to-image retrieval")


def _semantic_identity_tuple(
    values: object,
) -> tuple[PersonalExactSemanticIdentity, ...]:
    if not isinstance(values, tuple) or any(
        not isinstance(item, PersonalExactSemanticIdentity) for item in values
    ):
        raise TypeError("semantic_identities must contain PersonalExactSemanticIdentity values")
    if len(values) != 2:
        raise ValueError("semantic_identities must contain exactly two lanes")
    return cast(tuple[PersonalExactSemanticIdentity, ...], values)


def _query_tuple(values: object) -> tuple[PersonalExactOracleQuery, ...]:
    if not isinstance(values, tuple) or any(
        not isinstance(item, PersonalExactOracleQuery) for item in values
    ):
        raise TypeError("queries must contain PersonalExactOracleQuery values")
    return cast(tuple[PersonalExactOracleQuery, ...], values)


def _target_unit_id_tuple(values: object) -> tuple[str, ...]:
    if not isinstance(values, tuple) or any(not isinstance(item, str) for item in values):
        raise TypeError("target_unit_ids must contain strings")
    if len(values) != PERSONAL_EXACT_ORACLE_TARGET_COUNT:
        raise ValueError("target_unit_ids must contain exactly 10,000 values")
    normalized = cast(tuple[str, ...], values)
    for unit_id in normalized:
        _require_unit_id("target unit_id", unit_id)
    if len(set(normalized)) != len(normalized):
        raise ValueError("target_unit_ids must be unique")
    return normalized


def _require_relevance(name: str, values: object) -> None:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a non-empty tuple")
    if len(values) > _MAX_RELEVANCE_PER_LANE:
        raise ValueError(f"{name} must contain at most {_MAX_RELEVANCE_PER_LANE} entries")
    unit_ids: list[str] = []
    positive = False
    for item in values:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(f"{name} must contain (unit_id, grade) tuples")
        unit_id, grade = item
        _require_unit_id("relevance unit_id", unit_id)
        if isinstance(grade, bool) or not isinstance(grade, int) or not 0 <= grade <= 3:
            raise ValueError("relevance grades must be integers from zero through three")
        unit_ids.append(cast(str, unit_id))
        positive = positive or grade > 0
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError(f"{name} must not repeat a unit ID")
    if not positive:
        raise ValueError(f"{name} must contain positive relevance")


def _require_target_batch_size(value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= PERSONAL_EXACT_ORACLE_MAX_TARGET_BATCH_SIZE
    ):
        raise ValueError("maximum_target_batch_size must be from 1 through 1,000")


def _update_target_stream_digest(
    digest: object,
    row: ImageTargetRepresentation,
) -> None:
    hasher = cast("hashlib._Hash", digest)
    unit_id = row.unit_id.encode("utf-8")
    hasher.update(struct.pack(">Q", row.ordinal))
    hasher.update(struct.pack(">I", len(unit_id)))
    hasher.update(unit_id)
    hasher.update(bytes.fromhex(row.content_sha256))
    for embedding in row.embeddings:
        hasher.update(bytes.fromhex(_semantic_identity_sha256(embedding.identity)))
        for value in embedding.values:
            hasher.update(struct.pack("<f", value))


def _scored_target_rank_key(item: _ScoredTarget) -> tuple[float, int, str]:
    return (-item.score, item.ordinal, item.unit_id)


def _neighbor_rank_key(item: PersonalExactNeighbor) -> tuple[float, int, str]:
    score = struct.unpack(">f", bytes.fromhex(item.score_bits))[0]
    return (-score, item.ordinal, item.unit_id)


def _query_snapshot_sha256(queries: tuple[PersonalExactOracleQuery, ...]) -> str:
    for query in queries:
        for vector in query.vectors:
            _validate_query_vector(vector)
    return hashlib.sha256(_canonical_json([_query_document(item) for item in queries])).hexdigest()


def _subject_sha256(subject: PersonalExactOracleSubject) -> str:
    return hashlib.sha256(_canonical_json(_subject_document(subject))).hexdigest()


def _evidence_sha256(
    subject: PersonalExactOracleSubject,
    queries: tuple[PersonalExactOracleQuery, ...],
    oracles: tuple[PersonalExactLaneOracle, ...],
    target_stream_sha256: str,
) -> str:
    _require_sha256("target_stream_sha256", target_stream_sha256)
    document = {
        "lane_oracles": [_lane_oracle_document(item) for item in oracles],
        "queries": [_query_document(item) for item in queries],
        "subject": _subject_document(subject),
        "target_stream_sha256": target_stream_sha256,
    }
    return hashlib.sha256(_EVIDENCE_DOMAIN + _canonical_json(document)).hexdigest()


def _subject_document(subject: PersonalExactOracleSubject) -> dict[str, object]:
    return {
        "algorithm_id": subject.algorithm_id,
        "definition_sha256": subject.definition_sha256,
        "evaluation_id": subject.evaluation_id,
        "exact_target_count": subject.exact_target_count,
        "generation_manifest_sha256": subject.generation_manifest_sha256,
        "schema_version": subject.schema_version,
        "semantic_identities": [
            _exact_semantic_identity_document(item) for item in subject.semantic_identities
        ],
        "target_representation_manifest_sha256": (subject.target_representation_manifest_sha256),
        "target_selection_sha256": subject.target_selection_sha256,
        "target_unit_ids": list(subject.target_unit_ids),
        "unit_manifest_sha256": subject.unit_manifest_sha256,
    }


def _query_document(query: PersonalExactOracleQuery) -> dict[str, object]:
    return {
        "concept_relevance": [list(item) for item in query.concept_relevance],
        "instance_relevance": [list(item) for item in query.instance_relevance],
        "query_ref": query.query_ref,
        "query_sha256": query.query_sha256,
        "schema_version": query.schema_version,
        "vectors": [
            {
                "query_ref": item.query_ref,
                "query_sha256": item.query_sha256,
                "query_vector_sha256": item.query_vector_sha256,
                "schema_version": item.schema_version,
                "semantic_identity": _exact_semantic_identity_document(item.semantic_identity),
            }
            for item in query.vectors
        ],
    }


def _lane_oracle_document(oracle: PersonalExactLaneOracle) -> dict[str, object]:
    return {
        "algorithm_id": oracle.algorithm_id,
        "exact_target_count": oracle.exact_target_count,
        "generation_manifest_sha256": oracle.generation_manifest_sha256,
        "neighbors": [
            {
                "ordinal": item.ordinal,
                "schema_version": item.schema_version,
                "score_bits": item.score_bits,
                "unit_id": item.unit_id,
            }
            for item in oracle.neighbors
        ],
        "query_ref": oracle.query_ref,
        "query_sha256": oracle.query_sha256,
        "query_vector_sha256": oracle.query_vector_sha256,
        "schema_version": oracle.schema_version,
        "semantic_identity": _exact_semantic_identity_document(oracle.semantic_identity),
        "target_representation_manifest_sha256": (oracle.target_representation_manifest_sha256),
        "target_selection_sha256": oracle.target_selection_sha256,
        "unit_manifest_sha256": oracle.unit_manifest_sha256,
    }


def _receipt_document(
    receipt: PersonalExactOracleReceipt,
    *,
    include_digest: bool,
) -> dict[str, object]:
    document: dict[str, object] = {
        "algorithm_id": receipt.algorithm_id,
        "definition_sha256": receipt.definition_sha256,
        "evaluation_id": receipt.evaluation_id,
        "evidence_sha256": receipt.evidence_sha256,
        "exact_target_count": receipt.exact_target_count,
        "generation_manifest_sha256": receipt.generation_manifest_sha256,
        "query_count": receipt.query_count,
        "schema_version": receipt.schema_version,
        "semantic_identities": [
            _exact_semantic_identity_document(item) for item in receipt.semantic_identities
        ],
        "target_representation_manifest_sha256": (receipt.target_representation_manifest_sha256),
        "target_selection_sha256": receipt.target_selection_sha256,
        "target_stream_sha256": receipt.target_stream_sha256,
        "unit_manifest_sha256": receipt.unit_manifest_sha256,
    }
    if include_digest:
        document["receipt_sha256"] = receipt.receipt_sha256
    return document


def _exact_semantic_identity_document(
    identity: PersonalExactSemanticIdentity,
) -> dict[str, object]:
    return {
        "query_bundle_manifest_sha256": identity.query_bundle_manifest_sha256,
        "schema_version": identity.schema_version,
        "target_identity": _semantic_target_document(identity.target_identity),
    }


def _semantic_target_document(identity: SemanticTargetIdentity) -> dict[str, object]:
    space = identity.space
    return {
        "bundle_manifest_sha256": identity.bundle_manifest_sha256,
        "lane": identity.lane.value,
        "space": {
            "dimension": space.dimension,
            "distance_metric": space.distance_metric.value,
            "model_revision": space.model_revision,
            "normalization": space.normalization.value,
            "preprocessing_version": space.preprocessing_version,
            "provider_id": space.provider_id,
            "query_modalities": sorted(item.value for item in space.query_modalities),
            "schema_version": space.schema_version,
            "space_id": space.space_id,
            "target_modalities": sorted(item.value for item in space.target_modalities),
        },
    }


def _semantic_identity_sha256(identity: SemanticTargetIdentity) -> str:
    return hashlib.sha256(_canonical_json(_semantic_target_document(identity))).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _require_ref(name: str, value: object) -> None:
    _require_bounded_text(name, value, maximum_bytes=_MAX_REF_UTF8_BYTES)


def _require_unit_id(name: str, value: object) -> None:
    _require_bounded_text(name, value, maximum_bytes=_MAX_UNIT_ID_UTF8_BYTES)


def _require_bounded_text(name: str, value: object, *, maximum_bytes: int) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a non-blank string")
    # A Unicode code point always consumes at least one UTF-8 byte.  Rejecting
    # by character count first prevents hostile oversized values from forcing
    # an equally oversized temporary string or encoded bytes allocation.
    if len(value) > maximum_bytes:
        raise ValueError(f"{name} must be at most {maximum_bytes} UTF-8 bytes")
    if not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a non-blank string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must be valid UTF-8") from error
    if len(encoded) > maximum_bytes:
        raise ValueError(f"{name} must be at most {maximum_bytes} UTF-8 bytes")


def _require_sha256(name: str, value: object) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_schema_version(value: object) -> None:
    if isinstance(value, bool) or value != _SCHEMA_VERSION:
        raise ValueError("schema_version must be 1")


def _check_deadline(deadline: datetime) -> None:
    if (
        not isinstance(deadline, datetime)
        or deadline.tzinfo is None
        or deadline.utcoffset() is None
    ):
        raise ValueError("deadline must be a timezone-aware datetime")
    if datetime.now(UTC) >= deadline.astimezone(UTC):
        raise TimeoutError("personal exact-oracle deadline exceeded")


__all__ = [
    "PERSONAL_EXACT_ORACLE_ALGORITHM_ID",
    "PERSONAL_EXACT_ORACLE_MAX_QUERIES",
    "PERSONAL_EXACT_ORACLE_MAX_TARGET_BATCH_SIZE",
    "PERSONAL_EXACT_ORACLE_TARGET_COUNT",
    "PERSONAL_EXACT_ORACLE_TOP_K",
    "PersonalExactLaneOracle",
    "PersonalExactNeighbor",
    "PersonalExactOracleEvidence",
    "PersonalExactOracleQuery",
    "PersonalExactOracleReceipt",
    "PersonalExactOracleSubject",
    "PersonalExactQueryVector",
    "PersonalExactSemanticIdentity",
    "StagedRepresentationOracleIdentity",
    "StagedRepresentationOracleSource",
    "personal_exact_target_selection_sha256",
    "produce_personal_exact_oracle",
]
