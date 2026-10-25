"""Tests for profile-independent intent-routed reverse-image search."""

from __future__ import annotations

import hashlib
from concurrent.futures import Future
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest
from PIL import Image

from capicove.contracts.retrieval import (
    MAX_CURSOR_BYTES,
    MAX_REF_BYTES,
    DetailLevel,
    DistanceMetric,
    EmbeddingSpaceRef,
    ExecutionProfile,
    FullHit,
    HashSpaceRef,
    ImageQuery,
    InlineBytesSource,
    MatchKind,
    ProcessingFingerprint,
    QueryModality,
    SearchableUnit,
    SearchFailureCode,
    SearchFilters,
    SearchIntent,
    SearchRequest,
    SearchStatus,
    SearchWarningCode,
    TargetType,
    TextQuery,
    VectorNormalization,
    VideoClipQuery,
)
from capicove.contracts.tools import PrivacyClass
from capicove.retrieval import search as search_module
from capicove.retrieval.cursor import HmacSearchCursorCodec
from capicove.retrieval.fusion import WeightedReciprocalRankFusion
from capicove.retrieval.image_generations import GenerationSetManifest
from capicove.retrieval.search import CandidateEvaluationRetrievalService, RetrievalService
from capicove.retrieval.search_models import (
    Candidate,
    ContentCandidateProvenance,
    DeferredCleanupReservation,
    DeferredCleanupSupervisor,
    IndexedCandidateProvenance,
    InvocationAuthorizationContext,
    LaneDeadlineExceeded,
    PerceptualHashQuery,
    QueryVector,
    ResolvedImageQuery,
    RetrievalLane,
    SearchFailure,
)
from capicove.retrieval.search_ports import GenerationSetProvider
from capicove.runtime.retrieval import (
    BoundedLaneScheduler,
    CpuExecutionBudget,
    default_retrieval_profiles,
)


FUTURE = datetime.now(UTC) + timedelta(hours=1)
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _space(lane: RetrievalLane) -> EmbeddingSpaceRef:
    return EmbeddingSpaceRef(
        space_id=f"{lane.value}-space-v1",
        provider_id=f"{lane.value}-provider",
        model_revision="revision-1",
        preprocessing_version="rgb-v1",
        dimension=2,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.INNER_PRODUCT,
        query_modalities=frozenset({QueryModality.IMAGE}),
        target_modalities=frozenset({TargetType.IMAGE}),
    )


def _hash_space() -> HashSpaceRef:
    return HashSpaceRef(
        space_id="opencv-phash-64-v1",
        provider_id="opencv-phash",
        provider_version="4.12.0",
        preprocessing_version="opencv-phash-preprocess-v1",
        algorithm="phash",
        bit_width=64,
        hash_version="phash-64-v1",
        target_modalities=frozenset({TargetType.IMAGE}),
    )


def _unit(unit_id: str) -> SearchableUnit:
    return SearchableUnit(
        unit_id=unit_id,
        unit_type=TargetType.IMAGE,
        source_hash=SHA_B,
        artifact_ref=f"artifact:{unit_id}",
        evidence_ref=f"evidence:{unit_id}",
        representation_refs=(f"representation:{unit_id}",),
        processing_run_ref="processing-run-1",
        processing_version="image-v1",
        processing_fingerprint=ProcessingFingerprint(SHA_C),
    )


def _indexed_candidate(
    unit_id: str,
    ordinal: int,
    lane: RetrievalLane,
    rank: int,
) -> Candidate:
    semantic = lane in {RetrievalLane.INSTANCE, RetrievalLane.CONCEPT}
    return Candidate(
        unit_id=unit_id,
        unit_ordinal=ordinal,
        lane=lane,
        match_kind=(
            MatchKind.VISUAL
            if lane is RetrievalLane.INSTANCE
            else MatchKind.SEMANTIC
            if lane is RetrievalLane.CONCEPT
            else MatchKind.NEAR_DUPLICATE
        ),
        raw_score=0.9 if semantic else float(rank - 1),
        score_semantics="inner-product" if semantic else "hamming-distance",
        rank=rank,
        provenance=IndexedCandidateProvenance(
            activation_set_id="set-1",
            generation_id=f"generation-{lane.value}",
            provider_id=f"provider-{lane.value}",
        ),
        embedding_space=_space(lane) if semantic else None,
        hash_space=_hash_space() if lane is RetrievalLane.PERCEPTUAL else None,
    )


def _content_candidate(unit_id: str, ordinal: int) -> Candidate:
    return Candidate(
        unit_id=unit_id,
        unit_ordinal=ordinal,
        lane=RetrievalLane.CONTENT,
        match_kind=MatchKind.NEAR_DUPLICATE,
        raw_score=1.0,
        score_semantics="sha256-equality",
        rank=1,
        provenance=ContentCandidateProvenance("set-1", "catalog-1"),
        exact_content=True,
    )


@dataclass(frozen=True)
class _Selection:
    filter_sha256: str = SHA_A
    ordinals: tuple[int, ...] | None = None


class _Catalog:
    def __init__(
        self,
        content: tuple[Candidate, ...],
        units: tuple[SearchableUnit, ...],
        error: Exception | None = None,
        *,
        content_error: Exception | None = None,
        get_many_error: Exception | None = None,
        get_many_mode: str = "exact",
    ) -> None:
        self._content = content
        self._units = {unit.unit_id: unit for unit in units}
        self._error = error
        self._content_error = content_error
        self._get_many_error = get_many_error
        self._get_many_mode = get_many_mode
        self.unit_count = len(units)
        self.calls: list[str] = []

    def lookup_content_candidates(
        self,
        source_sha256: str,
        filters: SearchFilters,
        *,
        limit: int,
        deadline: datetime,
    ) -> tuple[Candidate, ...]:
        del filters, limit
        assert source_sha256 == SHA_A
        assert deadline is FUTURE
        self.calls.append("content")
        if self._content_error is not None:
            raise self._content_error
        return self._content

    def select_ordinals(
        self,
        filters: SearchFilters,
        *,
        deadline: datetime,
    ) -> _Selection:
        del filters
        assert deadline is FUTURE
        if self._error is not None:
            raise self._error
        return _Selection()

    def get_many(
        self,
        unit_ids: tuple[str, ...],
        *,
        deadline: datetime,
    ) -> tuple[SearchableUnit, ...]:
        assert deadline is FUTURE
        if self._get_many_error is not None:
            raise self._get_many_error
        units = tuple(self._units[unit_id] for unit_id in unit_ids)
        if self._get_many_mode == "missing":
            return units[:-1]
        if self._get_many_mode == "reversed":
            return tuple(reversed(units))
        return units


class _Reader:
    def __init__(
        self,
        lane: RetrievalLane,
        candidates: tuple[Candidate, ...] = (),
        failure: SearchFailureCode | None = None,
        error: Exception | None = None,
    ) -> None:
        self.lane = lane
        self.candidates = candidates
        self.failure = failure
        self.error = error
        self.calls = 0
        self.limits: list[int] = []

    def search(
        self,
        query: QueryVector | PerceptualHashQuery,
        selection: tuple[int, ...] | None,
        *,
        limit: int,
        deadline: datetime,
    ) -> tuple[Candidate, ...]:
        del query, selection
        assert deadline is FUTURE
        self.calls += 1
        self.limits.append(limit)
        if self.failure is not None:
            raise SearchFailure(self.failure)
        if self.error is not None:
            raise self.error
        return self.candidates[:limit]


class _Lease:
    def __init__(self, catalog: _Catalog, readers: dict[RetrievalLane, _Reader]) -> None:
        self.qualification_tier = "preview"
        self.manifest = cast(GenerationSetManifest, _manifest())
        self.unit_catalog = catalog
        self.readers = readers
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> _Lease:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class _Provider:
    def __init__(self, lease: _Lease) -> None:
        self.lease = lease
        self.calls = 0
        self.cleanup_reservations: list[DeferredCleanupReservation] = []

    def acquire(
        self,
        profile_ref: str,
        *,
        deadline: datetime,
        cleanup_reservation: DeferredCleanupReservation | None = None,
    ) -> _Lease:
        assert profile_ref in {"personal-lite", "local-default"}
        assert deadline is FUTURE
        assert cleanup_reservation is not None
        self.calls += 1
        self.cleanup_reservations.append(cleanup_reservation)
        return self.lease


class _CandidateProvider:
    def __init__(self, lease: _Lease) -> None:
        lease.qualification_tier = "candidate_evaluation"
        self.lease = lease
        self.calls = 0
        self.cleanup_reservations: list[DeferredCleanupReservation] = []

    def acquire_candidate_evaluation(
        self,
        profile_ref: str,
        *,
        deadline: datetime,
        cleanup_reservation: DeferredCleanupReservation | None = None,
    ) -> _Lease:
        assert profile_ref in {"personal-lite", "local-default"}
        assert deadline is FUTURE
        assert cleanup_reservation is not None
        self.calls += 1
        self.cleanup_reservations.append(cleanup_reservation)
        return self.lease


class _Resolver:
    def __init__(self) -> None:
        self.calls = 0

    def resolve(
        self,
        query: ImageQuery,
        *,
        authorization: InvocationAuthorizationContext,
        data_policy_ref: str,
        deadline: datetime,
    ) -> ResolvedImageQuery:
        del query
        assert authorization.workspace_id == "workspace-1"
        assert data_policy_ref == "local-only"
        assert deadline is FUTURE
        self.calls += 1
        return ResolvedImageQuery(
            Image.new("RGB", (2, 2), "white"),
            SHA_A,
            "image/png",
            16,
            2,
            2,
        )


class _PerceptualEncoder:
    def __init__(self) -> None:
        self.calls = 0
        self.space = _hash_space()

    def encode(
        self,
        image: Image.Image,
        source_sha256: str,
        max_hamming_distance: int,
        deadline: datetime,
    ) -> PerceptualHashQuery:
        assert image.mode == "RGB"
        assert source_sha256 == SHA_A
        assert max_hamming_distance == 2
        assert deadline is FUTURE
        self.calls += 1
        return PerceptualHashQuery(SHA_A, self.space, 0, 2)


@dataclass
class _SemanticEncoder:
    embedding_space: EmbeddingSpaceRef
    bundle_manifest_sha256: str
    calls: int = 0

    def encode(
        self,
        image: Image.Image,
        space: EmbeddingSpaceRef,
        deadline: datetime,
    ) -> QueryVector:
        assert image.mode == "RGB"
        assert space == self.embedding_space
        assert deadline is FUTURE
        self.calls += 1
        return QueryVector(space, (1.0, 0.0))


class _EncoderResolver:
    def __init__(self) -> None:
        self.resolved: list[RetrievalLane] = []
        self.encoders = {
            lane: _SemanticEncoder(_space(lane), SHA_B)
            for lane in (RetrievalLane.INSTANCE, RetrievalLane.CONCEPT)
        }

    def resolve(
        self,
        lane: RetrievalLane,
        space: EmbeddingSpaceRef,
        target_bundle_manifest_sha256: str,
        execution_profile: ExecutionProfile,
    ) -> _SemanticEncoder:
        assert space == _space(lane)
        assert target_bundle_manifest_sha256 == SHA_B
        assert execution_profile in {ExecutionProfile.AUTO, ExecutionProfile.CPU}
        self.resolved.append(lane)
        return self.encoders[lane]


def _manifest() -> Any:
    return SimpleNamespace(
        activation_set_id="set-1",
        workspace_id="workspace-1",
        retrieval_profile_ref="personal-lite",
        unit_catalog=SimpleNamespace(unit_count=4),
        lanes=tuple(
            SimpleNamespace(
                lane=lane,
                target_bundle_manifest_sha256=(None if lane is RetrievalLane.PERCEPTUAL else SHA_B),
                generation=SimpleNamespace(
                    generation_id=f"generation-{lane.value}",
                    retriever_id=f"provider-{lane.value}",
                    embedding_spaces=(() if lane is RetrievalLane.PERCEPTUAL else (_space(lane),)),
                    hash_spaces=((_hash_space(),) if lane is RetrievalLane.PERCEPTUAL else ()),
                ),
            )
            for lane in (
                RetrievalLane.PERCEPTUAL,
                RetrievalLane.INSTANCE,
                RetrievalLane.CONCEPT,
            )
        ),
    )


def _authorization() -> InvocationAuthorizationContext:
    return InvocationAuthorizationContext(
        request_id="request-1",
        workspace_id="workspace-1",
        actor_id="actor-1",
        environment_id="test",
        operation="capicove.search.image",
        authorization_scope_ref="scope-1",
        permissions=frozenset({"retrieval:search"}),
        allowed_data_policy_refs=frozenset({"local-only"}),
        allowed_source_root_refs=frozenset(),
        privacy_class=PrivacyClass.CONFIDENTIAL,
    )


def _request(
    intent: SearchIntent,
    *,
    detail: DetailLevel = DetailLevel.COMPACT,
    top_k: int = 20,
    cursor: str | None = None,
) -> SearchRequest:
    return SearchRequest(
        query=ImageQuery(InlineBytesSource(b"synthetic", "image/png")),
        targets=frozenset({TargetType.IMAGE}),
        top_k=top_k,
        cursor=cursor,
        detail_level=detail,
        retrieval_profile_ref="personal-lite",
        search_intent=intent,
    )


@dataclass
class _Fixture:
    service: RetrievalService
    provider: _Provider
    resolver: _Resolver
    phash: _PerceptualEncoder
    semantic: _EncoderResolver
    readers: dict[RetrievalLane, _Reader]


def _fixture(
    *,
    content: tuple[Candidate, ...] = (),
    perceptual: tuple[Candidate, ...] = (),
    instance: tuple[Candidate, ...] = (),
    concept: tuple[Candidate, ...] = (),
    concept_failure: SearchFailureCode | None = None,
    concept_error: Exception | None = None,
    catalog_error: Exception | None = None,
    content_error: Exception | None = None,
    get_many_error: Exception | None = None,
    get_many_mode: str = "exact",
    cleanup_supervisor: DeferredCleanupSupervisor | None = None,
) -> _Fixture:
    all_candidates = (*content, *perceptual, *instance, *concept)
    maximum_ordinal = max((item.unit_ordinal for item in all_candidates), default=3)
    units = tuple(_unit(f"unit-{index}") for index in range(1, maximum_ordinal + 2))
    catalog = _Catalog(
        content,
        units,
        catalog_error,
        content_error=content_error,
        get_many_error=get_many_error,
        get_many_mode=get_many_mode,
    )
    readers = {
        RetrievalLane.PERCEPTUAL: _Reader(RetrievalLane.PERCEPTUAL, perceptual),
        RetrievalLane.INSTANCE: _Reader(RetrievalLane.INSTANCE, instance),
        RetrievalLane.CONCEPT: _Reader(
            RetrievalLane.CONCEPT,
            concept,
            failure=concept_failure,
            error=concept_error,
        ),
    }
    provider = _Provider(_Lease(catalog, readers))
    resolver = _Resolver()
    phash = _PerceptualEncoder()
    semantic = _EncoderResolver()
    scheduler = BoundedLaneScheduler(
        CpuExecutionBudget(
            total_threads=1,
            max_parallel_lanes=1,
            per_lane_compute_threads=1,
        ).schedule_snapshot(schedule_id="test-search")
    )
    cursor_codec = HmacSearchCursorCodec(
        signing_key=b"k" * 32,
        workspace_id="workspace-1",
    )
    service = RetrievalService(
        workspace_id="workspace-1",
        generation_sets=provider,
        query_resolver=resolver,
        perceptual_encoder=phash,
        query_encoders=semantic,
        fusion=WeightedReciprocalRankFusion(),
        cursor_codec=cursor_codec,
        lane_scheduler=scheduler,
        profiles=default_retrieval_profiles(),
        clock=lambda: datetime.now(UTC),
        cleanup_supervisor=cleanup_supervisor,
    )
    return _Fixture(service, provider, resolver, phash, semantic, readers)


@pytest.mark.parametrize(
    ("case", "expected_code"),
    (
        ("untrusted_authorization", SearchFailureCode.PERMISSION_DENIED),
        ("wrong_workspace", SearchFailureCode.PERMISSION_DENIED),
        ("wrong_operation", SearchFailureCode.PERMISSION_DENIED),
        ("missing_permission", SearchFailureCode.PERMISSION_DENIED),
        ("disallowed_data_policy", SearchFailureCode.SOURCE_NOT_AUTHORIZED),
        ("untrusted_request", SearchFailureCode.INVALID_QUERY),
        ("naive_deadline", SearchFailureCode.INVALID_QUERY),
        ("expired_deadline", SearchFailureCode.DEADLINE_EXCEEDED),
    ),
)
def test_authorization_and_deadline_failures_stop_before_generation_access(
    case: str,
    expected_code: SearchFailureCode,
) -> None:
    fixture = _fixture()
    authorization_input: object = _authorization()
    request_input: object = _request(SearchIntent.INSTANCE)
    deadline = FUTURE

    if case == "untrusted_authorization":
        authorization_input = object()
    elif case == "wrong_workspace":
        authorization_input = replace(_authorization(), workspace_id="workspace-2")
    elif case == "wrong_operation":
        authorization_input = replace(_authorization(), operation="capicove.admin")
    elif case == "missing_permission":
        authorization_input = replace(_authorization(), permissions=frozenset())
    elif case == "disallowed_data_policy":
        request_input = replace(
            _request(SearchIntent.INSTANCE),
            data_policy_ref="policy-not-granted",
        )
    elif case == "untrusted_request":
        request_input = object()
    elif case == "naive_deadline":
        deadline = datetime(2026, 8, 27, 12, 0)
    elif case == "expired_deadline":
        deadline = datetime.now(UTC) - timedelta(seconds=1)
    else:  # pragma: no cover - closed parametrization guard
        raise AssertionError(f"unexpected case: {case}")

    result = fixture.service.search(
        cast(SearchRequest, request_input),
        authorization=cast(InvocationAuthorizationContext, authorization_input),
        deadline=deadline,
    )

    assert result.status is SearchStatus.FAILED
    assert result.failure_code is expected_code
    assert fixture.provider.calls == 0
    assert fixture.resolver.calls == 0
    assert not fixture.provider.lease.closed
    if case == "untrusted_authorization":
        assert result.query_id == "request:untrusted"


def test_cleanup_capacity_is_rejected_before_generation_acquisition() -> None:
    supervisor = DeferredCleanupSupervisor(max_pending=1)
    held = supervisor.reserve()
    fixture = _fixture(cleanup_supervisor=supervisor)

    result = fixture.service.search(
        _request(SearchIntent.INSTANCE),
        authorization=_authorization(),
        deadline=FUTURE,
    )

    assert result.status is SearchStatus.FAILED
    assert result.failure_code is SearchFailureCode.BACKPRESSURE
    assert fixture.provider.calls == 0
    assert supervisor.drain(timeout_seconds=0).admission_rejections == 1
    held.cancel()


def test_one_cleanup_slot_is_handed_to_provider_without_second_admission() -> None:
    supervisor = DeferredCleanupSupervisor(max_pending=1)
    fixture = _fixture(cleanup_supervisor=supervisor)

    result = fixture.service.search(
        _request(SearchIntent.INSTANCE),
        authorization=_authorization(),
        deadline=FUTURE,
    )

    assert result.status is SearchStatus.SUCCEEDED
    assert fixture.provider.calls == 1
    assert len(fixture.provider.cleanup_reservations) == 1
    report = supervisor.drain(timeout_seconds=0)
    assert report.completed
    assert report.admission_rejections == 0


def test_production_service_rejects_candidate_evaluation_lease() -> None:
    fixture = _fixture()
    fixture.provider.lease.qualification_tier = "candidate_evaluation"

    result = fixture.service.search(
        _request(SearchIntent.INSTANCE),
        authorization=_authorization(),
        deadline=FUTURE,
    )

    assert result.status is SearchStatus.FAILED
    assert result.failure_code is SearchFailureCode.INDEX_GENERATION_UNAVAILABLE
    assert fixture.provider.calls == 1
    assert fixture.resolver.calls == 0
    assert fixture.provider.lease.closed


def test_dedicated_offline_service_accepts_only_candidate_provider_capability() -> None:
    fixture = _fixture()
    provider = _CandidateProvider(fixture.provider.lease)
    service = CandidateEvaluationRetrievalService(
        workspace_id="workspace-1",
        candidate_generation_sets=provider,
        query_resolver=fixture.resolver,
        perceptual_encoder=fixture.phash,
        query_encoders=fixture.semantic,
        fusion=WeightedReciprocalRankFusion(),
        cursor_codec=HmacSearchCursorCodec(
            signing_key=b"k" * 32,
            workspace_id="workspace-1",
        ),
        lane_scheduler=BoundedLaneScheduler(
            CpuExecutionBudget(
                total_threads=1,
                max_parallel_lanes=1,
                per_lane_compute_threads=1,
            ).schedule_snapshot(schedule_id="candidate-test-search")
        ),
        profiles=default_retrieval_profiles(),
        clock=lambda: datetime.now(UTC),
    )

    assert not isinstance(provider, GenerationSetProvider)
    assert not isinstance(service, RetrievalService)
    assert not hasattr(service, "search")
    denied = service.evaluate_candidate(
        _request(SearchIntent.INSTANCE),
        authorization=_authorization(),
        deadline=FUTURE,
    )
    assert denied.status is SearchStatus.FAILED
    assert denied.failure_code is SearchFailureCode.PERMISSION_DENIED
    assert provider.calls == 0

    result = service.evaluate_candidate(
        _request(SearchIntent.INSTANCE),
        authorization=replace(
            _authorization(),
            operation="capicove.offline.evaluate_candidate",
            permissions=frozenset({"retrieval:evaluate_candidate"}),
        ),
        deadline=FUTURE,
    )

    assert result.status is SearchStatus.SUCCEEDED
    assert result.execution_summary is not None
    assert result.execution_summary.activation_tier == "candidate_evaluation"
    assert provider.calls == 1
    assert provider.lease.closed


@pytest.mark.parametrize(
    "request_id",
    ("r" * MAX_REF_BYTES, "界" * MAX_REF_BYTES, "surrogate-\ud800"),
)
def test_failure_fallback_hashes_request_ids_that_exceed_result_bounds(
    request_id: str,
) -> None:
    fixture = _fixture()
    authorization = replace(_authorization(), request_id=request_id)

    result = fixture.service.search(
        cast(SearchRequest, object()),
        authorization=authorization,
        deadline=FUTURE,
    )

    candidate = f"request:{request_id}"
    try:
        encoded = candidate.encode("utf-8")
    except UnicodeEncodeError:
        encoded = candidate.encode("utf-8", errors="surrogatepass")
    digest = hashlib.sha256(encoded).hexdigest()[:32]
    assert result.status is SearchStatus.FAILED
    assert result.failure_code is SearchFailureCode.INVALID_QUERY
    assert result.query_id == f"request:sha256:{digest}"
    assert len(result.query_id.encode("utf-8")) <= MAX_REF_BYTES
    assert fixture.provider.calls == 0


def test_failure_fallback_preserves_short_request_id() -> None:
    fixture = _fixture()

    result = fixture.service.search(
        cast(SearchRequest, object()),
        authorization=_authorization(),
        deadline=FUTURE,
    )

    assert result.query_id == "request:request-1"


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        ("workspace", SearchFailureCode.INDEX_GENERATION_UNAVAILABLE),
        ("profile", SearchFailureCode.INDEX_GENERATION_UNAVAILABLE),
        ("lane_set", SearchFailureCode.INDEX_GENERATION_UNAVAILABLE),
        ("unit_count", SearchFailureCode.CAPACITY_EXCEEDED),
    ),
)
def test_generation_contract_mismatches_fail_before_query_resolution(
    mutation: str,
    expected_code: SearchFailureCode,
) -> None:
    fixture = _fixture()
    manifest = _manifest()
    if mutation == "workspace":
        manifest.workspace_id = "workspace-2"
    elif mutation == "profile":
        manifest.retrieval_profile_ref = "profile-not-requested"
    elif mutation == "lane_set":
        manifest.lanes = manifest.lanes[:-1]
    elif mutation == "unit_count":
        manifest.unit_catalog.unit_count = 0
    else:  # pragma: no cover - closed parametrization guard
        raise AssertionError(f"unexpected mutation: {mutation}")
    fixture.provider.lease.manifest = cast(GenerationSetManifest, manifest)

    result = fixture.service.search(
        _request(SearchIntent.INSTANCE),
        authorization=_authorization(),
        deadline=FUTURE,
    )

    assert result.status is SearchStatus.FAILED
    assert result.failure_code is expected_code
    assert fixture.provider.calls == 1
    assert fixture.resolver.calls == 0
    assert fixture.provider.lease.closed


@pytest.mark.parametrize(
    ("missing_lane", "intent"),
    (
        (RetrievalLane.PERCEPTUAL, SearchIntent.INSTANCE),
        (RetrievalLane.INSTANCE, SearchIntent.INSTANCE),
    ),
)
def test_missing_native_reader_is_a_degraded_canonical_result(
    missing_lane: RetrievalLane,
    intent: SearchIntent,
) -> None:
    fixture = _fixture()
    del fixture.readers[missing_lane]

    result = fixture.service.search(
        _request(intent),
        authorization=_authorization(),
        deadline=FUTURE,
    )

    assert result.status is SearchStatus.SUCCEEDED
    assert result.degraded
    assert SearchWarningCode.INDEX_GENERATION_UNAVAILABLE in result.warning_codes
    assert fixture.provider.lease.closed


def test_perceptual_space_mismatch_is_degraded_without_query_substitution() -> None:
    fixture = _fixture()
    fixture.phash.space = replace(_hash_space(), provider_version="4.13.0")

    result = fixture.service.search(
        _request(SearchIntent.INSTANCE),
        authorization=_authorization(),
        deadline=FUTURE,
    )

    assert result.status is SearchStatus.SUCCEEDED
    assert result.degraded
    assert SearchWarningCode.HASH_SPACE_INCOMPATIBLE in result.warning_codes
    assert fixture.readers[RetrievalLane.PERCEPTUAL].calls == 0
    assert fixture.provider.lease.closed


@pytest.mark.parametrize(
    ("error", "expected_warning"),
    (
        (
            SearchFailure(SearchFailureCode.PROVIDER_UNAVAILABLE),
            SearchWarningCode.PROVIDER_UNAVAILABLE,
        ),
        (TimeoutError("content lookup timed out"), SearchWarningCode.DEADLINE_EXCEEDED),
        (ValueError("content catalog changed"), SearchWarningCode.INDEX_CORRUPT),
    ),
)
def test_content_lookup_failures_are_bounded_degraded_warnings(
    error: Exception,
    expected_warning: SearchWarningCode,
) -> None:
    fixture = _fixture(content_error=error)

    result = fixture.service.search(
        _request(SearchIntent.INSTANCE),
        authorization=_authorization(),
        deadline=FUTURE,
    )

    assert result.status is SearchStatus.SUCCEEDED
    assert result.degraded
    assert expected_warning in result.warning_codes
    assert SearchWarningCode.RETRIEVAL_PREVIEW in result.warning_codes
    assert fixture.provider.lease.closed


@pytest.mark.parametrize(
    ("error", "expected_code"),
    (
        (
            SearchFailure(SearchFailureCode.PROVIDER_UNAVAILABLE),
            SearchFailureCode.PROVIDER_UNAVAILABLE,
        ),
        (TimeoutError("ordinal selection timed out"), SearchFailureCode.DEADLINE_EXCEEDED),
    ),
)
def test_ordinal_selection_failures_preserve_canonical_failure_codes(
    error: Exception,
    expected_code: SearchFailureCode,
) -> None:
    fixture = _fixture(catalog_error=error)

    result = fixture.service.search(
        _request(SearchIntent.INSTANCE),
        authorization=_authorization(),
        deadline=FUTURE,
    )

    assert result.status is SearchStatus.FAILED
    assert result.failure_code is expected_code
    assert fixture.phash.calls == 0
    assert fixture.semantic.resolved == []
    assert fixture.provider.lease.closed


@pytest.mark.parametrize(
    ("error", "expected_code"),
    (
        (
            SearchFailure(SearchFailureCode.PROVIDER_UNAVAILABLE),
            SearchFailureCode.PROVIDER_UNAVAILABLE,
        ),
        (TimeoutError("catalog projection timed out"), SearchFailureCode.DEADLINE_EXCEEDED),
        (ValueError("catalog projection changed"), SearchFailureCode.INDEX_CORRUPT),
    ),
)
def test_catalog_projection_failures_do_not_escape_the_service_boundary(
    error: Exception,
    expected_code: SearchFailureCode,
) -> None:
    fixture = _fixture(
        perceptual=(_indexed_candidate("unit-1", 0, RetrievalLane.PERCEPTUAL, 1),),
        get_many_error=error,
    )

    result = fixture.service.search(
        _request(SearchIntent.INSTANCE),
        authorization=_authorization(),
        deadline=FUTURE,
    )

    assert result.status is SearchStatus.FAILED
    assert result.failure_code is expected_code
    assert fixture.provider.lease.closed


@pytest.mark.parametrize("get_many_mode", ("missing", "reversed"))
def test_catalog_projection_rejects_missing_or_reordered_units(
    get_many_mode: str,
) -> None:
    fixture = _fixture(
        perceptual=(
            _indexed_candidate("unit-1", 0, RetrievalLane.PERCEPTUAL, 1),
            _indexed_candidate("unit-2", 1, RetrievalLane.PERCEPTUAL, 2),
        ),
        get_many_mode=get_many_mode,
    )

    result = fixture.service.search(
        _request(SearchIntent.INSTANCE),
        authorization=_authorization(),
        deadline=FUTURE,
    )

    assert result.status is SearchStatus.FAILED
    assert result.failure_code is SearchFailureCode.INDEX_CORRUPT
    assert fixture.provider.lease.closed


@pytest.mark.parametrize(
    ("intent", "expected"),
    (
        (SearchIntent.INSTANCE, (RetrievalLane.INSTANCE,)),
        (SearchIntent.CONCEPT, (RetrievalLane.CONCEPT,)),
        (SearchIntent.BALANCED, (RetrievalLane.INSTANCE, RetrievalLane.CONCEPT)),
    ),
)
def test_agent_intent_selects_only_declared_semantic_lanes(
    intent: SearchIntent,
    expected: tuple[RetrievalLane, ...],
) -> None:
    fixture = _fixture()

    result = fixture.service.search(
        _request(intent),
        authorization=_authorization(),
        deadline=FUTURE,
    )

    assert result.status is SearchStatus.SUCCEEDED
    assert fixture.provider.calls == 1
    assert fixture.resolver.calls == 1
    assert fixture.phash.calls == 1
    assert tuple(fixture.semantic.resolved) == expected
    assert fixture.readers[RetrievalLane.PERCEPTUAL].calls == 1
    assert fixture.readers[RetrievalLane.INSTANCE].calls == (
        1 if intent != SearchIntent.CONCEPT else 0
    )
    assert fixture.readers[RetrievalLane.CONCEPT].calls == (
        1 if intent != SearchIntent.INSTANCE else 0
    )


def test_exact_content_precedes_semantic_and_full_projection_is_bounded() -> None:
    fixture = _fixture(
        content=(_content_candidate("unit-1", 0),),
        perceptual=(_indexed_candidate("unit-2", 1, RetrievalLane.PERCEPTUAL, 1),),
        instance=(_indexed_candidate("unit-2", 1, RetrievalLane.INSTANCE, 1),),
        concept=(_indexed_candidate("unit-3", 2, RetrievalLane.CONCEPT, 1),),
    )

    result = fixture.service.search(
        _request(SearchIntent.BALANCED, detail=DetailLevel.FULL),
        authorization=_authorization(),
        deadline=FUTURE,
    )

    assert result.status is SearchStatus.SUCCEEDED
    assert [hit.id for hit in result.hits] == ["unit-1", "unit-2", "unit-3"]
    assert all(isinstance(hit, FullHit) for hit in result.hits)
    assert result.execution_summary is not None
    assert result.execution_summary.activation_tier == "preview"
    encoded = repr(result.hits)
    assert "/private/" not in encoded
    assert "synthetic" not in encoded
    assert "(1.0, 0.0)" not in encoded


def test_completed_lanes_survive_one_semantic_failure() -> None:
    fixture = _fixture(
        perceptual=(_indexed_candidate("unit-2", 1, RetrievalLane.PERCEPTUAL, 1),),
        concept_failure=SearchFailureCode.PROVIDER_UNAVAILABLE,
    )

    result = fixture.service.search(
        _request(SearchIntent.BALANCED),
        authorization=_authorization(),
        deadline=FUTURE,
    )

    assert result.status is SearchStatus.SUCCEEDED
    assert result.degraded
    assert SearchWarningCode.PROVIDER_UNAVAILABLE in result.warning_codes
    assert [hit.id for hit in result.hits] == ["unit-2"]


def test_catalog_and_reader_corruption_return_canonical_failures() -> None:
    catalog_failure = _fixture(catalog_error=ValueError("catalog query failed")).service.search(
        _request(SearchIntent.INSTANCE),
        authorization=_authorization(),
        deadline=FUTURE,
    )
    assert catalog_failure.status is SearchStatus.FAILED
    assert catalog_failure.failure_code is SearchFailureCode.INDEX_CORRUPT

    degraded = _fixture(
        concept_error=ValueError("native index shape changed"),
    ).service.search(
        _request(SearchIntent.BALANCED),
        authorization=_authorization(),
        deadline=FUTURE,
    )
    assert degraded.status is SearchStatus.SUCCEEDED
    assert degraded.degraded
    assert SearchWarningCode.INDEX_CORRUPT in degraded.warning_codes


def test_zero_hits_is_preview_success_and_compact() -> None:
    fixture = _fixture()

    result = fixture.service.search(
        _request(SearchIntent.INSTANCE),
        authorization=_authorization(),
        deadline=FUTURE,
    )

    assert result.status is SearchStatus.SUCCEEDED
    assert result.hits == ()
    assert result.degraded
    assert SearchWarningCode.RETRIEVAL_PREVIEW in result.warning_codes


@pytest.mark.parametrize(
    ("query", "targets"),
    (
        (TextQuery("find this"), frozenset({TargetType.IMAGE})),
        (
            VideoClipQuery(InlineBytesSource(b"synthetic", "video/mp4")),
            frozenset({TargetType.IMAGE}),
        ),
        (
            ImageQuery(InlineBytesSource(b"synthetic", "image/png")),
            frozenset({TargetType.KEYFRAME, TargetType.VIDEO_SEGMENT}),
        ),
    ),
)
def test_unsupported_preview_scope_is_non_retryable_invalid_input(
    query: TextQuery | ImageQuery | VideoClipQuery,
    targets: frozenset[TargetType],
) -> None:
    fixture = _fixture()
    request = SearchRequest(
        query=query,
        targets=targets,
        retrieval_profile_ref="personal-lite",
    )

    result = fixture.service.search(
        request,
        authorization=_authorization(),
        deadline=FUTURE,
    )

    assert result.status is SearchStatus.FAILED
    assert result.failure_code is SearchFailureCode.INVALID_QUERY
    assert fixture.provider.calls == 0
    assert fixture.resolver.calls == 0


def test_signed_cursor_resumes_without_restarting_on_mismatch() -> None:
    fixture = _fixture(
        perceptual=tuple(
            _indexed_candidate(f"unit-{index}", index - 1, RetrievalLane.PERCEPTUAL, index)
            for index in range(1, 4)
        ),
    )
    first = fixture.service.search(
        _request(SearchIntent.INSTANCE, top_k=2),
        authorization=_authorization(),
        deadline=FUTURE,
    )
    assert [hit.id for hit in first.hits] == ["unit-1", "unit-2"]
    assert first.next_cursor is not None

    second = fixture.service.search(
        _request(SearchIntent.INSTANCE, top_k=2, cursor=first.next_cursor),
        authorization=_authorization(),
        deadline=FUTURE,
    )
    assert [hit.id for hit in second.hits] == ["unit-3"]
    assert second.next_cursor is None

    mismatched = fixture.service.search(
        _request(SearchIntent.CONCEPT, top_k=2, cursor=first.next_cursor),
        authorization=_authorization(),
        deadline=FUTURE,
    )
    assert mismatched.status is SearchStatus.FAILED
    assert mismatched.failure_code is SearchFailureCode.STALE_CURSOR


def test_large_generation_binding_returns_a_canonical_bounded_cursor() -> None:
    fixture = _fixture(
        perceptual=tuple(
            _indexed_candidate(f"unit-{index}", index - 1, RetrievalLane.PERCEPTUAL, index)
            for index in range(1, 4)
        ),
    )
    for index, lane in enumerate(fixture.provider.lease.manifest.lanes):
        cast(Any, lane.generation).generation_id = chr(ord("a") + index) * MAX_REF_BYTES

    first = fixture.service.search(
        _request(SearchIntent.INSTANCE, top_k=2),
        authorization=_authorization(),
        deadline=FUTURE,
    )

    assert first.status is SearchStatus.SUCCEEDED
    assert first.next_cursor is not None
    assert len(first.next_cursor.encode("ascii")) <= MAX_CURSOR_BYTES

    second = fixture.service.search(
        _request(SearchIntent.INSTANCE, top_k=2, cursor=first.next_cursor),
        authorization=_authorization(),
        deadline=FUTURE,
    )
    assert second.status is SearchStatus.SUCCEEDED
    assert [hit.id for hit in second.hits] == ["unit-3"]


def test_cursor_is_stale_when_the_completed_lane_state_changes() -> None:
    fixture = _fixture(
        perceptual=tuple(
            _indexed_candidate(f"unit-{index}", index - 1, RetrievalLane.PERCEPTUAL, index)
            for index in range(1, 4)
        ),
    )
    first = fixture.service.search(
        _request(SearchIntent.INSTANCE, top_k=2),
        authorization=_authorization(),
        deadline=FUTURE,
    )
    assert first.next_cursor is not None

    fixture.readers[RetrievalLane.INSTANCE].failure = SearchFailureCode.PROVIDER_UNAVAILABLE
    resumed = fixture.service.search(
        _request(SearchIntent.INSTANCE, top_k=2, cursor=first.next_cursor),
        authorization=_authorization(),
        deadline=FUTURE,
    )

    assert resumed.status is SearchStatus.FAILED
    assert resumed.failure_code is SearchFailureCode.STALE_CURSOR


def test_cursor_is_stale_when_the_ranked_candidate_set_changes() -> None:
    candidates = tuple(
        _indexed_candidate(f"unit-{index}", index - 1, RetrievalLane.PERCEPTUAL, index)
        for index in range(1, 4)
    )
    fixture = _fixture(perceptual=candidates)
    first = fixture.service.search(
        _request(SearchIntent.INSTANCE, top_k=2),
        authorization=_authorization(),
        deadline=FUTURE,
    )
    assert first.next_cursor is not None

    fixture.readers[RetrievalLane.PERCEPTUAL].candidates = candidates[:-1]
    resumed = fixture.service.search(
        _request(SearchIntent.INSTANCE, top_k=2, cursor=first.next_cursor),
        authorization=_authorization(),
        deadline=FUTURE,
    )

    assert resumed.status is SearchStatus.FAILED
    assert resumed.failure_code is SearchFailureCode.STALE_CURSOR


def test_top_k_100_returns_the_supported_depth_without_false_stale_cursor() -> None:
    candidates = tuple(
        _indexed_candidate(f"unit-{index}", index - 1, RetrievalLane.PERCEPTUAL, index)
        for index in range(1, 101)
    )
    fixture = _fixture(perceptual=candidates)

    result = fixture.service.search(
        _request(SearchIntent.INSTANCE, top_k=100),
        authorization=_authorization(),
        deadline=FUTURE,
    )

    assert result.status is SearchStatus.SUCCEEDED
    assert len(result.hits) == 100
    assert result.next_cursor is None


def test_cursor_continues_beyond_the_first_hundred_personal_candidates() -> None:
    candidates = tuple(
        _indexed_candidate(f"unit-{index}", index - 1, RetrievalLane.PERCEPTUAL, index)
        for index in range(1, 151)
    )
    fixture = _fixture(perceptual=candidates)

    first = fixture.service.search(
        _request(SearchIntent.INSTANCE, top_k=100),
        authorization=_authorization(),
        deadline=FUTURE,
    )
    assert first.status is SearchStatus.SUCCEEDED
    assert len(first.hits) == 100
    assert first.next_cursor is not None

    second = fixture.service.search(
        _request(SearchIntent.INSTANCE, top_k=100, cursor=first.next_cursor),
        authorization=_authorization(),
        deadline=FUTURE,
    )
    assert second.status is SearchStatus.SUCCEEDED
    assert [hit.id for hit in second.hits] == [f"unit-{index}" for index in range(101, 151)]
    assert second.next_cursor is None


def test_balanced_rrf_reads_the_full_personal_candidate_depth_before_fusing() -> None:
    unit_ids = tuple(f"unit-{index}" for index in range(1, 11))
    ordinal_by_id = {unit_id: index for index, unit_id in enumerate(unit_ids)}
    instance_order = (*unit_ids[:2], unit_ids[9], *unit_ids[4:9], unit_ids[3], unit_ids[2])
    concept_order = (
        unit_ids[2],
        unit_ids[3],
        unit_ids[9],
        *unit_ids[4:9],
        unit_ids[1],
        unit_ids[0],
    )
    fixture = _fixture(
        instance=tuple(
            _indexed_candidate(
                unit_id,
                ordinal_by_id[unit_id],
                RetrievalLane.INSTANCE,
                rank,
            )
            for rank, unit_id in enumerate(instance_order, start=1)
        ),
        concept=tuple(
            _indexed_candidate(
                unit_id,
                ordinal_by_id[unit_id],
                RetrievalLane.CONCEPT,
                rank,
            )
            for rank, unit_id in enumerate(concept_order, start=1)
        ),
    )

    result = fixture.service.search(
        _request(SearchIntent.BALANCED, top_k=1),
        authorization=_authorization(),
        deadline=FUTURE,
    )

    assert result.status is SearchStatus.SUCCEEDED
    assert [hit.id for hit in result.hits] == ["unit-10"]
    assert fixture.readers[RetrievalLane.INSTANCE].limits == [10]
    assert fixture.readers[RetrievalLane.CONCEPT].limits == [10]


def test_timed_out_lane_batch_retains_generation_lease_until_jobs_settle() -> None:
    lease = _fixture().provider.lease
    future: Future[object] = Future()
    supervisor = DeferredCleanupSupervisor()
    cleanup = supervisor.reserve()
    cleanup.bind(lease.close)

    with pytest.raises(LaneDeadlineExceeded):
        with search_module._retained_generation_lease(lease, cleanup):
            raise LaneDeadlineExceeded((future,))

    assert not lease.closed
    future.set_result(None)
    assert supervisor.drain(timeout_seconds=1).completed
    assert lease.closed


def test_lane_failure_summary_is_priority_ordered_and_warning_deduplicated() -> None:
    outcomes = [
        search_module._LaneOutcome(
            RetrievalLane.CONCEPT,
            failure_code=SearchFailureCode.PROVIDER_UNAVAILABLE,
        ),
        search_module._LaneOutcome(
            RetrievalLane.INSTANCE,
            failure_code=SearchFailureCode.INDEX_CORRUPT,
        ),
        search_module._LaneOutcome(
            RetrievalLane.PERCEPTUAL,
            failure_code=SearchFailureCode.PROVIDER_UNAVAILABLE,
        ),
    ]

    assert search_module._primary_failure(outcomes) is SearchFailureCode.INDEX_CORRUPT
    assert search_module._warnings_for(outcomes) == (
        SearchWarningCode.PROVIDER_UNAVAILABLE,
        SearchWarningCode.INDEX_CORRUPT,
    )


def test_unknown_lane_failure_is_reported_as_internal() -> None:
    class _UnknownFailure:
        value = "not-a-canonical-code"

    unknown = cast(SearchFailureCode, _UnknownFailure())
    outcome = search_module._LaneOutcome(
        RetrievalLane.INSTANCE,
        failure_code=unknown,
    )

    assert search_module._warning_for_failure(unknown) is SearchWarningCode.INTERNAL
    assert search_module._primary_failure([outcome]) is SearchFailureCode.INTERNAL
