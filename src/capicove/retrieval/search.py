"""Profile-independent intent-routed reverse-image retrieval orchestration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable

from PIL import Image

from capicove.contracts.retrieval import (
    MAX_REF_BYTES,
    CompactHit,
    DetailLevel,
    EmbeddingSpaceRef,
    ExecutionProfile,
    FullHit,
    HashSpaceRef,
    ImageQuery,
    LaneScore,
    SearchableUnit,
    SearchExecutionSummary,
    SearchFailureCode,
    SearchFilters,
    SearchIntent,
    SearchRequest,
    SearchResult,
    SearchStatus,
    SearchWarningCode,
    TargetType,
)
from capicove.retrieval.cursor import (
    SearchCursorBinding,
    SearchCursorPosition,
    StaleCursorError,
)
from capicove.retrieval.image_generations import LaneGeneration
from capicove.retrieval.profiles import RetrievalProfile, RetrievalProfileRegistry
from capicove.retrieval.search_models import (
    Candidate,
    CandidateLane,
    DeferredCleanupDrainReport,
    DeferredCleanupFailureSnapshot,
    DeferredCleanupReservation,
    DeferredCleanupSupervisor,
    FusedCandidate,
    InvocationAuthorizationContext,
    LaneDeadlineExceeded,
    RetrievalLane,
    ScheduledLaneJob,
    SearchFailure,
)
from capicove.retrieval.search_ports import (
    CursorCodec,
    FusionStrategy,
    GenerationSetProvider,
    ImageQueryResolver,
    LaneScheduler,
    OrdinalSelection,
    PerceptualQueryEncoder,
    QueryEncoderResolver,
    SearchGenerationLease,
    UnitCatalog,
)


_REQUIRED_OPERATION = "capicove.search.image"
_REQUIRED_PERMISSION = "retrieval:search"
_CANDIDATE_EVALUATION_OPERATION = "capicove.offline.evaluate_candidate"
_CANDIDATE_EVALUATION_PERMISSION = "retrieval:evaluate_candidate"
_CURSOR_TTL = timedelta(minutes=5)


@dataclass(frozen=True, slots=True)
class _LaneOutcome:
    lane: RetrievalLane
    candidates: tuple[Candidate, ...] = ()
    failure_code: SearchFailureCode | None = None

    @property
    def completed(self) -> bool:
        return self.failure_code is None


@runtime_checkable
class _CandidateEvaluationGenerationSetProvider(Protocol):
    """Acquire only explicitly bound offline candidate-evaluation leases."""

    def acquire_candidate_evaluation(
        self,
        profile_ref: str,
        *,
        deadline: datetime,
        cleanup_reservation: DeferredCleanupReservation | None = None,
    ) -> SearchGenerationLease: ...


class _CandidateEvaluationProviderAdapter:
    """Adapt the distinct offline capability only inside its dedicated service."""

    def __init__(self, provider: _CandidateEvaluationGenerationSetProvider) -> None:
        self._provider = provider

    def acquire(
        self,
        profile_ref: str,
        *,
        deadline: datetime,
        cleanup_reservation: DeferredCleanupReservation | None = None,
    ) -> SearchGenerationLease:
        return self._provider.acquire_candidate_evaluation(
            profile_ref,
            deadline=deadline,
            cleanup_reservation=cleanup_reservation,
        )


class RetrievalService:
    """Search one immutable image generation without adapter-specific branching."""

    _accepted_qualification_tiers = frozenset({"preview", "formal"})
    _required_operation = _REQUIRED_OPERATION
    _required_permission = _REQUIRED_PERMISSION

    def __init__(
        self,
        *,
        workspace_id: str,
        generation_sets: GenerationSetProvider,
        query_resolver: ImageQueryResolver,
        perceptual_encoder: PerceptualQueryEncoder,
        query_encoders: QueryEncoderResolver,
        fusion: FusionStrategy,
        cursor_codec: CursorCodec,
        lane_scheduler: LaneScheduler,
        profiles: RetrievalProfileRegistry,
        clock: Callable[[], datetime],
        cleanup_supervisor: DeferredCleanupSupervisor | None = None,
    ) -> None:
        if not isinstance(workspace_id, str) or not workspace_id.strip():
            raise ValueError("workspace_id must be a non-blank string")
        if not isinstance(generation_sets, GenerationSetProvider):
            raise TypeError("generation_sets must provide active generation leases")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if cleanup_supervisor is not None and not isinstance(
            cleanup_supervisor,
            DeferredCleanupSupervisor,
        ):
            raise TypeError("cleanup_supervisor must be a DeferredCleanupSupervisor")
        self._workspace_id = workspace_id
        self._generation_sets = generation_sets
        self._query_resolver = query_resolver
        self._perceptual_encoder = perceptual_encoder
        self._query_encoders = query_encoders
        self._fusion = fusion
        self._cursor_codec = cursor_codec
        self._lane_scheduler = lane_scheduler
        self._profiles = profiles
        self._clock = clock
        self._cleanup_supervisor = cleanup_supervisor or DeferredCleanupSupervisor()

    def deferred_cleanup_failures(
        self,
    ) -> tuple[DeferredCleanupFailureSnapshot, ...]:
        """Return message-free failures that require explicit operator recovery."""
        return self._cleanup_supervisor.failures()

    def drain_deferred_cleanups(
        self,
        *,
        timeout_seconds: float | int,
    ) -> DeferredCleanupDrainReport:
        """Wait boundedly for deferred lease cleanup and report remaining work."""
        return self._cleanup_supervisor.drain(timeout_seconds=timeout_seconds)

    def recover_deferred_cleanup(self, cleanup_id: str) -> bool:
        """Start one bounded retry run for a recorded cleanup failure."""
        return self._cleanup_supervisor.recover(cleanup_id)

    def search(
        self,
        request: SearchRequest,
        *,
        authorization: InvocationAuthorizationContext,
        deadline: datetime,
    ) -> SearchResult:
        """Return a privacy-bounded preview result under one absolute deadline."""
        fallback_query_id = _fallback_query_id(authorization)
        try:
            _check_deadline(deadline, self._clock())
            self._authorize(authorization, request)
            if not isinstance(request, SearchRequest):
                raise SearchFailure(SearchFailureCode.INVALID_QUERY)
            if not isinstance(request.query, ImageQuery):
                raise SearchFailure(SearchFailureCode.INVALID_QUERY)
            if TargetType.IMAGE not in request.targets:
                raise SearchFailure(SearchFailureCode.INVALID_QUERY)

            profile = self._profiles.resolve(request.retrieval_profile_ref)
            cleanup = self._cleanup_supervisor.reserve()
            try:
                lease = self._generation_sets.acquire(
                    profile.profile_ref,
                    deadline=deadline,
                    cleanup_reservation=cleanup,
                )
            except BaseException:
                cleanup.cancel_if_unused()
                raise
            try:
                cleanup.bind(lease.close)
            except BaseException:
                cleanup.cancel()
                lease.close()
                raise
            with _retained_generation_lease(lease, cleanup) as lease:
                self._validate_lease(lease, profile)
                resolved = self._query_resolver.resolve(
                    request.query,
                    authorization=authorization,
                    data_policy_ref=request.data_policy_ref,
                    deadline=deadline,
                )
                query_sha256 = _normalized_query_sha256(request, resolved.source_sha256)
                query_id = f"query:{query_sha256[:32]}"
                lane_generations = {item.lane: item for item in lease.manifest.lanes}
                generation_ids = tuple(
                    lane_generations[lane].generation.generation_id
                    for lane in (
                        RetrievalLane.PERCEPTUAL,
                        RetrievalLane.INSTANCE,
                        RetrievalLane.CONCEPT,
                    )
                )
                binding = SearchCursorBinding(
                    query_sha256=query_sha256,
                    workspace_id=self._workspace_id,
                    targets=tuple(sorted(target.value for target in request.targets)),
                    filters_sha256=_filters_sha256(request.filters),
                    search_intent=request.search_intent.value,
                    retrieval_profile_ref=profile.profile_ref,
                    data_policy_ref=request.data_policy_ref,
                    generation_ids=generation_ids,
                )
                start = self._page_start(request, binding, profile)
                selection = self._select_ordinals(
                    lease.unit_catalog,
                    request.filters,
                    deadline,
                )
                candidate_depth = max(
                    1,
                    min(
                        profile.candidate_depth_ceiling,
                        (
                            lease.unit_catalog.unit_count
                            if selection.ordinals is None
                            else len(selection.ordinals)
                        ),
                    ),
                )
                # Every page replays the same bounded ranking set. Reducing lane
                # depth to the visible page can change RRF order and would defeat
                # the cursor's full ranking-state change detection.
                outcomes = [
                    self._content_lane(
                        lease.unit_catalog,
                        resolved.source_sha256,
                        request.filters,
                        candidate_depth,
                        deadline,
                    )
                ]
                jobs = self._lane_jobs(
                    request=request,
                    resolved_image=resolved.image,
                    source_sha256=resolved.source_sha256,
                    lane_generations=lane_generations,
                    readers=lease.readers,
                    ordinal_selection=selection.ordinals,
                    limit=candidate_depth,
                    deadline=deadline,
                    profile=profile,
                )
                outcomes.extend(self._lane_scheduler.run(jobs, deadline=deadline))
                _check_deadline(deadline, self._clock())

                completed = tuple(outcome for outcome in outcomes if outcome.completed)
                if not completed:
                    primary = _primary_failure(outcomes)
                    return _failed(query_id, primary)
                warnings = _warnings_for(outcomes)
                if any(target is not TargetType.IMAGE for target in request.targets):
                    warnings = _append_warning(
                        warnings,
                        SearchWarningCode.PROVIDER_UNAVAILABLE,
                    )
                warnings = _append_warning(warnings, SearchWarningCode.RETRIEVAL_PREVIEW)
                lanes = tuple(
                    CandidateLane(
                        outcome.lane,
                        profile.lane_weight(outcome.lane),
                        outcome.candidates,
                    )
                    for outcome in completed
                )
                try:
                    fused = self._fusion.fuse(lanes, limit=candidate_depth)
                except ValueError as error:
                    raise SearchFailure(SearchFailureCode.INDEX_CORRUPT) from error
                page = self._resume_page(
                    request,
                    binding,
                    fused,
                    start,
                    outcomes=tuple(outcomes),
                )
                units = self._get_many(
                    lease.unit_catalog,
                    tuple(candidate.unit_id for candidate in page),
                    deadline,
                )
                hits = self._project(
                    request,
                    page,
                    units,
                    outcomes=completed,
                    warnings=warnings,
                )
                next_cursor = self._next_cursor(
                    request,
                    binding,
                    fused,
                    start=start,
                    page=page,
                    outcomes=tuple(outcomes),
                )
                summary = _execution_summary(
                    request,
                    profile,
                    lane_generations,
                    tuple(outcome.lane for outcome in outcomes),
                    qualification_tier=lease.qualification_tier,
                )
                return SearchResult(
                    status=SearchStatus.SUCCEEDED,
                    query_id=query_id,
                    degraded=bool(warnings),
                    warning_codes=warnings,
                    hits=hits,
                    next_cursor=next_cursor,
                    execution_summary=summary,
                )
        except SearchFailure as error:
            return _failed(fallback_query_id, error.code)
        except StaleCursorError:
            return _failed(fallback_query_id, SearchFailureCode.STALE_CURSOR)
        except TimeoutError:
            return _failed(fallback_query_id, SearchFailureCode.DEADLINE_EXCEEDED)

    def _authorize(
        self,
        authorization: InvocationAuthorizationContext,
        request: object,
    ) -> None:
        if not isinstance(authorization, InvocationAuthorizationContext):
            raise SearchFailure(SearchFailureCode.PERMISSION_DENIED)
        if authorization.workspace_id != self._workspace_id:
            raise SearchFailure(SearchFailureCode.PERMISSION_DENIED)
        if authorization.operation != self._required_operation:
            raise SearchFailure(SearchFailureCode.PERMISSION_DENIED)
        if self._required_permission not in authorization.permissions:
            raise SearchFailure(SearchFailureCode.PERMISSION_DENIED)
        if isinstance(request, SearchRequest):
            if request.data_policy_ref not in authorization.allowed_data_policy_refs:
                raise SearchFailure(SearchFailureCode.SOURCE_NOT_AUTHORIZED)

    def _validate_lease(
        self,
        lease: SearchGenerationLease,
        profile: RetrievalProfile,
    ) -> None:
        if lease.qualification_tier not in self._accepted_qualification_tiers:
            raise SearchFailure(SearchFailureCode.INDEX_GENERATION_UNAVAILABLE)
        manifest = lease.manifest
        if getattr(manifest, "workspace_id", None) != self._workspace_id:
            raise SearchFailure(SearchFailureCode.INDEX_GENERATION_UNAVAILABLE)
        if getattr(manifest, "retrieval_profile_ref", None) != profile.profile_ref:
            raise SearchFailure(SearchFailureCode.INDEX_GENERATION_UNAVAILABLE)
        unit_catalog = getattr(manifest, "unit_catalog", None)
        profile.validate_unit_count(getattr(unit_catalog, "unit_count", 0))
        lanes = getattr(manifest, "lanes", ())
        if {lane.lane for lane in lanes} != {
            RetrievalLane.PERCEPTUAL,
            RetrievalLane.INSTANCE,
            RetrievalLane.CONCEPT,
        }:
            raise SearchFailure(SearchFailureCode.INDEX_GENERATION_UNAVAILABLE)

    def _page_start(
        self,
        request: SearchRequest,
        binding: SearchCursorBinding,
        profile: RetrievalProfile,
    ) -> int:
        if request.cursor is None:
            start = 0
        else:
            try:
                position = self._cursor_codec.decode(
                    request.cursor,
                    expected=binding,
                    now=self._clock(),
                )
            except StaleCursorError as error:
                raise SearchFailure(SearchFailureCode.STALE_CURSOR) from error
            if position.page_size != request.top_k:
                raise SearchFailure(SearchFailureCode.STALE_CURSOR)
            start = position.consumed_count
        if start >= profile.candidate_depth_ceiling:
            raise SearchFailure(SearchFailureCode.STALE_CURSOR)
        return start

    @staticmethod
    def _content_lane(
        catalog: UnitCatalog,
        source_sha256: str,
        filters: SearchFilters,
        limit: int,
        deadline: datetime,
    ) -> _LaneOutcome:
        try:
            candidates = catalog.lookup_content_candidates(
                source_sha256,
                filters,
                limit=limit,
                deadline=deadline,
            )
            return _LaneOutcome(RetrievalLane.CONTENT, tuple(candidates))
        except SearchFailure as error:
            return _LaneOutcome(RetrievalLane.CONTENT, failure_code=error.code)
        except TimeoutError:
            return _LaneOutcome(
                RetrievalLane.CONTENT,
                failure_code=SearchFailureCode.DEADLINE_EXCEEDED,
            )
        except (KeyError, OSError, RuntimeError, ValueError):
            return _LaneOutcome(
                RetrievalLane.CONTENT,
                failure_code=SearchFailureCode.INDEX_CORRUPT,
            )

    @staticmethod
    def _select_ordinals(
        catalog: UnitCatalog,
        filters: SearchFilters,
        deadline: datetime,
    ) -> OrdinalSelection:
        try:
            return catalog.select_ordinals(filters, deadline=deadline)
        except SearchFailure:
            raise
        except TimeoutError as error:
            raise SearchFailure(SearchFailureCode.DEADLINE_EXCEEDED) from error
        except (KeyError, OSError, RuntimeError, ValueError) as error:
            raise SearchFailure(SearchFailureCode.INDEX_CORRUPT) from error

    @staticmethod
    def _get_many(
        catalog: UnitCatalog,
        unit_ids: tuple[str, ...],
        deadline: datetime,
    ) -> tuple[SearchableUnit, ...]:
        try:
            return tuple(catalog.get_many(unit_ids, deadline=deadline))
        except SearchFailure:
            raise
        except TimeoutError as error:
            raise SearchFailure(SearchFailureCode.DEADLINE_EXCEEDED) from error
        except (KeyError, OSError, RuntimeError, ValueError) as error:
            raise SearchFailure(SearchFailureCode.INDEX_CORRUPT) from error

    def _lane_jobs(
        self,
        *,
        request: SearchRequest,
        resolved_image: Image.Image,
        source_sha256: str,
        lane_generations: Mapping[RetrievalLane, LaneGeneration],
        readers: Mapping[RetrievalLane, object],
        ordinal_selection: tuple[int, ...] | None,
        limit: int,
        deadline: datetime,
        profile: RetrievalProfile,
    ) -> tuple[ScheduledLaneJob[_LaneOutcome], ...]:
        semantic_lanes = {
            SearchIntent.INSTANCE: (RetrievalLane.INSTANCE,),
            SearchIntent.CONCEPT: (RetrievalLane.CONCEPT,),
            SearchIntent.BALANCED: (RetrievalLane.INSTANCE, RetrievalLane.CONCEPT),
        }[request.search_intent]
        jobs: list[ScheduledLaneJob[_LaneOutcome]] = [
            ScheduledLaneJob(
                RetrievalLane.PERCEPTUAL.value,
                lambda: self._run_perceptual_lane(
                    resolved_image,
                    source_sha256,
                    lane_generations[RetrievalLane.PERCEPTUAL],
                    readers.get(RetrievalLane.PERCEPTUAL),
                    ordinal_selection,
                    limit,
                    deadline,
                    profile,
                ),
            )
        ]

        def semantic_callback(lane: RetrievalLane) -> Callable[[], _LaneOutcome]:
            def run() -> _LaneOutcome:
                return self._run_semantic_lane(
                    lane,
                    resolved_image,
                    request.execution_profile,
                    lane_generations[lane],
                    readers.get(lane),
                    ordinal_selection,
                    limit,
                    deadline,
                )

            return run

        jobs.extend(
            ScheduledLaneJob(lane.value, semantic_callback(lane)) for lane in semantic_lanes
        )
        return tuple(jobs)

    def _run_perceptual_lane(
        self,
        image: Image.Image,
        source_sha256: str,
        lane_generation: LaneGeneration,
        reader: object,
        selection: tuple[int, ...] | None,
        limit: int,
        deadline: datetime,
        profile: RetrievalProfile,
    ) -> _LaneOutcome:
        try:
            if reader is None:
                raise SearchFailure(SearchFailureCode.INDEX_GENERATION_UNAVAILABLE)
            representation = self._perceptual_encoder.encode(
                image,
                source_sha256,
                profile.phash_radius,
                deadline,
            )
            expected = lane_generation.generation.hash_spaces
            if expected != (representation.space,):
                raise SearchFailure(SearchFailureCode.HASH_SPACE_INCOMPATIBLE)
            candidates = reader.search(  # type: ignore[attr-defined]
                representation,
                selection,
                limit=limit,
                deadline=deadline,
            )
            return _LaneOutcome(RetrievalLane.PERCEPTUAL, tuple(candidates))
        except SearchFailure as error:
            return _LaneOutcome(RetrievalLane.PERCEPTUAL, failure_code=error.code)
        except TimeoutError:
            return _LaneOutcome(
                RetrievalLane.PERCEPTUAL,
                failure_code=SearchFailureCode.DEADLINE_EXCEEDED,
            )
        except (KeyError, OSError, RuntimeError, ValueError):
            return _LaneOutcome(
                RetrievalLane.PERCEPTUAL,
                failure_code=SearchFailureCode.INDEX_CORRUPT,
            )

    def _run_semantic_lane(
        self,
        lane: RetrievalLane,
        image: Image.Image,
        execution_profile: ExecutionProfile,
        lane_generation: LaneGeneration,
        reader: object,
        selection: tuple[int, ...] | None,
        limit: int,
        deadline: datetime,
    ) -> _LaneOutcome:
        try:
            if reader is None:
                raise SearchFailure(SearchFailureCode.INDEX_GENERATION_UNAVAILABLE)
            spaces = lane_generation.generation.embedding_spaces
            bundle_digest = lane_generation.target_bundle_manifest_sha256
            if len(spaces) != 1 or bundle_digest is None:
                raise SearchFailure(SearchFailureCode.EMBEDDING_SPACE_INCOMPATIBLE)
            encoder = self._query_encoders.resolve(
                lane,
                spaces[0],
                bundle_digest,
                execution_profile,
            )
            representation = encoder.encode(image, spaces[0], deadline)
            if representation.space != spaces[0]:
                raise SearchFailure(SearchFailureCode.EMBEDDING_SPACE_INCOMPATIBLE)
            candidates = reader.search(  # type: ignore[attr-defined]
                representation,
                selection,
                limit=limit,
                deadline=deadline,
            )
            return _LaneOutcome(lane, tuple(candidates))
        except SearchFailure as error:
            return _LaneOutcome(lane, failure_code=error.code)
        except TimeoutError:
            return _LaneOutcome(lane, failure_code=SearchFailureCode.DEADLINE_EXCEEDED)
        except (KeyError, OSError, RuntimeError, ValueError):
            return _LaneOutcome(lane, failure_code=SearchFailureCode.INDEX_CORRUPT)

    def _resume_page(
        self,
        request: SearchRequest,
        binding: SearchCursorBinding,
        fused: tuple[FusedCandidate, ...],
        start: int,
        *,
        outcomes: tuple[_LaneOutcome, ...],
    ) -> tuple[FusedCandidate, ...]:
        if request.cursor is not None:
            try:
                position = self._cursor_codec.decode(
                    request.cursor,
                    expected=binding,
                    now=self._clock(),
                )
            except StaleCursorError as error:
                raise SearchFailure(SearchFailureCode.STALE_CURSOR) from error
            if start < 1 or len(fused) < start:
                raise SearchFailure(SearchFailureCode.STALE_CURSOR)
            if position.ranking_state_sha256 != _ranking_state_sha256(fused, outcomes):
                raise SearchFailure(SearchFailureCode.STALE_CURSOR)
            previous = fused[start - 1]
            if not _position_matches(position, previous):
                raise SearchFailure(SearchFailureCode.STALE_CURSOR)
        return fused[start : start + request.top_k]

    def _next_cursor(
        self,
        request: SearchRequest,
        binding: SearchCursorBinding,
        fused: tuple[FusedCandidate, ...],
        *,
        start: int,
        page: tuple[FusedCandidate, ...],
        outcomes: tuple[_LaneOutcome, ...],
    ) -> str | None:
        consumed = start + len(page)
        if not page or len(fused) <= consumed:
            return None
        now = self._clock()
        _require_aware(now)
        last = page[-1]
        return self._cursor_codec.encode(
            SearchCursorPosition(
                binding=binding,
                last_exact_content=last.exact_content,
                last_fused_score_hex=last.fused_score.hex(),
                last_best_lane_rank=last.best_lane_rank,
                last_unit_id=last.unit_id,
                ranking_state_sha256=_ranking_state_sha256(fused, outcomes),
                consumed_count=consumed,
                page_size=request.top_k,
                issued_at=now,
                expires_at=now + _CURSOR_TTL,
            )
        )

    @staticmethod
    def _project(
        request: SearchRequest,
        fused: tuple[FusedCandidate, ...],
        units: tuple[SearchableUnit, ...],
        *,
        outcomes: tuple[_LaneOutcome, ...],
        warnings: tuple[SearchWarningCode, ...],
    ) -> tuple[CompactHit | FullHit, ...]:
        if len(units) != len(fused):
            raise SearchFailure(SearchFailureCode.INDEX_CORRUPT)
        candidates_by_unit: dict[str, tuple[Candidate, ...]] = {}
        for unit_id in (item.unit_id for item in fused):
            candidates_by_unit[unit_id] = tuple(
                candidate
                for outcome in outcomes
                for candidate in outcome.candidates
                if candidate.unit_id == unit_id
            )
        hits: list[CompactHit | FullHit] = []
        for candidate, unit in zip(fused, units, strict=True):
            if unit.unit_id != candidate.unit_id:
                raise SearchFailure(SearchFailureCode.INDEX_CORRUPT)
            if request.detail_level is DetailLevel.COMPACT:
                hits.append(
                    CompactHit(
                        id=unit.unit_id,
                        target_type=unit.unit_type,
                        brief_score=candidate.brief_score,
                        evidence_id=unit.evidence_ref,
                        timestamp_ms=unit.timestamp_ms,
                        time_range=unit.time_range,
                    )
                )
                continue
            native = candidates_by_unit[unit.unit_id]
            embedding_spaces = tuple(
                dict.fromkeys(
                    item.embedding_space for item in native if item.embedding_space is not None
                )
            )
            hits.append(
                FullHit(
                    id=unit.unit_id,
                    target_type=unit.unit_type,
                    brief_score=candidate.brief_score,
                    evidence_id=unit.evidence_ref,
                    timestamp_ms=unit.timestamp_ms,
                    time_range=unit.time_range,
                    artifact_ref=unit.artifact_ref,
                    searchable_unit_ref=unit.unit_id,
                    representation_refs=unit.representation_refs,
                    scene_range=unit.scene_range,
                    match_kinds=candidate.match_kinds,
                    lane_scores=tuple(
                        LaneScore(
                            lane_id=observation.lane.value,
                            raw_score=observation.raw_score,
                            normalized_score=min(1.0, observation.contribution),
                            score_semantics=observation.score_semantics,
                            calibration_version="rank-fusion-preview-v1",
                            rank=observation.rank,
                        )
                        for observation in candidate.lane_observations
                    ),
                    embedding_spaces=embedding_spaces,
                    processing_fingerprint=unit.processing_fingerprint,
                    index_generation_ids=candidate.generation_ids,
                    provider_ids=candidate.provider_ids,
                    execution_profile=request.execution_profile,
                    warning_codes=tuple(warning.value for warning in warnings),
                    lineage_refs=(f"activation:{candidate.activation_set_id}",),
                )
            )
        return tuple(hits)


class _CandidateEvaluationExecutionCore(RetrievalService):
    """Reuse retrieval execution behind a non-substitutable offline facade."""

    _accepted_qualification_tiers = frozenset({"candidate_evaluation"})
    _required_operation = _CANDIDATE_EVALUATION_OPERATION
    _required_permission = _CANDIDATE_EVALUATION_PERMISSION


class CandidateEvaluationRetrievalService:
    """Evaluate an explicitly bound candidate through an offline-only capability."""

    def __init__(
        self,
        *,
        workspace_id: str,
        candidate_generation_sets: _CandidateEvaluationGenerationSetProvider,
        query_resolver: ImageQueryResolver,
        perceptual_encoder: PerceptualQueryEncoder,
        query_encoders: QueryEncoderResolver,
        fusion: FusionStrategy,
        cursor_codec: CursorCodec,
        lane_scheduler: LaneScheduler,
        profiles: RetrievalProfileRegistry,
        clock: Callable[[], datetime],
        cleanup_supervisor: DeferredCleanupSupervisor | None = None,
    ) -> None:
        if not isinstance(
            candidate_generation_sets,
            _CandidateEvaluationGenerationSetProvider,
        ):
            raise TypeError(
                "candidate_generation_sets must provide offline candidate-evaluation leases"
            )
        self._core = _CandidateEvaluationExecutionCore(
            workspace_id=workspace_id,
            generation_sets=_CandidateEvaluationProviderAdapter(candidate_generation_sets),
            query_resolver=query_resolver,
            perceptual_encoder=perceptual_encoder,
            query_encoders=query_encoders,
            fusion=fusion,
            cursor_codec=cursor_codec,
            lane_scheduler=lane_scheduler,
            profiles=profiles,
            clock=clock,
            cleanup_supervisor=cleanup_supervisor,
        )

    def evaluate_candidate(
        self,
        request: SearchRequest,
        *,
        authorization: InvocationAuthorizationContext,
        deadline: datetime,
    ) -> SearchResult:
        """Execute one offline candidate evaluation under its distinct authorization."""
        return self._core.search(
            request,
            authorization=authorization,
            deadline=deadline,
        )

    def candidate_cleanup_failures(
        self,
    ) -> tuple[DeferredCleanupFailureSnapshot, ...]:
        return self._core.deferred_cleanup_failures()

    def drain_candidate_cleanups(
        self,
        *,
        timeout_seconds: float | int,
    ) -> DeferredCleanupDrainReport:
        return self._core.drain_deferred_cleanups(timeout_seconds=timeout_seconds)

    def recover_candidate_cleanup(self, cleanup_id: str) -> bool:
        return self._core.recover_deferred_cleanup(cleanup_id)


def _execution_summary(
    request: SearchRequest,
    profile: RetrievalProfile,
    generations: Mapping[RetrievalLane, LaneGeneration],
    attempted_lanes: tuple[RetrievalLane, ...],
    *,
    qualification_tier: str,
) -> SearchExecutionSummary:
    unique_lanes = tuple(dict.fromkeys(attempted_lanes))
    indexed = tuple(
        generations[lane]
        for lane in unique_lanes
        if lane in generations and lane is not RetrievalLane.CONTENT
    )
    embedding_spaces: tuple[EmbeddingSpaceRef, ...] = tuple(
        dict.fromkeys(space for item in indexed for space in item.generation.embedding_spaces)
    )
    hash_spaces: tuple[HashSpaceRef, ...] = tuple(
        dict.fromkeys(space for item in indexed for space in item.generation.hash_spaces)
    )
    bundle_digests = tuple(
        dict.fromkeys(
            item.target_bundle_manifest_sha256
            for item in indexed
            if item.target_bundle_manifest_sha256 is not None
        )
    )
    providers = tuple(dict.fromkeys(item.generation.retriever_id for item in indexed))
    generation_ids = tuple(dict.fromkeys(item.generation.generation_id for item in indexed))
    tier = (
        qualification_tier
        if qualification_tier in {"preview", "formal", "candidate_evaluation"}
        else "preview"
    )
    return SearchExecutionSummary(
        search_intent=request.search_intent,
        selected_lanes=tuple(lane.value for lane in unique_lanes),
        provider_ids=providers,
        embedding_spaces=embedding_spaces,
        hash_spaces=hash_spaces,
        index_generation_ids=generation_ids,
        bundle_manifest_sha256s=bundle_digests,
        resolved_profile_ref=profile.profile_ref,
        execution_profile=request.execution_profile,
        activation_tier=tier,
    )


def _position_matches(position: SearchCursorPosition, candidate: FusedCandidate) -> bool:
    return (
        position.last_exact_content == candidate.exact_content
        and position.last_fused_score_hex == candidate.fused_score.hex()
        and position.last_best_lane_rank == candidate.best_lane_rank
        and position.last_unit_id == candidate.unit_id
    )


def _ranking_state_sha256(
    fused: tuple[FusedCandidate, ...],
    outcomes: tuple[_LaneOutcome, ...],
) -> str:
    document = {
        "schema_version": 1,
        "lane_outcomes": [
            {
                "lane": outcome.lane.value,
                "completed": outcome.completed,
                "failure_code": (
                    None if outcome.failure_code is None else outcome.failure_code.value
                ),
            }
            for outcome in sorted(outcomes, key=lambda item: item.lane.value)
        ],
        "ranking": [
            {
                "unit_id": candidate.unit_id,
                "exact_content": candidate.exact_content,
                "fused_score_hex": candidate.fused_score.hex(),
                "best_lane_rank": candidate.best_lane_rank,
                "match_kinds": [item.value for item in candidate.match_kinds],
                "lane_ranks": [
                    {
                        "lane": observation.lane.value,
                        "rank": observation.rank,
                        "contribution_hex": observation.contribution.hex(),
                    }
                    for observation in candidate.lane_observations
                ],
            }
            for candidate in fused
        ],
    }
    return hashlib.sha256(_canonical_json(document)).hexdigest()


@contextmanager
def _retained_generation_lease(
    lease: SearchGenerationLease,
    cleanup: DeferredCleanupReservation,
) -> Iterator[SearchGenerationLease]:
    """Keep generation resources alive while timed-out lane jobs are still running."""
    try:
        yield lease
    except LaneDeadlineExceeded as error:
        cleanup.defer_until(error)
        raise
    except BaseException as body_error:
        try:
            cleanup.cleanup_now()
        except Exception:
            # cleanup_now records callback failures for explicit recovery. Keep
            # the search body's canonical failure as the externally visible one.
            raise body_error from None
        raise
    else:
        try:
            cleanup.cleanup_now()
        except Exception:
            # The supervisor retains the failure for explicit recovery. A cleanup
            # error must not replace an already-computed canonical search result.
            return


def _normalized_query_sha256(request: SearchRequest, source_sha256: str) -> str:
    document = {
        "schema_version": 1,
        "source_sha256": source_sha256,
        "targets": sorted(target.value for target in request.targets),
        "filters_sha256": _filters_sha256(request.filters),
        "search_intent": request.search_intent.value,
        "retrieval_profile_ref": request.retrieval_profile_ref,
        "execution_profile": request.execution_profile.value,
        "data_policy_ref": request.data_policy_ref,
        "detail_level": request.detail_level.value,
    }
    return hashlib.sha256(_canonical_json(document)).hexdigest()


def _filters_sha256(filters: SearchFilters) -> str:
    time_range = filters.time_range
    document = {
        "schema_version": filters.schema_version,
        "corpus_refs": sorted(filters.corpus_refs),
        "artifact_refs": sorted(filters.artifact_refs),
        "media_types": sorted(filters.media_types),
        "time_range": (
            None
            if time_range is None
            else {"start_ms": time_range.start_ms, "end_ms": time_range.end_ms}
        ),
    }
    return hashlib.sha256(_canonical_json(document)).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _warnings_for(outcomes: list[_LaneOutcome]) -> tuple[SearchWarningCode, ...]:
    warnings: tuple[SearchWarningCode, ...] = ()
    for outcome in outcomes:
        if outcome.failure_code is None:
            continue
        warnings = _append_warning(warnings, _warning_for_failure(outcome.failure_code))
    return warnings


def _warning_for_failure(code: SearchFailureCode) -> SearchWarningCode:
    try:
        return SearchWarningCode(code.value)
    except ValueError:
        return SearchWarningCode.INTERNAL


def _append_warning(
    warnings: tuple[SearchWarningCode, ...],
    warning: SearchWarningCode,
) -> tuple[SearchWarningCode, ...]:
    if warning in warnings:
        return warnings
    return (*warnings, warning)


def _primary_failure(outcomes: list[_LaneOutcome]) -> SearchFailureCode:
    ordered = (
        SearchFailureCode.DEADLINE_EXCEEDED,
        SearchFailureCode.INDEX_CORRUPT,
        SearchFailureCode.EMBEDDING_SPACE_INCOMPATIBLE,
        SearchFailureCode.HASH_SPACE_INCOMPATIBLE,
        SearchFailureCode.INDEX_GENERATION_UNAVAILABLE,
        SearchFailureCode.PROFILE_UNAVAILABLE,
        SearchFailureCode.PROVIDER_UNAVAILABLE,
    )
    failures = {outcome.failure_code for outcome in outcomes}
    return next((code for code in ordered if code in failures), SearchFailureCode.INTERNAL)


def _failed(query_id: str, code: SearchFailureCode) -> SearchResult:
    return SearchResult(
        status=SearchStatus.FAILED,
        query_id=query_id,
        failure_code=code,
    )


def _fallback_query_id(authorization: object) -> str:
    if isinstance(authorization, InvocationAuthorizationContext):
        candidate = f"request:{authorization.request_id}"
        try:
            encoded = candidate.encode("utf-8")
        except UnicodeEncodeError:
            encoded = candidate.encode("utf-8", errors="surrogatepass")
            digest = hashlib.sha256(encoded).hexdigest()[:32]
            return f"request:sha256:{digest}"
        if len(encoded) <= MAX_REF_BYTES:
            return candidate
        digest = hashlib.sha256(encoded).hexdigest()[:32]
        return f"request:sha256:{digest}"
    return "request:untrusted"


def _check_deadline(deadline: datetime, now: datetime) -> None:
    _require_aware(deadline)
    _require_aware(now)
    if now >= deadline:
        raise SearchFailure(SearchFailureCode.DEADLINE_EXCEEDED)


def _require_aware(value: object) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise SearchFailure(SearchFailureCode.INVALID_QUERY)


__all__ = ["RetrievalService"]
