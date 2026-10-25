"""Immutable values for production reverse-image retrieval."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from threading import Condition, Lock, Thread
from time import monotonic
from typing import Any, Generic, TypeVar

from PIL import Image

from capicove.contracts.retrieval import (
    EmbeddingSpaceRef,
    HashSpaceRef,
    MatchKind,
    QueryModality,
    SearchFailureCode,
    TargetType,
    VectorNormalization,
)
from capicove.contracts.tools import PrivacyClass


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MAX_VECTOR_DIMENSION = 16_384
_T = TypeVar("_T")


class RetrievalLane(StrEnum):
    """Identify one independent reverse-image retrieval lane."""

    CONTENT = "content"
    PERCEPTUAL = "perceptual"
    INSTANCE = "instance"
    CONCEPT = "concept"


@dataclass(frozen=True, slots=True)
class LaneScheduleSnapshot:
    """Record the bounded CPU policy used for one search execution."""

    schedule_id: str
    total_threads: int
    max_parallel_lanes: int
    per_lane_compute_threads: int
    onnx_inter_op_threads: int

    def __post_init__(self) -> None:
        _require_non_blank("schedule_id", self.schedule_id)
        for field_name in (
            "total_threads",
            "max_parallel_lanes",
            "per_lane_compute_threads",
            "onnx_inter_op_threads",
        ):
            _require_positive_int(field_name, getattr(self, field_name))
        if self.max_parallel_lanes * self.per_lane_compute_threads > self.total_threads:
            raise ValueError("lane schedule oversubscribes its CPU budget")


@dataclass(frozen=True, slots=True)
class ScheduledLaneJob(Generic[_T]):
    """Bind one named no-argument lane callback to the runtime scheduler."""

    lane_id: str
    callback: Callable[[], _T]

    def __post_init__(self) -> None:
        _require_non_blank("lane_id", self.lane_id)
        if not callable(self.callback):
            raise TypeError("callback must be callable")


@dataclass(frozen=True, slots=True)
class InvocationAuthorizationContext:
    """Carry host-created authority that a search request may only narrow."""

    request_id: str
    workspace_id: str
    actor_id: str
    environment_id: str
    operation: str
    authorization_scope_ref: str
    permissions: frozenset[str]
    allowed_data_policy_refs: frozenset[str]
    allowed_source_root_refs: frozenset[str]
    privacy_class: PrivacyClass

    def __post_init__(self) -> None:
        for field_name in (
            "request_id",
            "workspace_id",
            "actor_id",
            "environment_id",
            "operation",
            "authorization_scope_ref",
        ):
            _require_non_blank(field_name, getattr(self, field_name))
        object.__setattr__(
            self,
            "permissions",
            _unique_non_blank_frozenset("permission", self.permissions),
        )
        policies = _unique_non_blank_frozenset(
            "allowed_data_policy_ref",
            self.allowed_data_policy_refs,
        )
        if not policies:
            raise ValueError("allowed_data_policy_refs must not be empty")
        object.__setattr__(self, "allowed_data_policy_refs", policies)
        object.__setattr__(
            self,
            "allowed_source_root_refs",
            _unique_non_blank_frozenset(
                "allowed_source_root_ref",
                self.allowed_source_root_refs,
            ),
        )
        if not isinstance(self.privacy_class, PrivacyClass):
            raise TypeError("privacy_class must be a PrivacyClass")


@dataclass(frozen=True, slots=True)
class QueryVector:
    """Bind one L2-normalized query vector to its complete space identity."""

    space: EmbeddingSpaceRef
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.space, EmbeddingSpaceRef):
            raise TypeError("space must be an EmbeddingSpaceRef")
        if self.space.dimension > _MAX_VECTOR_DIMENSION:
            raise ValueError("query-vector dimension must be at most 16,384")
        if self.space.normalization is not VectorNormalization.L2:
            raise ValueError("query-vector space must use L2 normalization")
        if not self.space.accepts(QueryModality.IMAGE, TargetType.IMAGE):
            raise ValueError("query-vector space must support image-to-image retrieval")
        values = _finite_float_tuple("values", self.values)
        if len(values) != self.space.dimension:
            raise ValueError("query-vector dimension must match its complete space identity")
        norm_squared = sum(value * value for value in values)
        if not math.isclose(norm_squared, 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError("query vector must be L2-normalized")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class PerceptualHashQuery:
    """Carry one content-bound pHash query in a complete hash space."""

    source_sha256: str
    space: HashSpaceRef
    value: int
    max_hamming_distance: int

    def __post_init__(self) -> None:
        _require_sha256("source_sha256", self.source_sha256)
        if not isinstance(self.space, HashSpaceRef):
            raise TypeError("space must be a HashSpaceRef")
        if (
            isinstance(self.value, bool)
            or not isinstance(self.value, int)
            or not 0 <= self.value < 2**self.space.bit_width
        ):
            raise ValueError("value must be an unsigned integer within the hash width")
        if (
            isinstance(self.max_hamming_distance, bool)
            or not isinstance(self.max_hamming_distance, int)
            or not 0 <= self.max_hamming_distance <= 2
        ):
            raise ValueError("max_hamming_distance must be an integer from 0 to 2")


QueryRepresentation = QueryVector | PerceptualHashQuery


@dataclass(frozen=True, slots=True)
class ResolvedImageQuery:
    """Bind one authorized, single-read image to immutable source evidence."""

    image: Image.Image
    source_sha256: str
    media_type: str
    encoded_byte_size: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if not isinstance(self.image, Image.Image):
            raise TypeError("image must be a PIL Image")
        _require_sha256("source_sha256", self.source_sha256)
        _require_non_blank("media_type", self.media_type)
        _require_positive_int("encoded_byte_size", self.encoded_byte_size)
        _require_positive_int("width", self.width)
        _require_positive_int("height", self.height)
        if self.image.size != (self.width, self.height):
            raise ValueError("decoded dimensions must match the resolved image")


@dataclass(frozen=True, slots=True)
class IndexedCandidateProvenance:
    """Identify one candidate emitted by an immutable indexed lane."""

    activation_set_id: str
    generation_id: str
    provider_id: str

    def __post_init__(self) -> None:
        _require_non_blank("activation_set_id", self.activation_set_id)
        _require_non_blank("generation_id", self.generation_id)
        _require_non_blank("provider_id", self.provider_id)


@dataclass(frozen=True, slots=True)
class ContentCandidateProvenance:
    """Identify exact-content evidence without inventing an index generation."""

    activation_set_id: str
    unit_catalog_id: str

    def __post_init__(self) -> None:
        _require_non_blank("activation_set_id", self.activation_set_id)
        _require_non_blank("unit_catalog_id", self.unit_catalog_id)


CandidateProvenance = IndexedCandidateProvenance | ContentCandidateProvenance


@dataclass(frozen=True, slots=True)
class Candidate:
    """Contain one stable, lane-native candidate observation."""

    unit_id: str
    unit_ordinal: int
    lane: RetrievalLane
    match_kind: MatchKind
    raw_score: float
    score_semantics: str
    rank: int
    provenance: CandidateProvenance
    embedding_space: EmbeddingSpaceRef | None = None
    hash_space: HashSpaceRef | None = None
    exact_content: bool = False

    def __post_init__(self) -> None:
        _require_non_blank("unit_id", self.unit_id)
        _require_non_negative_int("unit_ordinal", self.unit_ordinal)
        if not isinstance(self.lane, RetrievalLane):
            raise TypeError("lane must be a RetrievalLane")
        if not isinstance(self.match_kind, MatchKind):
            raise TypeError("match_kind must be a MatchKind")
        _require_finite("raw_score", self.raw_score)
        _require_non_blank("score_semantics", self.score_semantics)
        _require_positive_int("rank", self.rank)
        if not isinstance(self.exact_content, bool):
            raise TypeError("exact_content must be a boolean")
        self._validate_lane_identity()

    def _validate_lane_identity(self) -> None:
        if self.lane is RetrievalLane.CONTENT:
            if not isinstance(self.provenance, ContentCandidateProvenance):
                raise ValueError("content lane requires content provenance")
            if self.match_kind is not MatchKind.NEAR_DUPLICATE or not self.exact_content:
                raise ValueError("content lane must carry exact near-duplicate evidence")
            if self.embedding_space is not None or self.hash_space is not None:
                raise ValueError("content lane must not carry an indexed space")
            return

        if not isinstance(self.provenance, IndexedCandidateProvenance):
            raise ValueError("indexed lanes require indexed provenance")
        if self.exact_content:
            raise ValueError("only the content lane may mark exact content")
        if self.lane is RetrievalLane.PERCEPTUAL:
            if self.match_kind is not MatchKind.NEAR_DUPLICATE:
                raise ValueError("perceptual lane requires a near-duplicate match")
            if not isinstance(self.hash_space, HashSpaceRef) or self.embedding_space is not None:
                raise ValueError("perceptual lane requires exactly one complete hash space")
            return
        if self.lane is RetrievalLane.INSTANCE:
            if self.match_kind is not MatchKind.VISUAL:
                raise ValueError("instance lane requires a visual match")
        elif self.lane is RetrievalLane.CONCEPT:
            if self.match_kind is not MatchKind.SEMANTIC:
                raise ValueError("concept lane requires a semantic match")
        else:  # pragma: no cover - StrEnum exhaustiveness guard
            raise ValueError("unsupported retrieval lane")
        if not isinstance(self.embedding_space, EmbeddingSpaceRef) or self.hash_space is not None:
            raise ValueError("semantic lanes require exactly one complete embedding space")


@dataclass(frozen=True, slots=True)
class CandidateLane:
    """Bind one ordered, duplicate-free lane result to its fusion weight."""

    lane: RetrievalLane
    weight: float
    candidates: tuple[Candidate, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.lane, RetrievalLane):
            raise TypeError("lane must be a RetrievalLane")
        _require_positive_finite("weight", self.weight)
        candidates = tuple(self.candidates)
        if any(not isinstance(candidate, Candidate) for candidate in candidates):
            raise TypeError("candidates must contain Candidate values")
        if any(candidate.lane is not self.lane for candidate in candidates):
            raise ValueError("every candidate must belong to the declared lane")
        if len({candidate.unit_id for candidate in candidates}) != len(candidates):
            raise ValueError("candidate lane must not contain a duplicate unit")
        object.__setattr__(self, "candidates", candidates)


@dataclass(frozen=True, slots=True)
class LaneObservation:
    """Preserve one lane's native score and rank without cross-score comparison."""

    lane: RetrievalLane
    raw_score: float
    score_semantics: str
    rank: int
    contribution: float

    def __post_init__(self) -> None:
        if not isinstance(self.lane, RetrievalLane):
            raise TypeError("lane must be a RetrievalLane")
        _require_finite("raw_score", self.raw_score)
        _require_non_blank("score_semantics", self.score_semantics)
        _require_positive_int("rank", self.rank)
        _require_positive_finite("contribution", self.contribution)


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    """Contain one rank-fused candidate and bounded lineage identity."""

    unit_id: str
    unit_ordinal: int
    exact_content: bool
    fused_score: float
    best_lane_rank: int
    match_kinds: tuple[MatchKind, ...]
    lane_observations: tuple[LaneObservation, ...]
    activation_set_id: str
    generation_ids: tuple[str, ...]
    provider_ids: tuple[str, ...]
    brief_score: float
    brief_score_version: str = "rank-reciprocal-v1"

    def __post_init__(self) -> None:
        _require_non_blank("unit_id", self.unit_id)
        _require_non_negative_int("unit_ordinal", self.unit_ordinal)
        if not isinstance(self.exact_content, bool):
            raise TypeError("exact_content must be a boolean")
        _require_positive_finite("fused_score", self.fused_score)
        _require_positive_int("best_lane_rank", self.best_lane_rank)
        match_kinds = tuple(self.match_kinds)
        if not match_kinds or any(not isinstance(kind, MatchKind) for kind in match_kinds):
            raise TypeError("match_kinds must contain MatchKind values")
        if len(set(match_kinds)) != len(match_kinds):
            raise ValueError("match_kinds must not contain duplicates")
        object.__setattr__(self, "match_kinds", match_kinds)
        observations = tuple(self.lane_observations)
        if not observations or any(
            not isinstance(observation, LaneObservation) for observation in observations
        ):
            raise TypeError("lane_observations must contain LaneObservation values")
        if len({observation.lane for observation in observations}) != len(observations):
            raise ValueError("lane_observations must not repeat a lane")
        object.__setattr__(self, "lane_observations", observations)
        _require_non_blank("activation_set_id", self.activation_set_id)
        object.__setattr__(
            self,
            "generation_ids",
            _unique_non_blank_tuple("generation_id", self.generation_ids),
        )
        object.__setattr__(
            self,
            "provider_ids",
            _unique_non_blank_tuple("provider_id", self.provider_ids),
        )
        _require_finite("brief_score", self.brief_score)
        if not 0.0 <= self.brief_score <= 1.0:
            raise ValueError("brief_score must be within [0, 1]")
        _require_non_blank("brief_score_version", self.brief_score_version)


class SearchFailure(RuntimeError):
    """Carry one stable canonical search failure across capability boundaries."""

    def __init__(self, code: SearchFailureCode) -> None:
        if not isinstance(code, SearchFailureCode):
            raise TypeError("code must be a SearchFailureCode")
        super().__init__(code.value)
        self.code = code


class LaneDeadlineExceeded(SearchFailure):
    """Keep timed-out lane resources alive until every submitted job settles."""

    def __init__(self, futures: tuple[Future[Any], ...]) -> None:
        materialized = tuple(futures)
        if not materialized or any(not isinstance(future, Future) for future in materialized):
            raise ValueError("futures must contain submitted lane futures")
        super().__init__(SearchFailureCode.DEADLINE_EXCEEDED)
        self._lock = Lock()
        self._pending = len(materialized)
        self._cleanup_callbacks: list[Callable[[], None]] = []
        for future in materialized:
            future.add_done_callback(self._job_settled)

    def defer_cleanup(self, callback: Callable[[], None]) -> None:
        """Run cleanup once after all timed-out jobs stop using their resources."""
        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._lock:
            if self._pending:
                self._cleanup_callbacks.append(callback)
                return
        callback()

    def _job_settled(self, _future: Future[Any]) -> None:
        callbacks: tuple[Callable[[], None], ...] = ()
        with self._lock:
            self._pending -= 1
            if self._pending == 0:
                callbacks = tuple(self._cleanup_callbacks)
                self._cleanup_callbacks.clear()
        for callback in callbacks:
            callback()


class DeferredCleanupState(StrEnum):
    """Describe one bounded deferred lease-cleanup lifecycle."""

    RESERVED = "reserved"
    WAITING_FOR_LANES = "waiting_for_lanes"
    READY = "ready"
    RUNNING = "running"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class DeferredCleanupFailureSnapshot:
    """Expose one message-free, explicitly recoverable cleanup failure."""

    cleanup_id: str
    failure_code: str
    attempt_count: int
    max_attempts_per_run: int
    recovery_count: int
    error_type: str


@dataclass(frozen=True, slots=True)
class DeferredCleanupDrainReport:
    """Report whether all reserved cleanup work reached successful completion."""

    completed: bool
    pending_count: int
    waiting_count: int
    running_count: int
    failed_count: int
    admission_rejections: int
    failures: tuple[DeferredCleanupFailureSnapshot, ...]


@dataclass(slots=True)
class _DeferredCleanupTask:
    cleanup_id: str
    state: DeferredCleanupState = DeferredCleanupState.RESERVED
    callback: Callable[[], None] | None = None
    attempt_count: int = 0
    attempts_this_run: int = 0
    recovery_count: int = 0
    error_type: str | None = None


class DeferredCleanupReservation:
    """Own one pre-admitted supervisor slot for exactly one search lease."""

    def __init__(self, supervisor: DeferredCleanupSupervisor, cleanup_id: str) -> None:
        self._supervisor = supervisor
        self.cleanup_id = cleanup_id

    def bind(self, callback: Callable[[], None]) -> None:
        """Bind the acquired lease closer to this reservation exactly once."""
        self._supervisor._bind(self.cleanup_id, callback)

    def cancel(self) -> None:
        """Release an unused reservation after lease acquisition fails."""
        self._supervisor._cancel(self.cleanup_id)

    def cancel_if_unused(self) -> bool:
        """Idempotently release this slot only while it remains unused."""
        return self._supervisor._cancel_if_unused(self.cleanup_id)

    def cleanup_now(self) -> None:
        """Run the ordinary request-thread cleanup once and retain failures."""
        self._supervisor._cleanup_now(self.cleanup_id)

    def defer_until(self, error: LaneDeadlineExceeded) -> None:
        """Schedule bounded cleanup only after every timed-out lane settles."""
        if not isinstance(error, LaneDeadlineExceeded):
            raise TypeError("error must be a LaneDeadlineExceeded")
        self._supervisor._wait_for_lanes(self.cleanup_id)
        error.defer_cleanup(lambda: self._supervisor._mark_ready(self.cleanup_id))


class DeferredCleanupSupervisor:
    """Bound, observe, drain, and explicitly recover deferred lease cleanup."""

    def __init__(self, *, max_pending: int = 64, max_attempts_per_run: int = 3) -> None:
        _require_positive_int("max_pending", max_pending)
        _require_positive_int("max_attempts_per_run", max_attempts_per_run)
        if max_pending > 1_024:
            raise ValueError("max_pending must be at most 1,024")
        if max_attempts_per_run > 10:
            raise ValueError("max_attempts_per_run must be at most 10")
        self._max_pending = max_pending
        self._max_attempts_per_run = max_attempts_per_run
        self._condition = Condition()
        self._tasks: dict[str, _DeferredCleanupTask] = {}
        self._next_id = 1
        self._worker_running = False
        self._admission_rejections = 0

    def reserve(self) -> DeferredCleanupReservation:
        """Reserve capacity before acquiring a lease, or fail without acquiring it."""
        with self._condition:
            if len(self._tasks) >= self._max_pending:
                self._admission_rejections += 1
                raise SearchFailure(SearchFailureCode.BACKPRESSURE)
            cleanup_id = f"lease-cleanup-{self._next_id}"
            self._next_id += 1
            self._tasks[cleanup_id] = _DeferredCleanupTask(cleanup_id)
            return DeferredCleanupReservation(self, cleanup_id)

    def failures(self) -> tuple[DeferredCleanupFailureSnapshot, ...]:
        """Return stable structured failures without exception messages or lease data."""
        with self._condition:
            return self._failure_snapshots_locked()

    def drain(self, *, timeout_seconds: float | int) -> DeferredCleanupDrainReport:
        """Wait boundedly for active work and return remaining recoverable state."""
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or not 0.0 <= float(timeout_seconds) <= 60.0
        ):
            raise ValueError("timeout_seconds must be finite and within [0, 60]")
        deadline = monotonic() + float(timeout_seconds)
        with self._condition:
            while self._has_active_work_locked():
                remaining = deadline - monotonic()
                if remaining <= 0.0:
                    break
                self._condition.wait(timeout=remaining)
            return self._report_locked()

    def recover(self, cleanup_id: str) -> bool:
        """Start one new bounded retry run for a recorded cleanup failure."""
        _require_non_blank("cleanup_id", cleanup_id)
        with self._condition:
            task = self._tasks.get(cleanup_id)
            if task is None:
                raise KeyError(cleanup_id)
            if task.state is not DeferredCleanupState.FAILED:
                return False
            task.state = DeferredCleanupState.READY
            task.attempts_this_run = 0
            task.recovery_count += 1
            task.error_type = None
            self._ensure_worker_locked()
            return True

    def _bind(self, cleanup_id: str, callback: Callable[[], None]) -> None:
        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._condition:
            task = self._require_state_locked(
                cleanup_id,
                DeferredCleanupState.RESERVED,
            )
            if task.callback is not None:
                raise RuntimeError("cleanup reservation is already bound")
            task.callback = callback

    def _cancel(self, cleanup_id: str) -> None:
        with self._condition:
            task = self._require_state_locked(
                cleanup_id,
                DeferredCleanupState.RESERVED,
            )
            if task.callback is not None:
                raise RuntimeError("a bound cleanup reservation cannot be cancelled")
            del self._tasks[cleanup_id]
            self._condition.notify_all()

    def _cancel_if_unused(self, cleanup_id: str) -> bool:
        with self._condition:
            task = self._tasks.get(cleanup_id)
            if (
                task is None
                or task.state is not DeferredCleanupState.RESERVED
                or task.callback is not None
            ):
                return False
            del self._tasks[cleanup_id]
            self._condition.notify_all()
            return True

    def _cleanup_now(self, cleanup_id: str) -> None:
        with self._condition:
            task = self._require_state_locked(
                cleanup_id,
                DeferredCleanupState.RESERVED,
            )
            callback = self._require_callback(task)
            task.state = DeferredCleanupState.RUNNING
            task.attempt_count += 1
            task.attempts_this_run = 1
        try:
            callback()
        except Exception as error:
            with self._condition:
                task.state = DeferredCleanupState.FAILED
                task.error_type = type(error).__name__
                self._condition.notify_all()
            raise
        with self._condition:
            del self._tasks[cleanup_id]
            self._condition.notify_all()

    def _wait_for_lanes(self, cleanup_id: str) -> None:
        with self._condition:
            task = self._require_state_locked(
                cleanup_id,
                DeferredCleanupState.RESERVED,
            )
            self._require_callback(task)
            task.state = DeferredCleanupState.WAITING_FOR_LANES
            self._condition.notify_all()

    def _mark_ready(self, cleanup_id: str) -> None:
        with self._condition:
            task = self._require_state_locked(
                cleanup_id,
                DeferredCleanupState.WAITING_FOR_LANES,
            )
            task.state = DeferredCleanupState.READY
            self._ensure_worker_locked()

    def _ensure_worker_locked(self) -> None:
        if self._worker_running:
            self._condition.notify_all()
            return
        self._worker_running = True
        worker = Thread(
            target=self._run_ready,
            name="capicove-lease-cleanup",
            daemon=True,
        )
        try:
            worker.start()
        except RuntimeError as error:
            self._worker_running = False
            for task in self._tasks.values():
                if task.state is DeferredCleanupState.READY:
                    task.state = DeferredCleanupState.FAILED
                    task.error_type = type(error).__name__
            self._condition.notify_all()

    def _run_ready(self) -> None:
        while True:
            with self._condition:
                task = next(
                    (
                        item
                        for item in self._tasks.values()
                        if item.state is DeferredCleanupState.READY
                    ),
                    None,
                )
                if task is None:
                    self._worker_running = False
                    self._condition.notify_all()
                    return
                task.state = DeferredCleanupState.RUNNING
                callback = self._require_callback(task)
            completed = False
            while task.attempts_this_run < self._max_attempts_per_run:
                with self._condition:
                    task.attempt_count += 1
                    task.attempts_this_run += 1
                try:
                    callback()
                except Exception as error:
                    with self._condition:
                        task.error_type = type(error).__name__
                    continue
                completed = True
                break
            with self._condition:
                if completed:
                    del self._tasks[task.cleanup_id]
                else:
                    task.state = DeferredCleanupState.FAILED
                self._condition.notify_all()

    def _require_state_locked(
        self,
        cleanup_id: str,
        expected: DeferredCleanupState,
    ) -> _DeferredCleanupTask:
        task = self._tasks.get(cleanup_id)
        if task is None:
            raise KeyError(cleanup_id)
        if task.state is not expected:
            raise RuntimeError(
                f"cleanup {cleanup_id} must be {expected.value}, got {task.state.value}"
            )
        return task

    @staticmethod
    def _require_callback(task: _DeferredCleanupTask) -> Callable[[], None]:
        if task.callback is None:
            raise RuntimeError("cleanup reservation is not bound")
        return task.callback

    def _has_active_work_locked(self) -> bool:
        return any(task.state is not DeferredCleanupState.FAILED for task in self._tasks.values())

    def _failure_snapshots_locked(self) -> tuple[DeferredCleanupFailureSnapshot, ...]:
        return tuple(
            DeferredCleanupFailureSnapshot(
                cleanup_id=task.cleanup_id,
                failure_code="lease_cleanup_failed",
                attempt_count=task.attempt_count,
                max_attempts_per_run=self._max_attempts_per_run,
                recovery_count=task.recovery_count,
                error_type=task.error_type or "UnknownError",
            )
            for task in self._tasks.values()
            if task.state is DeferredCleanupState.FAILED
        )

    def _report_locked(self) -> DeferredCleanupDrainReport:
        failures = self._failure_snapshots_locked()
        waiting = sum(
            task.state in {DeferredCleanupState.RESERVED, DeferredCleanupState.WAITING_FOR_LANES}
            for task in self._tasks.values()
        )
        running = sum(
            task.state in {DeferredCleanupState.READY, DeferredCleanupState.RUNNING}
            for task in self._tasks.values()
        )
        return DeferredCleanupDrainReport(
            completed=not self._tasks,
            pending_count=len(self._tasks),
            waiting_count=waiting,
            running_count=running,
            failed_count=len(failures),
            admission_rejections=self._admission_rejections,
            failures=failures,
        )


def _finite_float_tuple(field_name: str, values: object) -> tuple[float, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{field_name} must be numeric values")
    materialized: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field_name} must contain finite numbers")
        try:
            normalized = float(value)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(f"{field_name} must contain finite numbers") from error
        if not math.isfinite(normalized):
            raise ValueError(f"{field_name} must contain finite numbers")
        materialized.append(normalized)
    return tuple(materialized)


def _require_sha256(field_name: str, value: object) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hash")


def _require_non_blank(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")


def _require_positive_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


def _require_non_negative_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _require_finite(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be finite")
    try:
        finite = math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        finite = False
    if not finite:
        raise ValueError(f"{field_name} must be finite")


def _require_positive_finite(field_name: str, value: object) -> None:
    _require_finite(field_name, value)
    if float(value) <= 0:  # type: ignore[arg-type]
        raise ValueError(f"{field_name} must be positive")


def _unique_non_blank_tuple(field_name: str, values: tuple[str, ...]) -> tuple[str, ...]:
    materialized = tuple(values)
    if len(set(materialized)) != len(materialized):
        raise ValueError(f"{field_name} values must not contain duplicates")
    for value in materialized:
        _require_non_blank(field_name, value)
    return materialized


def _unique_non_blank_frozenset(
    field_name: str,
    values: object,
) -> frozenset[str]:
    if not isinstance(values, frozenset):
        raise TypeError(f"{field_name} values must be an iterable of strings")
    materialized = tuple(values)
    if len(set(materialized)) != len(materialized):
        raise ValueError(f"{field_name} values must not contain duplicates")
    for value in materialized:
        _require_non_blank(field_name, value)
    return frozenset(materialized)


__all__ = [
    "Candidate",
    "CandidateLane",
    "CandidateProvenance",
    "ContentCandidateProvenance",
    "DeferredCleanupDrainReport",
    "DeferredCleanupFailureSnapshot",
    "DeferredCleanupReservation",
    "DeferredCleanupState",
    "DeferredCleanupSupervisor",
    "FusedCandidate",
    "IndexedCandidateProvenance",
    "InvocationAuthorizationContext",
    "LaneDeadlineExceeded",
    "LaneObservation",
    "LaneScheduleSnapshot",
    "PerceptualHashQuery",
    "QueryRepresentation",
    "QueryVector",
    "ResolvedImageQuery",
    "RetrievalLane",
    "ScheduledLaneJob",
    "SearchFailure",
]
