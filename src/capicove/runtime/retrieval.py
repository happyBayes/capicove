"""Trusted local runtime policy for intent-routed reverse-image retrieval."""

from __future__ import annotations

import os
import re
import sqlite3
import threading
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any, Protocol, TypeVar

from capicove.contracts.retrieval import (
    EmbeddingSpaceRef,
    ExecutionProfile,
    SearchFailureCode,
)
from capicove.retrieval._adapters.sqlite_image_generations import (
    GenerationLeaseRecord,
    GenerationQualificationError,
    PublishedCandidateLeaseRecord,
    PublishedGenerationRecord,
)
from capicove.retrieval.image_generations import GenerationSetManifest, GenerationValidationError
from capicove.retrieval.profiles import (
    RetrievalProfile,
    RetrievalProfileRegistry,
    default_retrieval_profiles,
)
from capicove.retrieval.search_models import (
    DeferredCleanupReservation,
    DeferredCleanupSupervisor,
    LaneDeadlineExceeded,
    LaneScheduleSnapshot,
    RetrievalLane,
    ScheduledLaneJob,
    SearchFailure,
)
from capicove.retrieval.search_ports import IndexReader, QueryEncoder, UnitCatalog


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class CpuExecutionBudget:
    """Bound all lane-level and encoder-level native CPU concurrency."""

    total_threads: int
    max_parallel_lanes: int
    per_lane_compute_threads: int
    onnx_inter_op_threads: int = 1

    def __post_init__(self) -> None:
        for name in (
            "total_threads",
            "max_parallel_lanes",
            "per_lane_compute_threads",
            "onnx_inter_op_threads",
        ):
            _positive_int(name, getattr(self, name))
        if self.max_parallel_lanes * self.per_lane_compute_threads > self.total_threads:
            raise ValueError("lane workers oversubscribe the CPU budget")

    def schedule_snapshot(self, *, schedule_id: str) -> LaneScheduleSnapshot:
        _non_blank("schedule_id", schedule_id)
        return LaneScheduleSnapshot(
            schedule_id=schedule_id,
            total_threads=self.total_threads,
            max_parallel_lanes=self.max_parallel_lanes,
            per_lane_compute_threads=self.per_lane_compute_threads,
            onnx_inter_op_threads=self.onnx_inter_op_threads,
        )


def default_cpu_execution_budget() -> CpuExecutionBudget:
    """Derive the bounded personal default once at runtime composition."""
    total = min(4, os.cpu_count() or 1)
    parallel = min(2, total)
    return CpuExecutionBudget(
        total_threads=total,
        max_parallel_lanes=parallel,
        per_lane_compute_threads=max(1, total // parallel),
    )


class BoundedLaneScheduler:
    """Run retrieval work through the sole bounded lane thread pool."""

    def __init__(self, execution_policy: LaneScheduleSnapshot) -> None:
        if not isinstance(execution_policy, LaneScheduleSnapshot):
            raise TypeError("execution_policy must be a LaneScheduleSnapshot")
        self._execution_policy = execution_policy
        self._lane_slots = threading.BoundedSemaphore(execution_policy.max_parallel_lanes)

    @property
    def execution_policy(self) -> LaneScheduleSnapshot:
        return self._execution_policy

    def run(
        self,
        jobs: tuple[ScheduledLaneJob[_T], ...],
        *,
        deadline: datetime,
    ) -> tuple[_T, ...]:
        """Run bounded jobs and preserve submission order."""
        _deadline(deadline)
        materialized = tuple(jobs)
        if any(not isinstance(job, ScheduledLaneJob) for job in materialized):
            raise TypeError("jobs must contain ScheduledLaneJob values")
        if len({job.lane_id for job in materialized}) != len(materialized):
            raise ValueError("jobs must not repeat a lane_id")
        if not materialized:
            return ()

        futures: list[Future[_T]] = []
        executor = ThreadPoolExecutor(
            max_workers=self._execution_policy.max_parallel_lanes,
            thread_name_prefix="capicove-retrieval",
        )
        try:
            for job in materialized:
                _deadline(deadline)
                remaining = (deadline - datetime.now(UTC)).total_seconds()
                if not self._lane_slots.acquire(timeout=max(0.0, remaining)):
                    raise TimeoutError("lane admission deadline exceeded")
                permit = _LanePermit(self._lane_slots)
                try:
                    future = executor.submit(_run_lane_job, job.callback, permit)
                except BaseException:
                    permit.release()
                    raise
                future.add_done_callback(permit.release_if_cancelled)
                futures.append(future)
            results: list[_T] = []
            for future in futures:
                _deadline(deadline)
                remaining = (deadline - datetime.now(UTC)).total_seconds()
                results.append(future.result(timeout=max(0.0, remaining)))
        except TimeoutError as error:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            if not futures:
                raise SearchFailure(SearchFailureCode.DEADLINE_EXCEEDED) from error
            raise LaneDeadlineExceeded(tuple(futures)) from error
        except SearchFailure as error:
            for future in futures:
                future.cancel()
            if error.code is SearchFailureCode.DEADLINE_EXCEEDED and futures:
                executor.shutdown(wait=False, cancel_futures=True)
                raise LaneDeadlineExceeded(tuple(futures)) from error
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        executor.shutdown(wait=True)
        return tuple(results)


class _LanePermit:
    """Release one scheduler-wide lane slot exactly once."""

    def __init__(self, semaphore: threading.BoundedSemaphore) -> None:
        self._semaphore = semaphore
        self._lock = threading.Lock()
        self._released = False

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._semaphore.release()

    def release_if_cancelled(self, future: Future[Any]) -> None:
        if future.cancelled():
            self.release()


def _run_lane_job(callback: Callable[[], _T], permit: _LanePermit) -> _T:
    try:
        return callback()
    finally:
        permit.release()


QueryEncoderFactory = Callable[..., QueryEncoder]


@dataclass(frozen=True, slots=True)
class TrustedQueryEncoderRegistration:
    """Bind one exact semantic lane/space/bundle to a trusted CPU factory."""

    lane: RetrievalLane
    space: EmbeddingSpaceRef
    target_bundle_manifest_sha256: str
    execution_profile: ExecutionProfile
    factory: QueryEncoderFactory

    def __post_init__(self) -> None:
        if self.lane not in {RetrievalLane.INSTANCE, RetrievalLane.CONCEPT}:
            raise ValueError("query encoders may only register semantic lanes")
        if not isinstance(self.space, EmbeddingSpaceRef):
            raise TypeError("space must be an EmbeddingSpaceRef")
        if _SHA256.fullmatch(self.target_bundle_manifest_sha256) is None:
            raise ValueError("target_bundle_manifest_sha256 must be a lowercase SHA-256")
        if self.execution_profile is not ExecutionProfile.CPU:
            raise ValueError("the local registry accepts CPU registrations only")
        if not callable(self.factory):
            raise TypeError("factory must be callable")


class TrustedQueryEncoderRegistry:
    """Lazily resolve only exact verified CPU encoder registrations."""

    def __init__(
        self,
        registrations: tuple[TrustedQueryEncoderRegistration, ...],
        *,
        cpu_budget: CpuExecutionBudget,
    ) -> None:
        if not isinstance(cpu_budget, CpuExecutionBudget):
            raise TypeError("cpu_budget must be a CpuExecutionBudget")
        entries: dict[tuple[RetrievalLane, EmbeddingSpaceRef, str], object] = {}
        for registration in registrations:
            if not isinstance(registration, TrustedQueryEncoderRegistration):
                raise TypeError("registrations must contain trusted registration values")
            key = (
                registration.lane,
                registration.space,
                registration.target_bundle_manifest_sha256,
            )
            if key in entries:
                raise ValueError("query encoder registrations must have unique exact keys")
            entries[key] = registration
        self._registrations = MappingProxyType(entries)
        self._cpu_budget = cpu_budget
        self._instances: dict[tuple[RetrievalLane, EmbeddingSpaceRef, str], QueryEncoder] = {}
        self._lock = threading.Lock()

    def resolve(
        self,
        lane: RetrievalLane,
        space: EmbeddingSpaceRef,
        target_bundle_manifest_sha256: str,
        execution_profile: ExecutionProfile,
    ) -> QueryEncoder:
        """Resolve an exact lane, complete space, bundle digest, and CPU policy."""
        if execution_profile is ExecutionProfile.GPU:
            raise SearchFailure(SearchFailureCode.PROFILE_UNAVAILABLE)
        if execution_profile not in {ExecutionProfile.AUTO, ExecutionProfile.CPU}:
            raise SearchFailure(SearchFailureCode.PROFILE_UNAVAILABLE)
        key = (lane, space, target_bundle_manifest_sha256)
        registration = self._registrations.get(key)
        if not isinstance(registration, TrustedQueryEncoderRegistration):
            raise SearchFailure(SearchFailureCode.EMBEDDING_SPACE_INCOMPATIBLE)
        with self._lock:
            current = self._instances.get(key)
            if current is None:
                current = registration.factory(
                    intra_op_threads=self._cpu_budget.per_lane_compute_threads,
                    inter_op_threads=self._cpu_budget.onnx_inter_op_threads,
                )
                if (
                    current.embedding_space != space
                    or current.bundle_manifest_sha256 != target_bundle_manifest_sha256
                ):
                    raise SearchFailure(SearchFailureCode.EMBEDDING_SPACE_INCOMPATIBLE)
                self._instances[key] = current
            return current


class GenerationLifecycle(Protocol):
    """Persistent active-alias lifecycle consumed by production retrieval."""

    def acquire(
        self,
        *,
        workspace_id: str,
        profile_ref: str,
        lease_id: str,
        owner_id: str,
        owner_session_epoch: str,
        acquired_at: datetime,
        expires_at: datetime,
        deadline: datetime,
    ) -> GenerationLeaseRecord: ...

    def resolve_published(
        self,
        activation_set_id: str,
        *,
        deadline: datetime,
    ) -> PublishedGenerationRecord | None: ...

    def release(
        self,
        lease_id: str,
        *,
        released_at: datetime,
        deadline: datetime,
    ) -> None: ...


class CandidateGenerationLifecycle(Protocol):
    """Distinct persistent lifecycle for exact unqualified candidate evaluation."""

    def acquire_published(
        self,
        *,
        activation_set_id: str,
        expected_manifest_sha256: str,
        profile_ref: str,
        lease_id: str,
        owner_id: str,
        owner_session_epoch: str,
        acquired_at: datetime,
        expires_at: datetime,
        deadline: datetime,
    ) -> PublishedCandidateLeaseRecord: ...

    def resolve_published(
        self,
        activation_set_id: str,
        *,
        deadline: datetime,
    ) -> PublishedGenerationRecord | None: ...

    def release_published(
        self,
        lease_id: str,
        *,
        released_at: datetime,
        deadline: datetime,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class OpenedGenerationHandles:
    """Contain one fully verified complete set of process-local readers.

    Close callbacks must be idempotent and safe to retry after raising. Successful
    callbacks are removed from the pending close sequence and are never rerun.
    """

    manifest: GenerationSetManifest
    unit_catalog: UnitCatalog
    readers: Mapping[RetrievalLane, IndexReader]
    close_callbacks: tuple[Callable[[], None], ...]
    _close_lock: Any = field(init=False, repr=False, compare=False)
    _pending_close_callbacks: list[Callable[[], None]] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        readers = dict(self.readers)
        if set(readers) != {
            RetrievalLane.PERCEPTUAL,
            RetrievalLane.INSTANCE,
            RetrievalLane.CONCEPT,
        }:
            raise ValueError("readers must contain the complete personal lane set")
        object.__setattr__(self, "readers", MappingProxyType(readers))
        callbacks = tuple(self.close_callbacks)
        if not callbacks or any(not callable(callback) for callback in callbacks):
            raise ValueError("close_callbacks must contain callable handle closers")
        object.__setattr__(self, "close_callbacks", callbacks)
        object.__setattr__(self, "_close_lock", threading.Lock())
        object.__setattr__(self, "_pending_close_callbacks", list(callbacks))

    def close(self) -> None:
        """Close native/mmap handles once each in reverse dependency order."""
        with self._close_lock:
            while self._pending_close_callbacks:
                callback = self._pending_close_callbacks[-1]
                callback()
                self._pending_close_callbacks.pop()


class GenerationHandleFactory(Protocol):
    """Open all readers from one checksum-verified published generation."""

    def open(
        self,
        published: PublishedGenerationRecord,
        *,
        deadline: datetime,
    ) -> OpenedGenerationHandles: ...


@dataclass(slots=True)
class _LeaseReleaseState:
    record: GenerationLeaseRecord | PublishedCandidateLeaseRecord
    released: bool = False


@dataclass(slots=True)
class _HandleGroup:
    handles: OpenedGenerationHandles
    refcount: int
    closing: bool = False
    handles_closed: bool = False
    pending_release: _LeaseReleaseState | None = None


class ManagedGenerationLease:
    """Own one persistent lease and one reference to shared local readers."""

    def __init__(
        self,
        manager: GenerationSetManager,
        release_state: _LeaseReleaseState,
        handles: OpenedGenerationHandles,
    ) -> None:
        self._manager = manager
        self._release_state = release_state
        self._handles = handles
        self._closed = False

    @property
    def manifest(self) -> GenerationSetManifest:
        return self._handles.manifest

    @property
    def unit_catalog(self) -> UnitCatalog:
        return self._handles.unit_catalog

    @property
    def readers(self) -> Mapping[RetrievalLane, IndexReader]:
        return self._handles.readers

    @property
    def qualification_tier(self) -> str:
        record = self._release_state.record
        if not isinstance(
            record, GenerationLeaseRecord
        ):  # pragma: no cover - constructor invariant
            raise RuntimeError("active generation lease record kind is inconsistent")
        return record.alias_snapshot.qualification_tier

    @property
    def activation_set_id(self) -> str:
        record = self._release_state.record
        if not isinstance(
            record, GenerationLeaseRecord
        ):  # pragma: no cover - constructor invariant
            raise RuntimeError("active generation lease record kind is inconsistent")
        return record.alias_snapshot.activation_set_id

    def close(self) -> None:
        if self._closed and self._release_state.released:
            return
        self._closed = True
        try:
            self._manager._release(self._release_state)
        except SearchFailure:
            raise
        except (sqlite3.DatabaseError, OSError, RuntimeError, TimeoutError, ValueError) as error:
            raise SearchFailure(SearchFailureCode.INTERNAL) from error

    def __enter__(self) -> ManagedGenerationLease:
        if self._closed:
            raise RuntimeError("generation lease is closed")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class PublishedCandidateLease:
    """Own one persistent unqualified candidate lease and shared local readers."""

    def __init__(
        self,
        manager: GenerationSetManager,
        release_state: _LeaseReleaseState,
        handles: OpenedGenerationHandles,
    ) -> None:
        self._manager = manager
        self._release_state = release_state
        self._handles = handles
        self._closed = False

    @property
    def manifest(self) -> GenerationSetManifest:
        return self._handles.manifest

    @property
    def unit_catalog(self) -> UnitCatalog:
        return self._handles.unit_catalog

    @property
    def readers(self) -> Mapping[RetrievalLane, IndexReader]:
        return self._handles.readers

    @property
    def activation_set_id(self) -> str:
        record = self._candidate_record()
        return record.activation_set_id

    @property
    def profile_ref(self) -> str:
        return self._candidate_record().profile_ref

    def close(self) -> None:
        if self._closed and self._release_state.released:
            return
        self._closed = True
        try:
            self._manager._release(self._release_state)
        except SearchFailure:
            raise
        except (sqlite3.DatabaseError, OSError, RuntimeError, TimeoutError, ValueError) as error:
            raise SearchFailure(SearchFailureCode.INTERNAL) from error

    def __enter__(self) -> PublishedCandidateLease:
        if self._closed:
            raise RuntimeError("published candidate lease is closed")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _candidate_record(self) -> PublishedCandidateLeaseRecord:
        record = self._release_state.record
        if not isinstance(record, PublishedCandidateLeaseRecord):
            raise RuntimeError("published candidate lease record kind is inconsistent")
        return record


class GenerationSetManager:
    """Pin atomic aliases and retain local handles until the last lease closes."""

    def __init__(
        self,
        *,
        workspace_id: str,
        lifecycle: GenerationLifecycle,
        candidate_lifecycle: CandidateGenerationLifecycle | None = None,
        handle_factory: GenerationHandleFactory,
        owner_id: str,
        owner_session_epoch: str,
        clock: Callable[[], datetime],
        lease_id_factory: Callable[[], str] = lambda: str(uuid.uuid4()),
        lease_interval: timedelta = timedelta(minutes=5),
        close_timeout: timedelta = timedelta(seconds=30),
        cleanup_supervisor: DeferredCleanupSupervisor | None = None,
    ) -> None:
        for name, value in (
            ("workspace_id", workspace_id),
            ("owner_id", owner_id),
            ("owner_session_epoch", owner_session_epoch),
        ):
            _non_blank(name, value)
        if not callable(clock) or not callable(lease_id_factory):
            raise TypeError("clock and lease_id_factory must be callable")
        if not timedelta(0) < lease_interval <= timedelta(minutes=10):
            raise ValueError("lease_interval must be positive and at most 10 minutes")
        if not timedelta(0) < close_timeout <= timedelta(minutes=1):
            raise ValueError("close_timeout must be positive and at most one minute")
        if cleanup_supervisor is not None and not isinstance(
            cleanup_supervisor,
            DeferredCleanupSupervisor,
        ):
            raise TypeError("cleanup_supervisor must be a DeferredCleanupSupervisor")
        if candidate_lifecycle is not None and any(
            not callable(getattr(candidate_lifecycle, method, None))
            for method in (
                "acquire_published",
                "resolve_published",
                "release_published",
            )
        ):
            raise TypeError("candidate_lifecycle must provide the candidate lifecycle surface")
        self._workspace_id = workspace_id
        self._lifecycle = lifecycle
        self._candidate_lifecycle = candidate_lifecycle
        self._handle_factory = handle_factory
        self._owner_id = owner_id
        self._owner_session_epoch = owner_session_epoch
        self._clock = clock
        self._lease_id_factory = lease_id_factory
        self._lease_interval = lease_interval
        self._close_timeout = close_timeout
        self._cleanup_supervisor = cleanup_supervisor or DeferredCleanupSupervisor()
        self._groups: dict[tuple[str, str], _HandleGroup] = {}
        self._lock = threading.RLock()

    def acquire(
        self,
        profile_ref: str,
        *,
        deadline: datetime,
        cleanup_reservation: DeferredCleanupReservation | None = None,
    ) -> ManagedGenerationLease:
        """Acquire one persistent alias snapshot and its complete local readers."""
        _non_blank("profile_ref", profile_ref)
        if cleanup_reservation is not None and not isinstance(
            cleanup_reservation,
            DeferredCleanupReservation,
        ):
            raise TypeError("cleanup_reservation must be a DeferredCleanupReservation")
        _deadline(deadline)
        acquired_at = self._clock()
        _deadline(deadline)
        lease_id = self._lease_id_factory()
        _non_blank("lease_id", lease_id)
        owns_cleanup = cleanup_reservation is None
        cleanup = cleanup_reservation or self._cleanup_supervisor.reserve()
        try:
            record = self._lifecycle.acquire(
                workspace_id=self._workspace_id,
                profile_ref=profile_ref,
                lease_id=lease_id,
                owner_id=self._owner_id,
                owner_session_epoch=self._owner_session_epoch,
                acquired_at=acquired_at,
                expires_at=acquired_at + self._lease_interval,
                deadline=deadline,
            )
        except GenerationQualificationError as error:
            if owns_cleanup:
                cleanup.cancel()
            raise SearchFailure(SearchFailureCode.INDEX_GENERATION_UNAVAILABLE) from error
        except TimeoutError as error:
            if owns_cleanup:
                cleanup.cancel()
            raise SearchFailure(SearchFailureCode.DEADLINE_EXCEEDED) from error
        except sqlite3.DatabaseError as error:
            if owns_cleanup:
                cleanup.cancel()
            raise SearchFailure(SearchFailureCode.INTERNAL) from error
        except BaseException:
            if owns_cleanup:
                cleanup.cancel()
            raise
        alias = record.alias_snapshot
        key = (alias.activation_set_id, alias.manifest_sha256)
        try:
            published = self._lifecycle.resolve_published(
                alias.activation_set_id,
                deadline=deadline,
            )
            if (
                published is None
                or published.manifest_sha256 != alias.manifest_sha256
                or published.workspace_id != self._workspace_id
                or published.retrieval_profile_ref != profile_ref
            ):
                raise SearchFailure(SearchFailureCode.INDEX_GENERATION_UNAVAILABLE)
            with self._lock:
                group = self._groups.get(key)
                if group is not None and group.closing:
                    try:
                        self._recover_closing_group(key, group)
                    except (
                        sqlite3.DatabaseError,
                        OSError,
                        RuntimeError,
                        TimeoutError,
                        ValueError,
                    ) as error:
                        raise SearchFailure(SearchFailureCode.INTERNAL) from error
                    group = None
                elif group is not None and group.handles_closed:
                    raise RuntimeError("generation handle close state is inconsistent")
                if group is None:
                    handles = self._handle_factory.open(published, deadline=deadline)
                    manifest = handles.manifest
                    if (
                        manifest.activation_set_id != alias.activation_set_id
                        or manifest.workspace_id != self._workspace_id
                        or manifest.retrieval_profile_ref != profile_ref
                    ):
                        handles.close()
                        raise SearchFailure(SearchFailureCode.INDEX_GENERATION_UNAVAILABLE)
                    group = _HandleGroup(handles=handles, refcount=0)
                    self._groups[key] = group
                group.refcount += 1
                if owns_cleanup:
                    cleanup.cancel()
                return ManagedGenerationLease(
                    self,
                    _LeaseReleaseState(record),
                    group.handles,
                )
        except BaseException as error:
            mapped = _map_generation_acquire_error(error)
            cleanup.bind(lambda: self._release_failed_acquire(record))
            try:
                cleanup.cleanup_now()
            # Preserve the acquire error; the supervisor records cleanup failure for recovery.
            except Exception:  # nosec B110
                pass
            if mapped is not None:
                raise mapped from error
            raise

    def acquire_published(
        self,
        activation_set_id: str,
        expected_manifest_sha256: str,
        profile_ref: str,
        *,
        deadline: datetime,
        cleanup_reservation: DeferredCleanupReservation | None = None,
    ) -> PublishedCandidateLease:
        """Pin one exact published candidate without resolving an active alias."""
        _non_blank("activation_set_id", activation_set_id)
        _non_blank("profile_ref", profile_ref)
        if _SHA256.fullmatch(expected_manifest_sha256) is None:
            raise ValueError("expected_manifest_sha256 must be a lowercase SHA-256")
        if cleanup_reservation is not None and not isinstance(
            cleanup_reservation,
            DeferredCleanupReservation,
        ):
            raise TypeError("cleanup_reservation must be a DeferredCleanupReservation")
        _deadline(deadline)
        candidate_lifecycle = self._require_candidate_lifecycle()
        acquired_at = self._clock()
        _deadline(deadline)
        lease_id = self._lease_id_factory()
        _non_blank("lease_id", lease_id)
        owns_cleanup = cleanup_reservation is None
        cleanup = cleanup_reservation or self._cleanup_supervisor.reserve()
        try:
            record = candidate_lifecycle.acquire_published(
                activation_set_id=activation_set_id,
                expected_manifest_sha256=expected_manifest_sha256,
                profile_ref=profile_ref,
                lease_id=lease_id,
                owner_id=self._owner_id,
                owner_session_epoch=self._owner_session_epoch,
                acquired_at=acquired_at,
                expires_at=acquired_at + self._lease_interval,
                deadline=deadline,
            )
        except GenerationQualificationError as error:
            if owns_cleanup:
                cleanup.cancel()
            raise SearchFailure(SearchFailureCode.INDEX_GENERATION_UNAVAILABLE) from error
        except TimeoutError as error:
            if owns_cleanup:
                cleanup.cancel()
            raise SearchFailure(SearchFailureCode.DEADLINE_EXCEEDED) from error
        except sqlite3.DatabaseError as error:
            if owns_cleanup:
                cleanup.cancel()
            raise SearchFailure(SearchFailureCode.INTERNAL) from error
        except BaseException:
            if owns_cleanup:
                cleanup.cancel()
            raise
        key = (record.activation_set_id, record.manifest_sha256)
        try:
            if (
                record.activation_set_id != activation_set_id
                or record.manifest_sha256 != expected_manifest_sha256
                or record.profile_ref != profile_ref
            ):
                raise SearchFailure(SearchFailureCode.INDEX_GENERATION_UNAVAILABLE)
            published = candidate_lifecycle.resolve_published(
                record.activation_set_id,
                deadline=deadline,
            )
            if (
                published is None
                or published.manifest_sha256 != record.manifest_sha256
                or published.workspace_id != self._workspace_id
                or published.retrieval_profile_ref != profile_ref
            ):
                raise SearchFailure(SearchFailureCode.INDEX_GENERATION_UNAVAILABLE)
            with self._lock:
                group = self._groups.get(key)
                if group is not None and group.closing:
                    try:
                        self._recover_closing_group(key, group)
                    except (
                        sqlite3.DatabaseError,
                        OSError,
                        RuntimeError,
                        TimeoutError,
                        ValueError,
                    ) as error:
                        raise SearchFailure(SearchFailureCode.INTERNAL) from error
                    group = None
                elif group is not None and group.handles_closed:
                    raise RuntimeError("generation handle close state is inconsistent")
                if group is None:
                    handles = self._handle_factory.open(published, deadline=deadline)
                    manifest = handles.manifest
                    if (
                        manifest.activation_set_id != record.activation_set_id
                        or manifest.workspace_id != self._workspace_id
                        or manifest.retrieval_profile_ref != profile_ref
                    ):
                        handles.close()
                        raise SearchFailure(SearchFailureCode.INDEX_GENERATION_UNAVAILABLE)
                    group = _HandleGroup(handles=handles, refcount=0)
                    self._groups[key] = group
                group.refcount += 1
                if owns_cleanup:
                    cleanup.cancel()
                return PublishedCandidateLease(
                    self,
                    _LeaseReleaseState(record),
                    group.handles,
                )
        except BaseException as error:
            mapped = _map_generation_acquire_error(error)
            cleanup.bind(lambda: self._release_failed_acquire(record))
            try:
                cleanup.cleanup_now()
            # Preserve the acquire error; the supervisor records cleanup failure for recovery.
            except Exception:  # nosec B110
                pass
            if mapped is not None:
                raise mapped from error
            raise

    def _release_failed_acquire(
        self,
        record: GenerationLeaseRecord | PublishedCandidateLeaseRecord,
    ) -> None:
        released_at = self._clock()
        self._release_lease_record(record, released_at=released_at)

    def _recover_closing_group(
        self,
        key: tuple[str, str],
        group: _HandleGroup,
    ) -> None:
        pending = group.pending_release
        if not group.closing or group.refcount != 1 or pending is None:
            raise RuntimeError("closing generation handle state is inconsistent")
        if not group.handles_closed:
            group.handles.close()
            group.handles_closed = True
        self._release_record(pending.record)
        pending.released = True
        group.refcount = 0
        group.pending_release = None
        del self._groups[key]

    def _release(self, release_state: _LeaseReleaseState) -> None:
        record = release_state.record
        key = _lease_record_key(record)
        with self._lock:
            if release_state.released:
                return
            group = self._groups.get(key)
            if group is None or group.refcount < 1:
                raise RuntimeError("local generation lease ownership is inconsistent")
            if group.refcount == 1:
                if group.pending_release is None:
                    group.pending_release = release_state
                elif group.pending_release is not release_state:
                    raise RuntimeError("generation lease release state is inconsistent")
                group.closing = True
                self._recover_closing_group(key, group)
            else:
                if group.closing or group.handles_closed or group.pending_release is not None:
                    raise RuntimeError("generation handle reference state is inconsistent")
                self._release_record(record)
                release_state.released = True
                group.refcount -= 1

    def _release_record(
        self,
        record: GenerationLeaseRecord | PublishedCandidateLeaseRecord,
    ) -> None:
        released_at = self._clock()
        self._release_lease_record(record, released_at=released_at)

    def _release_lease_record(
        self,
        record: GenerationLeaseRecord | PublishedCandidateLeaseRecord,
        *,
        released_at: datetime,
    ) -> None:
        deadline = released_at + self._close_timeout
        if isinstance(record, PublishedCandidateLeaseRecord):
            self._require_candidate_lifecycle().release_published(
                record.lease_id,
                released_at=released_at,
                deadline=deadline,
            )
            return
        self._lifecycle.release(
            record.lease_id,
            released_at=released_at,
            deadline=deadline,
        )

    def _require_candidate_lifecycle(self) -> CandidateGenerationLifecycle:
        lifecycle = self._candidate_lifecycle
        if lifecycle is None:
            raise RuntimeError("candidate generation lifecycle is not configured")
        return lifecycle


class CandidateEvaluationLease:
    """Borrow one session-owned candidate without owning its persistent lease."""

    def __init__(
        self,
        provider: CandidateBoundGenerationSetProvider,
        candidate: PublishedCandidateLease,
    ) -> None:
        if not isinstance(candidate, PublishedCandidateLease):
            raise TypeError("candidate must be a PublishedCandidateLease")
        self._provider = provider
        self._candidate = candidate
        self._close_lock = threading.Lock()
        self._closed = False

    @property
    def manifest(self) -> GenerationSetManifest:
        return self._candidate.manifest

    @property
    def unit_catalog(self) -> UnitCatalog:
        return self._candidate.unit_catalog

    @property
    def readers(self) -> Mapping[RetrievalLane, IndexReader]:
        return self._candidate.readers

    @property
    def qualification_tier(self) -> str:
        return "candidate_evaluation"

    @property
    def activation_set_id(self) -> str:
        return self._candidate.activation_set_id

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._provider._release_borrow()
            self._closed = True

    def __enter__(self) -> CandidateEvaluationLease:
        with self._close_lock:
            if self._closed:
                raise RuntimeError("candidate evaluation lease is closed")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class CandidateBoundGenerationSetProvider:
    """Own one persistent candidate lease and issue process-local query borrows."""

    def __init__(self, candidate: PublishedCandidateLease) -> None:
        if not isinstance(candidate, PublishedCandidateLease):
            raise TypeError("candidate must be a PublishedCandidateLease")
        self._candidate = candidate
        self._profile_ref = candidate.profile_ref
        self._lock = threading.RLock()
        self._borrow_count = 0
        self._accepting_borrows = True
        self._closed = False

    def acquire_candidate_evaluation(
        self,
        profile_ref: str,
        *,
        deadline: datetime,
        cleanup_reservation: DeferredCleanupReservation | None = None,
    ) -> CandidateEvaluationLease:
        """Borrow the already-pinned candidate without persistent lifecycle I/O."""
        if profile_ref != self._profile_ref:
            raise SearchFailure(SearchFailureCode.PROFILE_UNAVAILABLE)
        if cleanup_reservation is not None and not isinstance(
            cleanup_reservation,
            DeferredCleanupReservation,
        ):
            raise TypeError("cleanup_reservation must be a DeferredCleanupReservation")
        _deadline(deadline)
        with self._lock:
            if not self._accepting_borrows or self._closed:
                raise SearchFailure(SearchFailureCode.INDEX_GENERATION_UNAVAILABLE)
            self._borrow_count += 1
            return CandidateEvaluationLease(self, self._candidate)

    def close(self) -> None:
        """Release the persistent candidate only after every local borrow closes."""
        with self._lock:
            if self._closed:
                return
            self._accepting_borrows = False
            if self._borrow_count:
                raise RuntimeError("candidate evaluation borrows remain open")
            self._candidate.close()
            self._closed = True

    def __enter__(self) -> CandidateBoundGenerationSetProvider:
        with self._lock:
            if not self._accepting_borrows or self._closed:
                raise RuntimeError("candidate evaluation session is closed")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _release_borrow(self) -> None:
        with self._lock:
            if self._borrow_count < 1:
                raise RuntimeError("candidate evaluation borrow state is inconsistent")
            self._borrow_count -= 1


def _lease_record_key(
    record: GenerationLeaseRecord | PublishedCandidateLeaseRecord,
) -> tuple[str, str]:
    if isinstance(record, GenerationLeaseRecord):
        alias = record.alias_snapshot
        return alias.activation_set_id, alias.manifest_sha256
    return record.activation_set_id, record.manifest_sha256


def _deadline(deadline: datetime) -> None:
    if (
        not isinstance(deadline, datetime)
        or deadline.tzinfo is None
        or deadline.utcoffset() is None
        or datetime.now(UTC) >= deadline
    ):
        raise SearchFailure(SearchFailureCode.DEADLINE_EXCEEDED)


def _map_generation_acquire_error(error: BaseException) -> SearchFailure | None:
    if isinstance(error, SearchFailure):
        return error
    if isinstance(error, TimeoutError):
        return SearchFailure(SearchFailureCode.DEADLINE_EXCEEDED)
    if isinstance(error, GenerationQualificationError):
        return SearchFailure(SearchFailureCode.INDEX_GENERATION_UNAVAILABLE)
    if isinstance(error, (GenerationValidationError, OSError, ValueError)):
        return SearchFailure(SearchFailureCode.INDEX_CORRUPT)
    if isinstance(error, (sqlite3.DatabaseError, RuntimeError)):
        return SearchFailure(SearchFailureCode.INTERNAL)
    return None


def _non_blank(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")


def _positive_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


__all__ = [
    "BoundedLaneScheduler",
    "CpuExecutionBudget",
    "GenerationHandleFactory",
    "GenerationLifecycle",
    "GenerationSetManager",
    "LaneScheduleSnapshot",
    "ManagedGenerationLease",
    "OpenedGenerationHandles",
    "RetrievalProfile",
    "RetrievalProfileRegistry",
    "ScheduledLaneJob",
    "TrustedQueryEncoderRegistration",
    "TrustedQueryEncoderRegistry",
    "default_cpu_execution_budget",
    "default_retrieval_profiles",
]
