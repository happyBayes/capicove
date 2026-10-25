"""Tests for trusted reverse-image retrieval runtime profiles."""

from __future__ import annotations

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from PIL import Image

from capicove.contracts.retrieval import (
    DistanceMetric,
    EmbeddingSpaceRef,
    ExecutionProfile,
    QueryModality,
    SearchFailureCode,
    TargetType,
    VectorNormalization,
)
from capicove.retrieval._adapters.sqlite_image_generations import (
    GenerationAliasSnapshot,
    GenerationLeaseRecord,
    GenerationQualificationError,
    PublishedCandidateLeaseRecord,
    PublishedGenerationRecord,
)
from capicove.retrieval.image_generations import GenerationSetManifest
from capicove.retrieval.search_models import (
    DeferredCleanupSupervisor,
    LaneDeadlineExceeded,
    QueryVector,
    RetrievalLane,
    SearchFailure,
)
from capicove.retrieval.search_ports import GenerationSetProvider, SearchGenerationLease
from capicove.runtime.retrieval import (
    BoundedLaneScheduler,
    CandidateBoundGenerationSetProvider,
    CpuExecutionBudget,
    GenerationSetManager,
    OpenedGenerationHandles,
    ScheduledLaneJob,
    TrustedQueryEncoderRegistration,
    TrustedQueryEncoderRegistry,
    default_cpu_execution_budget,
    default_retrieval_profiles,
)


FUTURE = datetime.now(UTC) + timedelta(hours=1)
PAST = datetime.now(UTC) - timedelta(seconds=1)
SHA_A = "a" * 64


def _space(space_id: str = "instance-v1") -> EmbeddingSpaceRef:
    return EmbeddingSpaceRef(
        space_id=space_id,
        provider_id="test-provider",
        model_revision="revision-1",
        preprocessing_version="rgb-v1",
        dimension=2,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.INNER_PRODUCT,
        query_modalities=frozenset({QueryModality.IMAGE}),
        target_modalities=frozenset({TargetType.IMAGE}),
    )


def test_local_default_is_a_fixed_alias_for_personal_lite() -> None:
    registry = default_retrieval_profiles()

    assert registry.resolve("local-default") == registry.resolve("personal-lite")
    personal = registry.resolve("personal-lite")
    assert personal.max_units == 10_000
    assert personal.phash_radius == 2
    assert dict(personal.lane_weights) == {
        RetrievalLane.CONTENT: 1.0,
        RetrievalLane.PERCEPTUAL: 2.0,
        RetrievalLane.INSTANCE: 1.0,
        RetrievalLane.CONCEPT: 1.0,
    }
    assert personal.semantic_formats == (
        (RetrievalLane.INSTANCE, "numpy-flat-f32-v1"),
        (RetrievalLane.CONCEPT, "numpy-flat-f32-v1"),
    )
    with pytest.raises(SearchFailure) as enterprise:
        registry.resolve("enterprise-disk")
    assert enterprise.value.code is SearchFailureCode.PROFILE_UNAVAILABLE


def test_capacity_failure_never_switches_profile() -> None:
    profile = default_retrieval_profiles().resolve("personal-lite")
    with pytest.raises(SearchFailure) as raised:
        profile.validate_unit_count(10_001)
    assert raised.value.code is SearchFailureCode.CAPACITY_EXCEEDED

    with pytest.raises(SearchFailure) as missing:
        default_retrieval_profiles().resolve("request-selected-backend")
    assert missing.value.code is SearchFailureCode.PROFILE_UNAVAILABLE


def test_unimplemented_enterprise_does_not_import_optional_backends() -> None:
    sys.modules.pop("faiss", None)
    sys.modules.pop("onnxruntime", None)

    with pytest.raises(SearchFailure) as raised:
        default_retrieval_profiles().resolve("enterprise-disk")

    assert raised.value.code is SearchFailureCode.PROFILE_UNAVAILABLE
    assert "faiss" not in sys.modules
    assert "onnxruntime" not in sys.modules


def test_scheduler_is_bounded_preserves_order_and_checks_deadline() -> None:
    budget = CpuExecutionBudget(total_threads=4, max_parallel_lanes=2, per_lane_compute_threads=2)
    scheduler = BoundedLaneScheduler(budget.schedule_snapshot(schedule_id="search-1"))
    lock = threading.Lock()
    barrier = threading.Barrier(2)
    active = 0
    peak = 0

    def task(value: str) -> str:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait(timeout=2)
        with lock:
            active -= 1
        return value

    results = scheduler.run(
        (
            ScheduledLaneJob("instance", lambda: task("first")),
            ScheduledLaneJob("concept", lambda: task("second")),
        ),
        deadline=FUTURE,
    )

    assert results == ("first", "second")
    assert peak == 2
    with pytest.raises(SearchFailure) as expired:
        scheduler.run((ScheduledLaneJob("never", lambda: "bad"),), deadline=PAST)
    assert expired.value.code is SearchFailureCode.DEADLINE_EXCEEDED


def test_scheduler_validates_contract_and_handles_empty_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert default_cpu_execution_budget() == CpuExecutionBudget(1, 1, 1)

    with pytest.raises(TypeError, match="execution_policy"):
        BoundedLaneScheduler(cast(Any, object()))

    policy = CpuExecutionBudget(1, 1, 1).schedule_snapshot(schedule_id="contract")
    scheduler = BoundedLaneScheduler(policy)
    assert scheduler.execution_policy is policy
    assert scheduler.run((), deadline=FUTURE) == ()

    with pytest.raises(TypeError, match="ScheduledLaneJob"):
        scheduler.run(
            cast(tuple[ScheduledLaneJob[str], ...], (object(),)),
            deadline=FUTURE,
        )
    with pytest.raises(ValueError, match="repeat"):
        scheduler.run(
            (
                ScheduledLaneJob("duplicate", lambda: "first"),
                ScheduledLaneJob("duplicate", lambda: "second"),
            ),
            deadline=FUTURE,
        )


@pytest.mark.parametrize(
    "error",
    (SearchFailure(SearchFailureCode.PERMISSION_DENIED), RuntimeError("lane failed")),
    ids=("search-failure", "runtime-error"),
)
def test_scheduler_releases_shared_slot_after_job_failure(error: Exception) -> None:
    scheduler = BoundedLaneScheduler(
        CpuExecutionBudget(1, 1, 1).schedule_snapshot(schedule_id="job-failure")
    )

    def fail() -> str:
        raise error

    with pytest.raises(type(error)) as raised:
        scheduler.run((ScheduledLaneJob("failing", fail),), deadline=FUTURE)
    if isinstance(error, SearchFailure):
        assert cast(SearchFailure, raised.value).code is SearchFailureCode.PERMISSION_DENIED

    assert scheduler.run(
        (ScheduledLaneJob("later", lambda: "released"),),
        deadline=FUTURE,
    ) == ("released",)


def test_scheduler_wraps_job_deadline_failure_and_runs_settled_cleanup() -> None:
    scheduler = BoundedLaneScheduler(
        CpuExecutionBudget(1, 1, 1).schedule_snapshot(schedule_id="job-deadline")
    )

    def fail() -> str:
        raise SearchFailure(SearchFailureCode.DEADLINE_EXCEEDED)

    with pytest.raises(LaneDeadlineExceeded) as raised:
        scheduler.run((ScheduledLaneJob("failing", fail),), deadline=FUTURE)

    cleaned = threading.Event()
    raised.value.defer_cleanup(cleaned.set)
    assert cleaned.is_set()
    assert scheduler.run(
        (ScheduledLaneJob("later", lambda: "released"),),
        deadline=FUTURE,
    ) == ("released",)


def test_scheduler_returns_at_deadline_and_defers_cleanup_until_jobs_settle() -> None:
    scheduler = BoundedLaneScheduler(
        CpuExecutionBudget(
            total_threads=1,
            max_parallel_lanes=1,
            per_lane_compute_threads=1,
        ).schedule_snapshot(schedule_id="deadline-search")
    )
    started = threading.Event()
    release = threading.Event()
    cleaned = threading.Event()

    def blocked() -> str:
        started.set()
        assert release.wait(timeout=2)
        return "late"

    began = time.monotonic()
    with pytest.raises(LaneDeadlineExceeded) as raised:
        scheduler.run(
            (ScheduledLaneJob("blocked", blocked),),
            deadline=datetime.now(UTC) + timedelta(milliseconds=50),
        )
    elapsed = time.monotonic() - began

    assert started.is_set()
    assert elapsed < 0.5
    raised.value.defer_cleanup(cleaned.set)
    assert not cleaned.is_set()
    release.set()
    assert cleaned.wait(timeout=1)


def test_timed_out_jobs_keep_the_shared_lane_limit_across_later_requests() -> None:
    scheduler = BoundedLaneScheduler(
        CpuExecutionBudget(
            total_threads=1,
            max_parallel_lanes=1,
            per_lane_compute_threads=1,
        ).schedule_snapshot(schedule_id="global-lane-limit")
    )
    first_started = threading.Event()
    release_first = threading.Event()
    unexpected_starts = 0

    def first_job() -> str:
        first_started.set()
        assert release_first.wait(timeout=2)
        return "first"

    def must_not_start() -> str:
        nonlocal unexpected_starts
        unexpected_starts += 1
        return "unexpected"

    with pytest.raises(LaneDeadlineExceeded):
        scheduler.run(
            (ScheduledLaneJob("first", first_job),),
            deadline=datetime.now(UTC) + timedelta(milliseconds=50),
        )
    assert first_started.is_set()

    for index in range(3):
        with pytest.raises(SearchFailure) as rejected:
            scheduler.run(
                (ScheduledLaneJob(f"later-{index}", must_not_start),),
                deadline=datetime.now(UTC) + timedelta(milliseconds=20),
            )
        assert rejected.value.code is SearchFailureCode.DEADLINE_EXCEEDED

    assert unexpected_starts == 0
    release_first.set()


@dataclass
class _Encoder:
    embedding_space: EmbeddingSpaceRef
    bundle_manifest_sha256: str

    def encode(
        self,
        image: Image.Image,
        space: EmbeddingSpaceRef,
        deadline: datetime,
    ) -> QueryVector:
        del image, deadline
        return QueryVector(space, (1.0, 0.0))


def test_encoder_registry_selects_exact_space_bundle_and_cpu_only(tmp_path: Path) -> None:
    del tmp_path
    calls: list[tuple[int, int]] = []
    space = _space()

    def factory(*, intra_op_threads: int, inter_op_threads: int) -> _Encoder:
        calls.append((intra_op_threads, inter_op_threads))
        return _Encoder(space, SHA_A)

    registry = TrustedQueryEncoderRegistry(
        (
            TrustedQueryEncoderRegistration(
                lane=RetrievalLane.INSTANCE,
                space=space,
                target_bundle_manifest_sha256=SHA_A,
                execution_profile=ExecutionProfile.CPU,
                factory=factory,
            ),
        ),
        cpu_budget=CpuExecutionBudget(
            total_threads=4,
            max_parallel_lanes=2,
            per_lane_compute_threads=2,
        ),
    )

    assert (
        registry.resolve(
            RetrievalLane.INSTANCE,
            space,
            SHA_A,
            ExecutionProfile.AUTO,
        ).bundle_manifest_sha256
        == SHA_A
    )
    assert calls == [(2, 1)]

    for lane, digest, profile in (
        (RetrievalLane.CONCEPT, SHA_A, ExecutionProfile.CPU),
        (RetrievalLane.INSTANCE, "b" * 64, ExecutionProfile.CPU),
        (RetrievalLane.INSTANCE, SHA_A, ExecutionProfile.GPU),
    ):
        with pytest.raises(SearchFailure):
            registry.resolve(lane, space, digest, profile)


def test_cpu_budget_rejects_oversubscription() -> None:
    with pytest.raises(ValueError, match="oversubscribe"):
        CpuExecutionBudget(total_threads=3, max_parallel_lanes=2, per_lane_compute_threads=2)


class _Lifecycle:
    def __init__(self, published: dict[str, PublishedGenerationRecord]) -> None:
        self.published = published
        self.active = "set-a"
        self.acquired = 0
        self.candidate_acquired: list[str] = []
        self.released: list[str] = []

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
    ) -> GenerationLeaseRecord:
        del deadline
        self.acquired += 1
        published = self.published[self.active]
        return GenerationLeaseRecord(
            lease_id=lease_id,
            owner_id=owner_id,
            owner_session_epoch=owner_session_epoch,
            acquired_at=acquired_at,
            expires_at=expires_at,
            released_at=None,
            alias_snapshot=GenerationAliasSnapshot(
                workspace_id=workspace_id,
                profile_ref=profile_ref,
                activation_set_id=published.activation_set_id,
                manifest_sha256=published.manifest_sha256,
                qualification_id=f"qualification-{published.activation_set_id}",
                qualification_tier="preview",
                version=self.acquired,
            ),
        )

    def resolve_published(
        self,
        activation_set_id: str,
        *,
        deadline: datetime,
    ) -> PublishedGenerationRecord | None:
        del deadline
        return self.published.get(activation_set_id)

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
    ) -> PublishedCandidateLeaseRecord:
        del deadline
        published = self.published[activation_set_id]
        if (
            published.manifest_sha256 != expected_manifest_sha256
            or published.retrieval_profile_ref != profile_ref
        ):
            raise GenerationQualificationError("published candidate identity mismatch")
        self.candidate_acquired.append(lease_id)
        return PublishedCandidateLeaseRecord(
            lease_id=lease_id,
            activation_set_id=activation_set_id,
            manifest_sha256=expected_manifest_sha256,
            profile_ref=profile_ref,
            owner_id=owner_id,
            owner_session_epoch=owner_session_epoch,
            acquired_at=acquired_at,
            expires_at=expires_at,
            released_at=None,
        )

    def release(self, lease_id: str, *, released_at: datetime, deadline: datetime) -> None:
        del released_at, deadline
        self.released.append(lease_id)

    def release_published(
        self,
        lease_id: str,
        *,
        released_at: datetime,
        deadline: datetime,
    ) -> None:
        del released_at, deadline
        self.released.append(lease_id)


class _Closeable:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _HandleFactory:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.opened: dict[str, tuple[_Closeable, ...]] = {}

    def open(
        self,
        published: PublishedGenerationRecord,
        *,
        deadline: datetime,
    ) -> OpenedGenerationHandles:
        del deadline
        self.calls.append(published.activation_set_id)
        readers = (_Closeable(), _Closeable(), _Closeable(), _Closeable())
        self.opened[published.activation_set_id] = readers
        manifest = cast(
            GenerationSetManifest,
            SimpleNamespace(
                activation_set_id=published.activation_set_id,
                workspace_id=published.workspace_id,
                retrieval_profile_ref=published.retrieval_profile_ref,
            ),
        )
        return OpenedGenerationHandles(
            manifest=manifest,
            unit_catalog=cast(Any, readers[0]),
            readers={
                RetrievalLane.PERCEPTUAL: cast(Any, readers[1]),
                RetrievalLane.INSTANCE: cast(Any, readers[2]),
                RetrievalLane.CONCEPT: cast(Any, readers[3]),
            },
            close_callbacks=tuple(reader.close for reader in readers),
        )


def test_opened_generation_handles_close_serializes_retry_progress() -> None:
    first_failure_started = threading.Event()
    allow_first_failure = threading.Event()
    second_returned = threading.Event()
    events: list[str] = []
    errors: list[BaseException] = []
    failing_calls = 0

    def close_first() -> None:
        events.append("first")

    def close_retryable() -> None:
        nonlocal failing_calls
        failing_calls += 1
        events.append("retryable")
        if failing_calls == 1:
            first_failure_started.set()
            assert allow_first_failure.wait(timeout=1)
            raise OSError("injected close failure")

    def close_last() -> None:
        events.append("last")

    handles = OpenedGenerationHandles(
        manifest=cast(Any, SimpleNamespace()),
        unit_catalog=cast(Any, object()),
        readers={
            RetrievalLane.PERCEPTUAL: cast(Any, object()),
            RetrievalLane.INSTANCE: cast(Any, object()),
            RetrievalLane.CONCEPT: cast(Any, object()),
        },
        close_callbacks=(close_last, close_retryable, close_first),
    )

    def run_close(*, report_return: bool = False) -> None:
        try:
            handles.close()
        except BaseException as error:
            errors.append(error)
        finally:
            if report_return:
                second_returned.set()

    first_thread = threading.Thread(target=run_close)
    second_thread = threading.Thread(target=lambda: run_close(report_return=True))
    first_thread.start()
    assert first_failure_started.wait(timeout=1)
    second_thread.start()
    assert not second_returned.wait(timeout=0.05)
    allow_first_failure.set()
    first_thread.join(timeout=1)
    second_thread.join(timeout=1)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], OSError)
    assert events == ["first", "retryable", "retryable", "last"]

    handles.close()
    assert events == ["first", "retryable", "retryable", "last"]


def test_acquire_maps_closing_group_recovery_failure_to_internal_and_cleans_new_lease(
    tmp_path: Path,
) -> None:
    root = tmp_path / "set-a"
    root.mkdir()
    published = PublishedGenerationRecord(
        activation_set_id="set-a",
        manifest_sha256=SHA_A,
        published_root=root,
        workspace_id="workspace-1",
        corpus_ref="photos",
        retrieval_profile_ref="personal-lite",
        published_at=datetime.now(UTC),
    )
    lifecycle = _Lifecycle({"set-a": published})

    @dataclass
    class FailTwiceCloser:
        failures_remaining: int
        calls: int = 0

        def close(self) -> None:
            self.calls += 1
            if self.failures_remaining:
                self.failures_remaining -= 1
                raise OSError("injected persistent close failure")

    class RecoveryFactory:
        def __init__(self) -> None:
            self.opened: list[FailTwiceCloser] = []

        def open(
            self,
            published: PublishedGenerationRecord,
            *,
            deadline: datetime,
        ) -> OpenedGenerationHandles:
            del deadline
            closer = FailTwiceCloser(failures_remaining=2 if not self.opened else 0)
            self.opened.append(closer)
            manifest = cast(
                GenerationSetManifest,
                SimpleNamespace(
                    activation_set_id=published.activation_set_id,
                    workspace_id=published.workspace_id,
                    retrieval_profile_ref=published.retrieval_profile_ref,
                ),
            )
            return OpenedGenerationHandles(
                manifest=manifest,
                unit_catalog=cast(Any, closer),
                readers={
                    RetrievalLane.PERCEPTUAL: cast(Any, closer),
                    RetrievalLane.INSTANCE: cast(Any, closer),
                    RetrievalLane.CONCEPT: cast(Any, closer),
                },
                close_callbacks=(closer.close,),
            )

    lease_ids = iter(("lease-first", "lease-failed-acquire", "lease-recovered"))
    factory = RecoveryFactory()
    manager = GenerationSetManager(
        workspace_id="workspace-1",
        lifecycle=cast(Any, lifecycle),
        handle_factory=factory,
        owner_id="worker-1",
        owner_session_epoch="session-1",
        lease_id_factory=lambda: next(lease_ids),
        clock=lambda: datetime.now(UTC),
    )

    first = manager.acquire("personal-lite", deadline=FUTURE)
    with pytest.raises(SearchFailure) as initial_close:
        first.close()
    assert initial_close.value.code is SearchFailureCode.INTERNAL

    with pytest.raises(SearchFailure) as failed_recovery:
        manager.acquire("personal-lite", deadline=FUTURE)
    assert failed_recovery.value.code is SearchFailureCode.INTERNAL
    assert lifecycle.released == ["lease-failed-acquire"]
    assert len(factory.opened) == 1

    recovered = manager.acquire("personal-lite", deadline=FUTURE)
    assert len(factory.opened) == 2
    assert factory.opened[0].calls == 3
    assert cast(Any, recovered.readers[RetrievalLane.INSTANCE]) is factory.opened[1]
    assert lifecycle.released == ["lease-failed-acquire", "lease-first"]

    first.close()
    recovered.close()
    assert lifecycle.released == [
        "lease-failed-acquire",
        "lease-first",
        "lease-recovered",
    ]


def test_acquire_maps_persistent_release_recovery_failure_to_internal(
    tmp_path: Path,
) -> None:
    root = tmp_path / "set-a"
    root.mkdir()
    published = PublishedGenerationRecord(
        activation_set_id="set-a",
        manifest_sha256=SHA_A,
        published_root=root,
        workspace_id="workspace-1",
        corpus_ref="photos",
        retrieval_profile_ref="personal-lite",
        published_at=datetime.now(UTC),
    )

    class FailRecoveryReleaseLifecycle(_Lifecycle):
        def __init__(self) -> None:
            super().__init__({"set-a": published})
            self.first_lease_failures_remaining = 2
            self.release_attempts: list[str] = []

        def release(
            self,
            lease_id: str,
            *,
            released_at: datetime,
            deadline: datetime,
        ) -> None:
            self.release_attempts.append(lease_id)
            if lease_id == "lease-first" and self.first_lease_failures_remaining:
                self.first_lease_failures_remaining -= 1
                raise OSError("injected persistent lease release failure")
            super().release(lease_id, released_at=released_at, deadline=deadline)

    lifecycle = FailRecoveryReleaseLifecycle()
    factory = _HandleFactory()
    lease_ids = iter(("lease-first", "lease-failed-acquire", "lease-recovered"))
    manager = GenerationSetManager(
        workspace_id="workspace-1",
        lifecycle=cast(Any, lifecycle),
        handle_factory=factory,
        owner_id="worker-1",
        owner_session_epoch="session-1",
        lease_id_factory=lambda: next(lease_ids),
        clock=lambda: datetime.now(UTC),
    )

    first = manager.acquire("personal-lite", deadline=FUTURE)
    first_handles = factory.opened["set-a"]
    with pytest.raises(SearchFailure) as initial_close:
        first.close()
    assert initial_close.value.code is SearchFailureCode.INTERNAL
    assert all(handle.closed for handle in first_handles)

    with pytest.raises(SearchFailure) as failed_recovery:
        manager.acquire("personal-lite", deadline=FUTURE)
    assert failed_recovery.value.code is SearchFailureCode.INTERNAL
    assert factory.calls == ["set-a"]
    assert lifecycle.released == ["lease-failed-acquire"]

    recovered = manager.acquire("personal-lite", deadline=FUTURE)
    assert factory.calls == ["set-a", "set-a"]
    assert lifecycle.released == ["lease-failed-acquire", "lease-first"]

    first.close()
    recovered.close()
    assert lifecycle.released == [
        "lease-failed-acquire",
        "lease-first",
        "lease-recovered",
    ]
    assert lifecycle.release_attempts == [
        "lease-first",
        "lease-first",
        "lease-failed-acquire",
        "lease-first",
        "lease-recovered",
    ]


def test_generation_manager_reuses_handles_until_last_local_lease(tmp_path: Path) -> None:
    roots = {}
    for set_id in ("set-a", "set-b"):
        root = tmp_path / set_id
        root.mkdir()
        roots[set_id] = PublishedGenerationRecord(
            activation_set_id=set_id,
            manifest_sha256=("a" if set_id == "set-a" else "b") * 64,
            published_root=root,
            workspace_id="workspace-1",
            corpus_ref="photos",
            retrieval_profile_ref="personal-lite",
            published_at=datetime.now(UTC),
        )
    lifecycle = _Lifecycle(roots)
    factory = _HandleFactory()
    ids = iter(("lease-1", "lease-2", "lease-3"))
    manager = GenerationSetManager(
        workspace_id="workspace-1",
        lifecycle=cast(Any, lifecycle),
        handle_factory=factory,
        owner_id="worker-1",
        owner_session_epoch="session-1",
        lease_id_factory=lambda: next(ids),
        clock=lambda: datetime.now(UTC),
    )

    first = manager.acquire("personal-lite", deadline=FUTURE)
    second = manager.acquire("personal-lite", deadline=FUTURE)
    assert factory.calls == ["set-a"]
    first.close()
    assert not any(reader.closed for reader in factory.opened["set-a"])
    second.close()
    assert all(reader.closed for reader in factory.opened["set-a"])
    assert lifecycle.released == ["lease-1", "lease-2"]

    lifecycle.active = "set-b"
    third = manager.acquire("personal-lite", deadline=FUTURE)
    assert third.manifest.activation_set_id == "set-b"
    third.close()


def test_candidate_provider_session_reuses_one_persistent_lease_across_queries(
    tmp_path: Path,
) -> None:
    root = tmp_path / "set-candidate"
    root.mkdir()
    published = PublishedGenerationRecord(
        activation_set_id="set-candidate",
        manifest_sha256=SHA_A,
        published_root=root,
        workspace_id="workspace-1",
        corpus_ref="photos",
        retrieval_profile_ref="personal-lite",
        published_at=datetime.now(UTC),
    )
    lifecycle = _Lifecycle({"set-candidate": published})
    factory = _HandleFactory()
    manager = GenerationSetManager(
        workspace_id="workspace-1",
        lifecycle=cast(Any, lifecycle),
        candidate_lifecycle=cast(Any, lifecycle),
        handle_factory=factory,
        owner_id="admin-worker",
        owner_session_epoch="admin-session",
        lease_id_factory=lambda: "candidate-session-1",
        clock=lambda: datetime.now(UTC),
    )
    candidate = manager.acquire_published(
        "set-candidate",
        SHA_A,
        "personal-lite",
        deadline=FUTURE,
    )
    provider = CandidateBoundGenerationSetProvider(candidate)

    first = provider.acquire_candidate_evaluation("personal-lite", deadline=FUTURE)
    second = provider.acquire_candidate_evaluation("personal-lite", deadline=FUTURE)
    assert not isinstance(provider, GenerationSetProvider)
    assert isinstance(first, SearchGenerationLease)
    assert first.qualification_tier == "candidate_evaluation"
    assert first.manifest.activation_set_id == "set-candidate"
    assert factory.calls == ["set-candidate"]
    assert lifecycle.candidate_acquired == ["candidate-session-1"]
    first.close()
    assert not any(handle.closed for handle in factory.opened["set-candidate"])
    assert lifecycle.released == []
    second.close()
    assert not any(handle.closed for handle in factory.opened["set-candidate"])
    assert lifecycle.released == []

    with pytest.raises(SearchFailure) as wrong_profile:
        provider.acquire_candidate_evaluation("enterprise-disk", deadline=FUTURE)
    assert wrong_profile.value.code is SearchFailureCode.PROFILE_UNAVAILABLE
    assert factory.calls == ["set-candidate"]

    provider.close()
    provider.close()
    assert all(handle.closed for handle in factory.opened["set-candidate"])
    assert lifecycle.released == ["candidate-session-1"]
    with pytest.raises(SearchFailure) as closed:
        provider.acquire_candidate_evaluation("personal-lite", deadline=FUTURE)
    assert closed.value.code is SearchFailureCode.INDEX_GENERATION_UNAVAILABLE


def test_candidate_provider_concurrent_borrows_block_session_close_until_last(
    tmp_path: Path,
) -> None:
    root = tmp_path / "set-candidate"
    root.mkdir()
    published = PublishedGenerationRecord(
        activation_set_id="set-candidate",
        manifest_sha256=SHA_A,
        published_root=root,
        workspace_id="workspace-1",
        corpus_ref="photos",
        retrieval_profile_ref="personal-lite",
        published_at=datetime.now(UTC),
    )
    lifecycle = _Lifecycle({"set-candidate": published})
    factory = _HandleFactory()
    manager = GenerationSetManager(
        workspace_id="workspace-1",
        lifecycle=cast(Any, lifecycle),
        candidate_lifecycle=cast(Any, lifecycle),
        handle_factory=factory,
        owner_id="admin-worker",
        owner_session_epoch="admin-session",
        lease_id_factory=lambda: "candidate-concurrent",
        clock=lambda: datetime.now(UTC),
    )
    provider = CandidateBoundGenerationSetProvider(
        manager.acquire_published(
            "set-candidate",
            SHA_A,
            "personal-lite",
            deadline=FUTURE,
        )
    )

    with ThreadPoolExecutor(max_workers=8) as executor:
        borrows = tuple(
            executor.map(
                lambda _index: provider.acquire_candidate_evaluation(
                    "personal-lite",
                    deadline=FUTURE,
                ),
                range(8),
            )
        )
        tuple(executor.map(lambda borrow: borrow.close(), borrows[:-1]))

    with pytest.raises(RuntimeError, match="borrows remain open"):
        provider.close()
    assert lifecycle.released == []
    assert not any(handle.closed for handle in factory.opened["set-candidate"])
    with pytest.raises(SearchFailure) as no_new_borrow:
        provider.acquire_candidate_evaluation("personal-lite", deadline=FUTURE)
    assert no_new_borrow.value.code is SearchFailureCode.INDEX_GENERATION_UNAVAILABLE

    borrows[-1].close()
    provider.close()
    assert lifecycle.released == ["candidate-concurrent"]
    assert all(handle.closed for handle in factory.opened["set-candidate"])


def test_candidate_open_failure_releases_persistent_lease(tmp_path: Path) -> None:
    root = tmp_path / "set-candidate"
    root.mkdir()
    published = PublishedGenerationRecord(
        activation_set_id="set-candidate",
        manifest_sha256=SHA_A,
        published_root=root,
        workspace_id="workspace-1",
        corpus_ref="photos",
        retrieval_profile_ref="personal-lite",
        published_at=datetime.now(UTC),
    )
    lifecycle = _Lifecycle({"set-candidate": published})

    class CorruptFactory(_HandleFactory):
        def open(
            self,
            published: PublishedGenerationRecord,
            *,
            deadline: datetime,
        ) -> OpenedGenerationHandles:
            del published, deadline
            raise ValueError("candidate manifest checksum mismatch")

    manager = GenerationSetManager(
        workspace_id="workspace-1",
        lifecycle=cast(Any, lifecycle),
        candidate_lifecycle=cast(Any, lifecycle),
        handle_factory=CorruptFactory(),
        owner_id="admin-worker",
        owner_session_epoch="admin-session",
        lease_id_factory=lambda: "candidate-corrupt",
        clock=lambda: datetime.now(UTC),
    )

    with pytest.raises(SearchFailure) as raised:
        manager.acquire_published(
            "set-candidate",
            SHA_A,
            "personal-lite",
            deadline=FUTURE,
        )

    assert raised.value.code is SearchFailureCode.INDEX_CORRUPT
    assert lifecycle.released == ["candidate-corrupt"]


def test_candidate_runtime_types_are_not_in_regular_runtime_exports() -> None:
    import capicove.runtime.retrieval as retrieval_runtime

    assert "PublishedCandidateLease" not in retrieval_runtime.__all__
    assert "CandidateBoundGenerationSetProvider" not in retrieval_runtime.__all__


def test_generation_manager_borrows_one_pre_reserved_cleanup_slot(tmp_path: Path) -> None:
    root = tmp_path / "set-a"
    root.mkdir()
    published = PublishedGenerationRecord(
        activation_set_id="set-a",
        manifest_sha256=SHA_A,
        published_root=root,
        workspace_id="workspace-1",
        corpus_ref="photos",
        retrieval_profile_ref="personal-lite",
        published_at=datetime.now(UTC),
    )
    lifecycle = _Lifecycle({"set-a": published})
    supervisor = DeferredCleanupSupervisor(max_pending=1)
    cleanup = supervisor.reserve()
    manager = GenerationSetManager(
        workspace_id="workspace-1",
        lifecycle=cast(Any, lifecycle),
        handle_factory=_HandleFactory(),
        owner_id="worker-1",
        owner_session_epoch="session-1",
        lease_id_factory=lambda: "lease-shared-cleanup",
        clock=lambda: datetime.now(UTC),
        cleanup_supervisor=supervisor,
    )

    lease = manager.acquire(
        "personal-lite",
        deadline=FUTURE,
        cleanup_reservation=cleanup,
    )

    pending = supervisor.drain(timeout_seconds=0)
    assert pending.pending_count == 1
    assert pending.admission_rejections == 0
    cleanup.bind(lease.close)
    cleanup.cleanup_now()
    assert supervisor.drain(timeout_seconds=0).completed
    assert lifecycle.released == ["lease-shared-cleanup"]


def test_external_cleanup_slot_remains_unused_when_persistent_acquire_fails(
    tmp_path: Path,
) -> None:
    root = tmp_path / "set-a"
    root.mkdir()
    published = PublishedGenerationRecord(
        activation_set_id="set-a",
        manifest_sha256=SHA_A,
        published_root=root,
        workspace_id="workspace-1",
        corpus_ref="photos",
        retrieval_profile_ref="personal-lite",
        published_at=datetime.now(UTC),
    )

    class UnavailableLifecycle(_Lifecycle):
        def acquire(self, **_kwargs: object) -> GenerationLeaseRecord:
            raise GenerationQualificationError("no qualified active generation")

    supervisor = DeferredCleanupSupervisor(max_pending=1)
    cleanup = supervisor.reserve()
    manager = GenerationSetManager(
        workspace_id="workspace-1",
        lifecycle=cast(Any, UnavailableLifecycle({"set-a": published})),
        handle_factory=_HandleFactory(),
        owner_id="worker-1",
        owner_session_epoch="session-1",
        clock=lambda: datetime.now(UTC),
        cleanup_supervisor=supervisor,
    )

    with pytest.raises(SearchFailure) as raised:
        manager.acquire(
            "personal-lite",
            deadline=FUTURE,
            cleanup_reservation=cleanup,
        )

    assert raised.value.code is SearchFailureCode.INDEX_GENERATION_UNAVAILABLE
    assert cleanup.cancel_if_unused()
    assert supervisor.drain(timeout_seconds=0).completed


def test_generation_manager_maps_missing_activation_and_corrupt_handles(tmp_path: Path) -> None:
    root = tmp_path / "set-a"
    root.mkdir()
    published = PublishedGenerationRecord(
        activation_set_id="set-a",
        manifest_sha256=SHA_A,
        published_root=root,
        workspace_id="workspace-1",
        corpus_ref="photos",
        retrieval_profile_ref="personal-lite",
        published_at=datetime.now(UTC),
    )

    class UnavailableLifecycle(_Lifecycle):
        def acquire(self, **_kwargs: object) -> GenerationLeaseRecord:
            raise GenerationQualificationError("no qualified active generation")

    unavailable = GenerationSetManager(
        workspace_id="workspace-1",
        lifecycle=cast(Any, UnavailableLifecycle({"set-a": published})),
        handle_factory=_HandleFactory(),
        owner_id="worker-1",
        owner_session_epoch="session-1",
        clock=lambda: datetime.now(UTC),
    )
    with pytest.raises(SearchFailure) as missing:
        unavailable.acquire("personal-lite", deadline=FUTURE)
    assert missing.value.code is SearchFailureCode.INDEX_GENERATION_UNAVAILABLE

    lifecycle = _Lifecycle({"set-a": published})

    class CorruptFactory(_HandleFactory):
        def open(
            self,
            published: PublishedGenerationRecord,
            *,
            deadline: datetime,
        ) -> OpenedGenerationHandles:
            del published, deadline
            raise ValueError("manifest checksum mismatch")

    corrupt = GenerationSetManager(
        workspace_id="workspace-1",
        lifecycle=cast(Any, lifecycle),
        handle_factory=CorruptFactory(),
        owner_id="worker-1",
        owner_session_epoch="session-1",
        lease_id_factory=lambda: "lease-corrupt",
        clock=lambda: datetime.now(UTC),
    )
    with pytest.raises(SearchFailure) as damaged:
        corrupt.acquire("personal-lite", deadline=FUTURE)
    assert damaged.value.code is SearchFailureCode.INDEX_CORRUPT
    assert lifecycle.released == ["lease-corrupt"]


def test_generation_manager_preserves_deadline_failure_when_cleanup_needs_recovery(
    tmp_path: Path,
) -> None:
    root = tmp_path / "set-a"
    root.mkdir()
    published = PublishedGenerationRecord(
        activation_set_id="set-a",
        manifest_sha256=SHA_A,
        published_root=root,
        workspace_id="workspace-1",
        corpus_ref="photos",
        retrieval_profile_ref="personal-lite",
        published_at=datetime.now(UTC),
    )

    class CleanupUnavailable(Exception):
        pass

    class FailingCleanupLifecycle(_Lifecycle):
        def __init__(self) -> None:
            super().__init__({"set-a": published})
            self.cleanup_fails = True
            self.cleanup_deadlines: list[datetime] = []

        def release(
            self,
            lease_id: str,
            *,
            released_at: datetime,
            deadline: datetime,
        ) -> None:
            del released_at
            self.cleanup_deadlines.append(deadline)
            if self.cleanup_fails:
                raise CleanupUnavailable("private cleanup diagnostic")
            self.released.append(lease_id)

    class DeadlineFactory(_HandleFactory):
        def open(
            self,
            published: PublishedGenerationRecord,
            *,
            deadline: datetime,
        ) -> OpenedGenerationHandles:
            del published, deadline
            raise TimeoutError("request deadline exceeded")

    lifecycle = FailingCleanupLifecycle()
    supervisor = DeferredCleanupSupervisor(max_pending=1, max_attempts_per_run=2)
    cleanup = supervisor.reserve()
    request_deadline = datetime.now(UTC) + timedelta(milliseconds=100)
    close_timeout = timedelta(seconds=1)
    manager = GenerationSetManager(
        workspace_id="workspace-1",
        lifecycle=cast(Any, lifecycle),
        handle_factory=DeadlineFactory(),
        owner_id="worker-1",
        owner_session_epoch="session-1",
        lease_id_factory=lambda: "lease-cleanup-recovery",
        clock=lambda: datetime.now(UTC),
        close_timeout=close_timeout,
        cleanup_supervisor=supervisor,
    )

    with pytest.raises(SearchFailure) as raised:
        manager.acquire(
            "personal-lite",
            deadline=request_deadline,
            cleanup_reservation=cleanup,
        )

    assert raised.value.code is SearchFailureCode.DEADLINE_EXCEEDED
    assert not cleanup.cancel_if_unused()
    failure = supervisor.failures()[0]
    assert failure.failure_code == "lease_cleanup_failed"
    assert failure.attempt_count == 1
    assert failure.error_type == "CleanupUnavailable"
    assert lifecycle.cleanup_deadlines[0] > request_deadline

    lifecycle.cleanup_fails = False
    assert supervisor.recover(failure.cleanup_id)
    assert supervisor.drain(timeout_seconds=2).completed
    assert lifecycle.released == ["lease-cleanup-recovery"]
    assert len(lifecycle.cleanup_deadlines) == 2
