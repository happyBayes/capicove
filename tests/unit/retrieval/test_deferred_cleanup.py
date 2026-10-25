"""Tests for bounded, recoverable deferred generation cleanup."""

from __future__ import annotations

from concurrent.futures import Future
from threading import Event, current_thread
from time import monotonic

import pytest

from capicove.contracts.retrieval import SearchFailureCode
from capicove.retrieval import search as search_module
from capicove.retrieval import search_models as search_models_module
from capicove.retrieval.search_models import (
    DeferredCleanupSupervisor,
    LaneDeadlineExceeded,
    SearchFailure,
)


def _defer(
    supervisor: DeferredCleanupSupervisor,
    callback: object,
    future: Future[object],
) -> str:
    reservation = supervisor.reserve()
    reservation.bind(callback)  # type: ignore[arg-type]
    with pytest.raises(LaneDeadlineExceeded):
        with search_module._retained_generation_lease(object(), reservation):  # type: ignore[arg-type]
            raise LaneDeadlineExceeded((future,))
    return reservation.cleanup_id


def test_failed_deferred_cleanup_is_bounded_observable_and_recoverable() -> None:
    supervisor = DeferredCleanupSupervisor(max_pending=1, max_attempts_per_run=3)
    future: Future[object] = Future()
    attempts = 0
    allow_success = False

    def cleanup() -> None:
        nonlocal attempts
        attempts += 1
        if not allow_success:
            raise OSError("private lease detail must not be exposed")

    began = monotonic()
    cleanup_id = _defer(supervisor, cleanup, future)
    assert monotonic() - began < 0.1
    waiting = supervisor.drain(timeout_seconds=0)
    assert (waiting.pending_count, waiting.waiting_count) == (1, 1)
    assert waiting.failures == ()

    future.set_result(None)
    failed = supervisor.drain(timeout_seconds=1)

    assert not failed.completed
    assert (failed.pending_count, failed.failed_count) == (1, 1)
    assert attempts == 3
    assert failed.failures[0].cleanup_id == cleanup_id
    assert failed.failures[0].failure_code == "lease_cleanup_failed"
    assert failed.failures[0].attempt_count == 3
    assert failed.failures[0].max_attempts_per_run == 3
    assert failed.failures[0].recovery_count == 0
    assert failed.failures[0].error_type == "OSError"
    assert "private lease detail" not in repr(failed.failures[0])

    allow_success = True
    assert supervisor.recover(cleanup_id)
    recovered = supervisor.drain(timeout_seconds=1)

    assert recovered.completed
    assert recovered.pending_count == 0
    assert recovered.failures == ()
    assert attempts == 4


def test_synchronous_cleanup_failure_is_retained_for_recovery() -> None:
    supervisor = DeferredCleanupSupervisor(max_pending=1, max_attempts_per_run=2)
    reservation = supervisor.reserve()
    allow_success = False

    def cleanup() -> None:
        if not allow_success:
            raise RuntimeError("private cleanup detail")

    reservation.bind(cleanup)
    with pytest.raises(RuntimeError, match="private cleanup detail"):
        reservation.cleanup_now()

    failure = supervisor.failures()[0]
    assert failure.cleanup_id == reservation.cleanup_id
    assert failure.attempt_count == 1
    assert failure.error_type == "RuntimeError"
    assert "private cleanup detail" not in repr(failure)
    assert not reservation.cancel_if_unused()

    allow_success = True
    assert supervisor.recover(reservation.cleanup_id)
    assert supervisor.drain(timeout_seconds=1).completed


def test_cleanup_failure_does_not_replace_search_body_failure() -> None:
    supervisor = DeferredCleanupSupervisor(max_pending=1)
    reservation = supervisor.reserve()
    original = SearchFailure(SearchFailureCode.INVALID_QUERY)

    def cleanup() -> None:
        raise RuntimeError("private cleanup detail")

    reservation.bind(cleanup)

    with pytest.raises(SearchFailure) as raised:
        with search_module._retained_generation_lease(object(), reservation):  # type: ignore[arg-type]
            raise original

    assert raised.value is original
    failure = supervisor.failures()[0]
    assert failure.cleanup_id == reservation.cleanup_id
    assert failure.error_type == "RuntimeError"
    assert "private cleanup detail" not in repr(failure)


def test_cleanup_failure_does_not_replace_successful_search_body() -> None:
    supervisor = DeferredCleanupSupervisor(max_pending=1)
    reservation = supervisor.reserve()

    def cleanup() -> None:
        raise RuntimeError("private cleanup detail")

    reservation.bind(cleanup)

    with search_module._retained_generation_lease(object(), reservation):  # type: ignore[arg-type]
        pass

    failure = supervisor.failures()[0]
    assert failure.cleanup_id == reservation.cleanup_id
    assert failure.error_type == "RuntimeError"
    assert "private cleanup detail" not in repr(failure)


def test_cancel_if_unused_is_idempotent_and_preserves_consumed_work() -> None:
    supervisor = DeferredCleanupSupervisor(max_pending=1)
    unused = supervisor.reserve()

    assert unused.cancel_if_unused()
    assert not unused.cancel_if_unused()
    assert supervisor.drain(timeout_seconds=0).completed

    bound = supervisor.reserve()
    bound.bind(lambda: None)
    assert not bound.cancel_if_unused()
    bound.cleanup_now()
    assert not bound.cancel_if_unused()
    assert supervisor.drain(timeout_seconds=0).completed


def test_reservation_lifecycle_rejects_invalid_transitions() -> None:
    supervisor = DeferredCleanupSupervisor(max_pending=1)
    unused = supervisor.reserve()

    assert not supervisor.recover(unused.cleanup_id)
    with pytest.raises(TypeError, match="callable"):
        unused.bind(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="LaneDeadlineExceeded"):
        unused.defer_until(object())  # type: ignore[arg-type]
    unused.cancel()
    assert supervisor.drain(timeout_seconds=0).completed
    with pytest.raises(KeyError, match=unused.cleanup_id):
        unused.cancel()
    with pytest.raises(KeyError, match=unused.cleanup_id):
        supervisor.recover(unused.cleanup_id)

    unbound = supervisor.reserve()
    with pytest.raises(RuntimeError, match="not bound"):
        unbound.cleanup_now()
    unbound.cancel()

    calls = 0

    def cleanup() -> None:
        nonlocal calls
        calls += 1

    bound = supervisor.reserve()
    bound.bind(cleanup)
    with pytest.raises(RuntimeError, match="already bound"):
        bound.bind(cleanup)
    with pytest.raises(RuntimeError, match="cannot be cancelled"):
        bound.cancel()
    bound.cleanup_now()
    assert calls == 1
    assert supervisor.drain(timeout_seconds=0).completed

    future: Future[object] = Future()
    waiting = supervisor.reserve()
    waiting.bind(cleanup)
    waiting.defer_until(LaneDeadlineExceeded((future,)))
    with pytest.raises(RuntimeError, match="got waiting_for_lanes"):
        waiting.cleanup_now()
    future.set_result(None)
    assert supervisor.drain(timeout_seconds=1).completed
    assert calls == 2


@pytest.mark.parametrize(
    ("max_pending", "max_attempts_per_run"),
    ((1_025, 1), (1, 11)),
)
def test_supervisor_rejects_excessive_bounds(
    max_pending: int,
    max_attempts_per_run: int,
) -> None:
    with pytest.raises(ValueError, match="at most"):
        DeferredCleanupSupervisor(
            max_pending=max_pending,
            max_attempts_per_run=max_attempts_per_run,
        )


def test_worker_start_failure_becomes_recoverable_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = DeferredCleanupSupervisor(max_pending=1)
    future: Future[object] = Future()

    class _FailingThread:
        def __init__(self, *, target: object, name: str, daemon: bool) -> None:
            assert callable(target)
            assert name == "capicove-lease-cleanup"
            assert daemon

        def start(self) -> None:
            raise RuntimeError("private worker detail")

    monkeypatch.setattr(search_models_module, "Thread", _FailingThread)
    cleanup_id = _defer(supervisor, lambda: None, future)
    future.set_result(None)

    report = supervisor.drain(timeout_seconds=0)
    assert not report.completed
    assert (report.pending_count, report.failed_count) == (1, 1)
    assert report.failures[0].cleanup_id == cleanup_id
    assert report.failures[0].error_type == "RuntimeError"
    assert "private worker detail" not in repr(report.failures[0])


def test_supervisor_uses_one_worker_and_rejects_over_capacity() -> None:
    supervisor = DeferredCleanupSupervisor(max_pending=2, max_attempts_per_run=1)
    first_future: Future[object] = Future()
    second_future: Future[object] = Future()
    first_started = Event()
    release_first = Event()
    second_started = Event()
    worker_names: list[str] = []

    def first_cleanup() -> None:
        worker_names.append(current_thread().name)
        first_started.set()
        assert release_first.wait(timeout=1)

    def second_cleanup() -> None:
        worker_names.append(current_thread().name)
        second_started.set()

    _defer(supervisor, first_cleanup, first_future)
    _defer(supervisor, second_cleanup, second_future)

    began = monotonic()
    first_future.set_result(None)
    assert monotonic() - began < 0.1
    assert first_started.wait(timeout=1)
    second_future.set_result(None)
    assert not second_started.wait(timeout=0.05)

    with pytest.raises(SearchFailure) as rejected:
        supervisor.reserve()
    assert rejected.value.code is SearchFailureCode.BACKPRESSURE
    bounded = supervisor.drain(timeout_seconds=0.01)
    assert bounded.pending_count == 2
    assert bounded.running_count == 2
    assert bounded.admission_rejections == 1

    release_first.set()
    assert supervisor.drain(timeout_seconds=1).completed
    assert second_started.is_set()
    assert worker_names == ["capicove-lease-cleanup", "capicove-lease-cleanup"]


@pytest.mark.parametrize("timeout_seconds", (-1, float("inf"), True, 61))
def test_drain_rejects_unbounded_waits(timeout_seconds: object) -> None:
    supervisor = DeferredCleanupSupervisor()

    with pytest.raises(ValueError, match="within"):
        supervisor.drain(timeout_seconds=timeout_seconds)  # type: ignore[arg-type]
