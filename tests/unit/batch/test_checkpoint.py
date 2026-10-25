from __future__ import annotations

import pytest

from capicove.batch.checkpoint import (
    BatchCheckpoint,
    CheckpointTransitionError,
    require_checkpoint_transition,
)
from capicove.batch.models import BatchFailure, BatchState


_PLAN_ID = "plan:" + "a" * 64
_ITEM_ID = "item:" + "b" * 64
_CACHE_KEY = "cache:" + "c" * 64
_SOURCE_SHA256 = "d" * 64


def _checkpoint(
    state: BatchState,
    *,
    attempts: int = 0,
    failure: BatchFailure | None = None,
) -> BatchCheckpoint:
    return BatchCheckpoint(
        plan_id=_PLAN_ID,
        item_id=_ITEM_ID,
        cache_key=_CACHE_KEY,
        source_sha256=_SOURCE_SHA256,
        state=state,
        attempts=attempts,
        failure=failure,
    )


def test_checkpoint_accepts_recovery_pending_with_preserved_attempt_count() -> None:
    checkpoint = _checkpoint(BatchState.PENDING, attempts=2)

    assert checkpoint.attempts == 2
    assert checkpoint.failure is None


def test_checkpoint_requires_failure_details_only_for_failed_terminal_states() -> None:
    failure = BatchFailure("TIMEOUT", "deadline expired", retryable=True)

    assert _checkpoint(BatchState.FAILED, attempts=1, failure=failure).failure == failure
    assert _checkpoint(BatchState.SOURCE_CHANGED, failure=failure).failure == failure
    with pytest.raises(ValueError, match="failure is required"):
        _checkpoint(BatchState.FAILED, attempts=1)
    with pytest.raises(ValueError, match="failure must be absent"):
        _checkpoint(BatchState.SUCCEEDED, attempts=1, failure=failure)


def test_checkpoint_rejects_running_without_an_attempt() -> None:
    with pytest.raises(ValueError, match="running checkpoint"):
        _checkpoint(BatchState.RUNNING)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (BatchState.PENDING, BatchState.RUNNING),
        (BatchState.PENDING, BatchState.SUCCEEDED),
        (BatchState.RUNNING, BatchState.RUNNING),
        (BatchState.RUNNING, BatchState.PENDING),
        (BatchState.RUNNING, BatchState.SUCCEEDED),
        (BatchState.RUNNING, BatchState.FAILED),
        (BatchState.FAILED, BatchState.RUNNING),
        (BatchState.FAILED, BatchState.SUCCEEDED),
        (BatchState.SUCCEEDED, BatchState.PENDING),
    ],
)
def test_checkpoint_allows_execution_and_recovery_transitions(
    current: BatchState,
    target: BatchState,
) -> None:
    require_checkpoint_transition(current, target)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (BatchState.PENDING, BatchState.PENDING),
        (BatchState.SUCCEEDED, BatchState.RUNNING),
        (BatchState.SKIPPED, BatchState.RUNNING),
        (BatchState.SOURCE_CHANGED, BatchState.RUNNING),
    ],
)
def test_checkpoint_rejects_unsafe_or_terminal_transitions(
    current: BatchState,
    target: BatchState,
) -> None:
    with pytest.raises(CheckpointTransitionError):
        require_checkpoint_transition(current, target)
