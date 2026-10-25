"""Database-independent checkpoint contracts and state transitions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from capicove.batch.models import BatchFailure, BatchPlan, BatchState


_PLAN_ID_PATTERN = re.compile(r"^plan:[0-9a-f]{64}$")
_ITEM_ID_PATTERN = re.compile(r"^item:[0-9a-f]{64}$")
_CACHE_KEY_PATTERN = re.compile(r"^cache:[0-9a-f]{64}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_FAILURE_STATES = frozenset({BatchState.FAILED, BatchState.SOURCE_CHANGED})
_ALLOWED_TRANSITIONS: dict[BatchState, frozenset[BatchState]] = {
    BatchState.PENDING: frozenset(
        {
            BatchState.RUNNING,
            BatchState.SUCCEEDED,
            BatchState.SKIPPED,
            BatchState.SOURCE_CHANGED,
        }
    ),
    BatchState.RUNNING: frozenset(
        {
            BatchState.RUNNING,
            BatchState.PENDING,
            BatchState.SUCCEEDED,
            BatchState.FAILED,
            BatchState.SKIPPED,
            BatchState.SOURCE_CHANGED,
        }
    ),
    BatchState.FAILED: frozenset(
        {BatchState.RUNNING, BatchState.SUCCEEDED, BatchState.SOURCE_CHANGED}
    ),
    BatchState.SUCCEEDED: frozenset({BatchState.PENDING}),
    BatchState.SKIPPED: frozenset(),
    BatchState.SOURCE_CHANGED: frozenset(),
}


class CheckpointTransitionError(RuntimeError):
    """Report an invalid or concurrently changed checkpoint transition."""


@dataclass(frozen=True, slots=True)
class BatchCheckpoint:
    """Record the current durable state of one planned batch item."""

    plan_id: str
    item_id: str
    cache_key: str
    source_sha256: str
    state: BatchState
    attempts: int = 0
    failure: BatchFailure | None = None

    def __post_init__(self) -> None:
        _require_pattern("plan_id", self.plan_id, _PLAN_ID_PATTERN)
        _require_pattern("item_id", self.item_id, _ITEM_ID_PATTERN)
        _require_pattern("cache_key", self.cache_key, _CACHE_KEY_PATTERN)
        _require_pattern("source_sha256", self.source_sha256, _SHA256_PATTERN)
        if self.attempts < 0:
            raise ValueError("attempts must not be negative")
        if self.state is BatchState.RUNNING and self.attempts < 1:
            raise ValueError("running checkpoint must record at least one attempt")
        if self.state in _FAILURE_STATES and self.failure is None:
            raise ValueError("failure is required for failed terminal states")
        if self.state not in _FAILURE_STATES and self.failure is not None:
            raise ValueError("failure must be absent outside failed terminal states")


class CheckpointStore(Protocol):
    """Persist checkpoint state without exposing a database contract."""

    def prepare(self, plan: BatchPlan) -> None:
        """Create missing items and recover interrupted running states."""

    def load(self, plan_id: str, item_id: str) -> BatchCheckpoint:
        """Load one prepared checkpoint."""

    def update(
        self,
        plan_id: str,
        item_id: str,
        *,
        expected_states: frozenset[BatchState],
        state: BatchState,
        attempts: int,
        failure: BatchFailure | None = None,
    ) -> BatchCheckpoint:
        """Apply one checked state transition and return the new value."""

    def list(self, plan_id: str) -> tuple[BatchCheckpoint, ...]:
        """Return checkpoints in deterministic item order."""

    def history(self, plan_id: str, item_id: str) -> tuple[BatchCheckpoint, ...]:
        """Return append-only state observations for one item."""


def require_checkpoint_transition(current: BatchState, target: BatchState) -> None:
    """Reject transitions that violate execution or recovery semantics."""
    if target not in _ALLOWED_TRANSITIONS[current]:
        raise CheckpointTransitionError(
            f"checkpoint cannot transition from {current.value} to {target.value}"
        )


def _require_pattern(field_name: str, value: str, pattern: re.Pattern[str]) -> None:
    if pattern.fullmatch(value) is None:
        raise ValueError(f"{field_name} has an invalid canonical identifier")
