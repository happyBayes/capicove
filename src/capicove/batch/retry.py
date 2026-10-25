"""Explicit bounded retry policy for batch operations."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from capicove.batch.models import BatchFailure


_MAX_RETRIES = 10


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Compute deterministic bounded exponential retry delays."""

    max_retries: int
    initial_backoff_seconds: float = 0.5
    multiplier: float = 2.0
    max_backoff_seconds: float = 30.0

    def __post_init__(self) -> None:
        if isinstance(self.max_retries, bool) or not 0 <= self.max_retries <= _MAX_RETRIES:
            raise ValueError(f"max_retries must be between 0 and {_MAX_RETRIES}")
        if not isfinite(self.initial_backoff_seconds) or self.initial_backoff_seconds < 0:
            raise ValueError("initial_backoff_seconds must be finite and non-negative")
        if not isfinite(self.multiplier) or self.multiplier < 1:
            raise ValueError("multiplier must be finite and at least one")
        if not isfinite(self.max_backoff_seconds) or self.max_backoff_seconds <= 0:
            raise ValueError("max_backoff_seconds must be finite and positive")

    @property
    def max_attempts(self) -> int:
        """Return the initial attempt plus the configured retries."""
        return self.max_retries + 1

    def delay_before_retry(self, failed_attempt: int) -> float:
        """Return the delay after one failed one-based attempt."""
        if not 1 <= failed_attempt <= self.max_retries:
            raise ValueError("failed_attempt must be within the retry budget")
        delay = self.initial_backoff_seconds * self.multiplier ** (failed_attempt - 1)
        return min(delay, self.max_backoff_seconds)


class BatchOperationError(RuntimeError):
    """Report an operation failure with explicit retry semantics."""

    def __init__(self, code: str, message: str, *, retryable: bool) -> None:
        failure = BatchFailure(code=code, message=message, retryable=retryable)
        super().__init__(failure.message)
        self.failure = failure


class BatchItemSkipped(RuntimeError):
    """Signal that an operation deliberately skipped one applicable item."""

    def __init__(self, reason: str) -> None:
        if not reason.strip():
            raise ValueError("reason must not be blank")
        super().__init__(reason)
        self.reason = reason
