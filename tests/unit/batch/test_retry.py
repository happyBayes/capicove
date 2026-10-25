from __future__ import annotations

import pytest

from capicove.batch.models import BatchFailure
from capicove.batch.retry import BatchItemSkipped, BatchOperationError, RetryPolicy


def test_retry_policy_bounds_exponential_backoff_and_total_attempts() -> None:
    policy = RetryPolicy(
        max_retries=4,
        initial_backoff_seconds=0.25,
        multiplier=2.0,
        max_backoff_seconds=1.0,
    )

    assert policy.max_attempts == 5
    assert [policy.delay_before_retry(attempt) for attempt in range(1, 5)] == [
        0.25,
        0.5,
        1.0,
        1.0,
    ]


@pytest.mark.parametrize(
    "values",
    [
        {"max_retries": -1},
        {"initial_backoff_seconds": -0.1},
        {"multiplier": 0.5},
        {"max_backoff_seconds": 0.0},
    ],
)
def test_retry_policy_rejects_unbounded_or_negative_values(values: dict[str, object]) -> None:
    defaults: dict[str, object] = {
        "max_retries": 2,
        "initial_backoff_seconds": 0.25,
        "multiplier": 2.0,
        "max_backoff_seconds": 10.0,
    }
    defaults.update(values)

    with pytest.raises(ValueError):
        RetryPolicy(**defaults)  # type: ignore[arg-type]


def test_retry_policy_rejects_delay_requests_outside_retry_budget() -> None:
    policy = RetryPolicy(max_retries=1)

    with pytest.raises(ValueError, match="failed_attempt"):
        policy.delay_before_retry(0)
    with pytest.raises(ValueError, match="failed_attempt"):
        policy.delay_before_retry(2)


def test_operation_error_exposes_only_explicit_retryability() -> None:
    error = BatchOperationError("RATE_LIMITED", "provider asked for backoff", retryable=True)

    assert error.failure == BatchFailure(
        code="RATE_LIMITED",
        message="provider asked for backoff",
        retryable=True,
    )
    assert str(error) == "provider asked for backoff"


def test_skip_signal_requires_a_bounded_reason() -> None:
    assert BatchItemSkipped("no speech").reason == "no speech"

    with pytest.raises(ValueError, match="reason"):
        BatchItemSkipped(" ")
