"""Cooperative deadline control for image-processing capabilities."""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator
from datetime import UTC, datetime


class ImageDeadlineExceeded(TimeoutError):
    """Raised when the active image invocation has reached its deadline."""


_DEADLINE: contextvars.ContextVar[datetime | None] = contextvars.ContextVar(
    "capicove_image_deadline",
    default=None,
)
_USAGE: contextvars.ContextVar[tuple[int, int, int] | None] = contextvars.ContextVar(
    "capicove_image_invocation_usage",
    default=None,
)
_MAX_INVOCATION_DECODED_PIXELS = 100_000_000
_MAX_INVOCATION_DECODED_BYTES = 300_000_000
_MAX_INVOCATION_MODEL_CALLS = 256


@contextlib.contextmanager
def image_invocation_deadline(deadline: datetime) -> Iterator[None]:
    """Bind a transport deadline to capability work in the current context."""
    deadline_token = _DEADLINE.set(deadline)
    usage_token = _USAGE.set((0, 0, 0))
    try:
        check_image_deadline()
        yield
        check_image_deadline()
    finally:
        _USAGE.reset(usage_token)
        _DEADLINE.reset(deadline_token)


def check_image_deadline() -> None:
    """Fail at a cooperative stage boundary after the active deadline."""
    deadline = _DEADLINE.get()
    if deadline is not None and datetime.now(UTC) >= deadline:
        raise ImageDeadlineExceeded("image invocation deadline exceeded")


def remaining_image_seconds(default: float) -> float:
    """Return the smaller of a configured timeout and active remaining time."""
    deadline = _DEADLINE.get()
    if deadline is None:
        return default
    remaining = (deadline - datetime.now(UTC)).total_seconds()
    if remaining <= 0:
        raise ImageDeadlineExceeded("image invocation deadline exceeded")
    return min(default, remaining)


def current_image_deadline() -> datetime | None:
    """Return the active absolute image deadline for worker propagation."""
    return _DEADLINE.get()


def account_decoded_image(width: int, height: int, channels: int = 3) -> None:
    """Charge one retained decoded image against invocation-level memory limits."""
    usage = _USAGE.get()
    if usage is None:
        return
    pixels, decoded_bytes, model_calls = usage
    image_pixels = width * height
    pixels += image_pixels
    decoded_bytes += image_pixels * channels
    if pixels > _MAX_INVOCATION_DECODED_PIXELS:
        raise ValueError("image invocation exceeds the cumulative decoded-pixel limit")
    if decoded_bytes > _MAX_INVOCATION_DECODED_BYTES:
        raise ValueError("image invocation exceeds the cumulative decoded-memory limit")
    _USAGE.set((pixels, decoded_bytes, model_calls))


def consume_image_model_calls(count: int = 1) -> None:
    """Charge bounded provider inference work to the active invocation."""
    if count <= 0:
        raise ValueError("model call count must be positive")
    usage = _USAGE.get()
    if usage is None:
        return
    pixels, decoded_bytes, model_calls = usage
    model_calls += count
    if model_calls > _MAX_INVOCATION_MODEL_CALLS:
        raise ValueError("image invocation exceeds the model-call limit")
    _USAGE.set((pixels, decoded_bytes, model_calls))
