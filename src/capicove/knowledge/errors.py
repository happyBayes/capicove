"""Stable, content-free temporal memory errors."""


class MemoryError(RuntimeError):
    """A stable error code safe to expose at the Tool boundary."""


class OptionalMemoryLaneUnavailable(RuntimeError):
    """An expected optional-provider outage, not a programming error."""
