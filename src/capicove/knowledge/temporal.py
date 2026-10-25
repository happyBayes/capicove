"""World-time ordering and interval relations, independent of storage."""

from datetime import datetime

from capicove.contracts.knowledge import TimeInterval


def later_start(candidate: datetime | None, original: datetime | None) -> bool:
    """Missing lower bounds denote negative infinity, not unknown ordering."""
    return candidate is not None and (original is None or candidate > original)


def overlaps(a: TimeInterval, b: TimeInterval) -> bool:
    return not (
        (a.end is not None and b.start is not None and a.end <= b.start)
        or (b.end is not None and a.start is not None and b.end <= a.start)
    )
