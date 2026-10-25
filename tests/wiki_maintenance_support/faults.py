"""Real process termination barriers and independent effect counters."""

import os
from dataclasses import dataclass
from pathlib import Path


BARRIERS = frozenset(
    {
        "before_journal",
        "after_destination",
        "before_state",
        "after_state",
        "before_marker_removal",
        "after_receipt",
    }
)


def crash_at(root: Path, name: str) -> None:
    """Persist an observation before terminating a child without cleanup."""
    if name not in BARRIERS:
        raise ValueError("unknown fault barrier")
    with (root / name).open("xb") as stream:
        stream.write(b"durable\n")
        stream.flush()
        os.fsync(stream.fileno())
    os._exit(73)


@dataclass
class Effects:
    """Counters injected at provider and filesystem observation boundaries."""

    provider_calls: int = 0
    egress_bytes: int = 0
    durable_writes: int = 0

    def snapshot(self) -> tuple[int, int, int]:
        """Return comparable effect counts."""
        return self.provider_calls, self.egress_bytes, self.durable_writes


@dataclass(frozen=True)
class FaultBarriers:
    """Reusable child-process crash interface for publication tests."""

    root: Path

    def stop_at(self, name: str) -> None:
        """Record a durable barrier and terminate the calling test child."""
        crash_at(self.root, name)
