"""Side-effect-free selection shared by model plugin registries."""

from collections.abc import Iterable
from importlib import metadata
from typing import Protocol, cast


class ModelEntryPoint(Protocol):
    """Describe the installed entry-point surface without loading its target."""

    name: str
    group: str

    def load(self) -> object:
        """Load the explicitly selected descriptor."""


def select_entry_points(
    group: str, name: str, entries: Iterable[object] | None
) -> tuple[ModelEntryPoint, ...]:
    """Enumerate metadata only; duplicate detection precedes import."""
    candidates = metadata.entry_points().select(group=group) if entries is None else entries
    return tuple(
        cast(ModelEntryPoint, entry)
        for entry in candidates
        if getattr(entry, "group", None) == group and getattr(entry, "name", None) == name
    )
