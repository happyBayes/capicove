"""Public dependency-light registry boundary for runtime skill loading."""

from __future__ import annotations

from typing import Protocol

from capicove.evolution.models import SkillVersion


class SkillRegistry(Protocol):
    """Resolve approved versions and record actual host loading."""

    def active(self, name: str) -> SkillVersion | None:
        """Resolve an approved active skill version."""
        ...

    def record_use(self, candidate_id: str, episode_id: str) -> None:
        """Record the exact version loaded for an episode."""
        ...
