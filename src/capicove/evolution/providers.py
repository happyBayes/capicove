"""Explicit opt-in ports for inference over bounded, sanitized evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from capicove.evolution.learning import Lesson


@dataclass(frozen=True, slots=True)
class SanitizedExample:
    """Visible text only; no raw goal, tool payload, media, or store access."""

    episode_id: str
    task_signature: str
    messages: tuple[str, ...]
    completed_status: str
    explicit_correction: bool


class ExperienceExtractor(Protocol):
    """An explicitly configured extractor, with host-owned network permissions."""

    def extract(self, examples: tuple[SanitizedExample, ...]) -> tuple[Lesson, ...]: ...
