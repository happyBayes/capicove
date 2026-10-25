"""Opt-in lifecycle orchestration from completed traces to draft skills."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

from capicove.evolution.background import BackgroundReview
from capicove.evolution.models import SkillVersion
from capicove.evolution.registry import EvolutionRegistry
from capicove.runtime.evolution import ExecutionResult, SkillRun, run_with_skills
from capicove.traces import EpisodeFeedback, TraceRecorder, TraceSanitizer


class SkillLearningLoop:
    """Connect explicit host lifecycle hooks to bounded candidate-only review.

    Each worker owns its SQLite connection. Cancellation is cooperative; review
    checks it before persistence, and never receives approval authority. Explicit
    evaluation and operator approval remain separate administrative operations.
    """

    def __init__(
        self,
        registry_root: Path,
        store: Path,
        *,
        enabled: bool = False,
        skill_names: Mapping[str, str],
    ) -> None:
        self._root = registry_root
        self._store = store
        self._enabled = enabled
        self._names = dict(skill_names)
        self._review = BackgroundReview()

    @property
    def failed(self) -> bool:
        """Expose background failure without leaking evidence or exception text."""
        return self._review.failed

    def schedule(self) -> bool:
        """Submit a single review only when learning was explicitly enabled."""
        return self._enabled and self._review.submit(self._learn, coalesce=True)

    def _learn(self, cancelled: threading.Event) -> None:
        from capicove.runtime.evolution_admin import distill_store, generate_candidate

        if cancelled.is_set():
            return
        with EvolutionRegistry(self._root) as registry:
            for signature, name in sorted(self._names.items()):
                if cancelled.is_set():
                    return
                batches = distill_store(registry, self._store, task_signature=signature)
                for batch_id in batches:
                    if cancelled.is_set():
                        return
                    generate_candidate(registry, batch_id, name, cancelled=cancelled.is_set)

    def on_task_completed(self) -> bool:
        """Schedule review after the host has durably completed its task."""
        return self.schedule()

    def on_feedback(self) -> bool:
        """Schedule review after a correction has been durably appended."""
        return self.schedule()

    def before_live_turn(self) -> bool:
        """Prioritize the live task with a bounded cooperative cancellation wait."""
        return self._review.cancel()

    def wait(self, *, timeout: float = 0.1) -> bool:
        """Wait up to five seconds for this optional review to finish."""
        return self._review.wait(timeout=timeout)

    def run(
        self,
        *,
        names: tuple[str, ...],
        goal: str,
        episode_id: str,
        executor: Callable[[str, tuple[SkillVersion, ...]], ExecutionResult],
    ) -> SkillRun:
        """Execute a traced host invocation and trigger optional review afterward."""
        from capicove.traces._adapters.sqlite import SqliteTraceStore

        self.before_live_turn()
        recorder = TraceRecorder(
            SqliteTraceStore(self._store.absolute()),
            sanitizer=TraceSanitizer(),
            clock=lambda: datetime.now(UTC),
        )
        with EvolutionRegistry(self._root) as registry:
            result = run_with_skills(
                registry,
                recorder,
                names=names,
                goal=goal,
                episode_id=episode_id,
                executor=executor,
            )
        self.on_task_completed()
        return result

    def feedback(self, run: SkillRun, feedback_id: str, payload: EpisodeFeedback) -> None:
        """Persist feedback before scheduling its use as learning evidence."""
        run.feedback(feedback_id, payload)
        self.on_feedback()

    def close(self) -> None:
        """Cancel optional work and prevent subsequent scheduling."""
        self._review.close()

    def __enter__(self) -> SkillLearningLoop:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
