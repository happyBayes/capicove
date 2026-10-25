"""Explicit host integration for approved skills and observable execution traces."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from capicove.evolution.models import SkillVersion
from capicove.evolution.registry import EvolutionRegistry
from capicove.traces import EpisodeFeedback, EpisodeStatus, MessageRole, TextContent
from capicove.traces.recorder import TraceRecorder, TraceSession


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Visible host output and independently determined terminal task status."""

    output: str
    succeeded: bool


@dataclass(frozen=True, slots=True)
class SkillRun:
    """Completed run retaining its session for append-only human feedback."""

    output: str
    succeeded: bool
    skills: tuple[SkillVersion, ...]
    session: TraceSession

    def feedback(self, feedback_id: str, payload: EpisodeFeedback) -> None:
        """Append a later correction without changing original source evidence."""
        self.session.record_feedback(feedback_id=feedback_id, payload=payload)


def run_with_skills(
    registry: EvolutionRegistry,
    recorder: TraceRecorder,
    *,
    names: tuple[str, ...],
    goal: str,
    episode_id: str,
    executor: Callable[[str, tuple[SkillVersion, ...]], ExecutionResult],
    workspace_id: str = "local",
    actor_id: str = "local-operator",
    environment_id: str = "explicit-skill-host",
    tool_catalog_sha256: str = "0" * 64,
) -> SkillRun:
    """Load verified active bytes, record use, then supply them to the chosen host.

    Loaded immutable versions form the invocation snapshot. Later approvals affect
    subsequent runs only. No candidate fallback, model call or network is implicit.
    """
    if len(names) != len(set(names)):
        raise ValueError("skill names must be unique")
    loaded = [registry.active(name) for name in names]
    if any(skill is None for skill in loaded):
        raise ValueError("requested skill has no active approved version")
    skills = tuple(skill for skill in loaded if skill is not None)
    session = recorder.start_session(
        episode_id=episode_id,
        trace_id=episode_id,
        goal=goal,
        workspace_id=workspace_id,
        actor_id=actor_id,
        environment_id=environment_id,
        environment_version=1,
        tool_catalog_sha256=tool_catalog_sha256,
        active_asset_versions={skill.name: skill.version for skill in skills},
    )
    try:
        for skill in skills:
            registry.record_use(skill.candidate_id, episode_id)
            session.record_message(
                message_id=f"skill-use-{skill.name}",
                role=MessageRole.SYSTEM,
                content=(
                    TextContent(
                        json.dumps(
                            {
                                "skill_name": skill.name,
                                "candidate_id": skill.candidate_id,
                                "version": skill.version,
                                "content_hash": skill.content_hash,
                            },
                            sort_keys=True,
                        )
                    ),
                ),
            )
        session.record_message(
            message_id="input", role=MessageRole.USER, content=(TextContent(goal),)
        )
        result = executor(goal, skills)
        session.record_message(
            message_id="output", role=MessageRole.ASSISTANT, content=(TextContent(result.output),)
        )
        session.complete(
            status=EpisodeStatus.SUCCEEDED if result.succeeded else EpisodeStatus.FAILED,
            outcome="host task result",
        )
    except Exception as error:
        try:
            session.complete(status=EpisodeStatus.FAILED, outcome="host execution failed")
        except Exception:
            error.add_note("The failure could not be recorded; the episode remains incomplete.")
        raise
    return SkillRun(result.output, result.succeeded, skills, session)
