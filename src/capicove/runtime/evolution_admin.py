"""Local composition for operator-driven learning and trace feedback."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from capicove.evolution.learning import batch_from_json, batch_to_json, distill, propose
from capicove.evolution.models import SkillVersion
from capicove.evolution.providers import ExperienceExtractor
from capicove.evolution.registry import EvolutionRegistry
from capicove.traces import TraceQuery


def distill_store(
    registry: EvolutionRegistry,
    store: Path,
    *,
    task_signature: str | None = None,
    extractor: ExperienceExtractor | None = None,
) -> tuple[str, ...]:
    """Read an existing trace store and persist validated evidence batches."""
    from capicove.traces._adapters.sqlite import SqliteTraceStore

    if not store.is_file():
        raise ValueError("trace_store_not_found")
    if extractor is not None and not task_signature:
        raise ValueError("model_extraction_requires_task_signature")

    def resolve(name: str, version: int) -> bool:
        return any(
            item.name == name
            and item.version == version
            and any(
                event["action"] == "approve" and event["candidate_id"] == item.candidate_id
                for event in registry.history(name)
            )
            for item in registry.list_candidates()
        )

    batches = distill(
        SqliteTraceStore(store.absolute()),
        TraceQuery(),
        task_signature=task_signature,
        resolve_version=resolve,
        extractor=extractor,
    )
    for batch in batches:
        registry.store_batch(batch.batch_id, batch_to_json(batch))
    return tuple(batch.batch_id for batch in batches)


def generate_candidate(
    registry: EvolutionRegistry,
    batch_id: str,
    name: str,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> SkillVersion | str:
    """Materialize a supported proposal without modifying an active version."""
    batch = batch_from_json(registry.get_batch(batch_id))
    baseline = registry.active(name)
    proposal = propose(
        batch,
        name=name,
        baseline_content=baseline.content if baseline else None,
        baseline_version=baseline.version if baseline else None,
    )
    if proposal.action == "no_change":
        return proposal.reason
    if cancelled is not None and cancelled():
        return "cancelled"
    return registry.create_candidate(
        name,
        proposal.content,
        proposal.source_episode_ids,
        expected_active=baseline.candidate_id if baseline else None,
        source_batch_id=proposal.batch_id,
    )


def run_fixture(
    registry: EvolutionRegistry,
    store: Path,
    *,
    names: tuple[str, ...],
    goal: str,
    episode_id: str,
    operation: str,
) -> dict[str, object]:
    """Run an explicit local fixture host using the actual approved instructions."""
    from dataclasses import asdict
    from datetime import UTC, datetime

    from capicove.evolution.evaluation import FixtureEnvironment
    from capicove.runtime.evolution import ExecutionResult, run_with_skills
    from capicove.traces import TraceRecorder, TraceSanitizer
    from capicove.traces._adapters.sqlite import SqliteTraceStore

    recorder = TraceRecorder(
        SqliteTraceStore(store.absolute()),
        sanitizer=TraceSanitizer(),
        clock=lambda: datetime.now(UTC),
    )
    environment = FixtureEnvironment()

    def execute(text: str, skills: tuple[SkillVersion, ...]) -> ExecutionResult:
        output = environment.execute(text, operation, "\n".join(skill.content for skill in skills))
        expected: dict[str, Callable[[], str]] = {
            "uppercase": text.upper,
            "lowercase": text.lower,
            "strip": text.strip,
            "sort-lines": lambda: "\n".join(sorted(text.splitlines())),
            "echo": lambda: text,
        }
        if operation not in expected:
            raise ValueError("unsupported fixture operation")
        return ExecutionResult(output, output == expected[operation]())

    result = run_with_skills(
        registry,
        recorder,
        names=names,
        goal=goal,
        episode_id=episode_id,
        executor=execute,
        environment_id=environment.evaluator_id,
    )
    return {
        "output": result.output,
        "succeeded": result.succeeded,
        "skills": [asdict(skill) for skill in result.skills],
        "episode_id": episode_id,
        "scope": environment.scope,
    }


def append_feedback(
    store: Path,
    *,
    episode_id: str,
    feedback_id: str,
    task_signature: str,
    lesson: str,
    applicability: str,
    exclusion: str,
    attribution: str,
    confidence: float,
) -> None:
    """Append a sanitized human correction to a previously completed Episode."""
    import math
    from datetime import UTC, datetime

    from capicove.traces import (
        EpisodeCompleted,
        EpisodeFeedback,
        TraceRecordDraft,
        TraceSanitizer,
    )
    from capicove.traces._adapters.sqlite import SqliteTraceStore

    if not store.is_file():
        raise ValueError("trace_store_not_found")
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("invalid_feedback_confidence")
    if any(
        not field.strip()
        for field in (episode_id, feedback_id, task_signature, lesson, applicability, exclusion)
    ):
        raise ValueError("feedback_fields_must_not_be_blank")
    traces = SqliteTraceStore(store.absolute())
    records = traces.get_episode(episode_id)
    if not records or not any(isinstance(record.payload, EpisodeCompleted) for record in records):
        raise ValueError("feedback_requires_completed_episode")
    sanitizer = TraceSanitizer(max_text_bytes=32768)
    payload = EpisodeFeedback(
        "correction",
        sanitizer.sanitize_json(
            {
                "task_signature": task_signature,
                "lesson": lesson,
                "applicability": applicability,
                "exclusion": exclusion,
                "attribution": attribution,
                "confidence": confidence,
            },
            field_name="feedback",
        ),
        "user",
    )
    traces.append(
        TraceRecordDraft(
            episode_id=episode_id,
            trace_id=records[0].trace_id,
            occurred_at=datetime.now(UTC),
            payload=payload,
            privacy_class=records[0].privacy_class,
        ),
        idempotency_key=f"feedback:{feedback_id}",
    )


def run_model(
    registry: EvolutionRegistry,
    store: Path,
    *,
    names: tuple[str, ...],
    goal: str,
    episode_id: str,
    operation: str,
    profile: Path,
    allow_network: bool,
    expected_output: str | None,
) -> dict[str, object]:
    """Execute a text-only model task with an independent operator-supplied oracle."""
    from dataclasses import asdict
    from datetime import UTC, datetime

    from capicove.runtime.evolution import ExecutionResult, run_with_skills
    from capicove.runtime.evolution_generation import CompatibleSkillExecutor
    from capicove.traces import TraceRecorder, TraceSanitizer
    from capicove.traces._adapters.sqlite import SqliteTraceStore

    environment = CompatibleSkillExecutor(profile, allow_network=allow_network)
    if expected_output is None:
        raise ValueError("model_run_requires_independent_expected_output")
    recorder = TraceRecorder(
        SqliteTraceStore(store.absolute()),
        sanitizer=TraceSanitizer(),
        clock=lambda: datetime.now(UTC),
    )

    def execute(text: str, skills: tuple[SkillVersion, ...]) -> ExecutionResult:
        output = environment.execute(text, operation, "\n".join(skill.content for skill in skills))
        return ExecutionResult(output, output == expected_output)

    result = run_with_skills(
        registry,
        recorder,
        names=names,
        goal=goal,
        episode_id=episode_id,
        executor=execute,
        environment_id=environment.evaluator_id,
    )
    return {
        "output": result.output,
        "succeeded": result.succeeded,
        "skills": [asdict(skill) for skill in result.skills],
        "episode_id": episode_id,
        "scope": environment.scope,
    }
