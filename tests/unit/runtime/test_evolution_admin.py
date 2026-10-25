"""Regression tests for local evolution host composition."""

from pathlib import Path
from typing import Any

import pytest

from capicove.evolution.registry import EvolutionRegistry
from capicove.runtime.evolution_admin import append_feedback, run_fixture
from capicove.traces import EpisodeCompleted, EpisodeStatus, TraceConflictError
from capicove.traces._adapters.sqlite import SqliteTraceStore


def test_fixture_records_unsatisfied_task_as_failure(tmp_path: Path) -> None:
    store = tmp_path / "trace" / "trace.sqlite3"
    with EvolutionRegistry(tmp_path / "evolution") as registry:
        result = run_fixture(
            registry, store, names=(), goal="lower case", episode_id="one", operation="uppercase"
        )
    assert not result["succeeded"]
    records = SqliteTraceStore(store).get_episode("one")
    completed = [r.payload for r in records if isinstance(r.payload, EpisodeCompleted)]
    assert completed[0].status == EpisodeStatus.FAILED


def test_feedback_retry_is_idempotent_and_changed_retry_rejected(tmp_path: Path) -> None:
    store = tmp_path / "trace" / "trace.sqlite3"
    with EvolutionRegistry(tmp_path / "evolution") as registry:
        run_fixture(registry, store, names=(), goal="hello", episode_id="one", operation="echo")
    fields: dict[str, Any] = dict(
        episode_id="one",
        feedback_id="feedback",
        task_signature="text",
        lesson="Use uppercase.",
        applicability="Text formatting",
        exclusion="Other tasks",
        attribution="skill_gap",
        confidence=1.0,
    )
    append_feedback(store, **fields)
    append_feedback(store, **fields)
    records = SqliteTraceStore(store).get_episode("one")
    assert sum(r.record_type.value == "episode.feedback" for r in records) == 1
    with pytest.raises(TraceConflictError, match="different trace content"):
        append_feedback(store, **dict(fields, lesson="Different rule"))


def test_model_run_uses_active_instructions_and_keeps_oracle_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from capicove.runtime import evolution_generation
    from capicove.runtime.evolution_admin import run_model

    observed: list[tuple[str, str, str]] = []

    class Model:
        evaluator_id = "test-explicit-provider"
        scope = "controlled test transport"

        def __init__(self, profile: Path, *, allow_network: bool = False) -> None:
            assert allow_network

        def execute(self, text: str, operation: str, instructions: str) -> str:
            observed.append((text, operation, instructions))
            return "independent answer"

    monkeypatch.setattr(evolution_generation, "CompatibleSkillExecutor", Model)
    with EvolutionRegistry(tmp_path / "registry") as registry:
        result = run_model(
            registry,
            tmp_path / "traces" / "episodes.sqlite3",
            names=(),
            goal="task",
            episode_id="model-one",
            operation="text",
            profile=tmp_path / "profile.json",
            allow_network=True,
            expected_output="independent answer",
        )
    assert result["succeeded"]
    assert observed == [("task", "text", "")]
