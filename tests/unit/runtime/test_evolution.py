"""Explicit skill runtime records exactly the content supplied to its host."""

from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from unittest.mock import Mock

from capicove.evolution.models import SkillVersion
from capicove.evolution.registry import EvolutionRegistry
from capicove.runtime.evolution import ExecutionResult, run_with_skills
from capicove.traces import EpisodeStarted, TraceSanitizer
from capicove.traces.ports import TraceStore
from capicove.traces.recorder import TraceRecorder
from tests.unit.traces.test_recorder import FakeStore


def test_no_active_skills_runs_explicit_host_and_records_empty_versions() -> None:
    store = FakeStore()
    recorder = TraceRecorder(
        cast(TraceStore, store), sanitizer=TraceSanitizer(), clock=lambda: datetime.now(UTC)
    )
    registry = Mock(spec=EvolutionRegistry)
    registry.active.return_value = None
    result = run_with_skills(
        registry,
        recorder,
        names=(),
        goal="hello",
        episode_id="run-1",
        executor=lambda goal, skills: ExecutionResult(goal, True),
    )
    assert result.output == "hello"
    started = store.records[0].payload
    assert isinstance(started, EpisodeStarted)
    assert not started.active_asset_versions
    registry.record_use.assert_not_called()


def test_approval_changes_host_behavior_and_feedback_is_append_only(tmp_path: Path) -> None:
    from capicove.evolution.evaluation import (
        EvaluationCampaign,
        FixtureEnvironment,
        ReplayCase,
        evaluate_candidate,
    )
    from capicove.traces import EpisodeFeedback

    store = FakeStore()
    recorder = TraceRecorder(
        cast(TraceStore, store), sanitizer=TraceSanitizer(), clock=lambda: datetime.now(UTC)
    )
    fixture = FixtureEnvironment()

    def host(goal: str, skills: tuple[SkillVersion, ...]) -> ExecutionResult:
        output = fixture.execute(goal, "uppercase", "\n".join(s.content for s in skills))
        return ExecutionResult(output, output == goal.upper())

    with EvolutionRegistry(tmp_path) as registry:
        candidate = registry.create_candidate(
            "format",
            "---\nname: format\ndescription: Format text\n---\nUse uppercase.\n",
            ("source",),
            expected_active=None,
        )
        before = run_with_skills(
            registry, recorder, names=(), goal="hello", episode_id="before", executor=host
        )
        assert before.output == "hello"
        campaign = EvaluationCampaign(
            tuple(ReplayCase(f"r{i}", "red", "RED", "uppercase") for i in range(2)),
            tuple(ReplayCase(f"h{i}", "blue", "BLUE", "uppercase") for i in range(2)),
        )
        assert evaluate_candidate(registry, candidate.candidate_id, campaign).passed
        registry.approve(
            candidate.candidate_id, "operator", "fixtures reviewed", expected_active=None
        )
        after = run_with_skills(
            registry, recorder, names=("format",), goal="hello", episode_id="after", executor=host
        )
        assert after.output == "HELLO"
        assert after.skills[0].candidate_id == candidate.candidate_id
        count = len(store.records)
        after.feedback(
            "correction",
            EpisodeFeedback(
                "correction",
                {
                    "lesson": "Strip whitespace before uppercase conversion.",
                    "task_signature": "format",
                    "attribution": "skill_gap",
                    "confidence": 1.0,
                    "applicability": "text formatting",
                    "exclusion": "verbatim quotations",
                },
                "user",
            ),
        )
        assert len(store.records) == count + 1
        assert isinstance(store.records[-1].payload, EpisodeFeedback)


def test_failed_host_records_failure_and_duplicate_names_are_rejected() -> None:
    import pytest

    from capicove.traces import EpisodeCompleted, EpisodeStatus

    store = FakeStore()
    recorder = TraceRecorder(
        cast(TraceStore, store), sanitizer=TraceSanitizer(), clock=lambda: datetime.now(UTC)
    )
    registry = Mock(spec=EvolutionRegistry)
    registry.active.return_value = None

    def fail(goal: str, skills: tuple[SkillVersion, ...]) -> ExecutionResult:
        raise RuntimeError("host unavailable")

    with pytest.raises(ValueError, match="unique"):
        run_with_skills(
            registry,
            recorder,
            names=("format", "format"),
            goal="x",
            episode_id="duplicate",
            executor=fail,
        )
    with pytest.raises(RuntimeError, match="host unavailable"):
        run_with_skills(registry, recorder, names=(), goal="x", episode_id="failure", executor=fail)
    completed = store.records[-1].payload
    assert isinstance(completed, EpisodeCompleted)
    assert completed.status == EpisodeStatus.FAILED


def test_explicit_unapproved_skill_is_refused_before_execution() -> None:
    import pytest

    registry = Mock(spec=EvolutionRegistry)
    registry.active.return_value = None
    recorder = Mock(spec=TraceRecorder)
    executor = Mock()
    with pytest.raises(ValueError, match="no active"):
        run_with_skills(
            registry, recorder, names=("draft",), goal="x", episode_id="absent", executor=executor
        )
    executor.assert_not_called()
    recorder.start_session.assert_not_called()


def test_recording_failure_does_not_hide_original_host_failure() -> None:
    import pytest

    registry = Mock(spec=EvolutionRegistry)
    recorder = Mock(spec=TraceRecorder)
    recorder.start_session.return_value.complete.side_effect = RuntimeError("trace unavailable")

    def fail(goal: str, skills: tuple[SkillVersion, ...]) -> ExecutionResult:
        raise ValueError("original host failure")

    with pytest.raises(ValueError, match="original host failure"):
        run_with_skills(registry, recorder, names=(), goal="x", episode_id="e", executor=fail)


def test_runtime_rollback_restores_actual_old_instructions(tmp_path: Path) -> None:
    from capicove.evolution.evaluation import (
        EvaluationCampaign,
        FixtureEnvironment,
        ReplayCase,
        evaluate_candidate,
    )

    fixture = FixtureEnvironment()
    store = FakeStore()
    recorder = TraceRecorder(
        cast(TraceStore, store), sanitizer=TraceSanitizer(), clock=lambda: datetime.now(UTC)
    )

    def campaign(operation: str, text: str, expected: str) -> EvaluationCampaign:
        return EvaluationCampaign(
            tuple(ReplayCase(f"r{i}", text, expected, operation) for i in range(2)),
            tuple(ReplayCase(f"h{i}", text, expected, operation) for i in range(2)),
        )

    def host(goal: str, skills: tuple[SkillVersion, ...]) -> ExecutionResult:
        output = fixture.execute(goal, "lowercase", skills[0].content)
        return ExecutionResult(output, output == goal.lower())

    with EvolutionRegistry(tmp_path) as registry:
        first = registry.create_candidate(
            "format",
            "---\nname: format\ndescription: Format text\n---\nUse uppercase.\n",
            ("source",),
            expected_active=None,
        )
        assert evaluate_candidate(
            registry, first.candidate_id, campaign("uppercase", "red", "RED")
        ).passed
        registry.approve(first.candidate_id, "operator", "reviewed", expected_active=None)
        second = registry.create_candidate(
            "format",
            first.content + "Use lowercase.\n",
            ("source2",),
            expected_active=first.candidate_id,
        )
        assert evaluate_candidate(
            registry, second.candidate_id, campaign("lowercase", "RED", "red")
        ).passed
        registry.approve(
            second.candidate_id, "operator", "reviewed", expected_active=first.candidate_id
        )
        run = run_with_skills(
            registry, recorder, names=("format",), goal="RED", episode_id="new", executor=host
        )
        assert run.output == "red"
        registry.rollback(
            "format",
            first.candidate_id,
            "operator",
            "restore prior behavior",
            expected_active=second.candidate_id,
        )
        restored = run_with_skills(
            registry, recorder, names=("format",), goal="RED", episode_id="restored", executor=host
        )
        assert restored.output == "RED"
        assert not restored.succeeded
        assert restored.skills[0].candidate_id == first.candidate_id
