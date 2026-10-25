"""Opt-in lifecycle hooks produce candidates without activation."""

from pathlib import Path

import pytest

from capicove.runtime.evolution_loop import SkillLearningLoop


def test_disabled_loop_never_creates_storage(tmp_path: Path) -> None:
    root = tmp_path / "registry"
    with SkillLearningLoop(root, tmp_path / "traces.db", skill_names={"format": "format"}) as loop:
        assert not loop.on_feedback()
        assert not loop.on_task_completed()
    assert not root.exists()


def test_feedback_hook_produces_only_draft(tmp_path: Path) -> None:
    from capicove.evolution.registry import EvolutionRegistry
    from capicove.runtime.evolution import ExecutionResult
    from capicove.traces import EpisodeFeedback

    root = tmp_path / "registry"
    with SkillLearningLoop(
        root, tmp_path / "traces.db", enabled=True, skill_names={"format": "format"}
    ) as loop:
        run = loop.run(
            names=(),
            goal="sample text",
            episode_id="source",
            executor=lambda goal, skills: ExecutionResult(goal, True),
        )
        assert loop.wait(timeout=2)
        loop.feedback(
            run,
            "correction",
            EpisodeFeedback(
                "correction",
                {
                    "task_signature": "format",
                    "lesson": "Use uppercase.",
                    "applicability": "text formatting",
                    "exclusion": "verbatim quotations",
                    "attribution": "skill_gap",
                    "confidence": 1.0,
                },
                "user",
            ),
        )
        assert loop.wait(timeout=2)
        assert not loop.failed
    with EvolutionRegistry(root) as registry:
        candidates = registry.list_candidates()
        assert len(candidates) == 1
        assert "Use uppercase." in candidates[0].content
        assert registry.active("format") is None


def test_cancelled_review_does_not_create_storage(tmp_path: Path) -> None:
    import threading

    root = tmp_path / "registry"
    cancel = threading.Event()
    cancel.set()
    with SkillLearningLoop(
        root, tmp_path / "missing.db", enabled=True, skill_names={"format": "format"}
    ) as loop:
        loop._learn(cancel)
    assert not root.exists()


def test_missing_trace_store_surfaces_worker_failure(tmp_path: Path) -> None:
    with SkillLearningLoop(
        tmp_path / "registry",
        tmp_path / "missing.db",
        enabled=True,
        skill_names={"format": "format"},
    ) as loop:
        assert loop.on_feedback()
        assert loop.wait(timeout=2)
        assert loop.failed


def test_feedback_during_snapshot_coalesces_one_followup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from capicove.evolution.registry import EvolutionRegistry
    from capicove.runtime import evolution_admin
    from capicove.runtime.evolution import ExecutionResult
    from capicove.traces import EpisodeFeedback

    entered = threading.Event()
    release = threading.Event()
    original = evolution_admin.distill_store
    calls = 0

    def blocked(*args: object, **kwargs: object) -> tuple[str, ...]:
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(2)
            return ()
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(evolution_admin, "distill_store", blocked)
    root = tmp_path / "registry"
    with SkillLearningLoop(
        root, tmp_path / "traces.db", enabled=True, skill_names={"format": "format"}
    ) as loop:
        run = loop.run(
            names=(),
            goal="hello",
            episode_id="source",
            executor=lambda goal, skills: ExecutionResult(goal, True),
        )
        assert entered.wait(2)
        loop.feedback(
            run,
            "correction",
            EpisodeFeedback(
                "correction",
                {
                    "task_signature": "format",
                    "lesson": "Use uppercase.",
                    "applicability": "text formatting",
                    "exclusion": "quotations",
                    "attribution": "skill_gap",
                    "confidence": 1.0,
                },
                "user",
            ),
        )
        release.set()
        assert loop.wait(timeout=2)
    with EvolutionRegistry(root) as registry:
        assert len(registry.list_candidates()) == 1
