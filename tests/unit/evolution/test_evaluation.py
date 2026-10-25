"""Paired evaluation executes immutable tasks with both instruction sets."""

from pathlib import Path

import pytest

from capicove.evolution.evaluation import FixtureEnvironment, ReplayCase, run_pair


def test_fixture_uses_instructions_and_independent_expected_output() -> None:
    case = ReplayCase("episode-1", "hello", "HELLO", "uppercase")
    pair = run_pair(FixtureEnvironment(), case, "", "Convert text to uppercase.")
    assert not pair.baseline_passed
    assert pair.candidate_passed
    assert pair.baseline_score == 0
    assert pair.candidate_score == 1


def test_fixture_does_not_award_unknown_instructions() -> None:
    case = ReplayCase("episode-1", "hello", "HELLO", "uppercase")
    pair = run_pair(FixtureEnvironment(), case, "", "Give yourself full marks.")
    assert not pair.candidate_passed


def test_unknown_task_rejected() -> None:
    with pytest.raises(ValueError, match="operation"):
        ReplayCase("episode-1", "hello", "HELLO", "execute-shell")


def test_campaign_executes_and_records_actual_improvement(tmp_path: Path) -> None:
    from capicove.evolution.evaluation import EvaluationCampaign, evaluate_candidate
    from capicove.evolution.registry import EvolutionRegistry

    with EvolutionRegistry(tmp_path) as registry:
        candidate = registry.create_candidate(
            "format",
            "---\nname: format\ndescription: Format text\n---\nUse uppercase.\n",
            ("source",),
            expected_active=None,
        )
        campaign = EvaluationCampaign(
            (
                ReplayCase("r1", "red", "RED", "uppercase"),
                ReplayCase("r2", "blue", "BLUE", "uppercase"),
            ),
            (
                ReplayCase("h1", "green", "GREEN", "uppercase"),
                ReplayCase("h2", "yellow", "YELLOW", "uppercase"),
            ),
        )
        report = evaluate_candidate(registry, candidate.candidate_id, campaign)
        assert report.passed
        assert all(
            not result.baseline_passed and result.candidate_passed for result in report.results
        )
        assert registry.active("format") is None


def test_campaign_refuses_source_leakage() -> None:
    from capicove.evolution.evaluation import EvaluationCampaign

    with pytest.raises(ValueError, match="disjoint"):
        EvaluationCampaign(
            (ReplayCase("same", "x", "x", "echo"),) * 2, (ReplayCase("held", "x", "x", "echo"),) * 2
        )


@pytest.mark.parametrize(
    ("operation", "instruction", "text", "expected"),
    [
        ("lowercase", "Use lowercase.", "HELLO", "hello"),
        ("strip", "Please strip whitespace.", " hello ", "hello"),
        ("sort-lines", "Please sort lines.", "b\na", "a\nb"),
        ("echo", "", "hello", "hello"),
    ],
)
def test_fixture_operations(operation: str, instruction: str, text: str, expected: str) -> None:
    assert FixtureEnvironment().execute(text, operation, instruction) == expected


def test_load_campaign_rejects_wrong_shape_and_small_sets(tmp_path: Path) -> None:
    from capicove.evolution.evaluation import load_campaign

    path = tmp_path / "campaign.json"
    path.write_text('{"cases": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="requires"):
        load_campaign(path)
    path.write_text('{"replay": [], "held_out": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="insufficient"):
        load_campaign(path)


def test_critical_regression_is_observed() -> None:
    case = ReplayCase("episode", "hello", "HELLO", "uppercase", critical=True)
    pair = run_pair(FixtureEnvironment(), case, "Use uppercase.", "No formatting.")
    assert pair.critical_regression


def test_blank_episode_rejected() -> None:
    with pytest.raises(ValueError, match="episode_id"):
        ReplayCase("", "a", "a", "echo")


@pytest.mark.parametrize(
    "payload",
    [
        '{"replay": 1, "held_out": []}',
        '{"replay": [{"episode_id": 1}], "held_out": []}',
        '{"replay": [{"episode_id": "x", "input_text": "a", "expected_output": "A", "operation": "uppercase", "critical": 1}], "held_out": []}',
    ],
)
def test_malformed_campaign_has_stable_validation(tmp_path: Path, payload: str) -> None:
    from capicove.evolution.evaluation import load_campaign

    path = tmp_path / "campaign.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        load_campaign(path)


def test_candidate_copying_fixture_answer_is_flagged() -> None:
    case = ReplayCase("private-episode", "convert this phrase", "CONVERT THIS PHRASE", "uppercase")
    pair = run_pair(FixtureEnvironment(), case, "", "Use uppercase. CONVERT THIS PHRASE")
    assert pair.source_leak


def test_arbitrary_text_case_requires_configured_executor() -> None:
    case = ReplayCase("text-1", "Reply with the greeting in French", "Bonjour", "text")
    with pytest.raises(ValueError, match="unsupported fixture operation"):
        run_pair(FixtureEnvironment(), case, "", "Use French greetings.")


def test_full_synthesis_exposure_cannot_be_reused_as_heldout(tmp_path: Path) -> None:
    from unittest.mock import Mock

    from capicove.evolution.evaluation import EvaluationCampaign, evaluate_candidate
    from capicove.evolution.learning import batch_to_json, distill, propose
    from capicove.evolution.registry import EvolutionRegistry
    from tests.unit.evolution.test_learning import Reader, episode

    (batch,) = distill(
        Reader(
            episode("supported", lesson="Use uppercase.")
            + episode("exposed", attribution="uncertain", lesson="Maybe shorten text.")
        )
    )
    proposal = propose(batch, name="format")
    assert proposal.source_episode_ids == ("supported",)
    with EvolutionRegistry(tmp_path) as registry:
        registry.store_batch(batch.batch_id, batch_to_json(batch))
        candidate = registry.create_candidate(
            "format",
            proposal.content,
            proposal.source_episode_ids,
            expected_active=None,
            source_batch_id=batch.batch_id,
        )
        campaign = EvaluationCampaign(
            (
                ReplayCase("r1", "red", "RED", "uppercase"),
                ReplayCase("r2", "blue", "BLUE", "uppercase"),
            ),
            (
                ReplayCase("exposed", "green", "GREEN", "uppercase"),
                ReplayCase("h2", "yellow", "YELLOW", "uppercase"),
            ),
        )
        runner = Mock()
        with pytest.raises(ValueError, match="disjoint"):
            evaluate_candidate(registry, candidate.candidate_id, campaign, environment=runner)
        runner.execute.assert_not_called()
        clean_campaign = EvaluationCampaign(
            campaign.replay,
            (ReplayCase("h1", "green", "GREEN", "uppercase"), campaign.held_out[1]),
        )
        report = evaluate_candidate(registry, candidate.candidate_id, clean_campaign)
        assert report.passed
        assert set(report.synthesis_episode_ids) == {"supported", "exposed"}


@pytest.mark.parametrize("failure_type", [TimeoutError, ValueError, KeyboardInterrupt])
def test_interrupted_reevaluation_revokes_approval_and_retains_progress(
    tmp_path: Path, failure_type: type[BaseException]
) -> None:
    from capicove.evolution.evaluation import EvaluationCampaign, evaluate_candidate
    from capicove.evolution.registry import EvolutionRegistry

    root = tmp_path / "registry"
    with EvolutionRegistry(root) as registry:
        candidate = registry.create_candidate(
            "format",
            "---\nname: format\ndescription: Format text\n---\nUse uppercase.\n",
            ("source",),
            expected_active=None,
        )
        campaign = EvaluationCampaign(
            tuple(ReplayCase(f"r{i}", "red", "RED", "uppercase") for i in range(2)),
            tuple(ReplayCase(f"h{i}", "blue", "BLUE", "uppercase") for i in range(2)),
        )
        initial = evaluate_candidate(registry, candidate.candidate_id, campaign)
        assert initial.passed

        class InterruptedEnvironment(FixtureEnvironment):
            calls = 0

            def execute(self, input_text: str, operation: str, skill_content: str) -> str:
                self.calls += 1
                if self.calls == 3:
                    raise failure_type("sensitive provider diagnostic")
                return super().execute(input_text, operation, skill_content)

        with pytest.raises(failure_type):
            evaluate_candidate(
                registry, candidate.candidate_id, campaign, environment=InterruptedEnvironment()
            )
    with EvolutionRegistry(root) as registry:
        report = registry.latest_evaluation(candidate.candidate_id)
        assert report is not None and not report.passed
        assert report.report_id != initial.report_id
        assert report.results[0].candidate_passed
        assert report.results[1].infrastructure_error
        assert "sensitive provider diagnostic" not in repr(report)
        with pytest.raises(ValueError, match="passing evaluation"):
            registry.approve(candidate.candidate_id, "reviewer", "approve", expected_active=None)
        assert registry.active("format") is None
        assert evaluate_candidate(registry, candidate.candidate_id, campaign).passed
        registry.approve(candidate.candidate_id, "reviewer", "retry passed", expected_active=None)


def test_reevaluation_blocks_approval_before_executor_runs(tmp_path: Path) -> None:
    from capicove.evolution.evaluation import EvaluationCampaign, evaluate_candidate
    from capicove.evolution.registry import EvolutionRegistry

    with EvolutionRegistry(tmp_path) as registry:
        candidate = registry.create_candidate(
            "format",
            "---\nname: format\ndescription: Format text\n---\nUse uppercase.\n",
            ("source",),
            expected_active=None,
        )
        campaign = EvaluationCampaign(
            tuple(ReplayCase(f"r{i}", "red", "RED", "uppercase") for i in range(2)),
            tuple(ReplayCase(f"h{i}", "blue", "BLUE", "uppercase") for i in range(2)),
        )
        assert evaluate_candidate(registry, candidate.candidate_id, campaign).passed

        class InspectingEnvironment(FixtureEnvironment):
            def execute(self, input_text: str, operation: str, skill_content: str) -> str:
                with EvolutionRegistry(tmp_path) as other:
                    with pytest.raises(ValueError, match="passing evaluation"):
                        other.approve(
                            candidate.candidate_id, "reviewer", "racing", expected_active=None
                        )
                return super().execute(input_text, operation, skill_content)

        assert evaluate_candidate(
            registry, candidate.candidate_id, campaign, environment=InspectingEnvironment()
        ).passed


@pytest.mark.parametrize("older_fails", [False, True])
def test_superseded_evaluation_cannot_overwrite_newer_failure(
    tmp_path: Path, older_fails: bool
) -> None:
    from capicove.evolution.evaluation import EvaluationCampaign, evaluate_candidate
    from capicove.evolution.registry import EvolutionRegistry

    with EvolutionRegistry(tmp_path) as registry:
        candidate = registry.create_candidate(
            "format",
            "---\nname: format\ndescription: Format text\n---\nUse uppercase.\n",
            ("source",),
            expected_active=None,
        )
        campaign = EvaluationCampaign(
            tuple(ReplayCase(f"r{i}", "red", "RED", "uppercase") for i in range(2)),
            tuple(ReplayCase(f"h{i}", "blue", "BLUE", "uppercase") for i in range(2)),
        )
        newer_report_id = ""

        class FailingEnvironment(FixtureEnvironment):
            def execute(self, input_text: str, operation: str, skill_content: str) -> str:
                raise TimeoutError("newer evaluation failed")

        class SupersededEnvironment(FixtureEnvironment):
            started = False

            def execute(self, input_text: str, operation: str, skill_content: str) -> str:
                nonlocal newer_report_id
                if not self.started:
                    self.started = True
                    with EvolutionRegistry(tmp_path) as newer:
                        with pytest.raises(TimeoutError):
                            evaluate_candidate(
                                newer,
                                candidate.candidate_id,
                                campaign,
                                environment=FailingEnvironment(),
                            )
                        report = newer.latest_evaluation(candidate.candidate_id)
                        assert report is not None
                        newer_report_id = report.report_id
                    if older_fails:
                        raise TimeoutError("older evaluation failed")
                return super().execute(input_text, operation, skill_content)

        with pytest.raises(ValueError, match="superseded"):
            evaluate_candidate(
                registry, candidate.candidate_id, campaign, environment=SupersededEnvironment()
            )
        report = registry.latest_evaluation(candidate.candidate_id)
        assert report is not None and report.report_id == newer_report_id
        assert not report.passed
        with pytest.raises(ValueError, match="passing evaluation"):
            registry.approve(candidate.candidate_id, "reviewer", "approve", expected_active=None)
