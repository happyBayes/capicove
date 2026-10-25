"""Frozen semantic fixtures and independently authored failure oracles."""

from dataclasses import replace

from tests.wiki_maintenance_support.scenarios import (
    Observation,
    evaluate,
    fixture_digest,
    scenarios,
)


def test_frozen_split_and_required_denominators() -> None:
    cases = scenarios()
    assert len(cases) == 60
    assert {
        language: sum(case.language == language for case in cases)
        for language in ("zh", "en", "mixed")
    } == {"zh": 30, "en": 20, "mixed": 10}
    held_out = [case for case in cases if case.split == "holdout"]
    assert len(held_out) == 40
    assert sum(case.primary in {"new", "no_material"} for case in held_out) >= 20
    assert sum(case.unanswerable for case in held_out) >= 10
    assert all(len(case.sources) == 4 for case in cases)
    # Frozen synthetic fixture SHA-256, not a credential.
    assert (
        fixture_digest()
        == "8de2f1902fed0ccc4cdb8c1d6644a0a53298cdb1078419013463f8cfd0c08d12"  # pragma: allowlist secret
    )


def test_oracles_detect_false_negative_dispute_unsupported_and_impact() -> None:
    cases = tuple(case for case in scenarios() if case.split == "holdout")
    observations = {
        case.scenario_id: Observation(
            case.scenario_id,
            case.primary,
            True,
            True,
            case.expected_affected,
            case.unanswerable,
            1,
            1,
            False,
        )
        for case in cases
    }
    result = evaluate(tuple(observations.values()))
    assert result["semantic_qualification"] == "not_established"
    novel = next(case for case in cases if case.primary == "new")
    observations[novel.scenario_id] = replace(
        observations[novel.scenario_id], primary="no_material"
    )
    dispute = next(case for case in cases if case.primary == "disputed")
    observations[dispute.scenario_id] = replace(
        observations[dispute.scenario_id],
        dispute_marked=False,
        supported_claims=0,
        affected_pages=(),
    )
    result = evaluate(tuple(observations.values()))
    assert result["no_material_false_negatives"] == 1
    assert result["unmarked_disputes"] == 1
    assert result["supported_claims"] < result["total_claims"]
    assert result["missing_affected_pages"] > 0
    assert not result["mechanical_oracle_pass"]


def test_missing_attempts_are_failures_not_dropped() -> None:
    result = evaluate(())
    assert result["held_out_scenarios"] == 40
    assert result["missing_attempts"] == 40
    assert not result["mechanical_oracle_pass"]
