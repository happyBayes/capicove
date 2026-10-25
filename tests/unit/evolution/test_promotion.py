import pytest

from capicove.contracts import (
    CandidateState,
    ChangeKind,
    EvolutionAssetKind,
    EvolutionCandidate,
    PromotionMode,
    RiskClass,
)
from capicove.evolution import (
    InvalidTransitionError,
    SupervisedReviewRequiredError,
    allowed_transitions,
    requires_supervised_review,
    transition,
)


def make_candidate(**overrides: object) -> EvolutionCandidate:
    values: dict[str, object] = {
        "candidate_id": "candidate-1",
        "asset_id": "search-grounded-evidence",
        "asset_kind": EvolutionAssetKind.SOP,
        "version": 2,
        "parent_version": 1,
        "content_hash": "b" * 64,
        "source_episode_ids": ("episode-1", "episode-2", "episode-3"),
        "state": CandidateState.EVALUATED,
        "risk_class": RiskClass.LOW,
        "change_kinds": frozenset({ChangeKind.INSTRUCTIONS}),
    }
    values.update(overrides)
    return EvolutionCandidate(**values)  # type: ignore[arg-type]


def test_mode_a_routes_evaluated_candidate_to_approval() -> None:
    updated = transition(
        make_candidate(),
        CandidateState.AWAITING_APPROVAL,
        PromotionMode.SUPERVISED,
    )

    assert updated.state is CandidateState.AWAITING_APPROVAL


def test_mode_b_routes_evaluated_candidate_to_sandbox() -> None:
    updated = transition(
        make_candidate(),
        CandidateState.SANDBOX,
        PromotionMode.AUTOMATIC_CANARY,
    )

    assert updated.state is CandidateState.SANDBOX


def test_mode_b_cannot_bypass_the_sandbox() -> None:
    with pytest.raises(InvalidTransitionError):
        transition(make_candidate(), CandidateState.ACTIVE, PromotionMode.AUTOMATIC_CANARY)


def test_permission_changes_require_supervised_review() -> None:
    assert requires_supervised_review({ChangeKind.NETWORK_PERMISSION}) is True
    assert requires_supervised_review({ChangeKind.INSTRUCTIONS}) is False


def test_mode_b_rejects_protected_changes() -> None:
    candidate = make_candidate(
        change_kinds=frozenset({ChangeKind.INSTRUCTIONS, ChangeKind.NETWORK_PERMISSION})
    )

    with pytest.raises(SupervisedReviewRequiredError):
        transition(candidate, CandidateState.SANDBOX, PromotionMode.AUTOMATIC_CANARY)


def test_active_candidate_can_be_rolled_back() -> None:
    candidate = make_candidate(state=CandidateState.ACTIVE)

    updated = transition(
        candidate,
        CandidateState.ROLLED_BACK,
        PromotionMode.AUTOMATIC_CANARY,
    )

    assert updated.state is CandidateState.ROLLED_BACK


def test_terminal_state_has_no_outgoing_transitions() -> None:
    assert (
        allowed_transitions(
            CandidateState.SUPERSEDED,
            PromotionMode.SUPERVISED,
        )
        == frozenset()
    )


def test_candidate_requires_source_episodes() -> None:
    with pytest.raises(ValueError, match="source_episode_ids"):
        make_candidate(source_episode_ids=())
