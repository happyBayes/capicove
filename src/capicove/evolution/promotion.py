"""Deterministic transition policy for controlled RSI promotion."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import replace

from capicove.contracts import (
    CandidateState,
    ChangeKind,
    EvolutionCandidate,
    PromotionMode,
)


_SUPERVISED_PATH = {
    CandidateState.DRAFT: CandidateState.VALIDATED,
    CandidateState.VALIDATED: CandidateState.EVALUATED,
    CandidateState.EVALUATED: CandidateState.AWAITING_APPROVAL,
    CandidateState.AWAITING_APPROVAL: CandidateState.ACTIVE,
    CandidateState.ACTIVE: CandidateState.SUPERSEDED,
}

_AUTOMATIC_PATH = {
    CandidateState.DRAFT: CandidateState.VALIDATED,
    CandidateState.VALIDATED: CandidateState.EVALUATED,
    CandidateState.EVALUATED: CandidateState.SANDBOX,
    CandidateState.SANDBOX: CandidateState.SHADOW,
    CandidateState.SHADOW: CandidateState.CANARY,
    CandidateState.CANARY: CandidateState.ACTIVE,
    CandidateState.ACTIVE: CandidateState.SUPERSEDED,
}

_PRE_ACTIVE_STATES = frozenset(
    {
        CandidateState.DRAFT,
        CandidateState.VALIDATED,
        CandidateState.EVALUATED,
        CandidateState.AWAITING_APPROVAL,
        CandidateState.SANDBOX,
        CandidateState.SHADOW,
        CandidateState.CANARY,
    }
)

_PROTECTED_CHANGES = frozenset(
    {
        ChangeKind.FILESYSTEM_PERMISSION,
        ChangeKind.NETWORK_PERMISSION,
        ChangeKind.CREDENTIAL_PERMISSION,
        ChangeKind.DATA_PERMISSION,
        ChangeKind.SUBPROCESS_PERMISSION,
        ChangeKind.TOOL_CONTRACT,
        ChangeKind.PERSISTENT_FORMAT,
        ChangeKind.SECURITY_BOUNDARY,
        ChangeKind.DEPENDENCY,
        ChangeKind.PRIVACY_POLICY,
        ChangeKind.AUDIT_POLICY,
    }
)

_SAFETY_TARGETS = frozenset(
    {
        CandidateState.REJECTED,
        CandidateState.QUARANTINED,
        CandidateState.ROLLED_BACK,
    }
)


class InvalidTransitionError(ValueError):
    """Report a state transition outside the selected promotion path."""


class SupervisedReviewRequiredError(PermissionError):
    """Report a protected change that Mode B cannot promote."""


def allowed_transitions(
    state: CandidateState,
    mode: PromotionMode,
) -> frozenset[CandidateState]:
    """Return every valid next state for a mode and current state."""
    path = _SUPERVISED_PATH if mode is PromotionMode.SUPERVISED else _AUTOMATIC_PATH
    targets: set[CandidateState] = set()
    path_target = path.get(state)
    if path_target is not None:
        targets.add(path_target)
    if state in _PRE_ACTIVE_STATES:
        targets.update({CandidateState.REJECTED, CandidateState.QUARANTINED})
    if state in {CandidateState.CANARY, CandidateState.ACTIVE}:
        targets.update({CandidateState.ROLLED_BACK, CandidateState.QUARANTINED})
    return frozenset(targets)


def requires_supervised_review(change_kinds: Collection[ChangeKind]) -> bool:
    """Return whether a candidate changes a protected trust boundary."""
    return not _PROTECTED_CHANGES.isdisjoint(change_kinds)


def transition(
    candidate: EvolutionCandidate,
    target: CandidateState,
    mode: PromotionMode,
) -> EvolutionCandidate:
    """Create a new candidate record at a valid next promotion state."""
    if target not in allowed_transitions(candidate.state, mode):
        raise InvalidTransitionError(
            f"invalid {mode.value} transition: {candidate.state.value} -> {target.value}"
        )
    if (
        mode is PromotionMode.AUTOMATIC_CANARY
        and target not in _SAFETY_TARGETS
        and requires_supervised_review(candidate.change_kinds)
    ):
        raise SupervisedReviewRequiredError(
            f"candidate requires supervised review: {candidate.candidate_id}"
        )
    return replace(candidate, state=target)
