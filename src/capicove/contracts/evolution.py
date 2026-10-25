"""Canonical values for versioned recursive self-improvement assets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from capicove.contracts._validation import (
    require_non_blank,
    require_positive_int,
    require_sha256,
)


class EvolutionAssetKind(StrEnum):
    """Classify reusable assets synthesized from Agent traces."""

    SOP = "sop"
    WORKFLOW = "workflow"
    SKILL = "skill"
    PITFALL = "pitfall"


class RiskClass(StrEnum):
    """Classify the review risk of an evolution candidate."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PRIVILEGED = "privileged"


class PromotionMode(StrEnum):
    """Select supervised or explicitly enabled automatic promotion."""

    SUPERVISED = "supervised"
    AUTOMATIC_CANARY = "automatic_canary"


class CandidateState(StrEnum):
    """Describe a candidate's immutable promotion-state record."""

    DRAFT = "draft"
    VALIDATED = "validated"
    EVALUATED = "evaluated"
    AWAITING_APPROVAL = "awaiting_approval"
    SANDBOX = "sandbox"
    SHADOW = "shadow"
    CANARY = "canary"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"
    ROLLED_BACK = "rolled_back"


class ChangeKind(StrEnum):
    """Classify the behavior or trust boundary changed by a candidate."""

    INSTRUCTIONS = "instructions"
    BRANCHING = "branching"
    TOOL_SELECTION = "tool_selection"
    PARAMETERS = "parameters"
    CONTEXT_TEMPLATE = "context_template"
    PITFALL = "pitfall"
    RECOVERY = "recovery"
    FILESYSTEM_PERMISSION = "filesystem_permission"
    NETWORK_PERMISSION = "network_permission"
    CREDENTIAL_PERMISSION = "credential_permission"
    DATA_PERMISSION = "data_permission"
    SUBPROCESS_PERMISSION = "subprocess_permission"
    TOOL_CONTRACT = "tool_contract"
    PERSISTENT_FORMAT = "persistent_format"
    SECURITY_BOUNDARY = "security_boundary"
    DEPENDENCY = "dependency"
    PRIVACY_POLICY = "privacy_policy"
    AUDIT_POLICY = "audit_policy"


@dataclass(frozen=True, slots=True)
class EvolutionCandidate:
    """Describe one trace-derived, immutable evolution-asset version."""

    candidate_id: str
    asset_id: str
    asset_kind: EvolutionAssetKind
    version: int
    parent_version: int | None
    content_hash: str
    source_episode_ids: tuple[str, ...]
    state: CandidateState
    risk_class: RiskClass
    change_kinds: frozenset[ChangeKind]

    def __post_init__(self) -> None:
        require_non_blank("candidate_id", self.candidate_id)
        require_non_blank("asset_id", self.asset_id)
        require_positive_int("version", self.version)
        require_sha256("content_hash", self.content_hash)
        if self.parent_version is None and self.version != 1:
            raise ValueError("parent_version is required after version 1")
        if self.parent_version is not None:
            require_positive_int("parent_version", self.parent_version)
            if self.parent_version >= self.version:
                raise ValueError("parent_version must be lower than version")
        if not self.source_episode_ids:
            raise ValueError("source_episode_ids must not be empty")
        for episode_id in self.source_episode_ids:
            require_non_blank("source_episode_id", episode_id)
        if not self.change_kinds:
            raise ValueError("change_kinds must not be empty")
