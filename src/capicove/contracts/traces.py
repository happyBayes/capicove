"""Canonical immutable records for Agentic RL tool-use traces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from capicove.contracts._validation import (
    freeze_mapping,
    require_canonical_tool_id,
    require_non_blank,
    require_non_negative_int,
    require_positive_int,
    require_sha256,
)


class StepStatus(StrEnum):
    """Describe the observable outcome of one Episode step."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    POLICY_REJECTED = "policy_rejected"


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Record one sanitized tool action selected by an agent."""

    request_id: str
    tool_id: str
    tool_version: int
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        require_non_blank("request_id", self.request_id)
        require_canonical_tool_id(self.tool_id)
        require_positive_int("tool_version", self.tool_version)
        object.__setattr__(self, "arguments", freeze_mapping(self.arguments))


@dataclass(frozen=True, slots=True)
class EpisodeStep:
    """Record an observable action and result without hidden model reasoning."""

    step_id: str
    sequence: int
    observation_refs: tuple[str, ...]
    action: str
    tool_call: ToolCall | None
    result_ref: str | None
    status: StepStatus

    def __post_init__(self) -> None:
        require_non_blank("step_id", self.step_id)
        require_non_negative_int("sequence", self.sequence)
        require_non_blank("action", self.action)
        for observation_ref in self.observation_refs:
            require_non_blank("observation_ref", observation_ref)
        if self.result_ref is not None:
            require_non_blank("result_ref", self.result_ref)


@dataclass(frozen=True, slots=True)
class Episode:
    """Describe one reproducible goal-directed agent interaction."""

    episode_id: str
    goal: str
    environment_id: str
    environment_version: int
    tool_catalog_hash: str
    active_asset_versions: Mapping[str, int]
    steps: tuple[EpisodeStep, ...] = ()
    outcome: str | None = None
    reward: float | None = None

    def __post_init__(self) -> None:
        require_non_blank("episode_id", self.episode_id)
        require_non_blank("goal", self.goal)
        require_non_blank("environment_id", self.environment_id)
        require_positive_int("environment_version", self.environment_version)
        require_sha256("tool_catalog_hash", self.tool_catalog_hash)
        for asset_id, version in self.active_asset_versions.items():
            require_non_blank("asset_id", asset_id)
            require_positive_int("asset_version", version)
        if tuple(step.sequence for step in self.steps) != tuple(range(len(self.steps))):
            raise ValueError("Episode step sequence must be contiguous and start at zero")
        if self.outcome is not None:
            require_non_blank("outcome", self.outcome)
        object.__setattr__(
            self,
            "active_asset_versions",
            freeze_mapping(self.active_asset_versions),
        )
