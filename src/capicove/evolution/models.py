"""Dependency-light immutable records for supervised skill evolution."""

from __future__ import annotations

import math
from dataclasses import dataclass

from capicove.contracts import CandidateState


@dataclass(frozen=True, slots=True)
class SkillVersion:
    """Bind retained skill bytes to their evidence and version lineage."""

    candidate_id: str
    name: str
    version: int
    parent_id: str | None
    content: str
    content_hash: str
    source_episode_ids: tuple[str, ...]
    state: CandidateState
    source_batch_id: str | None = None

    @property
    def baseline_id(self) -> str | None:
        """Return the baseline used when creating this candidate."""
        return self.parent_id


@dataclass(frozen=True, slots=True)
class PairedResult:
    """Observable baseline and candidate outcomes for one evaluation episode."""

    episode_id: str
    baseline_passed: bool
    candidate_passed: bool
    critical_regression: bool = False
    permission_expansion: bool = False
    source_leak: bool = False
    baseline_score: float = 1.0
    candidate_score: float = 1.0
    baseline_tool_errors: int = 0
    candidate_tool_errors: int = 0
    baseline_policy_rejections: int = 0
    candidate_policy_rejections: int = 0
    infrastructure_error: bool = False


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Retain paired observations; derive the gate rather than accepting a verdict."""

    report_id: str
    candidate_id: str
    content_hash: str
    baseline_id: str | None
    evaluator_id: str
    synthesis_episode_ids: tuple[str, ...]
    replay_episode_ids: tuple[str, ...]
    held_out_episode_ids: tuple[str, ...]
    results: tuple[PairedResult, ...]
    scope: str = "local deterministic evaluation"
    campaign_sha256: str = ""

    @property
    def passed(self) -> bool:
        """Evaluate independent split quality, regression, and integrity gates."""
        synthesis = set(self.synthesis_episode_ids)
        replay = set(self.replay_episode_ids)
        held = set(self.held_out_episode_ids)
        if not self.report_id.strip() or not self.evaluator_id.strip() or not self.scope.strip():
            return False
        if not synthesis or len(replay) < 2 or len(held) < 2:
            return False
        if synthesis & replay or synthesis & held or replay & held:
            return False
        if len(self.results) != len(replay | held):
            return False
        if {r.episode_id for r in self.results} != replay | held:
            return False
        for result in self.results:
            if any(
                (
                    result.critical_regression,
                    result.permission_expansion,
                    result.source_leak,
                    result.infrastructure_error,
                )
            ):
                return False
            if not all(
                math.isfinite(s) and 0 <= s <= 1
                for s in (result.baseline_score, result.candidate_score)
            ):
                return False
            if (
                min(
                    result.baseline_tool_errors,
                    result.candidate_tool_errors,
                    result.baseline_policy_rejections,
                    result.candidate_policy_rejections,
                )
                < 0
            ):
                return False
        if self.baseline_id is None and not any(
            (not row.baseline_passed and row.candidate_passed)
            or row.candidate_score > row.baseline_score
            for row in self.results
        ):
            return False
        for split in (replay, held):
            rows = [r for r in self.results if r.episode_id in split]
            if sum(r.candidate_passed for r in rows) < sum(r.baseline_passed for r in rows):
                return False
            baseline = sum(r.baseline_score for r in rows) / len(rows)
            candidate = sum(r.candidate_score for r in rows) / len(rows)
            if candidate + 1e-9 < (1.0 if baseline == 1.0 else baseline + 0.02):
                return False
            if sum(r.candidate_tool_errors for r in rows) > sum(
                r.baseline_tool_errors for r in rows
            ):
                return False
            if sum(r.candidate_policy_rejections for r in rows) > sum(
                r.baseline_policy_rejections for r in rows
            ):
                return False
            major = sum(r.baseline_score - r.candidate_score >= 0.2 for r in rows)
            if major > 1 or major / len(rows) > 0.05:
                return False
        return True
