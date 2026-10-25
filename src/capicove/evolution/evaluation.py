"""Paired local replay with independently specified behavioral verifiers.

The fixture adapter qualifies only its explicit text operations, never general model
quality. Hosts opt into other environments by passing a ReplayEnvironment instance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from capicove.evolution.learning import batch_from_json
from capicove.evolution.models import EvaluationReport, PairedResult
from capicove.evolution.registry import EvolutionRegistry


@dataclass(frozen=True, slots=True)
class ReplayCase:
    """An immutable task with an expected answer unavailable to the executor."""

    episode_id: str
    input_text: str
    expected_output: str
    operation: str
    critical: bool = False

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str)
            for value in (self.episode_id, self.input_text, self.expected_output, self.operation)
        ):
            raise ValueError("case text fields must be strings")
        if not isinstance(self.critical, bool):
            raise ValueError("critical must be a boolean")
        if any(
            len(value.encode("utf-8")) > 32768
            for value in (self.episode_id, self.input_text, self.expected_output)
        ):
            raise ValueError("case text exceeds bounds")
        if not self.episode_id.strip():
            raise ValueError("episode_id must not be blank")
        if self.operation not in {"uppercase", "lowercase", "strip", "sort-lines", "echo", "text"}:
            raise ValueError("unsupported fixture operation")


@dataclass(frozen=True, slots=True)
class EvaluationCampaign:
    """Explicit replay and held-out fixtures; neither is synthesis input."""

    replay: tuple[ReplayCase, ...]
    held_out: tuple[ReplayCase, ...]

    def __post_init__(self) -> None:
        ids = [case.episode_id for case in self.replay + self.held_out]
        if len(ids) != len(set(ids)):
            raise ValueError("evaluation episode sets must be disjoint")
        if len(self.replay) < 2 or len(self.held_out) < 2:
            raise ValueError("insufficient evidence: need two replay and two held-out cases")


class ReplayEnvironment(Protocol):
    """Host-owned deterministic execution; no expected answer is passed in."""

    evaluator_id: str
    scope: str

    def execute(self, input_text: str, operation: str, skill_content: str) -> str:
        """Execute identical task conditions with supplied immutable instructions."""
        ...


class FixtureEnvironment:
    """Local text fixture adapter with an intentionally narrow instruction grammar."""

    evaluator_id = "capicove.text-fixture.v1"
    scope = "local text fixture qualification only; not general model quality"

    def execute(self, input_text: str, operation: str, skill_content: str) -> str:
        """Apply a task transformation only if its instruction is present."""
        if operation not in {"uppercase", "lowercase", "strip", "sort-lines", "echo"}:
            raise ValueError("unsupported fixture operation; configure a text executor")
        instructions = skill_content.casefold()
        if operation == "uppercase" and "uppercase" in instructions:
            return input_text.upper()
        if operation == "lowercase" and "lowercase" in instructions:
            return input_text.lower()
        if operation == "strip" and "strip whitespace" in instructions:
            return input_text.strip()
        if operation == "sort-lines" and "sort lines" in instructions:
            return "\n".join(sorted(input_text.splitlines()))
        return input_text


def run_pair(
    environment: ReplayEnvironment,
    case: ReplayCase,
    baseline: str,
    candidate: str,
) -> PairedResult:
    """Execute both branches and score outputs against the external fixture oracle."""
    baseline_output = environment.execute(case.input_text, case.operation, baseline)
    candidate_output = environment.execute(case.input_text, case.operation, candidate)
    baseline_passed = baseline_output == case.expected_output
    candidate_passed = candidate_output == case.expected_output
    return PairedResult(
        episode_id=case.episode_id,
        baseline_passed=baseline_passed,
        candidate_passed=candidate_passed,
        critical_regression=case.critical and baseline_passed and not candidate_passed,
        source_leak=any(
            len(value.strip()) >= 8 and value in candidate
            for value in (case.episode_id, case.input_text, case.expected_output)
        ),
        baseline_score=float(baseline_passed),
        candidate_score=float(candidate_passed),
    )


def load_campaign(path: Path) -> EvaluationCampaign:
    """Load explicit authored fixtures without executing commands or using a network."""
    with path.open("rb") as handle:
        raw = handle.read(1_048_577)
    if len(raw) > 1_048_576:
        raise ValueError("campaign exceeds one MiB")
    data = json.loads(raw)
    if not isinstance(data, dict) or set(data) != {"replay", "held_out"}:
        raise ValueError("campaign requires replay and held_out arrays")
    for key in ("replay", "held_out"):
        if not isinstance(data[key], list) or len(data[key]) > 1000:
            raise ValueError("campaign splits must be bounded arrays")
    try:
        return EvaluationCampaign(
            tuple(ReplayCase(**item) for item in data["replay"]),
            tuple(ReplayCase(**item) for item in data["held_out"]),
        )
    except TypeError as error:
        raise ValueError("invalid campaign case fields") from error


def evaluate_candidate(
    registry: EvolutionRegistry,
    candidate_id: str,
    campaign: EvaluationCampaign,
    *,
    environment: ReplayEnvironment | None = None,
) -> EvaluationReport:
    """Verify stored bytes, execute paired cases and persist the resulting report."""
    candidate = registry.get_candidate(candidate_id)
    baseline = registry.active(candidate.name)
    if (baseline.candidate_id if baseline else None) != candidate.parent_id:
        raise ValueError("candidate baseline is stale")
    cases = campaign.replay + campaign.held_out
    synthesis_episode_ids = (
        batch_from_json(registry.get_batch(candidate.source_batch_id)).source_episode_ids
        if candidate.source_batch_id is not None
        else candidate.source_episode_ids
    )
    if set(synthesis_episode_ids).intersection(case.episode_id for case in cases):
        raise ValueError("synthesis and evaluation episodes must be disjoint")
    runner = environment or FixtureEnvironment()
    report = EvaluationReport(
        report_id=str(uuid4()),
        candidate_id=candidate.candidate_id,
        content_hash=candidate.content_hash,
        baseline_id=candidate.parent_id,
        evaluator_id=runner.evaluator_id,
        synthesis_episode_ids=synthesis_episode_ids,
        replay_episode_ids=tuple(case.episode_id for case in campaign.replay),
        held_out_episode_ids=tuple(case.episode_id for case in campaign.held_out),
        results=(),
        scope=runner.scope,
        campaign_sha256=hashlib.sha256(
            json.dumps(asdict(campaign), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    )
    # Persist an incomplete, non-passing report before invoking external code.
    # This revokes previous approval readiness even if the process is terminated.
    registry.record_evaluation(report)
    results: list[PairedResult] = []
    for case in cases:
        try:
            results.append(
                run_pair(runner, case, baseline.content if baseline else "", candidate.content)
            )
        except BaseException:
            # Retain completed observations without persisting provider diagnostics.
            results.append(
                PairedResult(
                    episode_id=case.episode_id,
                    baseline_passed=False,
                    candidate_passed=False,
                    baseline_score=0.0,
                    candidate_score=0.0,
                    infrastructure_error=True,
                )
            )
            registry.record_evaluation(
                replace(report, report_id=str(uuid4()), results=tuple(results)),
                expected_latest_id=report.report_id,
            )
            raise
    completed = replace(report, report_id=str(uuid4()), results=tuple(results))
    registry.record_evaluation(completed, expected_latest_id=report.report_id)
    return completed
