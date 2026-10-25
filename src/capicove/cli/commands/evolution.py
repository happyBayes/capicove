"""Local operator interface for supervised, versioned Skill evolution."""

from __future__ import annotations

import argparse
import difflib
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any


def add_parser(parent: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register explicit local Skill administration commands."""
    parser = parent.add_parser("evolution", help="Learn, evaluate and approve versioned Skills.")
    commands = parser.add_subparsers(dest="evolution_command", required=True)
    for name in (
        "init",
        "status",
        "active",
        "distill",
        "generate",
        "evaluate",
        "rollback",
        "run",
        "feedback",
    ):
        command = commands.add_parser(name)
        _root(command)
        if name in {"active", "rollback"}:
            command.add_argument("name")
        if name == "distill":
            command.add_argument("--store", type=Path, required=True)
            command.add_argument("--task-signature")
        if name == "generate":
            command.add_argument("batch_id")
            command.add_argument("--name", required=True)
        if name == "evaluate":
            command.add_argument("candidate_id")
            command.add_argument("--campaign", type=Path, required=True)
        if name == "rollback":
            command.add_argument("--to", dest="candidate_id", required=True)
            _decision(command)
        if name in {"run", "feedback"}:
            command.add_argument("--store", type=Path, required=True)
        if name == "run":
            command.add_argument("goal")
            command.add_argument("--episode-id", required=True)
            command.add_argument("--skill", action="append", default=[])
            command.add_argument(
                "--operation",
                required=True,
            )
        if name in {"run", "evaluate", "distill"}:
            command.add_argument("--generation-profile", type=Path)
            command.add_argument("--allow-network", action="store_true")
        if name == "run":
            command.add_argument(
                "--expected-output", help="Independent expected output for model tasks."
            )
        if name == "feedback":
            command.add_argument("episode_id")
            for field in ("feedback-id", "task-signature", "lesson", "applicability", "exclusion"):
                command.add_argument("--" + field, required=True)
            command.add_argument(
                "--attribution",
                choices=("skill_gap", "agent_misuse", "environment_failure", "uncertain"),
                required=True,
            )
            command.add_argument("--confidence", type=float, default=1.0)
    review = commands.add_parser("review")
    reviews = review.add_subparsers(dest="review_command", required=True)
    for name in ("show", "approve", "reject"):
        command = reviews.add_parser(name)
        _root(command)
        command.add_argument("candidate_id")
        if name in {"approve", "reject"}:
            _decision(command)
        if name == "approve":
            command.add_argument("--expect-state", choices=("awaiting_approval",), required=True)


def _root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, required=True)


def _decision(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--expect-active", required=True, help="Current candidate ID, or 'none'.")


def run(arguments: argparse.Namespace) -> int:
    """Render bounded administration results without enabling model or network use."""
    from capicove.evolution.registry import EvolutionRegistry

    try:
        with EvolutionRegistry(arguments.root.absolute()) as registry:
            result = _dispatch(arguments, registry)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except sqlite3.Error:
        print(json.dumps({"status": "error", "error": "evolution_storage_unavailable"}))
        return 1
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        return 1


def _dispatch(arguments: argparse.Namespace, registry: Any) -> dict[str, Any]:
    command = arguments.evolution_command
    if command == "init":
        return {"status": "initialized"}
    if command == "status":
        return {
            "candidates": [asdict(item) for item in registry.list_candidates()],
            "batches": list(registry.list_batches()),
        }
    if command == "active":
        skill = registry.active(arguments.name)
        if skill is None:
            raise ValueError("skill_not_active")
        return asdict(skill)
    if command == "distill":
        from capicove.runtime.evolution_admin import distill_store

        extractor = None
        if arguments.generation_profile is not None:
            from capicove.runtime.evolution_generation import CompatibleExperienceExtractor

            extractor = CompatibleExperienceExtractor(
                arguments.generation_profile, allow_network=arguments.allow_network
            )
        batches = distill_store(
            registry, arguments.store, task_signature=arguments.task_signature, extractor=extractor
        )
        return {"batches": list(batches)}
    if command == "generate":
        from capicove.runtime.evolution_admin import generate_candidate

        candidate = generate_candidate(registry, arguments.batch_id, arguments.name)
        if isinstance(candidate, str):
            return {"action": "no_change", "reason": candidate}
        return asdict(candidate)
    if command == "evaluate":
        from capicove.evolution.evaluation import evaluate_candidate, load_campaign

        environment = None
        if arguments.generation_profile is not None:
            from capicove.runtime.evolution_generation import CompatibleSkillExecutor

            environment = CompatibleSkillExecutor(
                arguments.generation_profile, allow_network=arguments.allow_network
            )
        report = evaluate_candidate(
            registry,
            arguments.candidate_id,
            load_campaign(arguments.campaign),
            environment=environment,
        )
        return {**asdict(report), "passed": report.passed}
    if command == "run":
        from capicove.runtime.evolution_admin import run_fixture, run_model

        if arguments.generation_profile is not None:
            return run_model(
                registry,
                arguments.store,
                names=tuple(arguments.skill),
                goal=arguments.goal,
                episode_id=arguments.episode_id,
                operation=arguments.operation,
                profile=arguments.generation_profile,
                allow_network=arguments.allow_network,
                expected_output=arguments.expected_output,
            )
        return run_fixture(
            registry,
            arguments.store,
            names=tuple(arguments.skill),
            goal=arguments.goal,
            episode_id=arguments.episode_id,
            operation=arguments.operation,
        )
    if command == "feedback":
        from capicove.runtime.evolution_admin import append_feedback

        append_feedback(
            arguments.store,
            episode_id=arguments.episode_id,
            feedback_id=arguments.feedback_id,
            task_signature=arguments.task_signature,
            lesson=arguments.lesson,
            applicability=arguments.applicability,
            exclusion=arguments.exclusion,
            attribution=arguments.attribution,
            confidence=arguments.confidence,
        )
        return {"status": "recorded", "episode_id": arguments.episode_id}
    expected = (
        None
        if getattr(arguments, "expect_active", None) == "none"
        else getattr(arguments, "expect_active", None)
    )
    if command == "rollback":
        return asdict(
            registry.rollback(
                arguments.name,
                arguments.candidate_id,
                reviewer=arguments.reviewer,
                reason=arguments.reason,
                expected_active=expected,
            )
        )
    if arguments.review_command == "show":
        candidate = registry.get_candidate(arguments.candidate_id)
        baseline = registry.get_candidate(candidate.parent_id) if candidate.parent_id else None
        report = registry.evaluation_report(candidate.candidate_id)
        diff = "".join(
            difflib.unified_diff(
                baseline.content.splitlines(keepends=True) if baseline else [],
                candidate.content.splitlines(keepends=True),
                fromfile="baseline",
                tofile="candidate",
            )
        )
        return {
            "candidate": asdict(candidate),
            "diff": diff,
            "evaluation": {**asdict(report), "passed": report.passed} if report else None,
        }
    if arguments.review_command == "approve":
        candidate = registry.get_candidate(arguments.candidate_id)
        if candidate.state.value != arguments.expect_state:
            raise ValueError("candidate_state_changed")
        return asdict(
            registry.approve(
                arguments.candidate_id,
                reviewer=arguments.reviewer,
                reason=arguments.reason,
                expected_active=expected,
            )
        )
    return asdict(
        registry.reject(
            arguments.candidate_id,
            reviewer=arguments.reviewer,
            reason=arguments.reason,
            expected_active=expected,
        )
    )
