"""Experimental ``capicove eval run|score|compare`` for offline retrieval evaluation.

Runs build their index in a private scratch directory and never touch a network.
Records name the dataset, profile, index and model identities; deterministic
fakes are labelled and real-model validation is always reported as not run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def add_parser(parent: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the evaluation commands without importing any stack runtime."""
    from capicove.evaluation.adapters import STACKS

    parser = parent.add_parser(
        "eval", help="Run, score and compare offline retrieval evaluations (experimental)."
    )
    commands = parser.add_subparsers(dest="eval_command", required=True)
    run = commands.add_parser("run", help="Build an offline index and run a frozen query set.")
    run.add_argument("--stack", required=True, choices=STACKS)
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True, help="New run record (not replaced).")
    run.add_argument("--k", type=int, default=10, help="Rank cutoff, 1 to 100.")
    run.add_argument(
        "--profile", type=Path, help="Pipeline profile JSON for kb/offline-text/unified."
    )
    run.add_argument(
        "--reranker",
        choices=("none", "deterministic-fake", "bge"),
        default="none",
        help=(
            "deterministic-fake exercises reranking mechanics (not a model); bge loads "
            "verified local BGE assets from --reranker-model and fails if they are missing."
        ),
    )
    run.add_argument(
        "--reranker-model", type=Path, help="Local BGE reranker asset directory (bge only)."
    )
    run.add_argument(
        "--fake-dense", action="store_true", help="kb only: add the fake hashed dense lane."
    )
    run.add_argument("--corpus-repo", type=Path, help="code only: local Git repository.")
    run.add_argument("--git", help="code only: absolute Git executable path.")
    run.add_argument("--workdir", type=Path, help="Parent directory for private scratch data.")
    score = commands.add_parser("score", help="Score one run record.")
    score.add_argument("run", type=Path)
    score.add_argument("--output", type=Path, help="New score file; prints when omitted.")
    compare = commands.add_parser("compare", help="Pair two runs over the same frozen dataset.")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--output", type=Path, help="New comparison file; prints when omitted.")


def run(arguments: argparse.Namespace) -> int:
    """Dispatch one evaluation command; failures print a stable message and exit 2."""
    from capicove.evaluation.harness import (
        EvaluationError,
        RunOptions,
        compare_runs,
        load_dataset,
        read_json,
        run_evaluation,
        score_run,
        write_json_exclusive,
    )
    from capicove.retrieval.pipeline_profiles import ProfileError

    try:
        if arguments.eval_command == "run":
            options = RunOptions(
                k=arguments.k,
                profile_path=arguments.profile,
                reranker=arguments.reranker,
                corpus_repo=arguments.corpus_repo,
                git=arguments.git,
                fake_dense=arguments.fake_dense,
                reranker_model=arguments.reranker_model,
            )
            if arguments.output.exists():
                raise EvaluationError("output already exists")
            record = run_evaluation(
                arguments.stack,
                load_dataset(arguments.dataset),
                options,
                workdir=arguments.workdir,
            )
            write_json_exclusive(arguments.output, record)
            _print(_summary(record, score_run(record)))
            return 0
        if arguments.eval_command == "score":
            record = score_run(read_json(arguments.run))
            return _emit(record, arguments.output)
        comparison = compare_runs(read_json(arguments.baseline), read_json(arguments.candidate))
        return _emit(comparison, arguments.output)
    except (EvaluationError, ProfileError, OSError) as error:
        print(f"evaluation failed: {_message(error)}", file=sys.stderr)
        return 2


def _message(error: Exception) -> str:
    if isinstance(error, OSError):
        return error.strerror or type(error).__name__
    return str(error)


def _summary(record: dict[str, Any], score: dict[str, Any]) -> dict[str, Any]:
    keys = ("mrr", "recall@1", "recall@10", "ndcg@10", "candidate_pool_recall")
    return {
        "stack": record["stack"],
        "dataset": record["dataset"],
        "profile": record["profile"],
        "model": record["model"],
        "results_sha256": record["results_sha256"],
        "metrics": {key: score["metrics"][key] for key in keys if key in score["metrics"]},
        "no_answer": score["no_answer"],
        "latency": score["latency"],
    }


def _emit(value: dict[str, Any], output: Path | None) -> int:
    from capicove.evaluation.harness import write_json_exclusive

    if output is None:
        _print(value)
    else:
        write_json_exclusive(output, value)
    return 0


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


__all__ = ["add_parser", "run"]
