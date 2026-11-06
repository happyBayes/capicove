"""Experimental offline retrieval evaluation harness (RFC-0034, Proposed).

The harness runs frozen query sets through one retrieval stack, records the
dataset, profile and index identities with every run, scores runs with the shared
retrieval metrics and compares two runs query by query. Stack adapters import
their optional runtimes lazily. Deterministic fakes stand in for models and are
labelled as such in every record; their numbers are engineering baselines, never
model quality.
"""

from capicove.evaluation.harness import (
    EVAL_RUN_SCHEMA,
    EvalDataset,
    EvalQuery,
    QueryOutcome,
    RankedItem,
    compare_runs,
    load_dataset,
    run_evaluation,
    score_run,
)


__all__ = [
    "EVAL_RUN_SCHEMA",
    "EvalDataset",
    "EvalQuery",
    "QueryOutcome",
    "RankedItem",
    "compare_runs",
    "load_dataset",
    "run_evaluation",
    "score_run",
]
