"""Deterministic run, score and compare operations for offline retrieval evaluation.

A run record separates deterministic ``results`` (ranked item keys, satisfied
judgment labels, candidate-pool coverage) from non-deterministic ``timing``. The
``results_sha256`` digest covers only the deterministic part, so two runs with the
same dataset, profile and index content can be compared byte for byte.

For MRR and nDCG each ranked item credits at most one not-yet-credited judged
label, highest grade first, so ranks stay positional when the shared metrics in
:mod:`capicove.retrieval.evaluation` are applied. ``recall@k`` and
``complete_support@k`` credit every label an item satisfies, because one chunk
can contain several labelled facts; ``single_credit_recall@k`` reports the
shared single-credit recall for reference. Queries without positive judgments are no-answer queries: they are
excluded from the ranking metrics and reported as false positives when any item
is returned.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from capicove.__about__ import __version__


EVAL_DATASET_SCHEMA = "capicove.retrieval-eval.v1"
EVAL_RUN_SCHEMA = "capicove.retrieval-eval-run.v1"
EVAL_SCORE_SCHEMA = "capicove.retrieval-eval-score.v1"
EVAL_COMPARISON_SCHEMA = "capicove.retrieval-eval-comparison.v1"
MAX_K = 100
MAX_QUERIES = 5_000
BOOTSTRAP_SEED = 0
REAL_MODEL_VALIDATION = "not_run"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/#+-]{0,255}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STATUSES = frozenset({"succeeded", "failed"})


class EvaluationError(ValueError):
    """Report a stable, path-free evaluation input or integrity failure."""


def canonical_json(value: Any) -> bytes:
    """Encode JSON deterministically: sorted keys, compact separators, no NaN."""
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    """Return the SHA-256 of the canonical JSON encoding."""
    return hashlib.sha256(canonical_json(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class EvalQuery:
    """One frozen query with graded judgments keyed by stable label identity.

    ``judgments`` maps labels to grades from zero through three. An empty or
    all-zero mapping marks a no-answer query. ``payload`` carries closed
    adapter-specific query fields such as authored lane candidates.
    """

    query_id: str
    category: str
    text: str | None
    judgments: Mapping[str, int]
    payload: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        _identifier("query_id", self.query_id)
        _identifier("category", self.category)
        if self.text is not None and (not isinstance(self.text, str) or not self.text.strip()):
            raise EvaluationError("query text must be nonblank when present")
        judgments = dict(self.judgments)
        for label, grade in judgments.items():
            if not isinstance(label, str) or not label or len(label) > 1024:
                raise EvaluationError("judgment labels must be nonblank strings")
            if type(grade) is not int or not 0 <= grade <= 3:
                raise EvaluationError("judgment grades must be integers from 0 through 3")
        object.__setattr__(self, "judgments", MappingProxyType(judgments))
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

    @property
    def positive_labels(self) -> tuple[str, ...]:
        """Return labels with positive grades in stable order."""
        return tuple(sorted(label for label, grade in self.judgments.items() if grade > 0))


@dataclass(frozen=True, slots=True)
class EvalDataset:
    """A loaded frozen dataset bound to the SHA-256 of its exact source bytes."""

    dataset_id: str
    schema: str
    stacks: tuple[str, ...]
    sha256: str
    queries: tuple[EvalQuery, ...]
    corpus: Mapping[str, Any]
    source: Path

    def __post_init__(self) -> None:
        _identifier("dataset_id", self.dataset_id)
        if not _SHA256.fullmatch(self.sha256):
            raise EvaluationError("dataset digest must be a SHA-256 value")
        if not self.queries or len(self.queries) > MAX_QUERIES:
            raise EvaluationError(f"datasets must contain 1 to {MAX_QUERIES} queries")
        identities = [query.query_id for query in self.queries]
        if len(set(identities)) != len(identities):
            raise EvaluationError("dataset query IDs must be unique")


@dataclass(frozen=True, slots=True)
class RankedItem:
    """One ranked result with a stable key and the judgment labels it satisfies."""

    key: str
    labels: tuple[str, ...] = ()
    chars: int | None = None


@dataclass(frozen=True, slots=True)
class QueryOutcome:
    """Deterministic outcome of one query at one cutoff.

    ``pool_labels`` are the judged labels satisfied by any candidate considered
    before final truncation (and before any reranking), so candidate-pool recall
    bounds what reranking could recover. ``search_ns`` optionally times only the
    stack's search call, excluding the adapter's evidence resolution; it is
    recorded with timing, never with results.
    """

    ranked: tuple[RankedItem, ...]
    pool_size: int
    pool_labels: tuple[str, ...]
    status: str = "succeeded"
    diagnostics: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    search_ns: int | None = None


@dataclass(frozen=True, slots=True)
class RunOptions:
    """Caller-selected run controls; adapters reject options they do not support."""

    k: int = 10
    profile_path: Path | None = None
    reranker: str = "none"
    corpus_repo: Path | None = None
    git: str | None = None
    fake_dense: bool = False
    reranker_model: Path | None = None

    def __post_init__(self) -> None:
        if type(self.k) is not int or not 1 <= self.k <= MAX_K:
            raise EvaluationError(f"k must be an integer from 1 to {MAX_K}")
        if self.reranker not in {"none", "deterministic-fake", "bge"}:
            raise EvaluationError("reranker must be none, deterministic-fake or bge")
        if (self.reranker == "bge") != (self.reranker_model is not None):
            raise EvaluationError("--reranker bge requires --reranker-model, and only it")


class PreparedStack(Protocol):
    """A built offline index that answers frozen queries."""

    identity: Mapping[str, Any]

    def search(self, query: EvalQuery, *, k: int) -> QueryOutcome: ...

    def close(self) -> None: ...


class StackAdapter(Protocol):
    """Build one stack's offline index from a frozen dataset."""

    stack: str
    adapter_version: str

    def prepare(
        self, dataset: EvalDataset, workdir: Path, options: RunOptions
    ) -> PreparedStack: ...


def _adapter(stack: str) -> StackAdapter:
    from capicove.evaluation.adapters import resolve_adapter

    return resolve_adapter(stack)


def load_dataset(path: Path) -> EvalDataset:
    """Load an authored generic dataset or a supported existing frozen set."""
    from capicove.evaluation.datasets import load_dataset as load

    return load(path)


def run_evaluation(
    stack: str,
    dataset: EvalDataset,
    options: RunOptions,
    *,
    workdir: Path | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    """Build the stack's index offline, run every query, and return a run record."""
    adapter = _adapter(stack)
    if stack not in dataset.stacks:
        raise EvaluationError(f"dataset {dataset.dataset_id} does not support stack {stack}")
    with tempfile.TemporaryDirectory(prefix="capicove-eval-", dir=workdir) as scratch:
        root = Path(scratch).resolve()
        started = clock()
        prepared = adapter.prepare(dataset, root, options)
        built = clock()
        try:
            results: list[dict[str, Any]] = []
            latencies: dict[str, float] = {}
            searches: dict[str, float] = {}
            for query in dataset.queries:
                before = clock()
                outcome = prepared.search(query, k=options.k)
                latencies[query.query_id] = (clock() - before) / 1e6
                if outcome.search_ns is not None:
                    searches[query.query_id] = outcome.search_ns / 1e6
                results.append(_result_row(query, outcome, options.k))
            identity = dict(prepared.identity)
        finally:
            prepared.close()
    for name in ("index", "model", "profile"):
        if not isinstance(identity.get(name), Mapping):
            raise EvaluationError(f"adapter identity must declare {name}")
    return {
        "schema": EVAL_RUN_SCHEMA,
        "experimental": True,
        "stack": stack,
        "adapter": {"id": stack, "version": adapter.adapter_version},
        "dataset": {
            "id": dataset.dataset_id,
            "schema": dataset.schema,
            "sha256": dataset.sha256,
            "queries": len(dataset.queries),
        },
        "profile": dict(identity["profile"]),
        "index": dict(identity["index"]),
        "model": {
            **identity["model"],
            # A local model run is a measurement awaiting acceptance; anything
            # involving a deterministic fake never validates a real model.
            "real_model_validation": "measured_unaccepted"
            if identity["model"].get("kind") == "local-model"
            else REAL_MODEL_VALIDATION,
        },
        "options": {
            "k": options.k,
            "reranker": options.reranker,
            "fake_dense": options.fake_dense,
        },
        "results": results,
        "results_sha256": sha256_json(results),
        "timing": {
            "build_ms": (built - started) / 1e6,
            "query_ms": latencies,
            "stats": latency_stats(list(latencies.values())),
            "search_ms": searches,
            "search_stats": latency_stats(list(searches.values())),
        },
        "environment": {
            "capicove_version": __version__,
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "machine": platform.machine(),
            "cpus": os.cpu_count(),
        },
    }


def _result_row(query: EvalQuery, outcome: QueryOutcome, k: int) -> dict[str, Any]:
    if not isinstance(outcome, QueryOutcome) or outcome.status not in _STATUSES:
        raise EvaluationError("adapter returned an invalid query outcome")
    keys = [item.key for item in outcome.ranked]
    if len(keys) > k or len(set(keys)) != len(keys):
        raise EvaluationError("ranked items must be unique and within the cutoff")
    known = set(query.judgments)
    for item in outcome.ranked:
        if not isinstance(item.key, str) or not item.key or not set(item.labels) <= known:
            raise EvaluationError("ranked items must reference judged labels only")
    if not set(outcome.pool_labels) <= known or type(outcome.pool_size) is not int:
        raise EvaluationError("pool labels must reference judged labels only")
    if outcome.pool_size < len(keys):
        raise EvaluationError("the candidate pool cannot be smaller than the ranking")
    return {
        "query_id": query.query_id,
        "category": query.category,
        "status": outcome.status,
        "judgments": dict(sorted(query.judgments.items())),
        "ranked": [_item(item) for item in outcome.ranked],
        "pool": {"size": outcome.pool_size, "labels": sorted(set(outcome.pool_labels))},
        "diagnostics": json.loads(canonical_json(dict(outcome.diagnostics))),
    }


def _item(item: RankedItem) -> dict[str, Any]:
    row: dict[str, Any] = {"key": item.key, "labels": sorted(set(item.labels))}
    if item.chars is not None:
        if type(item.chars) is not int or item.chars < 0:
            raise EvaluationError("ranked item sizes must be non-negative integers")
        row["chars"] = item.chars
    return row


def validate_run(run: Mapping[str, Any]) -> None:
    """Reject a run record whose schema or deterministic digest does not verify."""
    if not isinstance(run, Mapping) or run.get("schema") != EVAL_RUN_SCHEMA:
        raise EvaluationError("unsupported evaluation run schema")
    results = run.get("results")
    if not isinstance(results, list) or not results:
        raise EvaluationError("evaluation run has no results")
    if sha256_json(results) != run.get("results_sha256"):
        raise EvaluationError("evaluation run results digest mismatch")
    for name in ("dataset", "profile", "index", "model", "options", "timing"):
        if not isinstance(run.get(name), Mapping):
            raise EvaluationError(f"evaluation run lacks {name}")


def credited_ranking(row: Mapping[str, Any]) -> list[str]:
    """Credit each ranked item with at most one not-yet-credited judged label."""
    judgments: Mapping[str, int] = row["judgments"]
    credited: set[str] = set()
    ranking: list[str] = []
    for position, item in enumerate(row["ranked"]):
        labels = sorted(
            (label for label in item["labels"] if label not in credited and judgments[label] > 0),
            key=lambda label: (-judgments[label], label),
        )
        if labels:
            credited.add(labels[0])
            ranking.append(labels[0])
        else:
            ranking.append(f"~unjudged:{position}")
    return ranking


def cutoffs(k: int) -> tuple[int, ...]:
    """Return the reported rank cutoffs for a run cutoff."""
    return tuple(sorted({value for value in (1, 3, 5, 10, k) if value <= k}))


def _query_metrics(row: Mapping[str, Any], k: int, *, sizes: bool = False) -> dict[str, float]:
    from capicove.retrieval.evaluation import mean_reciprocal_rank, ndcg_at_k, recall_at_k

    judgments = {label: grade for label, grade in row["judgments"].items()}
    positive = {label for label, grade in judgments.items() if grade > 0}
    ranking = {"q": credited_ranking(row) if row["status"] == "succeeded" else []}
    graded = {"q": judgments}
    metrics = {"mrr": mean_reciprocal_rank(ranking, graded)}
    items = row["ranked"] if row["status"] == "succeeded" else []
    for cutoff in cutoffs(k):
        # Recall credits every label an item satisfies; one chunk may hold two facts.
        found = {label for item in items[:cutoff] for label in item["labels"]} & positive
        metrics[f"recall@{cutoff}"] = len(found) / len(positive)
        metrics[f"single_credit_recall@{cutoff}"] = recall_at_k(ranking, graded, k=cutoff)
        metrics[f"ndcg@{cutoff}"] = ndcg_at_k(ranking, graded, k=cutoff)
        metrics[f"complete_support@{cutoff}"] = float(found == positive)
        if sizes:
            # Evidence characters handed to a reader: a context-budget view.
            metrics[f"context_chars@{cutoff}"] = float(
                sum(item.get("chars") or 0 for item in items[:cutoff])
            )
    pool = set(row["pool"]["labels"]) if row["status"] == "succeeded" else set()
    metrics["candidate_pool_recall"] = len(pool & positive) / len(positive)
    return metrics


def latency_stats(values: Sequence[float]) -> dict[str, float | int | None]:
    """Summarize wall-clock milliseconds with nearest-rank percentiles."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}

    def percentile(fraction: float) -> float:
        return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]

    return {
        "count": len(ordered),
        "mean_ms": sum(ordered) / len(ordered),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "max_ms": ordered[-1],
    }


def score_run(run: Mapping[str, Any]) -> dict[str, Any]:
    """Score a verified run record with the shared ranking metrics."""
    validate_run(run)
    k = run["options"]["k"]
    per_query: dict[str, dict[str, float]] = {}
    no_answer: dict[str, int] = {}
    categories: dict[str, list[str]] = {}
    failed = 0
    sizes = any("chars" in item for row in run["results"] for item in row["ranked"])
    for row in run["results"]:
        failed += row["status"] != "succeeded"
        if any(grade > 0 for grade in row["judgments"].values()):
            per_query[row["query_id"]] = _query_metrics(row, k, sizes=sizes)
            categories.setdefault(row["category"], []).append(row["query_id"])
        else:
            no_answer[row["query_id"]] = len(row["ranked"])
    names = sorted({name for metrics in per_query.values() for name in metrics})

    def mean(ids: Sequence[str]) -> dict[str, float]:
        return {name: sum(per_query[i][name] for i in ids) / len(ids) for name in names}

    return {
        "schema": EVAL_SCORE_SCHEMA,
        "experimental": True,
        "run": {
            "stack": run["stack"],
            "results_sha256": run["results_sha256"],
            "dataset": dict(run["dataset"]),
            "profile": dict(run["profile"]),
            "index": dict(run["index"]),
            "model": dict(run["model"]),
            "options": dict(run["options"]),
        },
        "answerable_queries": len(per_query),
        "failed_queries": failed,
        "metrics": mean(sorted(per_query)) if per_query else {},
        "by_category": {name: mean(ids) for name, ids in sorted(categories.items())},
        "no_answer": {
            "queries": len(no_answer),
            "false_positive_queries": sum(1 for count in no_answer.values() if count),
        },
        "per_query": dict(sorted(per_query.items())),
        "latency": dict(run["timing"]["stats"]),
        "search_latency": dict(run["timing"].get("search_stats", latency_stats([]))),
    }


def sign_test_p_value(wins: int, losses: int) -> float:
    """Return the exact two-sided sign-test p-value, ignoring ties."""
    trials = wins + losses
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, count) for count in range(min(wins, losses) + 1))
    return min(1.0, float(2 * tail) / float(1 << trials))


def compare_runs(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Pair two runs over the same frozen dataset query by query."""
    from capicove.knowledge.memory_evaluation import summarize_paired

    base_score, candidate_score = score_run(baseline), score_run(candidate)
    if baseline["dataset"]["sha256"] != candidate["dataset"]["sha256"]:
        raise EvaluationError("runs must evaluate the same frozen dataset bytes")
    base_rows = {row["query_id"]: row for row in baseline["results"]}
    candidate_rows = {row["query_id"]: row for row in candidate["results"]}
    if base_rows.keys() != candidate_rows.keys():
        raise EvaluationError("runs must contain the same query IDs")
    ids = sorted(base_score["per_query"])
    shared = sorted(
        set(base_score["metrics"]) & set(candidate_score["metrics"]),
        key=lambda name: (name.split("@")[0], int(name.split("@")[1]) if "@" in name else 0),
    )
    metrics: dict[str, Any] = {}
    for name in shared:
        before = {i: base_score["per_query"][i][name] for i in ids}
        after = {i: candidate_score["per_query"][i][name] for i in ids}
        wins = sum(after[i] > before[i] for i in ids)
        losses = sum(after[i] < before[i] for i in ids)
        if name.startswith("context_chars@"):
            # Unbounded sizes: report paired means; lower is a smaller context.
            base_mean = sum(before.values()) / len(ids)
            candidate_mean = sum(after.values()) / len(ids)
            metrics[name] = {
                "baseline": base_mean,
                "candidate": candidate_mean,
                "delta": candidate_mean - base_mean,
                "wins": wins,
                "losses": losses,
                "ties": len(ids) - wins - losses,
            }
            continue
        summary = summarize_paired(ids, before, after, {i: i for i in ids}, seed=BOOTSTRAP_SEED)
        metrics[name] = {
            "baseline": summary["baseline_score"],
            "candidate": summary["candidate_score"],
            "delta": summary["delta"],
            "paired_bootstrap_95ci": summary["paired_cluster_bootstrap_95ci"],
            "wins": wins,
            "losses": losses,
            "ties": len(ids) - wins - losses,
            "sign_test_p": sign_test_p_value(wins, losses),
        }
    changed = sorted(
        query_id
        for query_id in base_rows
        if [item["key"] for item in base_rows[query_id]["ranked"]]
        != [item["key"] for item in candidate_rows[query_id]["ranked"]]
    )
    return {
        "schema": EVAL_COMPARISON_SCHEMA,
        "experimental": True,
        "dataset": dict(baseline["dataset"]),
        "baseline": base_score["run"],
        "candidate": candidate_score["run"],
        "paired_queries": len(ids),
        "bootstrap": {"seed": BOOTSTRAP_SEED, "samples": 2000, "groups": "query"},
        "metrics": metrics,
        "rankings": {
            "results_identical": baseline["results_sha256"] == candidate["results_sha256"],
            "changed_queries": changed,
        },
        "no_answer": {
            "baseline_false_positive_queries": base_score["no_answer"]["false_positive_queries"],
            "candidate_false_positive_queries": candidate_score["no_answer"][
                "false_positive_queries"
            ],
        },
        "latency": {"baseline": base_score["latency"], "candidate": candidate_score["latency"]},
        "interpretation": (
            "Engineering comparison over a frozen set; deterministic fakes are not model quality"
        ),
    }


def write_json_exclusive(path: Path, value: Any) -> None:
    """Write indented JSON to a new private file, refusing to overwrite."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def read_json(path: Path, *, max_bytes: int = 64 * 1024 * 1024) -> Any:
    """Read bounded JSON that rejects duplicate keys and nonstandard numbers."""
    payload = path.read_bytes()
    if len(payload) > max_bytes:
        raise EvaluationError("evaluation file exceeds the size limit")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EvaluationError("evaluation file contains a duplicate key")
            result[key] = value
        return result

    def constant(value: str) -> Any:
        raise EvaluationError("evaluation file contains a nonstandard number")

    try:
        return json.loads(payload, object_pairs_hook=unique, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise EvaluationError("evaluation file is not valid JSON") from None


def _identifier(name: str, value: object) -> None:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise EvaluationError(f"{name} must be a stable identifier")


__all__ = [
    "EVAL_COMPARISON_SCHEMA",
    "EVAL_DATASET_SCHEMA",
    "EVAL_RUN_SCHEMA",
    "EVAL_SCORE_SCHEMA",
    "REAL_MODEL_VALIDATION",
    "EvalDataset",
    "EvalQuery",
    "EvaluationError",
    "PreparedStack",
    "QueryOutcome",
    "RankedItem",
    "RunOptions",
    "StackAdapter",
    "canonical_json",
    "compare_runs",
    "credited_ranking",
    "cutoffs",
    "latency_stats",
    "load_dataset",
    "read_json",
    "run_evaluation",
    "score_run",
    "sha256_json",
    "sign_test_p_value",
    "validate_run",
    "write_json_exclusive",
]
