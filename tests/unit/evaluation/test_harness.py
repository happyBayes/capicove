"""Deterministic run, score and compare behavior of the evaluation harness."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from capicove.evaluation import harness
from capicove.evaluation.harness import (
    EvalDataset,
    EvalQuery,
    EvaluationError,
    QueryOutcome,
    RankedItem,
    RunOptions,
    compare_runs,
    credited_ranking,
    latency_stats,
    read_json,
    run_evaluation,
    score_run,
    sha256_json,
    sign_test_p_value,
    validate_run,
    write_json_exclusive,
)


def query(query_id: str, judgments: dict[str, int], category: str = "c") -> EvalQuery:
    return EvalQuery(query_id, category, "text " + query_id, judgments)


def dataset(*queries: EvalQuery, digest: str = "a" * 64) -> EvalDataset:
    return EvalDataset("set-v1", "schema", ("fake",), digest, queries, {}, Path("set.json"))


@dataclass
class Prepared:
    outcomes: Mapping[str, QueryOutcome]
    identity: Mapping[str, Any] = field(
        default_factory=lambda: {
            "profile": {"id": "p", "version": 1, "fingerprint": "f" * 64},
            "index": {"kind": "fake"},
            "model": {"kind": "none", "components": []},
        }
    )
    closed: bool = False

    def search(self, query: EvalQuery, *, k: int) -> QueryOutcome:
        return self.outcomes[query.query_id]

    def close(self) -> None:
        self.closed = True


class Adapter:
    stack = "fake"
    adapter_version = "1"

    def __init__(self, prepared: Prepared) -> None:
        self.prepared = prepared

    def prepare(self, dataset: EvalDataset, workdir: Path, options: RunOptions) -> Prepared:
        assert workdir.is_dir()
        return self.prepared


def run_with(
    monkeypatch: pytest.MonkeyPatch,
    data: EvalDataset,
    outcomes: Mapping[str, QueryOutcome],
    *,
    k: int = 3,
) -> dict[str, Any]:
    prepared = Prepared(outcomes)
    monkeypatch.setattr(harness, "_adapter", lambda stack: Adapter(prepared))
    ticks = iter(range(0, 10_000_000, 1_000_000))
    record = run_evaluation("fake", data, RunOptions(k=k), clock=lambda: next(ticks))
    assert prepared.closed
    return record


def standard(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    data = dataset(
        query("q1", {"a": 3, "b": 1}),
        query("q2", {"c": 2}, category="d"),
        query("q3", {}),
    )
    return run_with(
        monkeypatch,
        data,
        {
            "q1": QueryOutcome(
                (RankedItem("x", ()), RankedItem("y", ("a", "b"), 10)), 4, ("a", "b")
            ),
            "q2": QueryOutcome((RankedItem("z", (), 5),), 1, ()),
            "q3": QueryOutcome((RankedItem("w", (), 2),), 1, ()),
        },
    )


def test_run_record_separates_deterministic_results_from_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = standard(monkeypatch)
    assert record["schema"] == "capicove.retrieval-eval-run.v1"
    assert record["model"]["real_model_validation"] == "not_run"
    assert record["results_sha256"] == sha256_json(record["results"])
    assert record["timing"]["build_ms"] == 1.0
    assert record["timing"]["query_ms"] == {"q1": 1.0, "q2": 1.0, "q3": 1.0}
    assert record["results"][0]["ranked"] == [
        {"key": "x", "labels": []},
        {"key": "y", "labels": ["a", "b"], "chars": 10},
    ]
    again = standard(monkeypatch)
    assert again["results_sha256"] == record["results_sha256"]


def test_scores_credit_labels_and_report_no_answer_and_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score = score_run(standard(monkeypatch))
    q1 = score["per_query"]["q1"]
    assert q1["mrr"] == 0.5
    # Both labels sit in one item: recall credits both, single credit only one.
    assert q1["recall@1"] == 0.0 and q1["recall@3"] == 1.0
    assert q1["single_credit_recall@3"] == 0.5
    assert q1["complete_support@3"] == 1.0
    assert q1["candidate_pool_recall"] == 1.0
    assert q1["context_chars@3"] == 10.0
    assert score["per_query"]["q2"]["mrr"] == 0.0
    assert score["answerable_queries"] == 2
    assert score["no_answer"] == {"queries": 1, "false_positive_queries": 1}
    assert set(score["by_category"]) == {"c", "d"}
    assert score["metrics"]["mrr"] == 0.25
    assert math.isclose(q1["ndcg@3"], (7 / math.log2(3)) / (7 + 1 / math.log2(3)))


def test_credited_ranking_assigns_highest_grade_first_once() -> None:
    row = {
        "judgments": {"a": 1, "b": 3, "z": 0},
        "ranked": [{"labels": ["a", "b"]}, {"labels": ["b"]}, {"labels": ["z"]}],
    }
    assert credited_ranking(row) == ["b", "~unjudged:1", "~unjudged:2"]


def test_failed_queries_score_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    record = run_with(
        monkeypatch,
        dataset(query("q1", {"a": 1})),
        {"q1": QueryOutcome((), 0, (), status="failed", diagnostics={"error": "x"})},
    )
    score = score_run(record)
    assert score["failed_queries"] == 1
    assert score["metrics"]["recall@3"] == 0.0
    assert score["metrics"]["candidate_pool_recall"] == 0.0


@pytest.mark.parametrize(
    "outcome",
    [
        QueryOutcome((RankedItem("x"), RankedItem("x")), 2, ()),
        QueryOutcome((RankedItem("x", ("unknown",)),), 1, ()),
        QueryOutcome((RankedItem("x"),), 1, ("unknown",)),
        QueryOutcome((RankedItem("a"), RankedItem("b")), 1, ()),
        QueryOutcome((RankedItem("a"),), 1, (), status="odd"),
        QueryOutcome(tuple(RankedItem(str(i)) for i in range(4)), 4, ()),
        QueryOutcome((RankedItem("a", (), -1),), 1, ()),
    ],
)
def test_invalid_adapter_outcomes_are_rejected(
    monkeypatch: pytest.MonkeyPatch, outcome: QueryOutcome
) -> None:
    with pytest.raises(EvaluationError):
        run_with(monkeypatch, dataset(query("q1", {"a": 1})), {"q1": outcome})


def test_stack_must_be_declared_by_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harness, "_adapter", lambda stack: Adapter(Prepared({})))
    with pytest.raises(EvaluationError, match="does not support"):
        run_evaluation("other", dataset(query("q1", {"a": 1})), RunOptions())


def test_adapter_identity_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = Prepared({"q1": QueryOutcome((), 0, ())}, identity={"profile": {}})
    monkeypatch.setattr(harness, "_adapter", lambda stack: Adapter(prepared))
    with pytest.raises(EvaluationError, match="identity"):
        run_evaluation("fake", dataset(query("q1", {"a": 1})), RunOptions())


def test_validate_run_detects_tampering(monkeypatch: pytest.MonkeyPatch) -> None:
    record = standard(monkeypatch)
    tampered = copy.deepcopy(record)
    tampered["results"][0]["ranked"].reverse()
    with pytest.raises(EvaluationError, match="digest"):
        validate_run(tampered)
    with pytest.raises(EvaluationError, match="schema"):
        validate_run({**record, "schema": "other"})
    with pytest.raises(EvaluationError, match="no results"):
        validate_run({**record, "results": []})
    with pytest.raises(EvaluationError, match="lacks"):
        validate_run({key: value for key, value in record.items() if key != "index"})


def test_compare_identical_runs_and_changed_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = standard(monkeypatch)
    same = compare_runs(baseline, baseline)
    assert same["rankings"] == {"results_identical": True, "changed_queries": []}
    assert same["metrics"]["mrr"]["delta"] == 0.0
    assert same["metrics"]["mrr"]["ties"] == 2
    assert same["metrics"]["context_chars@3"]["delta"] == 0.0
    candidate = run_with(
        monkeypatch,
        dataset(query("q1", {"a": 3, "b": 1}), query("q2", {"c": 2}, "d"), query("q3", {})),
        {
            "q1": QueryOutcome((RankedItem("y", ("a", "b"), 10),), 4, ("a", "b")),
            "q2": QueryOutcome((RankedItem("c", ("c",), 5),), 1, ("c",)),
            "q3": QueryOutcome((), 0, ()),
        },
    )
    comparison = compare_runs(baseline, candidate)
    assert comparison["rankings"]["changed_queries"] == ["q1", "q2", "q3"]
    assert comparison["metrics"]["mrr"]["wins"] == 2
    assert comparison["metrics"]["mrr"]["delta"] == 0.75
    assert comparison["metrics"]["mrr"]["sign_test_p"] == 0.5
    assert comparison["no_answer"] == {
        "baseline_false_positive_queries": 1,
        "candidate_false_positive_queries": 0,
    }
    low, high = comparison["metrics"]["mrr"]["paired_bootstrap_95ci"]
    assert low <= 0.75 <= high


def test_compare_requires_same_dataset_and_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = standard(monkeypatch)
    other = copy.deepcopy(baseline)
    other["dataset"]["sha256"] = "b" * 64
    with pytest.raises(EvaluationError, match="same frozen dataset"):
        compare_runs(baseline, other)
    fewer = run_with(
        monkeypatch,
        dataset(query("q1", {"a": 1})),
        {"q1": QueryOutcome((), 0, ())},
    )
    with pytest.raises(EvaluationError, match="same query IDs"):
        compare_runs(baseline, fewer)


def test_latency_and_sign_test_helpers() -> None:
    assert latency_stats([]) == {
        "count": 0,
        "mean_ms": None,
        "p50_ms": None,
        "p95_ms": None,
        "max_ms": None,
    }
    stats = latency_stats([float(value) for value in range(1, 21)])
    assert stats["p50_ms"] == 10.0 and stats["p95_ms"] == 19.0 and stats["max_ms"] == 20.0
    assert sign_test_p_value(0, 0) == 1.0
    assert sign_test_p_value(5, 0) == 0.0625
    assert sign_test_p_value(3, 3) == 1.0


@pytest.mark.parametrize(
    "values",
    [
        {"k": 0},
        {"k": 101},
        {"k": True},
        {"reranker": "bge"},
    ],
)
def test_run_options_are_validated(values: dict[str, Any]) -> None:
    with pytest.raises(EvaluationError):
        RunOptions(**values)


@pytest.mark.parametrize(
    "values",
    [
        {"query_id": "bad id"},
        {"category": ""},
        {"text": " "},
        {"judgments": {"a": 4}},
        {"judgments": {"a": True}},
        {"judgments": {"": 1}},
    ],
)
def test_queries_are_validated(values: dict[str, Any]) -> None:
    arguments: dict[str, Any] = {
        "query_id": "q",
        "category": "c",
        "text": "t",
        "judgments": {"a": 1},
    }
    arguments.update(values)
    with pytest.raises(EvaluationError):
        EvalQuery(**arguments)


def test_datasets_are_validated() -> None:
    with pytest.raises(EvaluationError, match="digest"):
        dataset(query("q", {"a": 1}), digest="bad")
    with pytest.raises(EvaluationError, match="queries"):
        dataset()
    with pytest.raises(EvaluationError, match="unique"):
        dataset(query("q", {"a": 1}), query("q", {"a": 1}))
    assert query("q", {"b": 1, "a": 2, "z": 0}).positive_labels == ("a", "b")


def test_json_files_are_exclusive_and_strict(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    write_json_exclusive(path, {"b": 1, "a": [1.5]})
    assert json.loads(path.read_text()) == {"a": [1.5], "b": 1}
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        write_json_exclusive(path, {})
    assert read_json(path) == {"a": [1.5], "b": 1}
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a": 1, "a": 2}')
    with pytest.raises(EvaluationError, match="duplicate"):
        read_json(duplicate)
    nan = tmp_path / "nan.json"
    nan.write_text('{"a": NaN}')
    with pytest.raises(EvaluationError, match="nonstandard"):
        read_json(nan)
    broken = tmp_path / "broken.json"
    broken.write_bytes(b"\xff")
    with pytest.raises(EvaluationError, match="not valid JSON"):
        read_json(broken)
    with pytest.raises(EvaluationError, match="size"):
        read_json(path, max_bytes=2)


def test_unknown_stack_is_rejected() -> None:
    from capicove.evaluation.adapters import resolve_adapter

    with pytest.raises(EvaluationError, match="unsupported"):
        resolve_adapter("nope")


def test_search_only_timing_is_kept_out_of_results(monkeypatch: pytest.MonkeyPatch) -> None:
    record = run_with(
        monkeypatch,
        dataset(query("q1", {"a": 1})),
        {"q1": QueryOutcome((RankedItem("a", ("a",)),), 1, ("a",), search_ns=2_000_000)},
    )
    assert record["timing"]["search_ms"] == {"q1": 2.0}
    assert record["timing"]["search_stats"]["p50_ms"] == 2.0
    assert "search_ns" not in json.dumps(record["results"])
    assert score_run(record)["search_latency"]["count"] == 1
