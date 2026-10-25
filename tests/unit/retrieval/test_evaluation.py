"""Tests for deterministic retrieval ranking metrics."""

from __future__ import annotations

import math

import pytest

from capicove.retrieval import (
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)


def test_mean_reciprocal_rank_averages_first_relevant_result() -> None:
    rankings = {
        "first": ("irrelevant", "relevant", "also-relevant"),
        "second": ("miss",),
    }
    judgments = {
        "first": {"relevant": 3, "also-relevant": 1},
        "second": {"not-returned": 2},
    }

    assert mean_reciprocal_rank(rankings, judgments) == pytest.approx(0.25)


def test_ndcg_at_k_uses_exponential_gain_and_logarithmic_discount() -> None:
    rankings = {"query": ("relevant", "highly-relevant", "irrelevant")}
    judgments = {
        "query": {
            "highly-relevant": 3,
            "relevant": 1,
            "irrelevant": 0,
        }
    }
    expected_dcg = 1.0 + (7.0 / math.log2(3.0))
    expected_idcg = 7.0 + (1.0 / math.log2(3.0))

    assert ndcg_at_k(rankings, judgments, k=3) == pytest.approx(expected_dcg / expected_idcg)


def test_recall_at_k_counts_unique_relevant_results() -> None:
    rankings = {"query": ("relevant-a", "irrelevant", "relevant-b")}
    judgments = {
        "query": {
            "relevant-a": 1,
            "relevant-b": 3,
            "not-returned": 2,
        }
    }

    assert recall_at_k(rankings, judgments, k=2) == pytest.approx(1.0 / 3.0)


def test_cutoff_metrics_macro_average_queries() -> None:
    rankings = {
        "hit": ("relevant",),
        "miss": ("irrelevant",),
    }
    judgments = {
        "hit": {"relevant": 1},
        "miss": {"not-returned": 1},
    }

    assert ndcg_at_k(rankings, judgments, k=1) == pytest.approx(0.5)
    assert recall_at_k(rankings, judgments, k=1) == pytest.approx(0.5)


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_cutoff_metrics_reject_invalid_k(k: object) -> None:
    rankings = {"query": ("relevant",)}
    judgments = {"query": {"relevant": 1}}

    with pytest.raises(ValueError, match="positive integer"):
        ndcg_at_k(rankings, judgments, k=k)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integer"):
        recall_at_k(rankings, judgments, k=k)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("rankings", "judgments", "message"),
    [
        ({}, {}, "at least one query"),
        ({"query": ()}, {"other": {"relevant": 1}}, "same query IDs"),
        ({"query": ("duplicate", "duplicate")}, {"query": {"duplicate": 1}}, "duplicate"),
        ({"query": ()}, {"query": {"irrelevant": 0}}, "positive relevance"),
        ({"query": ()}, {"query": {"relevant": 4}}, "between 0 and 3"),
        ({"query": ()}, {"query": {"relevant": 1.0}}, "integer"),
    ],
)
def test_metrics_reject_invalid_evaluation_data(
    rankings: dict[str, tuple[str, ...]],
    judgments: dict[str, dict[str, object]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        mean_reciprocal_rank(rankings, judgments)  # type: ignore[arg-type]
