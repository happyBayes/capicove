"""Pure metrics for evaluating ranked retrieval results."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


Rankings = Mapping[str, Sequence[str]]
RelevanceJudgments = Mapping[str, Mapping[str, int]]


def mean_reciprocal_rank(
    rankings: Rankings,
    relevance_judgments: RelevanceJudgments,
) -> float:
    """Return the macro-averaged reciprocal rank across queries.

    For each query, let `r` be the one-based rank of its first result with
    positive relevance. Its reciprocal rank is `1 / r`, or zero if no relevant
    result exists. `MRR = sum(reciprocal ranks) / number of queries`.
    MRR is dimensionless and lies in `[0, 1]`.

    Args:
        rankings: Ranked document IDs for every query.
        relevance_judgments: Integer relevance grades from zero through three.

    Returns:
        The macro mean reciprocal rank.

    Raises:
        ValueError: If the evaluation data is invalid.
    """
    _validate_evaluation_data(rankings, relevance_judgments)
    total = 0.0
    for query_id, ranked_ids in rankings.items():
        judgments = relevance_judgments[query_id]
        for rank, document_id in enumerate(ranked_ids, start=1):
            if judgments.get(document_id, 0) > 0:
                total += 1.0 / rank
                break
    return total / len(rankings)


def ndcg_at_k(
    rankings: Rankings,
    relevance_judgments: RelevanceJudgments,
    *,
    k: int,
) -> float:
    """Return macro-averaged normalized discounted cumulative gain at `k`.

    For one-based ranks `i` through `k`, sum `(2 ** relevance[i] - 1) / log2(i + 1)`
    to obtain DCG. `NDCG@k = DCG@k / IDCG@k`, where IDCG uses the same grades in
    ideal descending order. Average the per-query NDCG values for the final
    score. NDCG@k is dimensionless and lies in `[0, 1]`.

    Args:
        rankings: Ranked document IDs for every query.
        relevance_judgments: Integer relevance grades from zero through three.
        k: Positive rank cutoff.

    Returns:
        The macro mean NDCG@k.

    Raises:
        ValueError: If ``k`` or the evaluation data is invalid.
    """
    _validate_k(k)
    _validate_evaluation_data(rankings, relevance_judgments)
    total = 0.0
    for query_id, ranked_ids in rankings.items():
        judgments = relevance_judgments[query_id]
        actual_grades = tuple(judgments.get(document_id, 0) for document_id in ranked_ids[:k])
        ideal_grades = tuple(sorted(judgments.values(), reverse=True)[:k])
        total += _discounted_cumulative_gain(actual_grades) / _discounted_cumulative_gain(
            ideal_grades
        )
    return total / len(rankings)


def recall_at_k(
    rankings: Rankings,
    relevance_judgments: RelevanceJudgments,
    *,
    k: int,
) -> float:
    """Return macro-averaged recall at `k` across queries.

    For each query, `Recall@k = relevant documents in the top k / total relevant
    documents`. Average these per-query values for the final score. Recall@k is
    dimensionless and lies in `[0, 1]`.

    Args:
        rankings: Ranked document IDs for every query.
        relevance_judgments: Integer relevance grades from zero through three.
        k: Positive rank cutoff.

    Returns:
        The macro mean Recall@k.

    Raises:
        ValueError: If ``k`` or the evaluation data is invalid.
    """
    _validate_k(k)
    _validate_evaluation_data(rankings, relevance_judgments)
    total = 0.0
    for query_id, ranked_ids in rankings.items():
        relevant_ids = {
            document_id for document_id, grade in relevance_judgments[query_id].items() if grade > 0
        }
        retrieved_relevant = relevant_ids.intersection(ranked_ids[:k])
        total += len(retrieved_relevant) / len(relevant_ids)
    return total / len(rankings)


def _discounted_cumulative_gain(grades: Sequence[int]) -> float:
    total = 0.0
    for rank, grade in enumerate(grades, start=1):
        gain = float(2**grade - 1)
        total += gain / math.log2(rank + 1)
    return total


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")


def _validate_evaluation_data(
    rankings: Rankings,
    relevance_judgments: RelevanceJudgments,
) -> None:
    if not rankings:
        raise ValueError("evaluation data must contain at least one query")
    if rankings.keys() != relevance_judgments.keys():
        raise ValueError("rankings and relevance judgments must contain the same query IDs")

    for query_id, ranked_ids in rankings.items():
        if len(ranked_ids) != len(set(ranked_ids)):
            raise ValueError(f"rankings for query {query_id!r} contain a duplicate document ID")

        judgments = relevance_judgments[query_id]
        for grade in judgments.values():
            if isinstance(grade, bool) or not isinstance(grade, int):
                raise ValueError("relevance grades must be integers")
            if not 0 <= grade <= 3:
                raise ValueError("relevance grades must be between 0 and 3")
        if not any(grade > 0 for grade in judgments.values()):
            raise ValueError(
                f"query {query_id!r} must have at least one positive relevance judgment"
            )
