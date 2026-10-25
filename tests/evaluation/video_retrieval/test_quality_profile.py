"""Frozen authored relevance is an engineering gate, not model accuracy."""

from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest

from capicove.retrieval.video_evaluation import evaluate_rankings
from tests.evaluation.video_retrieval.fixtures import (
    FIXTURES,
    Catalog,
    load_fixture,
    make_lanes,
    read_rows,
    run_queries,
)


pytestmark = pytest.mark.evaluation


def test_frozen_weighted_rrf_quality() -> None:
    profile, queries, judgments = load_fixture()
    rankings = run_queries(queries)
    report = evaluate_rankings(
        profile=profile,
        judgments=judgments,
        rankings=rankings,
        recorded_at=datetime(2026, 9, 15, tzinfo=UTC),
    )
    assert len(judgments) == 30
    assert len({query["scenario"] for query in queries}) == 10
    assert report.metrics.text_keyframe_recall_at_20 is not None
    assert report.metrics.text_keyframe_recall_at_20 >= 0.90
    assert report.metrics.segment_recall_at_20 is not None
    assert report.metrics.segment_recall_at_20 >= 0.85
    assert report.metrics.segment_temporal_iou_at_1 is not None
    assert report.metrics.segment_temporal_iou_at_1 >= 0.50
    assert report.metrics.mean_reciprocal_rank == 1
    assert not report.passed  # No model, capacity or endpoint evidence in this test.


def test_removing_visual_lane_loses_visual_only_query() -> None:
    _, queries, _ = load_fixture()
    query = next(row for row in queries if row["id"] == "text-keyframe-0")
    assert run_queries([query])[query["id"]]
    assert not run_queries([{**query, "lanes": []}])[query["id"]]


def test_every_frozen_judgment_order_matches() -> None:
    _, queries, _ = load_fixture()
    rankings = run_queries(queries)
    for judgment in read_rows(FIXTURES, "judgments"):
        assert [item.unit_id for item in rankings[judgment["id"]]] == judgment["expected"]


def test_raw_score_addition_is_not_the_fusion_policy() -> None:
    _, queries, _ = load_fixture()
    query = deepcopy(next(row for row in queries if row["id"] == "visual-asr-0"))
    query["lanes"] = [
        {"id": "keyframe_visual", "candidates": [{"id": "k2", "rank": 1, "score": 0.5}]},
        {"id": "segment_visual", "candidates": [{"id": "s2", "rank": 1, "score": 0.5}]},
        {"id": "asr_lexical", "candidates": [{"id": "asr1", "rank": 1, "score": 1000.0}]},
    ]
    assert run_queries([query])[query["id"]][0].unit_id == "s2"
    raw_scores = {"s1": 1000.0, "s2": 0.5 + 0.5}
    assert max(raw_scores, key=lambda key: raw_scores[key]) == "s1"


@pytest.mark.parametrize("mutation", ("field", "rank", "nan", "unknown", "duplicate"))
def test_invalid_candidate_metadata_rejected(mutation: str) -> None:
    _, queries, _ = load_fixture()
    query = deepcopy(queries[0])
    candidate = query["lanes"][0]["candidates"][0]
    if mutation == "field":
        candidate["unrecognized"] = True
    elif mutation == "rank":
        candidate["rank"] = 2
    elif mutation == "nan":
        candidate["score"] = float("nan")
    elif mutation == "unknown":
        candidate["id"] = "missing"
    else:
        query["lanes"].append(deepcopy(query["lanes"][0]))
    with pytest.raises(ValueError):
        make_lanes(query, Catalog(read_rows(FIXTURES, "corpus"), 0))


@pytest.mark.parametrize("mutation", ("range", "relation", "cross-source"))
def test_invalid_catalog_rejected(mutation: str) -> None:
    corpus = read_rows(FIXTURES, "corpus")
    if mutation == "range":
        corpus[0]["end_ms"] = 0
    elif mutation == "relation":
        corpus[4]["segments"] = ["missing"]
    else:
        corpus[4]["segments"] = ["other"]
    with pytest.raises(ValueError):
        Catalog(corpus, 0)


@pytest.mark.parametrize("name", ("corpus", "queries", "judgments"))
def test_tampered_fixture_rejected(tmp_path: Path, name: str) -> None:
    from tests.evaluation.video_retrieval.fixtures import FIXTURES

    for source in FIXTURES.glob("*.jsonl"):
        (tmp_path / source.name).write_bytes(source.read_bytes())
    path = tmp_path / f"{name}.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="digest"):
        load_fixture(tmp_path)
