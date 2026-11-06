"""Strict loading of authored and existing frozen evaluation sets."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from capicove.evaluation.datasets import load_dataset
from capicove.evaluation.harness import EvaluationError


ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / "docs" / "development" / "eval"
VIDEO = ROOT / "tests" / "evaluation" / "video_retrieval" / "fixtures"
AUTHORED = (
    "personal-notes-eval-v1.json",
    "unified-mixed-eval-v1.json",
    "image-synthetic-eval-v1.json",
    "multihop-graph-eval-v1.json",
)


def generic(**changes: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": "capicove.retrieval-eval.v1",
        "dataset_id": "tiny-v1",
        "stacks": ["kb"],
        "frozen_at": "2026-10-29",
        "provenance": "test",
        "corpus": {"documents": []},
        "queries": [
            {
                "id": "q1",
                "category": "c",
                "text": "hello",
                "relevant": [{"label": "l", "document": "d", "contains": "x", "grade": 2}],
            }
        ],
    }
    value.update(changes)
    return value


def write(tmp_path: Path, value: Any) -> Path:
    path = tmp_path / "set.json"
    path.write_text(json.dumps(value))
    return path


@pytest.mark.parametrize("name", AUTHORED)
def test_authored_sets_load_with_byte_identity(name: str) -> None:
    path = EVAL / name
    loaded = load_dataset(path)
    assert loaded.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert loaded.dataset_id == name.removesuffix(".json")
    for query in loaded.queries:
        assert set(query.judgments) == {label["label"] for label in query.payload["labels"]}


def test_generic_set_carries_labels_and_payload(tmp_path: Path) -> None:
    value = generic()
    value["queries"][0]["payload"] = {"intent": "concept"}
    loaded = load_dataset(write(tmp_path, value))
    query = loaded.queries[0]
    assert query.judgments == {"l": 2}
    assert query.payload["intent"] == "concept"
    assert query.payload["labels"][0]["contains"] == "x"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda v: v.update(extra=1),
        lambda v: v.update(stacks=["code"]),
        lambda v: v.update(stacks=[]),
        lambda v: v.update(stacks=["kb", "kb"]),
        lambda v: v.update(corpus=[]),
        lambda v: v["queries"][0].update(extra=1),
        lambda v: v["queries"][0].update(relevant="x"),
        lambda v: v["queries"][0]["relevant"][0].update(extra=1),
        lambda v: v["queries"][0]["relevant"][0].update(document=""),
        lambda v: v["queries"][0]["relevant"][0].update(contains=" "),
        lambda v: v["queries"][0]["relevant"].append(dict(v["queries"][0]["relevant"][0])),
        lambda v: v.update(schema="other"),
    ],
)
def test_generic_set_rejects_malformed_documents(tmp_path: Path, mutate: Any) -> None:
    value = generic()
    mutate(value)
    with pytest.raises(EvaluationError):
        load_dataset(write(tmp_path, value))


def test_non_object_and_oversized_sets_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="object"):
        load_dataset(write(tmp_path, []))
    big = tmp_path / "big.json"
    big.write_bytes(b" " * (16 * 1024 * 1024 + 1))
    with pytest.raises(EvaluationError, match="size"):
        load_dataset(big)


def test_existing_code_search_sets_map_path_and_symbol_labels() -> None:
    tuning = load_dataset(ROOT / "docs" / "development" / "code-search-eval-v1.json")
    assert tuning.stacks == ("code",)
    assert tuning.corpus["repository_commit"] == "53fb8e0"
    assert len(tuning.queries) == 30
    first = tuning.queries[0]
    assert first.judgments == {"src/capicove/retrieval/fusion.py#WeightedReciprocalRankFusion": 1}
    assert any(not query.judgments for query in tuning.queries)
    held_out = load_dataset(ROOT / "docs" / "development" / "code-search-eval-heldout-v1.json")
    assert held_out.corpus["repository_commit"] == "53fb8e0"
    assert len(held_out.queries) == 29


def test_code_search_set_requires_a_commit(tmp_path: Path) -> None:
    value = json.loads((ROOT / "docs" / "development" / "code-search-eval-v1.json").read_text())
    value["corpus"] = "no revision here"
    with pytest.raises(EvaluationError, match="commit"):
        load_dataset(write(tmp_path, value))
    value["corpus"] = "commit abcdef0"
    value["queries"] = ["bad"]
    with pytest.raises(EvaluationError, match="objects"):
        load_dataset(write(tmp_path, value))


def test_video_fixture_directory_is_digest_pinned(tmp_path: Path) -> None:
    loaded = load_dataset(VIDEO)
    assert loaded.stacks == ("video",)
    assert len(loaded.queries) == 30
    first = next(query for query in loaded.queries if query.query_id == "text-keyframe-0")
    assert first.judgments == {"k1": 3, "k2": 2}
    copied = tmp_path / "fixtures"
    shutil.copytree(VIDEO, copied)
    with (copied / "queries.jsonl").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(EvaluationError, match="digest"):
        load_dataset(copied)


def test_generic_rows_keep_their_original_order(tmp_path: Path) -> None:
    value = generic()
    second = copy.deepcopy(value["queries"][0])
    second["id"] = "q0"
    value["queries"].append(second)
    loaded = load_dataset(write(tmp_path, value))
    assert [query.query_id for query in loaded.queries] == ["q1", "q0"]
