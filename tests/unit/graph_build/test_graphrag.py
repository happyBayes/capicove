"""Experimental GraphRAG local/global queries over a scripted native snapshot.

The snapshot is produced by the real native graph pipeline; only the generator,
tokenizer and embedder are deterministic fakes. Nothing here measures model or
extraction quality.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from capicove.contracts.graph_build import GraphSnapshotRef
from capicove.evaluation.adapters.graphrag import build_index
from capicove.evaluation.harness import EvalDataset
from capicove.graph_build.graphrag import (
    GraphQueryError,
    GraphQueryIndex,
    GraphQueryProfile,
    answer_graph_result,
    load_graph_query_profile,
    query_terms,
    validate_cited_answer,
)


def document(identity: str, text: str, entities: Any, relations: Any) -> dict[str, Any]:
    return {
        "id": identity,
        "path": f"docs/{identity}.md",
        "text": text,
        "graph": {
            "entities": [{"name": name, "type": kind} for name, kind in entities],
            "relations": [
                {"source": s, "target": t, "description": d, "quote": q} for s, t, d, q in relations
            ],
        },
    }


CORPUS = [
    document(
        "ann",
        "Ann Lee works at Orbit Labs.\n\nAnn Lee enjoys chess.\n",
        [("Ann Lee", "PERSON"), ("Orbit Labs", "CONCEPT"), ("chess", "CONCEPT")],
        [
            ("Ann Lee", "Orbit Labs", "works at", "Ann Lee works at Orbit Labs"),
            ("Ann Lee", "chess", "enjoys", "Ann Lee enjoys chess"),
        ],
    ),
    document(
        "orbit",
        "Orbit Labs is headquartered in Lyon.\n",
        [("Orbit Labs", "CONCEPT"), ("Lyon", "CONCEPT")],
        [("Orbit Labs", "Lyon", "is headquartered in", "Orbit Labs is headquartered in Lyon")],
    ),
    document(
        "lyon",
        "Lyon hosts the River Festival.\n",
        [("Lyon", "CONCEPT"), ("River Festival", "CONCEPT")],
        [("Lyon", "River Festival", "hosts", "Lyon hosts the River Festival")],
    ),
    document("other", "A note about soup.\n", [], []),
]
TEXT = {row["id"]: row["text"] for row in CORPUS}


@pytest.fixture(scope="module")
def index(tmp_path_factory: pytest.TempPathFactory) -> GraphQueryIndex:
    built, generator, receipt = build_index(_dataset(), tmp_path_factory.mktemp("graph"))
    assert receipt["status"] == "succeeded"
    assert "user" in generator.calls and "report" in generator.calls
    assert isinstance(built, GraphQueryIndex)
    return built


def _dataset() -> EvalDataset:
    from capicove.evaluation.harness import EvalQuery

    return EvalDataset(
        "graph-test-v1",
        "capicove.retrieval-eval.v1",
        ("graphrag-local", "graphrag-global"),
        "0" * 64,
        (EvalQuery("q", "c", "unused", {}),),
        {"documents": CORPUS},
        Path("graph.json"),
    )


def assert_exact(evidence: list[dict[str, Any]]) -> None:
    for citation in evidence:
        source = TEXT[citation["document_ref"]]
        assert source[citation["start_char"] : citation["end_char"]] == citation["quote"]
        assert citation["citation_id"].startswith("graphcite:")


def test_local_search_expands_bounded_neighborhood_with_exact_quotes(
    index: GraphQueryIndex,
) -> None:
    result = index.local_search("Which city is the employer of Ann Lee headquartered in?")
    assert result["status"] == "succeeded"
    assert result["mode"] == "local" and result["experimental"] is True
    assert [seed["name"] for seed in result["seeds"]] == ["Ann Lee"]
    quotes = [row["quote"] for row in result["evidence"]]
    assert quotes[:2] == ["Ann Lee works at Orbit Labs", "Orbit Labs is headquartered in Lyon"]
    assert_exact(result["evidence"])
    pin = result["pin"]
    assert pin["snapshot_id"].startswith("graph:")
    assert len(pin["manifest_sha256"]) == 64 and len(pin["graph_profile_fingerprint"]) == 64
    assert pin["query_profile"] == GraphQueryProfile().identity()
    names = {row["name"]: row for row in result["entities"]}
    assert names["Ann Lee"]["hops"] == 0
    assert names["Lyon"]["hops"] == 2
    assert "River Festival" not in names  # three hops away with max_hops=2
    bridged = [row for row in result["entities"] if row["bridged_from"]]
    assert bridged, "same-name entities from other documents are bridged"
    again = index.local_search("Which city is the employer of Ann Lee headquartered in?")
    assert again["result_sha256"] == result["result_sha256"]


def test_local_bounds_and_bridge_switch(index: GraphQueryIndex) -> None:
    question = "Where is the employer of Ann Lee headquartered?"
    one_hop = index.local_search(question, GraphQueryProfile(max_hops=1))
    quotes = [row["quote"] for row in one_hop["evidence"]]
    assert "Ann Lee works at Orbit Labs" in quotes
    assert "Orbit Labs is headquartered in Lyon" not in quotes
    two_hops = [row["quote"] for row in index.local_search(question)["evidence"]]
    assert "Orbit Labs is headquartered in Lyon" in two_hops
    unbridged = index.local_search(question, GraphQueryProfile(bridge_same_name=False))
    assert {row["quote"] for row in unbridged["evidence"]} == {
        "Ann Lee works at Orbit Labs",
        "Ann Lee enjoys chess",
    }
    capped = index.local_search(
        question, GraphQueryProfile(max_evidence=1, max_neighborhood_entities=2)
    )
    assert len(capped["evidence"]) == 1 and len(capped["entities"]) == 2
    missing = index.local_search("What is the boiling point of nitrogen?")
    assert missing["status"] == "no_entity_match" and missing["evidence"] == []


def test_global_search_ranks_reports_and_cites_sources(index: GraphQueryIndex) -> None:
    result = index.global_search("Lyon festival")
    assert result["status"] == "succeeded"
    assert result["reports"]
    top = result["reports"][0]
    assert top["term_overlap"] >= 1
    cited = {cid for statement in top["statements"] for cid in statement["citations"]}
    assert cited <= {row["citation_id"] for row in result["evidence"]} | cited
    assert result["evidence"][0]["quote"] == "Lyon hosts the River Festival"
    assert_exact(result["evidence"])
    assert index.global_search("zzz qqq")["status"] == "no_report_match"


def test_citations_and_errors(index: GraphQueryIndex) -> None:
    entity = next(iter(index.entities))
    assert index.citations_for(entity)
    with pytest.raises(GraphQueryError, match="lineage"):
        index.citations_for("missing")
    with pytest.raises(GraphQueryError, match="invalid_question"):
        index.local_search(" ")
    with pytest.raises(GraphQueryError, match="invalid_question"):
        index.global_search("x" * 4001)


def test_answers_are_opt_in_and_quote_validated(index: GraphQueryIndex) -> None:
    result = index.local_search("Where is Orbit Labs headquartered?")
    assert answer_graph_result(result, None)["status"] == "evidence_only"
    first = result["evidence"][0]

    def answerer(question: str, evidence: list[dict[str, Any]], context: Any) -> dict[str, Any]:
        assert evidence[0]["evidence_id"] == first["citation_id"]
        assert context
        return {
            "claims": [
                {
                    "text": "It is in Lyon.",
                    "citations": [{"evidence_id": first["citation_id"], "quote": "in Lyon"}],
                }
            ]
        }

    answered = answer_graph_result(result, answerer)
    assert answered["status"] == "answered"
    assert answered["pin"] == result["pin"]
    assert answer_graph_result(result, lambda *_: {"claims": []})["status"] == (
        "insufficient_evidence"
    )
    empty = index.local_search("What about nitrogen?")
    assert answer_graph_result(empty, answerer)["status"] == "no_evidence"
    with pytest.raises(GraphQueryError, match="invalid_answer"):
        answer_graph_result(
            result,
            lambda *_: {
                "claims": [
                    {
                        "text": "x",
                        "citations": [{"evidence_id": first["citation_id"], "quote": "Paris"}],
                    }
                ]
            },
        )


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"claims": "x"},
        {"claims": [{"text": "x"}]},
        {"claims": [{"text": " ", "citations": [{"evidence_id": "e", "quote": "q"}]}]},
        {"claims": [{"text": "x", "citations": []}]},
        {"claims": [{"text": "x", "citations": [{"evidence_id": "nope", "quote": "q"}]}]},
        {"claims": [{"text": "x", "citations": [{"evidence_id": "e", "quote": " "}]}]},
        {"claims": [{"text": "x", "citations": ["e"]}]},
    ],
)
def test_validate_cited_answer_rejects_unsupported_payloads(payload: Any) -> None:
    with pytest.raises(GraphQueryError):
        validate_cited_answer(payload, [{"evidence_id": "e", "quote": "quoted text"}])


def test_snapshot_tampering_and_ref_mismatch_fail_closed(
    index: GraphQueryIndex, tmp_path: Path
) -> None:
    built, _, _ = build_index(_dataset(), tmp_path)
    from capicove.evaluation.adapters.graphrag import _OpenGate

    with pytest.raises(GraphQueryError, match="digest"):
        GraphQueryIndex.open(
            _catalog(tmp_path),
            _workspace(tmp_path),
            _OpenGate(built.scope),
            _Sources(),
            GraphSnapshotRef(built.pin["snapshot_id"], "0" * 64),
            built.scope,
        )


def _catalog(root: Path) -> Any:
    from capicove.evaluation.adapters.graphrag import _OpenGate
    from capicove.graph_build.native import open_native_graph

    scope = _scope()
    return open_native_graph(root / "graph-evaluation", gate=_OpenGate(scope))[0]


def _workspace(root: Path) -> Any:
    from capicove.processing import LocalProcessingWorkspace

    return LocalProcessingWorkspace(root / "graph-evaluation" / "processing")


def _scope() -> Any:
    from capicove.contracts.graph_build import GraphScope

    return GraphScope("evaluation", "graphrag")


class _Sources:
    def read(self, document_ref: str, scope: Any) -> Any:
        raise AssertionError("sources must not be read before the manifest verifies")

    def require_access(self, refs: tuple[str, ...], scope: Any) -> None:
        return None


def test_profile_documents_are_strict(tmp_path: Path) -> None:
    profile = GraphQueryProfile(max_hops=3)
    path = tmp_path / "profile.json"
    import json

    path.write_text(json.dumps(profile.to_mapping()))
    assert load_graph_query_profile(path) == profile
    assert profile.fingerprint != GraphQueryProfile().fingerprint
    for raw in ('{"a": 1, "a": 2}', "[", json.dumps({**profile.to_mapping(), "extra": 1})):
        path.write_text(raw)
        with pytest.raises(GraphQueryError):
            load_graph_query_profile(path)
    path.write_text(json.dumps({**profile.to_mapping(), "schema": "x"}))
    with pytest.raises(GraphQueryError, match="schema"):
        load_graph_query_profile(path)
    path.write_text(json.dumps({**profile.to_mapping(), "max_hops": "2"}))
    with pytest.raises(GraphQueryError):
        load_graph_query_profile(path)
    path.write_bytes(b" " * 65_537)
    with pytest.raises(GraphQueryError, match="64 KiB"):
        load_graph_query_profile(path)
    for values in (
        {"profile_id": "Bad"},
        {"version": 2},
        {"min_entity_match": 0},
        {"min_entity_match": True},
        {"max_hops": 5},
        {"bridge_same_name": 1},
    ):
        with pytest.raises(GraphQueryError):
            replace(GraphQueryProfile(), **values)


def test_query_terms_strip_stopwords_and_suffixes() -> None:
    assert query_terms("Where were the studies funded?") == {"study", "fund"}
    assert query_terms("Funding hosts 東京") == {"fund", "host", "東", "京"}
