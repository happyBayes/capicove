"""Experimental KB GraphRAG queries over a native snapshot built with scripted fakes.

The knowledge base, gate, source fencing, graph pipeline and activation are
real; the generator, tokenizer and embedder are deterministic fakes.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from capicove.evaluation.fakes import fake_text_encoder
from capicove.evaluation.graph_fakes import (
    HashedEmbedder,
    ScriptedGenerationProfile,
    ScriptedGraphGenerator,
    ScriptedTokenizer,
    hashed_text_space,
)
from capicove.graph_build.graphrag import GraphQueryProfile
from capicove.runtime import knowledge_graph
from capicove.runtime.native_knowledge import KnowledgeBase, KnowledgeConfiguration


ENTITIES = [
    {"name": "Ann Lee", "type": "PERSON"},
    {"name": "Orbit Labs", "type": "CONCEPT"},
    {"name": "Lyon", "type": "CONCEPT"},
]
RELATIONS = [
    {
        "source": "Ann Lee",
        "target": "Orbit Labs",
        "description": "works at",
        "quote": "Ann Lee works at Orbit Labs",
    },
    {
        "source": "Orbit Labs",
        "target": "Lyon",
        "description": "is headquartered in",
        "quote": "Orbit Labs is headquartered in Lyon",
    },
]


def configured(tmp_path: Path, *, graph: bool = True) -> KnowledgeConfiguration:
    source = tmp_path / "docs"
    source.mkdir(parents=True)
    (source / "ann.md").write_text("Ann Lee works at Orbit Labs.")
    (source / "orbit.md").write_text("Orbit Labs is headquartered in Lyon.")
    return KnowledgeConfiguration.from_mapping(
        {
            "schema_version": 1,
            "workspace_id": "test",
            "storage_root": str(tmp_path / "store"),
            "source_roots": {"docs": str(source)},
            "models": str(tmp_path / "models.json") if graph else None,
            "generation": str(tmp_path / "generation.json") if graph else None,
            "audio": None,
            "graph": graph,
            "network_enabled": graph,
        }
    )


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "capicove.runtime.knowledge_models.encode_texts",
        lambda path, texts, role: fake_text_encoder(texts, role),
    )

    def build(kb: KnowledgeBase, records: list[dict[str, Any]], revision: int) -> Any:
        return knowledge_graph.build_graph_with(
            kb,
            records,
            revision,
            generation=ScriptedGenerationProfile(),
            embedder=HashedEmbedder(),
            space=hashed_text_space(),
            generator=ScriptedGraphGenerator(ENTITIES, RELATIONS),
            tokenizer=ScriptedTokenizer(),
        )

    monkeypatch.setattr(knowledge_graph, "build_graph", build)


def build_kb(config: KnowledgeConfiguration, **keywords: Any) -> KnowledgeBase:
    kb = KnowledgeBase(config, **keywords)
    for name in ("ann.md", "orbit.md"):
        assert kb.update("docs", name)["status"] == "succeeded"
    return kb


def test_local_and_global_graph_queries_cite_original_sources(
    tmp_path: Path, scripted: None
) -> None:
    with build_kb(configured(tmp_path)) as kb:
        assert kb.status()["graph_status"] == "active"
        local = kb.graph_query("Where is the employer of Ann Lee headquartered?")
        assert local["status"] == "succeeded"
        quotes = [row["quote"] for row in local["evidence"]]
        assert quotes[:2] == ["Ann Lee works at Orbit Labs", "Orbit Labs is headquartered in Lyon"]
        for row in local["evidence"]:
            evidence = kb.store.records()
            assert any(row["document_ref"] == record["source"] for record in evidence)
        assert local["pin"]["snapshot_id"] == kb.status()["graph"]["snapshot_id"]
        overview = kb.graph_query("Lyon headquartered", mode="global")
        assert overview["mode"] == "global" and overview["evidence"]
        bounded = kb.graph_query(
            "Where is the employer of Ann Lee headquartered?",
            profile=GraphQueryProfile(max_hops=1),
        )
        assert "Orbit Labs is headquartered in Lyon" not in [
            row["quote"] for row in bounded["evidence"]
        ]
        with pytest.raises(ValueError, match="invalid_graph_query_mode"):
            kb.graph_query("Ann Lee", mode="drift")
        with pytest.raises(ValueError, match="invalid_question"):
            kb.graph_query(" ")


def test_answers_require_opt_in_and_exact_quotes(tmp_path: Path, scripted: None) -> None:
    def answerer(question: str, evidence: list[dict[str, Any]], context: Any) -> dict[str, Any]:
        first = evidence[0]
        return {
            "claims": [
                {
                    "text": "Ann Lee works at Orbit Labs.",
                    "citations": [{"evidence_id": first["evidence_id"], "quote": "Orbit Labs"}],
                }
            ]
        }

    with build_kb(configured(tmp_path), answerer=answerer) as kb:
        unanswered = kb.graph_query("Who employs Ann Lee?")
        assert "answer" not in unanswered
        answered = kb.graph_query("Who employs Ann Lee?", answer=True)
        assert answered["answer"]["status"] == "answered"
        assert answered["answer"]["pin"] == answered["pin"]


def test_source_change_and_missing_graph_fail_closed(tmp_path: Path, scripted: None) -> None:
    config = configured(tmp_path)
    with build_kb(config) as kb:
        (config.source_roots["docs"] / "ann.md").write_text("Changed.")
        with pytest.raises(ValueError, match="source_changed"):
            kb.graph_query("Ann Lee")
    plain = configured(tmp_path / "plain", graph=False)
    with build_kb(plain) as kb:
        with pytest.raises(ValueError, match="graph_unavailable"):
            kb.graph_query("Ann Lee")


def test_unrecorded_root_or_snapshot_drift_requires_rebuild(tmp_path: Path, scripted: None) -> None:
    with build_kb(configured(tmp_path)) as kb:
        revision, records, graph = kb._view()
        with pytest.raises(ValueError, match="graph_root_unrecorded"):
            knowledge_graph.query_graph(
                kb, {**graph, "graph_root": "../x"}, records, revision, "Ann", mode="local"
            )
        drifted = {**graph, "receipt": {**graph["receipt"], "snapshot_id": "graph:" + "0" * 64}}
        with pytest.raises(ValueError, match="graph_snapshot_changed"):
            knowledge_graph.query_graph(kb, drifted, records, revision, "Ann", mode="local")


def test_cli_graph_query(
    tmp_path: Path, scripted: None, capsys: pytest.CaptureFixture[str]
) -> None:
    from capicove.cli.app import main

    config = configured(tmp_path)
    with build_kb(config):
        pass
    path = tmp_path / "knowledge.json"
    path.write_text(json.dumps(config.to_mapping()))
    path.chmod(0o600)
    assert main(["kb", "graph-query", "--config", str(path), "Ann Lee employer"]) == 0
    local = json.loads(capsys.readouterr().out)
    assert local["schema"] == "capicove.graphrag-result.v1"
    profile_path = tmp_path / "graph-profile.json"
    profile_path.write_text(json.dumps(replace(GraphQueryProfile(), max_reports=1).to_mapping()))
    assert (
        main(
            [
                "kb",
                "graph-query",
                "--config",
                str(path),
                "--mode",
                "global",
                "--profile",
                str(profile_path),
                "Lyon",
            ]
        )
        == 0
    )
    assert len(json.loads(capsys.readouterr().out)["reports"]) <= 1
    profile_path.write_text("{}")
    assert (
        main(["kb", "graph-query", "--config", str(path), "--profile", str(profile_path), "x"]) == 1
    )
    assert json.loads(capsys.readouterr().out)["code"] == "invalid_graph_query_profile"
    missing = tmp_path / "missing.json"
    assert main(["kb", "graph-query", "--config", str(path), "--profile", str(missing), "x"]) == 1
    assert json.loads(capsys.readouterr().out)["code"] == "graph_query_profile_unreadable"
