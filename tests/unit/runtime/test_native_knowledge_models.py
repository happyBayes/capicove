"""Model composition, graph publication, and transport security regressions."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from capicove.runtime.native_knowledge import KnowledgeBase
from tests.unit.runtime.test_native_knowledge import configured


def test_hybrid_retrieval_and_embedding_space_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(configured(tmp_path), models=tmp_path / "models.json")
    (config.source_roots["docs"] / "one.md").write_text("The sun is a star.")
    space = "space-one"

    def encode(path: Path, texts: list[str], role: str) -> Any:
        return space, [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr("capicove.runtime.knowledge_models.encode_texts", encode)
    with KnowledgeBase(config) as kb:
        kb.update("docs", "one.md")
        assert kb.search("solar body")["evidence"]
        space = "space-two"
        with pytest.raises(ValueError, match="embedding_profile_changed"):
            kb.search("solar body")
        kb.rebuild()
        assert kb.search("solar body")["retrieval"] == "hybrid"


def test_partial_graph_never_activates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = replace(
        configured(tmp_path),
        graph=True,
        models=tmp_path / "models.json",
        generation=tmp_path / "generation.json",
        network_enabled=True,
    )
    (config.source_roots["docs"] / "one.md").write_text("The sun is a star.")
    monkeypatch.setattr(
        "capicove.runtime.knowledge_models.encode_texts",
        lambda path, texts, role: ("space", [[1.0] for _ in texts]),
    )
    monkeypatch.setattr(
        "capicove.runtime.knowledge_graph.build_graph",
        lambda *args: {
            "receipt": {"status": "partial", "issue_codes": ["extraction_failed"]},
            "context": [],
        },
    )
    with KnowledgeBase(config) as kb:
        assert kb.update("docs", "one.md")["status"] == "partial"
        assert kb.status()["sources"] == 0
        assert kb.status()["last_run"]["status"] == "partial"


def test_generation_request_is_canonical_and_excludes_host_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from capicove.graph_build.inference import RawGeneration
    from capicove.runtime.knowledge_answer import generate_answer
    from tests.unit.graph_build.test_remote_generation import profile_at

    profile = profile_at(tmp_path)
    monkeypatch.setattr(
        "capicove.graph_build.remote_generation.load_remote_generation_profile",
        lambda path: profile,
    )
    captured = []

    class Generator:
        def __init__(self, profile: Any) -> None:
            pass

        def generate(self, request: Any, *, control: Any) -> RawGeneration:
            captured.append(request)
            return RawGeneration(b'{"claims":[]}', "stop", 1, 1)

    monkeypatch.setattr(
        "capicove.graph_build.remote_generation.OpenAICompatibleGenerator", Generator
    )
    result = generate_answer(
        tmp_path / "config",
        "question",
        [
            {
                "evidence_id": "kb:test",
                "quote": "original",
                "locator": {"kind": "text"},
                "path": "/private/source",
                "lineage": {"secret": "never-send"},  # pragma: allowlist secret
            }
        ],
        [],
    )
    assert result == {"claims": []}
    assert b"/private" not in captured[0].canonical_payload
    assert b"never-send" not in captured[0].canonical_payload


def test_real_graph_composition_with_authored_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from capicove.contracts import (
        DistanceMetric,
        EmbeddingSpaceRef,
        QueryModality,
        TargetType,
        VectorNormalization,
    )
    from capicove.graph_build.native import build_native_graph
    from capicove.graph_build.representations import text_representation
    from tests.e2e.test_document_graph_pipeline import (
        FIXTURE,
        SCOPE,
        CharacterTokenizer,
        Gate,
        LocalEmbeddingProvider,
        ScriptedProvider,
        WireTokenizer,
    )

    fixture = json.loads(FIXTURE.read_text())
    from capicove.graph_build.inference import RawGeneration

    class NativeProvider(ScriptedProvider):
        def generate(self, request: Any, *, control: Any) -> RawGeneration:
            if request.role == "report":
                task = json.loads(request.input_segments[0][2])
                refs = [row["ref"] for row in task["sources"] + task["children"]]
                value = {
                    "title": "Native evidence report",
                    "statements": [
                        {
                            "text": "The supplied evidence supports a reviewable source claim.",
                            "references": refs[:1],
                        }
                    ],
                }
                return RawGeneration(json.dumps(value).encode(), "stop", 100, 100)
            response = super().generate(request, control=control)
            if request.role != "user":
                return response
            payload = json.loads(response.payload)
            names = {item["local_id"]: item["name"] for item in payload["entities"]}
            native = {
                "entities": [
                    {key: item[key] for key in ("name", "entity_type", "description")}
                    | {"quote": request.input_segments[0][2]}
                    for item in payload["entities"]
                ],
                "relationships": [
                    {
                        "source": names[item["source_local_id"]],
                        "target": names[item["target_local_id"]],
                        "description": item["description"],
                        "quote": request.input_segments[0][2],
                        "polarity": "positive",
                    }
                    for item in payload["relationships"]
                ],
            }
            return RawGeneration(json.dumps(native).encode(), "stop", 100, 100)

    provider = NativeProvider(fixture)

    class Tokenizer(CharacterTokenizer, WireTokenizer):
        pass

    monkeypatch.setattr(
        "capicove.graph_build.native.OpenAICompatibleTokenizer", lambda profile: Tokenizer()
    )
    monkeypatch.setattr(
        "capicove.graph_build.native.OpenAICompatibleGenerator", lambda profile: provider
    )
    profile = SimpleNamespace(
        provider_id=provider.provider_id,
        model=provider.model_id,
        model_revision=provider.model_revision,
        context_limit=65536,
    )
    space = EmbeddingSpaceRef(
        "test",
        "local",
        "revision",
        "preprocess",
        16,
        VectorNormalization.L2,
        DistanceMetric.COSINE,
        frozenset({QueryModality.TEXT}),
        frozenset({TargetType.TEXT_CHUNK}),
    )
    result = build_native_graph(
        tmp_path / "graph",
        scope=SCOPE,
        representations=tuple(
            text_representation(row["id"], row["text"], SCOPE) for row in fixture["documents"]
        ),
        gate=Gate(),
        generation=profile,  # type: ignore[arg-type]
        embedder=LocalEmbeddingProvider(),
        space=space,
    )
    assert result["receipt"]["status"] == "succeeded"
    assert {row["kind"] for row in result["context"]} == {"resolved_entities", "community_reports"}


def test_mcp_host_queries_and_rejects_forged_actor(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta

    from capicove.contracts import InvocationContext, ToolStatus
    from capicove.runtime.knowledge_host import host_configuration
    from capicove.runtime.mcp_config import load_mcp_host_config
    from capicove.runtime.mcp_host import compose_mcp_host

    config = configured(tmp_path)
    config_path = tmp_path / "knowledge.json"
    config_path.write_text(json.dumps(config.to_mapping()))
    config_path.chmod(0o600)
    host_path = tmp_path / "host.json"
    host_path.write_text(json.dumps(host_configuration(config, config_path)))
    host_path.chmod(0o600)
    (config.source_roots["docs"] / "one.md").write_text("Capicove stores evidence.")
    with KnowledgeBase(config) as kb:
        kb.update("docs", "one.md")
    host = load_mcp_host_config(host_path)
    with compose_mcp_host(host) as bindings:
        assert len(bindings) == 4
        search = next(item for item in bindings if item.spec.tool_id.endswith(".search"))
        context = InvocationContext(
            "req",
            "trace",
            config.workspace_id,
            "knowledge-owner",
            "local",
            datetime.now(UTC) + timedelta(seconds=30),
            frozenset({"evidence:read", "evidence:search"}),
        )
        assert search.handler(context, {"question": "evidence"}).status == ToolStatus.SUCCEEDED
        forged = replace(context, actor_id="other-user")
        assert search.handler(forged, {"question": "evidence"}).status == ToolStatus.REJECTED
        assert (
            search.handler(context, {"question": "evidence", "root": "/etc"}).status
            == ToolStatus.REJECTED
        )


def test_expired_request_never_calls_generator(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta

    config = configured(tmp_path)
    (config.source_roots["docs"] / "one.md").write_text("Evidence.")
    called = []

    def answerer(*args: Any) -> Any:
        called.append(True)
        return {"claims": []}

    with KnowledgeBase(config, answerer=answerer) as kb:
        kb.update("docs", "one.md")
        kb.deadline = datetime.now(UTC) - timedelta(seconds=1)
        with pytest.raises(TimeoutError):
            kb.ask("Evidence")
    assert not called


def test_scanned_pdf_uses_page_ocr_without_weakening_parser_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image

    from capicove.runtime.knowledge_prepare import prepare_source
    from capicove.runtime.media_handoff import MediaIntake

    config = replace(configured(tmp_path), models=tmp_path / "models.json")
    path = config.source_roots["docs"] / "scan.pdf"
    Image.new("RGB", (200, 100), "white").save(path, "PDF")
    observed = []

    def scanned(config: Any, source: Any) -> Any:
        observed.append(source.sha256)
        return [{"text": "Scanned evidence", "kind": "pdf", "page_index": 0}], {
            "adapter": "pdf-page-ocr"
        }

    monkeypatch.setattr("capicove.runtime.knowledge_prepare.prepare_scanned_pdf", scanned)
    intake = MediaIntake(
        read_roots=(path.parent,),
        staging_root=tmp_path / "stage",
        max_input_bytes=100000,
        accepts=lambda p, head: True,
    )
    with intake.stage((path,)) as staged:
        segments, lineage = prepare_source(config, staged[0], tmp_path / "store")
    assert observed and segments[0]["page_index"] == 0
    assert lineage["adapter"] == "pdf-page-ocr"


def test_query_host_opens_storage_read_only(tmp_path: Path) -> None:
    import sqlite3

    config = configured(tmp_path)
    (config.source_roots["docs"] / "one.md").write_text("Read-only evidence.")
    with KnowledgeBase(config) as kb:
        kb.update("docs", "one.md")
    with KnowledgeBase(config, read_only=True) as kb:
        assert kb.search("evidence")["evidence"]
        with pytest.raises(sqlite3.OperationalError):
            kb.forget("docs", "one.md")
    with KnowledgeBase(config) as kb:
        assert kb.status()["sources"] == 1


def test_generation_profile_change_is_rejected_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from capicove.runtime.knowledge_answer import generate_answer
    from tests.unit.graph_build.test_remote_generation import profile_at

    profile = profile_at(tmp_path)
    monkeypatch.setattr(
        "capicove.graph_build.remote_generation.load_remote_generation_profile",
        lambda path: profile,
    )
    with pytest.raises(ValueError, match="configuration_changed"):
        generate_answer(
            tmp_path / "profile.json", "Question", [], [], expected_provider="different-profile"
        )
