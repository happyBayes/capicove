"""Optional post-fusion reranking in KB, offline text and unified text search.

Only deterministic fakes are used. Real BGE reranker quality is not validated
here (real-model evaluation: not run).
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from capicove.contracts.retrieval import (
    SearchRequest,
    SearchStatus,
    SearchWarningCode,
    TargetType,
    TextQuery,
)
from capicove.evaluation.adapters._common import host_config, invocation_context, render_image
from capicove.evaluation.fakes import (
    FAKE_RERANKER_MODEL_ID,
    DeterministicFakeReranker,
    fake_text_encoder,
)
from capicove.retrieval.pipeline_profiles import (
    KnowledgeSearchProfile,
    OfflineTextSearchProfile,
    RerankPolicy,
    UnifiedSearchProfile,
)
from capicove.runtime.native_knowledge import KnowledgeBase, KnowledgeConfiguration


FAKE = RerankPolicy(enabled=True, model_id=FAKE_RERANKER_MODEL_ID)
DEGRADE = replace(FAKE, on_unavailable="degrade")
DOCUMENTS = {
    "a.md": "Bicycle chain wear notes.",
    "b.md": "Bicycle tyre pressure and chain lubrication schedule for the chain.",
    "c.md": "Garden tomato notes.",
}


def knowledge(tmp_path: Path, **keywords: Any) -> KnowledgeBase:
    source = tmp_path / "docs"
    source.mkdir(parents=True, exist_ok=True)
    for name, text in DOCUMENTS.items():
        (source / name).write_text(text)
    config = KnowledgeConfiguration.from_mapping(
        {
            "schema_version": 1,
            "workspace_id": "test",
            "storage_root": str(tmp_path / "store"),
            "source_roots": {"docs": str(source)},
            "models": None,
            "generation": None,
            "audio": None,
            "graph": False,
            "network_enabled": False,
        }
    )
    kb = KnowledgeBase(config, **keywords)
    for name in DOCUMENTS:
        kb.update("docs", name)
    return kb


def sources(result: dict[str, Any]) -> list[str]:
    return [row["source"] for row in result["evidence"]]


def test_kb_default_profile_matches_explicit_default(tmp_path: Path) -> None:
    with knowledge(tmp_path) as kb:
        default = kb.search("bicycle chain pressure")
        assert "rerank" not in default and "profile" not in default
    with knowledge(tmp_path / "explicit", profile=KnowledgeSearchProfile()) as kb:
        explicit = kb.search("bicycle chain pressure")
    assert sources(explicit) == sources(default)


def test_kb_rerank_applies_fails_or_degrades(tmp_path: Path) -> None:
    profile = KnowledgeSearchProfile(profile_id="kb-fake-v1", rerank=FAKE)
    with knowledge(tmp_path / "plain") as kb:
        fused = sources(kb.search("tyre pressure lubrication"))
    with knowledge(
        tmp_path / "applied", profile=profile, reranker=DeterministicFakeReranker()
    ) as kb:
        result = kb.search("tyre pressure lubrication")
        assert result["rerank"]["status"] == "applied"
        assert result["rerank"]["model_id"] == FAKE_RERANKER_MODEL_ID
        assert result["profile"] == profile.identity()
        assert sources(result)[0] == "docs:b.md"
        assert set(sources(result)) == set(fused)
    with knowledge(tmp_path / "fail", profile=profile) as kb:
        with pytest.raises(ValueError, match="reranker_unavailable"):
            kb.search("tyre pressure")
    with knowledge(tmp_path / "degrade", profile=replace(profile, rerank=DEGRADE)) as kb:
        degraded = kb.search("tyre pressure lubrication")
        assert degraded["rerank"]["status"] == "degraded"
        assert degraded["rerank"]["reason"] == "reranker_unavailable"
        assert sources(degraded) == fused


def test_kb_profile_limits_and_injected_dense_lane(tmp_path: Path) -> None:
    profile = KnowledgeSearchProfile(profile_id="kb-small-v1", default_limit=1, max_limit=2)
    with knowledge(tmp_path, profile=profile, text_encoder=fake_text_encoder) as kb:
        result = kb.search("bicycle")
        assert len(result["evidence"]) == 1
        assert result["retrieval"] == "hybrid"
        assert kb.status()["semantic"] is True
        assert len(kb.search("bicycle", limit=2)["evidence"]) == 2
        with pytest.raises(ValueError, match="invalid_search_limit"):
            kb.search("bicycle", limit=3)
    with pytest.raises(TypeError):
        KnowledgeBase(kb.config, profile=OfflineTextSearchProfile())  # type: ignore[arg-type]


def test_kb_rerank_tries_configured_models_then_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = KnowledgeSearchProfile(profile_id="kb-fake-v1", rerank=FAKE)
    kb = knowledge(tmp_path)
    configured = replace(kb.config, models=tmp_path / "missing-models.json")
    kb.__exit__(None, None, None)
    monkeypatch.setattr(
        "capicove.runtime.knowledge_models.encode_texts",
        lambda path, texts, role: fake_text_encoder(texts, role),
    )
    with KnowledgeBase(configured, profile=profile) as reloaded:
        reloaded.rebuild()
        with pytest.raises(ValueError, match="reranker_unavailable"):
            reloaded.search("bicycle")


def test_kb_cli_accepts_a_profile(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import json

    from capicove.cli.app import main

    with knowledge(tmp_path) as kb:
        config_path = tmp_path / "knowledge.json"
        config_path.write_text(json.dumps(kb.config.to_mapping()))
        config_path.chmod(0o600)
    profile = KnowledgeSearchProfile(profile_id="kb-fake-v1", rerank=DEGRADE)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile.to_mapping()))
    assert (
        main(
            ["kb", "search", "--config", str(config_path), "--profile", str(profile_path), "chain"]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["rerank"]["status"] == "degraded"
    profile_path.write_text("{}")
    assert (
        main(
            ["kb", "search", "--config", str(config_path), "--profile", str(profile_path), "chain"]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().out)["code"] == "invalid_pipeline_profile"
    assert (
        main(
            [
                "kb",
                "search",
                "--config",
                str(config_path),
                "--profile",
                str(tmp_path / "no.json"),
                "x",
            ]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().out)["code"] == "pipeline_profile_unreadable"


def offline_runtime(tmp_path: Path, **keywords: Any) -> Any:
    from capicove.runtime import local_retrieval

    workspace = (tmp_path / "workspace").resolve()
    corpus = (tmp_path / "corpus").resolve()
    workspace.mkdir(mode=0o700, parents=True)
    corpus.mkdir()
    for name, text in DOCUMENTS.items():
        (corpus / name).write_text(text)
    config = host_config(
        workspace,
        {
            "retrieval": {
                "schema_version": 1,
                "source_roots": {"corpus": str(corpus)},
                "index_root": str(workspace / "index"),
                "corpus_ref": "corpus:test",
                "retrieval_profile_ref": "local-default",
                "max_source_bytes": 4096,
            }
        },
        ["capicove.get_evidence", "capicove.search"],
    )
    for name in DOCUMENTS:
        local_retrieval.import_source(config, alias="corpus", relative_path=name)
    local_retrieval.build_index(config)
    return local_retrieval.open_local_retrieval(config, **keywords)


def offline_search(runtime: Any, text: str) -> Any:
    return runtime.search_handler(invocation_context(), {"query": text, "top_k": 3})


def test_offline_text_rerank_applies_fails_or_degrades(tmp_path: Path) -> None:
    plain = offline_search(offline_runtime(tmp_path / "plain"), "tyre lubrication bicycle")
    profile = OfflineTextSearchProfile(profile_id="offline-text-fake-v1", rerank=FAKE)
    applied = offline_search(
        offline_runtime(
            tmp_path / "applied", pipeline_profile=profile, reranker=DeterministicFakeReranker()
        ),
        "tyre lubrication bicycle",
    )
    assert applied.status.value == "succeeded"
    assert {hit["id"] for hit in applied.result["hits"]} == {
        hit["id"] for hit in plain.result["hits"]
    }
    assert not applied.result["degraded"]
    failed = offline_search(
        offline_runtime(tmp_path / "fail", pipeline_profile=profile), "tyre lubrication"
    )
    assert failed.status.value == "failed"
    assert failed.error.code.value == "unavailable"
    degraded = offline_search(
        offline_runtime(tmp_path / "degrade", pipeline_profile=replace(profile, rerank=DEGRADE)),
        "tyre lubrication bicycle",
    )
    assert degraded.result["degraded"]
    assert "retrieval_lane_unavailable" in degraded.result["warning_codes"]
    with pytest.raises(TypeError):
        offline_runtime(tmp_path / "wrong", pipeline_profile=KnowledgeSearchProfile())


def test_offline_text_rerank_text_source_failure_respects_policy(tmp_path: Path) -> None:
    from capicove.retrieval.text_search import TextSearchError, TextSearchService

    service = TextSearchService.__new__(TextSearchService)
    service._rerank_policy = FAKE
    service._reranker = DeterministicFakeReranker()

    def broken(generation: Any, candidates: Any) -> tuple[str, ...]:
        raise OSError("gone")

    service._rerank_texts = broken
    with pytest.raises(TextSearchError, match="provider_unavailable"):
        service._rerank(TextQuery("x"), (), object())
    service._rerank_policy = DEGRADE
    assert service._rerank(TextQuery("x"), (), object()) == ((), True)
    service._rerank_texts = lambda generation, candidates: ("extra",)
    assert service._rerank(TextQuery("x"), (), object()) == ((), True)


FFMPEG, FFPROBE = shutil.which("ffmpeg"), shutil.which("ffprobe")


@pytest.mark.skipif(FFMPEG is None or FFPROBE is None, reason="requires FFmpeg paths")
def test_unified_rerank_reorders_text_chunks_only(tmp_path: Path) -> None:
    from capicove.evaluation.adapters.unified import fake_model_factory
    from capicove.runtime import unified_retrieval

    workspace = (tmp_path / "workspace").resolve()
    media = workspace / "media"
    media.mkdir(parents=True)
    workspace.chmod(0o700)
    for name, text in DOCUMENTS.items():
        (media / name).write_text(text)
    (media / "red.png").write_bytes(render_image({"color": "red", "variant": 1, "size": 32}))
    config = host_config(
        workspace,
        {
            "multimodal": {
                "schema_version": 1,
                "profile": "unified-local-v1",
                "source_roots": {"media": str(media)},
                "index_root": str(workspace / "index"),
                "staging_root": str(workspace / "staging"),
                "corpus_ref": "corpus:test",
                "model_config": None,
                "model_binding": "unified",
                "ffmpeg": FFMPEG,
                "ffprobe": FFPROBE,
                "max_source_bytes": 1024 * 1024,
            }
        },
        ["capicove.search.multimodal"],
    )
    models = fake_model_factory(tmp_path)
    try:
        for name in (*DOCUMENTS, "red.png"):
            unified_retrieval.import_media(config, "media", name)
        built = unified_retrieval.build_generation(config, models=models)
        unified_retrieval.activate_generation(config, built["generation_id"])
        request = SearchRequest(
            TextQuery("red image tyre lubrication"),
            frozenset({TargetType.TEXT_CHUNK, TargetType.IMAGE}),
            top_k=10,
            retrieval_profile_ref="unified-local-v1",
        )
        plain = unified_retrieval.open_unified_retrieval(config, models=models).search(request)
        profile = UnifiedSearchProfile(profile_id="unified-fake-v1", rerank=FAKE)
        applied = unified_retrieval.open_unified_retrieval(
            config, models=models, pipeline_profile=profile, reranker=DeterministicFakeReranker()
        ).search(request)
        assert applied.status is SearchStatus.SUCCEEDED
        positions = [
            index for index, hit in enumerate(plain.hits) if hit.target_type is TargetType.IMAGE
        ]
        assert positions == [
            index for index, hit in enumerate(applied.hits) if hit.target_type is TargetType.IMAGE
        ]
        assert {hit.id for hit in applied.hits} == {hit.id for hit in plain.hits}
        assert not applied.degraded
        failed = unified_retrieval.open_unified_retrieval(
            config, models=models, pipeline_profile=profile
        ).search(request)
        assert failed.status is SearchStatus.FAILED
        assert failed.failure_code is not None
        assert failed.failure_code.value == "provider_unavailable"
        degraded = unified_retrieval.open_unified_retrieval(
            config, models=models, pipeline_profile=replace(profile, rerank=DEGRADE)
        ).search(request)
        assert degraded.degraded
        assert SearchWarningCode.RETRIEVAL_LANE_UNAVAILABLE in degraded.warning_codes
        assert [hit.id for hit in degraded.hits] == [hit.id for hit in plain.hits]
        with pytest.raises(TypeError):
            unified_retrieval.open_unified_retrieval(
                config,
                models=models,
                pipeline_profile=KnowledgeSearchProfile(),  # type: ignore[arg-type]
            )
    finally:
        models.close()
