"""Strict, versioned, fingerprinted pipeline profiles (RFC-0034, Proposed)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from capicove.retrieval.pipeline_profiles import (
    BGE_RERANKER_MODEL_ID,
    KnowledgeSearchProfile,
    OfflineTextSearchProfile,
    ProfileError,
    RerankPolicy,
    UnifiedSearchProfile,
    default_knowledge_profile,
    default_offline_text_profile,
    default_unified_profile,
    load_pipeline_profile,
    parse_pipeline_profile,
)


# Changing a default changes its fingerprint; these pins make that a reviewed edit.
DEFAULT_FINGERPRINTS = {
    "kb": "4bd38a59f3651d5f50e4df4a3a016f315a86f7f7318f2e34b9ee9973510a44f0",
    "offline-text": "503dd16beb010cd665a78a5aec47e2718b9b23f371043d1ffb395720f8f6bb5f",
    "unified": "26d189d2a5cb88cf6b27d8f505d820803fc56b9b4e3a01568fc73d3387696232",
}


def test_defaults_reproduce_the_previous_constants() -> None:
    kb = default_knowledge_profile()
    assert (kb.rrf_k, kb.lexical_weight, kb.dense_weight) == (60, 1.0, 1.0)
    assert (kb.default_limit, kb.max_limit, kb.graph_context_limit) == (8, 20, 8)
    text = default_offline_text_profile()
    assert (text.rrf_k, text.lexical_candidate_count) == (60, 100)
    unified = default_unified_profile()
    assert (unified.rrf_k, unified.candidate_depth, unified.max_lexical_tokens) == (60, 100, 64)
    assert (unified.dense_weight, unified.lexical_weight) == (1.0, 1.0)
    for profile in (kb, text, unified):
        assert profile.rerank == RerankPolicy()
        assert not profile.rerank.enabled
        assert profile.rerank.model_id == BGE_RERANKER_MODEL_ID
        assert profile.rerank.on_unavailable == "fail"


@pytest.mark.parametrize(
    ("stack", "profile"),
    [
        ("kb", default_knowledge_profile()),
        ("offline-text", default_offline_text_profile()),
        ("unified", default_unified_profile()),
    ],
)
def test_fingerprints_are_pinned_and_documents_round_trip(stack: str, profile: Any) -> None:
    assert profile.fingerprint == DEFAULT_FINGERPRINTS[stack]
    document = profile.to_mapping()
    assert document["schema"] == "capicove.retrieval-pipeline-profile.v1"
    assert document["stack"] == stack
    assert parse_pipeline_profile(document, stack) == profile
    assert profile.identity() == {
        "id": profile.profile_id,
        "version": 1,
        "fingerprint": profile.fingerprint,
    }
    changed = replace(profile, rerank=RerankPolicy(enabled=True))
    assert changed.fingerprint != profile.fingerprint


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(extra=1),
        lambda d: d.pop("rrf_k"),
        lambda d: d.update(schema="other"),
        lambda d: d.update(stack="unified"),
        lambda d: d.update(version=2),
        lambda d: d.update(profile_id="Bad Id"),
        lambda d: d.update(rrf_k=0),
        lambda d: d.update(rrf_k=True),
        lambda d: d.update(rrf_k=60.0),
        lambda d: d.update(lexical_weight=-1),
        lambda d: d.update(lexical_weight=float("inf")),
        lambda d: d.update(lexical_weight="1"),
        lambda d: d.update(lexical_weight=0, dense_weight=0),
        lambda d: d.update(default_limit=21),
        lambda d: d.update(max_limit=101),
        lambda d: d.update(graph_context_limit=0),
        lambda d: d["rerank"].update(enabled=1),
        lambda d: d["rerank"].update(candidate_depth=129),
        lambda d: d["rerank"].update(on_unavailable="skip"),
        lambda d: d["rerank"].update(model_id="bad model"),
        lambda d: d["rerank"].update(binding="Bad"),
        lambda d: d["rerank"].pop("binding"),
        lambda d: d.update(rerank=[]),
    ],
)
def test_knowledge_profiles_reject_invalid_documents(mutate: Any) -> None:
    document = default_knowledge_profile().to_mapping()
    mutate(document)
    with pytest.raises(ProfileError):
        parse_pipeline_profile(document, "kb")


@pytest.mark.parametrize(
    ("factory", "values"),
    [
        (OfflineTextSearchProfile, {"lexical_candidate_count": 1001}),
        (OfflineTextSearchProfile, {"rrf_k": 10_001}),
        (UnifiedSearchProfile, {"dense_weight": 0.0}),
        (UnifiedSearchProfile, {"candidate_depth": 0}),
        (UnifiedSearchProfile, {"max_lexical_tokens": 257}),
        (KnowledgeSearchProfile, {"rerank": "on"}),
    ],
)
def test_stack_specific_bounds(factory: Any, values: dict[str, Any]) -> None:
    with pytest.raises(ProfileError):
        factory(**values)


def test_unknown_stack_and_wrong_types_are_rejected() -> None:
    with pytest.raises(ProfileError, match="stack"):
        parse_pipeline_profile({}, "code")
    document = default_unified_profile().to_mapping()
    document["rerank"]["candidate_depth"] = "20"
    with pytest.raises(ProfileError):
        parse_pipeline_profile(document, "unified")


def test_profile_files_are_bounded_and_strict(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    enabled = replace(
        default_offline_text_profile(),
        profile_id="offline-text-rerank-v1",
        rerank=RerankPolicy(enabled=True, on_unavailable="degrade"),
    )
    path.write_text(json.dumps(enabled.to_mapping()))
    assert load_pipeline_profile(path, "offline-text") == enabled
    path.write_text('{"schema": 1, "schema": 2}')
    with pytest.raises(ProfileError, match="duplicate"):
        load_pipeline_profile(path, "offline-text")
    path.write_text('{"rrf_k": NaN}')
    with pytest.raises(ProfileError, match="nonstandard"):
        load_pipeline_profile(path, "offline-text")
    path.write_bytes(b"\xff")
    with pytest.raises(ProfileError, match="JSON"):
        load_pipeline_profile(path, "offline-text")
    path.write_bytes(b" " * 65_537)
    with pytest.raises(ProfileError, match="64 KiB"):
        load_pipeline_profile(path, "offline-text")
