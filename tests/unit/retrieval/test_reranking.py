"""Typed, deterministic reranking without optional model runtimes."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from capicove.model import ModelCapability, ModelConfiguration, ModelFactory, ProviderDefinition
from capicove.model.registry import ProviderRegistry
from capicove.retrieval.reranking import RerankResult, rank_scores, validate_rerank_request


def test_scores_keep_original_indices_and_stable_ties() -> None:
    result = rank_scores((0.5, 2.0, 2.0, -3.0), top_n=3)
    assert [r.index for r in result] == [1, 2, 0]
    assert [r.relevance_score for r in result] == [2.0, 2.0, 0.5]


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_nonfinite_scores_are_rejected(score: float) -> None:
    with pytest.raises(ValueError, match="invalid_reranker_output"):
        rank_scores((score,), top_n=1)


@pytest.mark.parametrize(
    "query,documents,top_n",
    [
        ("", ("a",), None),
        ("x", (), None),
        ("x", ("",), None),
        ("x", ("a",) * 129, None),
        ("x", ("a",), 0),
        ("x", ("a",), True),
        ("x", ("a",), 2),
        ("\ud800", ("a",), None),
        ("x", ("\ud800",), None),
        ("x", ("a", 1), None),
    ],
)
def test_invalid_inputs_are_rejected(query: Any, documents: Any, top_n: Any) -> None:
    with pytest.raises(ValueError, match="invalid_rerank_request"):
        validate_rerank_request(query, documents, top_n)


def test_candidate_bytes_are_bounded() -> None:
    with pytest.raises(ValueError, match="invalid_rerank_request"):
        validate_rerank_request("query", ("a" * (2 * 1024 * 1024),), None)
    assert validate_rerank_request("query", ("a", "a"), None) == 2


def test_factory_supports_a_typed_reranker_binding(tmp_path: Path) -> None:
    class Provider:
        model_id = "test"
        model_revision = "test-revision"
        closed = False

        def rerank(
            self, query: str, documents: tuple[str, ...], *, top_n: int | None = None
        ) -> RerankResult:
            return RerankResult(self.model_id, self.model_revision, rank_scores((1.0,), top_n=1), 3)

        def close(self) -> None:
            self.closed = True

    provider = Provider()
    config = ModelConfiguration.from_mapping(
        {
            "schema_version": 1,
            "profiles": {
                "ranker": {
                    "provider": "test-reranker",
                    "capability": "text_reranking",
                    "root": str(tmp_path),
                    "manifest_sha256": "a" * 64,
                }
            },
            "bindings": {"retrieval.rerank": "ranker"},
        }
    )
    registry = ProviderRegistry()
    registry.register(
        ProviderDefinition("test-reranker", "1", ModelCapability.TEXT_RERANKING, lambda p: provider)
    )
    with ModelFactory(config, registry=registry) as factory:
        with factory.text_reranker("retrieval.rerank") as ranker:
            assert ranker.rerank("q", ("d",)).results[0].index == 0
    assert provider.closed
