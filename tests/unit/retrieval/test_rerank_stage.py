"""Post-fusion rerank stage: applied, disabled, or explicitly failed/degraded."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from capicove.evaluation.fakes import FAKE_RERANKER_MODEL_ID, DeterministicFakeReranker
from capicove.retrieval.pipeline_profiles import RerankPolicy
from capicove.retrieval.rerank_stage import (
    RerankUnavailableError,
    acquire_reranker,
    rerank_fused,
)
from capicove.retrieval.reranking import RerankItem, RerankResult


ENABLED = RerankPolicy(enabled=True, model_id=FAKE_RERANKER_MODEL_ID, candidate_depth=2)
DEGRADE = RerankPolicy(enabled=True, model_id=FAKE_RERANKER_MODEL_ID, on_unavailable="degrade")


class Broken:
    model_id = FAKE_RERANKER_MODEL_ID
    model_revision = "x"

    def __init__(self, result: RerankResult | None = None) -> None:
        self.result = result

    def rerank(
        self, query: str, documents: tuple[str, ...], *, top_n: int | None = None
    ) -> RerankResult:
        if self.result is None:
            raise RuntimeError("boom")
        return self.result


def test_disabled_policy_keeps_fused_order_without_a_reranker() -> None:
    outcome = rerank_fused("q", ["a", "b"], RerankPolicy(), None)
    assert outcome.order == (0, 1)
    assert outcome.status == "disabled"
    assert outcome.diagnostics()["status"] == "disabled"


def test_applied_rerank_reorders_only_the_head() -> None:
    texts = ["blue car", "red bike", "red bike pump"]
    outcome = rerank_fused("red bike", texts, ENABLED, DeterministicFakeReranker())
    assert outcome.order == (1, 0, 2)
    assert outcome.status == "applied"
    assert outcome.depth == 2
    assert outcome.diagnostics() == {
        "status": "applied",
        "reason": None,
        "model_id": FAKE_RERANKER_MODEL_ID,
        "model_revision": "fake-1",
        "depth": 2,
    }


def test_empty_candidates_are_reported_explicitly() -> None:
    assert rerank_fused("q", [], ENABLED, None).status == "no_candidates"


@pytest.mark.parametrize(
    ("reranker", "reason"),
    [
        (None, "reranker_unavailable"),
        (
            type("Other", (DeterministicFakeReranker,), {"model_id": "x/y"})(),
            "reranker_model_mismatch",
        ),
        (Broken(), "reranker_failed"),
        (
            Broken(RerankResult(FAKE_RERANKER_MODEL_ID, "x", (RerankItem(0, 1.0),), 1)),
            "invalid_reranker_output",
        ),
    ],
)
def test_unusable_rerankers_fail_or_degrade_but_never_skip(reranker: Any, reason: str) -> None:
    with pytest.raises(RerankUnavailableError) as raised:
        rerank_fused("q", ["a", "b"], ENABLED, reranker)
    assert raised.value.code == reason
    outcome = rerank_fused("q", ["a", "b"], DEGRADE, reranker)
    assert outcome.status == "degraded"
    assert outcome.reason == reason
    assert outcome.order == (0, 1)


class Factory:
    def __init__(self, fail: bool) -> None:
        self.fail = fail
        self.released = False

    def text_reranker(self, binding: str) -> Any:
        assert binding == "retrieval.reranker"
        if self.fail:
            raise LookupError("binding not configured")
        factory = self

        @contextmanager
        def lease() -> Iterator[DeterministicFakeReranker]:
            yield DeterministicFakeReranker()
            factory.released = True

        return lease()


def test_acquire_reranker_loads_only_when_enabled() -> None:
    with acquire_reranker(None, ENABLED) as reranker:
        assert reranker is None
    factory = Factory(fail=False)
    with acquire_reranker(factory, RerankPolicy()) as reranker:  # type: ignore[arg-type]
        assert reranker is None
    with acquire_reranker(factory, ENABLED) as reranker:  # type: ignore[arg-type]
        assert isinstance(reranker, DeterministicFakeReranker)
    assert factory.released
    with acquire_reranker(Factory(fail=True), ENABLED) as reranker:  # type: ignore[arg-type]
        assert reranker is None
