"""Resolve evaluation run options into a pipeline profile and optional fake reranker."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from capicove.evaluation.harness import RunOptions


def resolve_profile(stack: str, options: RunOptions) -> tuple[Any, Any, dict[str, Any]]:
    """Return ``(profile, reranker, model)``; ``profile`` is ``None`` for defaults.

    ``None`` means the stack's untouched default code path. A ``deterministic-fake``
    reranker enables reranking with the fake model identity unless a profile file
    supplies its own rerank policy.
    """
    from capicove.retrieval.pipeline_profiles import (
        PROFILE_TYPES,
        RerankPolicy,
        load_pipeline_profile,
    )

    if options.profile_path is None and options.reranker == "none":
        return None, None, {"kind": "none", "components": []}
    if options.profile_path is not None:
        profile = load_pipeline_profile(options.profile_path, stack)
    else:
        profile = PROFILE_TYPES[stack]()
    if options.reranker == "bge":
        return _bge(profile, options)
    if options.reranker == "deterministic-fake":
        from capicove.evaluation.fakes import FAKE_RERANKER_MODEL_ID, DeterministicFakeReranker

        if options.profile_path is None:
            profile = replace(
                profile,
                profile_id=profile.profile_id + "-fake-rerank",
                rerank=RerankPolicy(enabled=True, model_id=FAKE_RERANKER_MODEL_ID),
            )
        return (
            profile,
            DeterministicFakeReranker(),
            {"kind": "deterministic-fake", "components": [f"reranker:{FAKE_RERANKER_MODEL_ID}"]},
        )
    return profile, None, {"kind": "none", "components": []}


def _bge(profile: Any, options: RunOptions) -> tuple[Any, Any, dict[str, Any]]:
    """Load the verified local BGE reranker; a missing model fails explicitly."""
    from capicove.evaluation.harness import EvaluationError
    from capicove.retrieval.pipeline_profiles import BGE_RERANKER_MODEL_ID, RerankPolicy

    assert options.reranker_model is not None
    try:
        from capicove.retrieval.bge_reranker import BgeReranker

        reranker = BgeReranker(options.reranker_model.resolve())
    except Exception as error:
        raise EvaluationError(
            f"BGE reranker unavailable ({type(error).__name__}); no evaluation was run"
        ) from None
    if options.profile_path is None:
        profile = replace(
            profile,
            profile_id=profile.profile_id + "-bge-rerank",
            rerank=RerankPolicy(enabled=True, model_id=BGE_RERANKER_MODEL_ID),
        )
    return (
        profile,
        reranker,
        {
            "kind": "local-model",
            "components": [f"reranker:{reranker.model_id}@{reranker.model_revision}"],
        },
    )


__all__ = ["resolve_profile"]
