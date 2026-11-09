"""Optional post-fusion reranking shared by text search paths (RFC-0034, Proposed).

The stage reorders only the first ``candidate_depth`` fused candidates using a
:class:`~capicove.retrieval.reranking.TextReranker`; later candidates keep their
fused order. The outcome always states what happened: ``disabled``, ``applied``,
``no_candidates`` or ``degraded`` (with a reason). A missing, mismatched or
failing reranker raises :class:`RerankUnavailableError` under the ``fail`` policy.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from capicove.retrieval.pipeline_profiles import RerankPolicy
from capicove.retrieval.reranking import TextReranker


if TYPE_CHECKING:
    from capicove.model.factory import ModelFactory


class RerankUnavailableError(RuntimeError):
    """Report that an enabled reranker could not be used under a fail policy."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class RerankOutcome:
    """A permutation of fused candidate positions plus an explicit status."""

    order: tuple[int, ...]
    status: str
    reason: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    depth: int = 0

    def diagnostics(self) -> dict[str, Any]:
        """Return a JSON-ready description of the rerank decision."""
        return {
            "status": self.status,
            "reason": self.reason,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "depth": self.depth,
        }


def rerank_fused(
    query: str,
    texts: Sequence[str],
    policy: RerankPolicy,
    reranker: TextReranker | None,
) -> RerankOutcome:
    """Rerank the fused head, or explain precisely why the fused order stands."""
    identity = tuple(range(len(texts)))
    if not policy.enabled:
        return RerankOutcome(identity, "disabled")
    depth = min(policy.candidate_depth, len(texts))
    if depth == 0:
        return RerankOutcome(identity, "no_candidates")
    if reranker is None:
        return _unavailable(policy, "reranker_unavailable", identity)
    if getattr(reranker, "model_id", None) != policy.model_id:
        return _unavailable(policy, "reranker_model_mismatch", identity)
    try:
        result = reranker.rerank(query, tuple(texts[:depth]))
    except Exception:
        return _unavailable(policy, "reranker_failed", identity)
    head = [item.index for item in result.results]
    if sorted(head) != list(range(depth)):
        return _unavailable(policy, "invalid_reranker_output", identity)
    return RerankOutcome(
        (*head, *range(depth, len(texts))),
        "applied",
        model_id=result.model,
        model_revision=result.model_revision,
        depth=depth,
    )


def _unavailable(policy: RerankPolicy, reason: str, identity: tuple[int, ...]) -> RerankOutcome:
    if policy.on_unavailable == "fail":
        raise RerankUnavailableError(reason)
    return RerankOutcome(identity, "degraded", reason=reason)


@contextmanager
def acquire_reranker(
    models: ModelFactory | None, policy: RerankPolicy
) -> Iterator[TextReranker | None]:
    """Borrow the configured reranker, yielding ``None`` when it cannot be loaded.

    Callers pass the result to :func:`rerank_fused`, which then fails or degrades
    as the policy requires. Nothing is loaded while reranking is disabled.
    """
    if not policy.enabled or models is None:
        yield None
        return
    with ExitStack() as stack:
        reranker: TextReranker | None
        try:
            reranker = stack.enter_context(models.text_reranker(policy.binding))
        except Exception:
            reranker = None
        yield reranker


__all__ = ["RerankOutcome", "RerankUnavailableError", "acquire_reranker", "rerank_fused"]
