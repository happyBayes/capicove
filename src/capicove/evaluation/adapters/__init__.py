"""Lazily imported stack adapters for the offline evaluation harness."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from capicove.evaluation.harness import EvaluationError


if TYPE_CHECKING:
    from capicove.evaluation.harness import StackAdapter


ADAPTER_MODULES = {
    "offline-text": "capicove.evaluation.adapters.offline_text",
    "kb": "capicove.evaluation.adapters.knowledge_base",
    "unified": "capicove.evaluation.adapters.unified",
    "code": "capicove.evaluation.adapters.code",
    "image": "capicove.evaluation.adapters.image",
    "video": "capicove.evaluation.adapters.video",
    "graphrag-local": "capicove.evaluation.adapters.graphrag",
    "graphrag-global": "capicove.evaluation.adapters.graphrag",
}
STACKS = tuple(ADAPTER_MODULES)


def resolve_adapter(stack: str) -> StackAdapter:
    """Import only the selected stack's adapter and its optional runtime."""
    try:
        module = importlib.import_module(ADAPTER_MODULES[stack])
    except KeyError:
        raise EvaluationError(f"unsupported evaluation stack: {stack}") from None
    adapter: StackAdapter = module.adapter(stack)
    return adapter


__all__ = ["ADAPTER_MODULES", "STACKS", "resolve_adapter"]
