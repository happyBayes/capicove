"""Ports consumed by the backend-independent batch capability."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from capicove.batch.models import (
    BatchItem,
    BatchItemResult,
    OperationFingerprint,
    PlannedBatchItem,
)


class BatchOperation(Protocol):
    """Execute one batch item through an injected capability operation."""

    @property
    def fingerprint(self) -> OperationFingerprint:
        """Return every value that affects result reuse."""

    def execute(self, item: BatchItem, /) -> Mapping[str, object]:
        """Return a transport-neutral result for one source observation."""


@runtime_checkable
class AttemptAwareBatchOperation(BatchOperation, Protocol):
    """Execute an item with its checkpoint-owned one-based attempt number."""

    def execute_attempt(
        self,
        item: BatchItem,
        attempt: int,
        /,
    ) -> Mapping[str, object]:
        """Return a result for one explicit durable execution attempt."""


@runtime_checkable
class CachedResultRebinder(BatchOperation, Protocol):
    """Rebind content-cached metadata to the current source item."""

    def rebind_cached_result(
        self,
        item: BatchItem,
        cached: Mapping[str, object],
        /,
    ) -> Mapping[str, object]:
        """Return a cache payload whose source metadata names ``item``."""


@runtime_checkable
class CachedResultValidator(BatchOperation, Protocol):
    """Validate operation-owned resources referenced by a cached result."""

    def is_cached_result_valid(
        self,
        item: BatchItem,
        cached: Mapping[str, object],
        /,
    ) -> bool:
        """Return whether the cached result remains safe to reuse."""


class BatchExecutionObserver(Protocol):
    """Observe each terminal item after its durable checkpoint is written."""

    def on_item_completed(
        self,
        plan_id: str,
        planned: PlannedBatchItem,
        result: BatchItemResult,
        /,
    ) -> None:
        """Receive one terminal item without changing batch-owned policy."""
