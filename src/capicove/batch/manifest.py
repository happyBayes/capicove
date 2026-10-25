"""In-memory terminal manifest summaries for batch executions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from capicove.batch.models import BatchExecution, BatchItemResult, BatchState


@dataclass(frozen=True, slots=True)
class BatchManifest:
    """Summarize one execution without defining a persistent file format."""

    plan_id: str
    total: int
    cache_hits: int
    counts: Mapping[BatchState, int]
    items: tuple[BatchItemResult, ...]

    def __post_init__(self) -> None:
        materialized_counts = dict(self.counts)
        if set(materialized_counts) != set(BatchState):
            raise ValueError("counts must include every batch state")
        if any(count < 0 for count in materialized_counts.values()):
            raise ValueError("manifest counts must not be negative")
        if self.total != len(self.items) or self.total != sum(materialized_counts.values()):
            raise ValueError("manifest total must match items and state counts")
        if self.cache_hits != sum(item.cache_hit for item in self.items):
            raise ValueError("manifest cache_hits must match item results")
        object.__setattr__(self, "counts", MappingProxyType(materialized_counts))


def build_batch_manifest(execution: BatchExecution) -> BatchManifest:
    """Build a complete state and cache summary from terminal results."""
    counts = {state: 0 for state in BatchState}
    for item in execution.items:
        counts[item.state] += 1
    return BatchManifest(
        plan_id=execution.plan_id,
        total=len(execution.items),
        cache_hits=sum(item.cache_hit for item in execution.items),
        counts=counts,
        items=execution.items,
    )
