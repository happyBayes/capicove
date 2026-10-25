"""Deterministic planning for backend-independent directory batches."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from capicove.batch.models import (
    BatchCacheScope,
    BatchItem,
    BatchPlan,
    OperationFingerprint,
    PlannedBatchItem,
)


class BatchPlanner:
    """Build stable plans and content-based cache keys from discovered items."""

    def plan(
        self,
        items: Iterable[BatchItem],
        operation: OperationFingerprint,
    ) -> BatchPlan:
        """Return an order-independent plan for one operation fingerprint."""
        ordered_items = tuple(
            sorted(
                items,
                key=lambda item: (
                    item.modality.value,
                    item.relative_path.as_posix(),
                    item.item_id,
                ),
            )
        )
        seen_item_ids: set[str] = set()
        planned_items: list[PlannedBatchItem] = []
        for item in ordered_items:
            if item.item_id in seen_item_ids:
                raise ValueError(f"duplicate item_id in batch plan: {item.item_id}")
            seen_item_ids.add(item.item_id)
            if item.modality not in operation.modalities:
                raise ValueError(
                    f"operation {operation.operation_id} does not support {item.modality.value}"
                )
            planned_items.append(
                PlannedBatchItem(
                    item=item,
                    cache_key=_cache_key(item, operation),
                )
            )
        planned = tuple(planned_items)
        return BatchPlan(
            plan_id=_plan_id(operation, planned),
            operation=operation,
            items=planned,
        )


def _cache_key(item: BatchItem, operation: OperationFingerprint) -> str:
    payload = (
        "capicove.batch.cache.v2",
        operation.cache_scope.value,
        item.item_id if operation.cache_scope is BatchCacheScope.ITEM else None,
        item.sha256,
        operation.operation_id,
        operation.operation_version,
        operation.parameters_sha256,
        operation.provider_fingerprint,
    )
    return f"cache:{_canonical_digest(payload)}"


def _plan_id(
    operation: OperationFingerprint,
    items: tuple[PlannedBatchItem, ...],
) -> str:
    payload: tuple[object, ...] = (
        "capicove.batch.plan.v2",
        operation.operation_id,
        operation.operation_version,
        operation.parameters_sha256,
        operation.provider_fingerprint,
        operation.cache_scope.value,
        tuple(sorted(modality.value for modality in operation.modalities)),
        tuple((planned.item.item_id, planned.cache_key) for planned in items),
    )
    return f"plan:{_canonical_digest(payload)}"


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
