from __future__ import annotations

from capicove.batch.manifest import build_batch_manifest
from capicove.batch.models import (
    BatchExecution,
    BatchFailure,
    BatchItemResult,
    BatchState,
)


_PLAN_ID = "plan:" + "a" * 64


def test_manifest_counts_every_terminal_state_and_cache_reuse() -> None:
    execution = BatchExecution(
        plan_id=_PLAN_ID,
        items=(
            BatchItemResult(
                item_id="item:" + "a" * 64,
                state=BatchState.SUCCEEDED,
                attempts=1,
                result={"text": "fresh"},
            ),
            BatchItemResult(
                item_id="item:" + "b" * 64,
                state=BatchState.SUCCEEDED,
                attempts=0,
                cache_hit=True,
                result={"text": "cached"},
            ),
            BatchItemResult(
                item_id="item:" + "c" * 64,
                state=BatchState.FAILED,
                attempts=2,
                failure=BatchFailure("TIMEOUT", "deadline expired", retryable=True),
            ),
            BatchItemResult(
                item_id="item:" + "d" * 64,
                state=BatchState.SKIPPED,
                attempts=1,
                detail="not applicable",
            ),
            BatchItemResult(
                item_id="item:" + "e" * 64,
                state=BatchState.SOURCE_CHANGED,
                attempts=0,
                failure=BatchFailure("SOURCE_CHANGED", "changed", retryable=False),
            ),
        ),
    )

    manifest = build_batch_manifest(execution)

    assert manifest.plan_id == _PLAN_ID
    assert manifest.total == 5
    assert manifest.cache_hits == 1
    assert manifest.counts == {
        BatchState.PENDING: 0,
        BatchState.RUNNING: 0,
        BatchState.SUCCEEDED: 2,
        BatchState.FAILED: 1,
        BatchState.SKIPPED: 1,
        BatchState.SOURCE_CHANGED: 1,
    }
    assert manifest.items == execution.items


def test_manifest_supports_an_empty_plan() -> None:
    manifest = build_batch_manifest(BatchExecution(plan_id=_PLAN_ID, items=()))

    assert manifest.total == 0
    assert manifest.cache_hits == 0
    assert all(count == 0 for count in manifest.counts.values())
