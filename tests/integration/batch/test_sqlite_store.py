from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import pytest

from capicove.batch._adapters.sqlite import SqliteBatchStore
from capicove.batch.cache import CacheConflictError
from capicove.batch.checkpoint import CheckpointTransitionError
from capicove.batch.models import (
    BatchFailure,
    BatchItem,
    BatchModality,
    BatchPlan,
    BatchState,
    OperationFingerprint,
    SourceSnapshot,
)
from capicove.batch.planner import BatchPlanner


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64


def _plan() -> BatchPlan:
    item = BatchItem(
        item_id="item:" + _SHA_A,
        modality=BatchModality.AUDIO,
        source_path=Path("/data/meeting.wav"),
        relative_path=PurePosixPath("meeting.wav"),
        media_type="audio/wav",
        sha256=_SHA_C,
        snapshot=SourceSnapshot(1, 2, 3, 4, 5),
    )
    operation = OperationFingerprint(
        operation_id="capicove.audio.transcribe",
        operation_version="1",
        modalities=frozenset({BatchModality.AUDIO}),
        parameters_sha256=_SHA_A,
        provider_fingerprint=_SHA_B,
    )
    return BatchPlanner().plan((item,), operation)


def test_sqlite_store_persists_checkpoints_and_append_only_history(tmp_path: Path) -> None:
    database = tmp_path / "batch.sqlite3"
    plan = _plan()
    store = SqliteBatchStore(database)
    store.prepare(plan)
    item_id = plan.items[0].item.item_id

    running = store.update(
        plan.plan_id,
        item_id,
        expected_states=frozenset({BatchState.PENDING}),
        state=BatchState.RUNNING,
        attempts=1,
    )
    succeeded = store.update(
        plan.plan_id,
        item_id,
        expected_states=frozenset({BatchState.RUNNING}),
        state=BatchState.SUCCEEDED,
        attempts=1,
    )

    reopened = SqliteBatchStore(database)
    assert reopened.load(plan.plan_id, item_id) == succeeded
    assert running.state is BatchState.RUNNING
    assert [event.state for event in reopened.history(plan.plan_id, item_id)] == [
        BatchState.PENDING,
        BatchState.RUNNING,
        BatchState.SUCCEEDED,
    ]


def test_prepare_recovers_only_interrupted_running_items(tmp_path: Path) -> None:
    plan = _plan()
    store = SqliteBatchStore(tmp_path / "batch.sqlite3")
    store.prepare(plan)
    item_id = plan.items[0].item.item_id
    store.update(
        plan.plan_id,
        item_id,
        expected_states=frozenset({BatchState.PENDING}),
        state=BatchState.RUNNING,
        attempts=2,
    )

    store.prepare(plan)

    recovered = store.load(plan.plan_id, item_id)
    assert recovered.state is BatchState.PENDING
    assert recovered.attempts == 2
    assert [event.state for event in store.history(plan.plan_id, item_id)] == [
        BatchState.PENDING,
        BatchState.RUNNING,
        BatchState.PENDING,
    ]


def test_sqlite_update_rejects_stale_expected_state(tmp_path: Path) -> None:
    plan = _plan()
    store = SqliteBatchStore(tmp_path / "batch.sqlite3")
    store.prepare(plan)
    item_id = plan.items[0].item.item_id
    store.update(
        plan.plan_id,
        item_id,
        expected_states=frozenset({BatchState.PENDING}),
        state=BatchState.RUNNING,
        attempts=1,
    )

    with pytest.raises(CheckpointTransitionError, match="expected"):
        store.update(
            plan.plan_id,
            item_id,
            expected_states=frozenset({BatchState.PENDING}),
            state=BatchState.FAILED,
            attempts=1,
            failure=BatchFailure("PERMANENT", "failed", retryable=False),
        )


def test_sqlite_claim_allows_only_one_concurrent_pending_transition(tmp_path: Path) -> None:
    plan = _plan()
    store = SqliteBatchStore(tmp_path / "batch.sqlite3")
    store.prepare(plan)
    item_id = plan.items[0].item.item_id

    def claim() -> bool:
        try:
            store.update(
                plan.plan_id,
                item_id,
                expected_states=frozenset({BatchState.PENDING}),
                state=BatchState.RUNNING,
                attempts=1,
            )
        except CheckpointTransitionError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as executor:
        claimed = tuple(executor.map(lambda _: claim(), range(2)))

    assert sorted(claimed) == [False, True]
    assert store.load(plan.plan_id, item_id).state is BatchState.RUNNING


def test_sqlite_cache_is_persistent_idempotent_and_conflict_safe(tmp_path: Path) -> None:
    database = tmp_path / "batch.sqlite3"
    store = SqliteBatchStore(database)
    cache_key = "cache:" + _SHA_A
    result = {
        "text": "hello",
        "segments": [{"start_seconds": 0.0, "end_seconds": 1.0}],
        "lineage": {"model_sha256": _SHA_B},
    }

    store.put(cache_key, result)
    store.put(cache_key, result)

    assert SqliteBatchStore(database).get(cache_key) == result
    with pytest.raises(CacheConflictError, match="different result"):
        store.put(cache_key, {"text": "stale"})
