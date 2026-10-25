"""Tests for the batch-owned terminal completion observer port."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from capicove.batch import (
    BatchExecutionObserver,
    BatchExecutor,
    BatchItem,
    BatchModality,
    BatchPlanner,
    DirectoryDiscoverer,
    OperationFingerprint,
    RetryPolicy,
)
from capicove.batch._adapters.sqlite import SqliteBatchStore


SHA256_A = "a" * 64
SHA256_B = "b" * 64


class Operation:
    """Execute one deterministic result while exposing call count."""

    def __init__(self, fingerprint: OperationFingerprint) -> None:
        self._fingerprint = fingerprint
        self.calls = 0

    @property
    def fingerprint(self) -> OperationFingerprint:
        return self._fingerprint

    def execute(self, item: BatchItem, /) -> Mapping[str, object]:
        self.calls += 1
        return {"sha256": item.sha256}


class Observer:
    """Capture exact terminal callback arguments."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[object, object, object]] = []

    def on_item_completed(self, plan_id: str, planned: object, result: object, /) -> None:
        self.calls.append((plan_id, planned, result))
        if self.fail:
            raise RuntimeError("observer unavailable")


def _plan(root: Path):  # type: ignore[no-untyped-def]
    root.mkdir()
    (root / "one.wav").write_bytes(b"one")
    items = DirectoryDiscoverer({".wav": "audio/wav"}).discover(root, BatchModality.AUDIO)
    fingerprint = OperationFingerprint(
        operation_id="capicove.test.operation",
        operation_version="1",
        modalities=frozenset({BatchModality.AUDIO}),
        parameters_sha256=SHA256_A,
        provider_fingerprint=SHA256_B,
    )
    return BatchPlanner().plan(items, fingerprint)


def _executor(
    store: SqliteBatchStore,
    observer: BatchExecutionObserver | None = None,
) -> BatchExecutor:
    return BatchExecutor(
        store,
        store,
        max_workers=1,
        retry_policy=RetryPolicy(max_retries=0),
        observer=observer,
    )


def test_observer_receives_original_plan_item_and_terminal_result_once(tmp_path: Path) -> None:
    plan = _plan(tmp_path / "input")
    store = SqliteBatchStore(tmp_path / "batch.sqlite3")
    observer = Observer()
    operation = Operation(plan.operation)

    execution = _executor(store, observer).execute(plan, operation)

    assert len(observer.calls) == 1
    plan_id, planned, result = observer.calls[0]
    assert plan_id == plan.plan_id
    assert planned is plan.items[0]
    assert result is execution.items[0]


def test_absent_observer_preserves_existing_execution_result(tmp_path: Path) -> None:
    plan = _plan(tmp_path / "input")
    store = SqliteBatchStore(tmp_path / "batch.sqlite3")

    execution = _executor(store).execute(plan, Operation(plan.operation))

    assert execution.items[0].result == {"sha256": plan.items[0].item.sha256}


def test_raising_observer_leaves_terminal_checkpoint_safe_for_resume(tmp_path: Path) -> None:
    plan = _plan(tmp_path / "input")
    store = SqliteBatchStore(tmp_path / "batch.sqlite3")
    first_operation = Operation(plan.operation)

    with pytest.raises(RuntimeError, match="observer"):
        _executor(store, Observer(fail=True)).execute(plan, first_operation)

    resumed_operation = Operation(plan.operation)
    resumed_observer = Observer()
    resumed = _executor(store, resumed_observer).execute(plan, resumed_operation)

    assert first_operation.calls == 1
    assert resumed_operation.calls == 0
    assert resumed.items[0].cache_hit is True
    assert len(resumed_observer.calls) == 1
