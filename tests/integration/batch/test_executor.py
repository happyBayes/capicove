from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import BinaryIO

import pytest

import capicove.batch.discovery as discovery
from capicove.batch._adapters.sqlite import SqliteBatchStore
from capicove.batch.discovery import DirectoryDiscoverer
from capicove.batch.executor import BatchExecutor
from capicove.batch.models import (
    BatchItem,
    BatchModality,
    BatchPlan,
    BatchState,
    OperationFingerprint,
)
from capicove.batch.planner import BatchPlanner
from capicove.batch.retry import BatchItemSkipped, BatchOperationError, RetryPolicy


_SHA_A = "a" * 64
_SHA_B = "b" * 64
Behavior = Callable[[BatchItem, int], Mapping[str, object]]


class _Operation:
    def __init__(self, fingerprint: OperationFingerprint, behavior: Behavior) -> None:
        self._fingerprint = fingerprint
        self._behavior = behavior
        self._lock = threading.Lock()
        self.call_counts: dict[str, int] = {}

    @property
    def fingerprint(self) -> OperationFingerprint:
        return self._fingerprint

    def execute(self, item: BatchItem, /) -> Mapping[str, object]:
        with self._lock:
            attempt = self.call_counts.get(item.item_id, 0) + 1
            self.call_counts[item.item_id] = attempt
        return self._behavior(item, attempt)


class _SourceAwareOperation(_Operation):
    def rebind_cached_result(
        self,
        item: BatchItem,
        cached: Mapping[str, object],
        /,
    ) -> Mapping[str, object]:
        return {
            **cached,
            "source": {
                "item_id": item.item_id,
                "relative_path": item.relative_path.as_posix(),
            },
        }


def _fingerprint(
    modalities: frozenset[BatchModality] = frozenset({BatchModality.AUDIO}),
) -> OperationFingerprint:
    return OperationFingerprint(
        operation_id="capicove.test.operation",
        operation_version="1",
        modalities=modalities,
        parameters_sha256=_SHA_A,
        provider_fingerprint=_SHA_B,
    )


def _audio_plan(
    root: Path,
    files: Mapping[str, bytes],
    fingerprint: OperationFingerprint | None = None,
) -> BatchPlan:
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    items = DirectoryDiscoverer({".wav": "audio/wav"}).discover(
        root,
        BatchModality.AUDIO,
    )
    return BatchPlanner().plan(items, fingerprint or _fingerprint())


def _executor(
    store: SqliteBatchStore,
    *,
    max_workers: int = 2,
    retry_policy: RetryPolicy | None = None,
    modality_limits: Mapping[BatchModality, int] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> BatchExecutor:
    return BatchExecutor(
        store,
        store,
        max_workers=max_workers,
        modality_limits=modality_limits,
        retry_policy=retry_policy or RetryPolicy(max_retries=0),
        sleep=sleep or time.sleep,
    )


def test_executor_coalesces_concurrent_items_with_the_same_cache_key(tmp_path: Path) -> None:
    plan = _audio_plan(tmp_path / "input", {"a.wav": b"same", "b.wav": b"same"})
    store = SqliteBatchStore(tmp_path / "batch.sqlite3")
    operation = _Operation(plan.operation, lambda _item, _attempt: {"text": "same"})

    execution = _executor(store).execute(plan, operation)

    assert sum(operation.call_counts.values()) == 1
    assert [item.state for item in execution.items] == [
        BatchState.SUCCEEDED,
        BatchState.SUCCEEDED,
    ]
    assert [item.cache_hit for item in execution.items] == [False, True]


def test_executor_rebinds_content_cache_results_to_each_source(tmp_path: Path) -> None:
    plan = _audio_plan(tmp_path / "input", {"a.wav": b"same", "b.wav": b"same"})
    store = SqliteBatchStore(tmp_path / "batch.sqlite3")
    operation = _SourceAwareOperation(
        plan.operation,
        lambda item, _attempt: {
            "source": {
                "item_id": item.item_id,
                "relative_path": item.relative_path.as_posix(),
            },
            "text": "same",
        },
    )

    execution = _executor(store).execute(plan, operation)

    assert [result.result["source"] for result in execution.items if result.result] == [
        {
            "item_id": planned.item.item_id,
            "relative_path": planned.item.relative_path.as_posix(),
        }
        for planned in plan.items
    ]


def test_executor_recomputes_when_operation_rejects_cached_result(tmp_path: Path) -> None:
    plan = _audio_plan(tmp_path / "input", {"meeting.wav": b"audio"})
    store = SqliteBatchStore(tmp_path / "batch.sqlite3")
    first = _Operation(plan.operation, lambda _item, _attempt: {"generation": 1})
    _executor(store).execute(plan, first)

    class RejectingCacheOperation(_Operation):
        def is_cached_result_valid(
            self,
            item: BatchItem,
            cached: Mapping[str, object],
            /,
        ) -> bool:
            del item, cached
            return False

    resumed_operation = RejectingCacheOperation(
        plan.operation,
        lambda _item, _attempt: {"generation": 2},
    )

    resumed = _executor(store).execute(plan, resumed_operation)

    assert sum(resumed_operation.call_counts.values()) == 1
    assert resumed.items[0].cache_hit is False
    assert resumed.items[0].result == {"generation": 2}


def test_executor_resume_reuses_completed_results_without_calling_operation(
    tmp_path: Path,
) -> None:
    plan = _audio_plan(tmp_path / "input", {"meeting.wav": b"audio"})
    store = SqliteBatchStore(tmp_path / "batch.sqlite3")
    first_operation = _Operation(plan.operation, lambda _item, _attempt: {"text": "hello"})
    _executor(store).execute(plan, first_operation)
    resumed_operation = _Operation(plan.operation, lambda _item, _attempt: {"text": "wrong"})

    resumed = _executor(store).execute(plan, resumed_operation)

    assert resumed_operation.call_counts == {}
    assert resumed.items[0].state is BatchState.SUCCEEDED
    assert resumed.items[0].cache_hit is True
    assert resumed.items[0].result == {"text": "hello"}


def test_executor_retries_only_explicit_transient_failures(tmp_path: Path) -> None:
    plan = _audio_plan(tmp_path / "input", {"meeting.wav": b"audio"})
    store = SqliteBatchStore(tmp_path / "batch.sqlite3")
    delays: list[float] = []

    def transient_then_success(_item: BatchItem, attempt: int) -> Mapping[str, object]:
        if attempt < 3:
            raise BatchOperationError("BUSY", "provider is busy", retryable=True)
        return {"text": "hello"}

    operation = _Operation(plan.operation, transient_then_success)
    execution = _executor(
        store,
        retry_policy=RetryPolicy(
            max_retries=2,
            initial_backoff_seconds=0.25,
            max_backoff_seconds=1.0,
        ),
        sleep=delays.append,
    ).execute(plan, operation)

    assert execution.items[0].state is BatchState.SUCCEEDED
    assert execution.items[0].attempts == 3
    assert delays == [0.25, 0.5]
    assert [
        checkpoint.state for checkpoint in store.history(plan.plan_id, plan.items[0].item.item_id)
    ] == [
        BatchState.PENDING,
        BatchState.RUNNING,
        BatchState.RUNNING,
        BatchState.RUNNING,
        BatchState.SUCCEEDED,
    ]


def test_executor_isolates_permanent_failure_and_preserves_other_results(tmp_path: Path) -> None:
    plan = _audio_plan(
        tmp_path / "input",
        {"fail.wav": b"fail", "pass.wav": b"pass"},
    )
    store = SqliteBatchStore(tmp_path / "batch.sqlite3")

    def fail_one(item: BatchItem, _attempt: int) -> Mapping[str, object]:
        if item.relative_path.name == "fail.wav":
            raise BatchOperationError("INVALID_AUDIO", "invalid audio", retryable=False)
        return {"text": "accepted"}

    execution = _executor(
        store,
        retry_policy=RetryPolicy(max_retries=2),
    ).execute(plan, _Operation(plan.operation, fail_one))

    assert [item.state for item in execution.items] == [
        BatchState.FAILED,
        BatchState.SUCCEEDED,
    ]
    assert execution.items[0].attempts == 1
    assert execution.items[0].failure is not None
    assert execution.items[0].failure.code == "INVALID_AUDIO"
    assert execution.items[1].result == {"text": "accepted"}


def test_executor_does_not_retry_persisted_non_retryable_failure(
    tmp_path: Path,
) -> None:
    plan = _audio_plan(tmp_path / "input", {"meeting.wav": b"audio"})
    store = SqliteBatchStore(tmp_path / "batch.sqlite3")

    def fail(_item: BatchItem, _attempt: int) -> Mapping[str, object]:
        raise BatchOperationError("INVALID_AUDIO", "invalid audio", retryable=False)

    first_operation = _Operation(plan.operation, fail)
    first = _executor(
        store,
        retry_policy=RetryPolicy(max_retries=2),
    ).execute(plan, first_operation)
    resumed_operation = _Operation(plan.operation, fail)

    resumed = _executor(
        store,
        retry_policy=RetryPolicy(max_retries=2),
    ).execute(plan, resumed_operation)

    assert first.items[0].attempts == 1
    assert resumed.items[0].state is BatchState.FAILED
    assert resumed.items[0].attempts == 1
    assert resumed_operation.call_counts == {}


def test_executor_records_skip_without_retrying_on_resume(tmp_path: Path) -> None:
    plan = _audio_plan(tmp_path / "input", {"silent.wav": b"silent"})
    store = SqliteBatchStore(tmp_path / "batch.sqlite3")

    def skip(_item: BatchItem, _attempt: int) -> Mapping[str, object]:
        raise BatchItemSkipped("no speech")

    first_operation = _Operation(plan.operation, skip)
    first = _executor(store).execute(plan, first_operation)
    resumed_operation = _Operation(plan.operation, skip)
    resumed = _executor(store).execute(plan, resumed_operation)

    assert first.items[0].state is BatchState.SKIPPED
    assert first.items[0].detail == "no speech"
    assert resumed.items[0].state is BatchState.SKIPPED
    assert resumed_operation.call_counts == {}


def test_executor_rejects_changed_source_before_cache_or_operation(tmp_path: Path) -> None:
    root = tmp_path / "input"
    plan = _audio_plan(root, {"meeting.wav": b"before"})
    plan.items[0].item.source_path.write_bytes(b"after")
    store = SqliteBatchStore(tmp_path / "batch.sqlite3")
    operation = _Operation(plan.operation, lambda _item, _attempt: {"text": "stale"})

    execution = _executor(store).execute(plan, operation)

    assert operation.call_counts == {}
    assert execution.items[0].state is BatchState.SOURCE_CHANGED
    assert execution.items[0].failure is not None
    assert execution.items[0].failure.code == "SOURCE_CHANGED"
    assert store.get(plan.items[0].cache_key) is None


def test_executor_rejects_replaced_source_without_reading_its_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "input"
    plan = _audio_plan(root, {"meeting.wav": b"authorized"})
    source = plan.items[0].item.source_path
    source.unlink()
    source.write_bytes(b"foreign replacement")
    store = SqliteBatchStore(tmp_path / "batch.sqlite3")
    operation = _Operation(plan.operation, lambda _item, _attempt: {"text": "must not run"})
    original_open = discovery._open_regular_file_no_follow
    read_attempts: list[int] = []

    class GuardedStream:
        def __init__(self, stream: BinaryIO) -> None:
            self._stream = stream

        def __enter__(self) -> GuardedStream:
            return self

        def __exit__(self, *_args: object) -> None:
            self._stream.close()

        def fileno(self) -> int:
            return self._stream.fileno()

        def read(self, size: int = -1) -> bytes:
            read_attempts.append(size)
            raise AssertionError("replaced source bytes were read")

    def guarded_open(path: Path) -> GuardedStream:
        return GuardedStream(original_open(path))

    monkeypatch.setattr(discovery, "_open_regular_file_no_follow", guarded_open)

    execution = _executor(store).execute(plan, operation)

    assert execution.items[0].state is BatchState.SOURCE_CHANGED
    assert operation.call_counts == {}
    assert read_attempts == []


def test_executor_rejects_same_content_source_replacement(tmp_path: Path) -> None:
    root = tmp_path / "input"
    plan = _audio_plan(root, {"meeting.wav": b"same"})
    source = plan.items[0].item.source_path
    replacement = root / "replacement.wav"
    replacement.write_bytes(b"same")
    replacement.replace(source)
    store = SqliteBatchStore(tmp_path / "batch.sqlite3")
    operation = _Operation(plan.operation, lambda _item, _attempt: {"text": "stale"})

    execution = _executor(store).execute(plan, operation)

    assert operation.call_counts == {}
    assert execution.items[0].state is BatchState.SOURCE_CHANGED


def test_executor_discards_result_when_source_changes_during_operation(tmp_path: Path) -> None:
    root = tmp_path / "input"
    plan = _audio_plan(root, {"meeting.wav": b"before"})
    store = SqliteBatchStore(tmp_path / "batch.sqlite3")

    def mutate(item: BatchItem, _attempt: int) -> Mapping[str, object]:
        item.source_path.write_bytes(b"after")
        return {"text": "stale"}

    execution = _executor(store).execute(plan, _Operation(plan.operation, mutate))

    assert execution.items[0].state is BatchState.SOURCE_CHANGED
    assert store.get(plan.items[0].cache_key) is None


def test_executor_enforces_global_and_per_modality_concurrency(tmp_path: Path) -> None:
    root = tmp_path / "input"
    for index in range(4):
        (root / f"audio-{index}.wav").parent.mkdir(parents=True, exist_ok=True)
        (root / f"audio-{index}.wav").write_bytes(str(index).encode())
        (root / f"image-{index}.png").write_bytes(f"image-{index}".encode())
    items = (
        *DirectoryDiscoverer({".wav": "audio/wav"}).discover(root, BatchModality.AUDIO),
        *DirectoryDiscoverer({".png": "image/png"}).discover(root, BatchModality.IMAGE),
    )
    fingerprint = _fingerprint(frozenset({BatchModality.AUDIO, BatchModality.IMAGE}))
    plan = BatchPlanner().plan(items, fingerprint)
    store = SqliteBatchStore(tmp_path / "batch.sqlite3")
    lock = threading.Lock()
    active_total = 0
    active_by_modality = {BatchModality.AUDIO: 0, BatchModality.IMAGE: 0}
    maximum_total = 0
    maximum_by_modality = {BatchModality.AUDIO: 0, BatchModality.IMAGE: 0}
    three_started = threading.Event()

    def observe(item: BatchItem, _attempt: int) -> Mapping[str, object]:
        nonlocal active_total, maximum_total
        with lock:
            active_total += 1
            active_by_modality[item.modality] += 1
            maximum_total = max(maximum_total, active_total)
            maximum_by_modality[item.modality] = max(
                maximum_by_modality[item.modality],
                active_by_modality[item.modality],
            )
            if active_total == 3:
                three_started.set()
        three_started.wait(timeout=5)
        with lock:
            active_total -= 1
            active_by_modality[item.modality] -= 1
        return {"sha256": item.sha256}

    _executor(
        store,
        max_workers=3,
        modality_limits={BatchModality.AUDIO: 1, BatchModality.IMAGE: 2},
    ).execute(plan, _Operation(plan.operation, observe))

    assert three_started.is_set()
    assert maximum_total == 3
    assert maximum_by_modality == {BatchModality.AUDIO: 1, BatchModality.IMAGE: 2}


def test_executor_passes_persisted_attempt_number_to_attempt_aware_operation(
    tmp_path: Path,
) -> None:
    plan = _audio_plan(tmp_path / "input", {"meeting.wav": b"audio"})
    store = SqliteBatchStore(tmp_path / "batch.sqlite3")

    class AttemptAwareOperation:
        def __init__(self) -> None:
            self.attempts: list[int] = []

        @property
        def fingerprint(self) -> OperationFingerprint:
            return plan.operation

        def execute(self, item: BatchItem, /) -> Mapping[str, object]:
            del item
            raise AssertionError("executor bypassed execute_attempt")

        def execute_attempt(
            self,
            item: BatchItem,
            attempt: int,
            /,
        ) -> Mapping[str, object]:
            del item
            self.attempts.append(attempt)
            if attempt == 1:
                raise BatchOperationError("BUSY", "provider is busy", retryable=True)
            return {"text": "complete"}

    operation = AttemptAwareOperation()

    execution = _executor(
        store,
        retry_policy=RetryPolicy(max_retries=1),
        sleep=lambda _: None,
    ).execute(plan, operation)

    assert operation.attempts == [1, 2]
    assert execution.items[0].attempts == 2
    assert execution.items[0].result == {"text": "complete"}
