"""Bounded in-process execution for deterministic batch plans."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

from capicove.batch.cache import ResultCache
from capicove.batch.checkpoint import BatchCheckpoint, CheckpointStore
from capicove.batch.discovery import DirectoryDiscoveryError, _observe_regular_file
from capicove.batch.models import (
    BatchExecution,
    BatchFailure,
    BatchItem,
    BatchItemResult,
    BatchModality,
    BatchPlan,
    BatchState,
    PlannedBatchItem,
)
from capicove.batch.ports import (
    AttemptAwareBatchOperation,
    BatchExecutionObserver,
    BatchOperation,
    CachedResultRebinder,
    CachedResultValidator,
)
from capicove.batch.retry import BatchItemSkipped, BatchOperationError, RetryPolicy


_MAX_WORKERS = 64
_SOURCE_CHANGED_FAILURE = BatchFailure(
    code="SOURCE_CHANGED",
    message="source changed after batch discovery",
    retryable=False,
)
_UNEXPECTED_FAILURE = BatchFailure(
    code="INTERNAL",
    message="batch operation failed unexpectedly",
    retryable=False,
)


class BatchExecutor:
    """Execute one plan with bounded workers, cache reuse, and checkpoints."""

    def __init__(
        self,
        checkpoint_store: CheckpointStore,
        result_cache: ResultCache,
        *,
        max_workers: int,
        retry_policy: RetryPolicy,
        modality_limits: Mapping[BatchModality, int] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        observer: BatchExecutionObserver | None = None,
    ) -> None:
        if isinstance(max_workers, bool) or not 1 <= max_workers <= _MAX_WORKERS:
            raise ValueError(f"max_workers must be between 1 and {_MAX_WORKERS}")
        limits = dict(modality_limits or {})
        for modality, limit in limits.items():
            if not isinstance(modality, BatchModality):
                raise ValueError("modality_limits keys must be BatchModality values")
            if isinstance(limit, bool) or limit < 1:
                raise ValueError("modality concurrency limits must be positive")
        self._checkpoint_store = checkpoint_store
        self._result_cache = result_cache
        self._max_workers = max_workers
        self._retry_policy = retry_policy
        self._modality_limits = limits
        self._sleep = sleep
        self._observer = observer

    def execute(
        self,
        plan: BatchPlan,
        operation: BatchOperation,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> BatchExecution:
        """Execute or resume one plan and return results in deterministic order."""
        if operation.fingerprint != plan.operation:
            raise ValueError("operation fingerprint does not match the batch plan")
        self._checkpoint_store.prepare(plan)
        pending = deque(plan.items)
        results: dict[str, BatchItemResult] = {}
        futures: dict[Future[BatchItemResult], PlannedBatchItem] = {}
        active_by_modality = {modality: 0 for modality in BatchModality}
        active_cache_keys: set[str] = set()

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            while pending or futures:
                while len(futures) < self._max_workers:
                    planned = self._take_eligible(
                        pending,
                        active_by_modality,
                        active_cache_keys,
                    )
                    if planned is None:
                        break
                    active_by_modality[planned.item.modality] += 1
                    active_cache_keys.add(planned.cache_key)
                    future = pool.submit(
                        self._execute_item, plan.plan_id, planned, operation, should_cancel
                    )
                    futures[future] = planned
                if not futures:
                    raise RuntimeError("batch scheduler could not select a pending item")
                completed, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                for future in completed:
                    planned = futures.pop(future)
                    active_by_modality[planned.item.modality] -= 1
                    active_cache_keys.remove(planned.cache_key)
                    result = future.result()
                    if self._observer is not None:
                        self._observer.on_item_completed(plan.plan_id, planned, result)
                    results[planned.item.item_id] = result

        return BatchExecution(
            plan_id=plan.plan_id,
            items=tuple(results[planned.item.item_id] for planned in plan.items),
        )

    def _take_eligible(
        self,
        pending: deque[PlannedBatchItem],
        active_by_modality: Mapping[BatchModality, int],
        active_cache_keys: set[str],
    ) -> PlannedBatchItem | None:
        for _ in range(len(pending)):
            planned = pending.popleft()
            limit = self._modality_limits.get(planned.item.modality, self._max_workers)
            if (
                active_by_modality[planned.item.modality] < limit
                and planned.cache_key not in active_cache_keys
            ):
                return planned
            pending.append(planned)
        return None

    def _execute_item(
        self,
        plan_id: str,
        planned: PlannedBatchItem,
        operation: BatchOperation,
        should_cancel: Callable[[], bool] | None,
    ) -> BatchItemResult:
        item = planned.item
        checkpoint = self._checkpoint_store.load(plan_id, item.item_id)
        if checkpoint.state in {BatchState.SKIPPED, BatchState.SOURCE_CHANGED}:
            return _terminal_checkpoint_result(checkpoint)
        if should_cancel is not None and should_cancel():
            return self._cancelled(checkpoint)
        if not _source_is_unchanged(item):
            return self._source_changed(checkpoint)

        cached = self._result_cache.get(planned.cache_key)
        if cached is not None and isinstance(operation, CachedResultValidator):
            if not operation.is_cached_result_valid(item, cached):
                self._result_cache.evict(planned.cache_key)
                cached = None
        if cached is not None:
            if isinstance(operation, CachedResultRebinder):
                cached = operation.rebind_cached_result(item, cached)
            if checkpoint.state is not BatchState.SUCCEEDED:
                checkpoint = self._checkpoint_store.update(
                    plan_id,
                    item.item_id,
                    expected_states=frozenset({checkpoint.state}),
                    state=BatchState.SUCCEEDED,
                    attempts=checkpoint.attempts,
                )
            return BatchItemResult(
                item_id=item.item_id,
                state=BatchState.SUCCEEDED,
                attempts=checkpoint.attempts,
                cache_hit=True,
                result=cached,
            )

        if checkpoint.state is BatchState.SUCCEEDED:
            checkpoint = self._checkpoint_store.update(
                plan_id,
                item.item_id,
                expected_states=frozenset({BatchState.SUCCEEDED}),
                state=BatchState.PENDING,
                attempts=checkpoint.attempts,
            )
        if checkpoint.state is BatchState.FAILED and (
            checkpoint.failure is None
            or not checkpoint.failure.retryable
            or checkpoint.attempts >= self._retry_policy.max_attempts
        ):
            return _terminal_checkpoint_result(checkpoint)
        return self._run_operation(checkpoint, planned, operation, should_cancel)

    def _run_operation(
        self,
        checkpoint: BatchCheckpoint,
        planned: PlannedBatchItem,
        operation: BatchOperation,
        should_cancel: Callable[[], bool] | None,
    ) -> BatchItemResult:
        item = planned.item
        reuse_interrupted_attempt = (
            checkpoint.state is BatchState.PENDING and checkpoint.attempts > 0
        )
        while True:
            if should_cancel is not None and should_cancel():
                return self._cancelled(checkpoint)
            attempts = checkpoint.attempts if reuse_interrupted_attempt else checkpoint.attempts + 1
            reuse_interrupted_attempt = False
            checkpoint = self._checkpoint_store.update(
                checkpoint.plan_id,
                item.item_id,
                expected_states=frozenset({checkpoint.state}),
                state=BatchState.RUNNING,
                attempts=attempts,
            )
            try:
                if isinstance(operation, AttemptAwareBatchOperation):
                    result = operation.execute_attempt(item, attempts)
                else:
                    result = operation.execute(item)
                if not isinstance(result, Mapping):
                    raise BatchOperationError(
                        "INVALID_RESULT",
                        "batch operation returned a non-object result",
                        retryable=False,
                    )
            except BatchItemSkipped as skipped:
                self._checkpoint_store.update(
                    checkpoint.plan_id,
                    item.item_id,
                    expected_states=frozenset({BatchState.RUNNING}),
                    state=BatchState.SKIPPED,
                    attempts=attempts,
                )
                return BatchItemResult(
                    item_id=item.item_id,
                    state=BatchState.SKIPPED,
                    attempts=attempts,
                    detail=skipped.reason,
                )
            except BatchOperationError as error:
                if error.failure.retryable and attempts < self._retry_policy.max_attempts:
                    self._sleep(self._retry_policy.delay_before_retry(attempts))
                    continue
                return self._record_failure(checkpoint, attempts, error.failure)
            except Exception:
                return self._record_failure(checkpoint, attempts, _UNEXPECTED_FAILURE)

            if not _source_is_unchanged(item):
                return self._source_changed(checkpoint)
            materialized = dict(result)
            self._result_cache.put(planned.cache_key, materialized)
            self._checkpoint_store.update(
                checkpoint.plan_id,
                item.item_id,
                expected_states=frozenset({BatchState.RUNNING}),
                state=BatchState.SUCCEEDED,
                attempts=attempts,
            )
            return BatchItemResult(
                item_id=item.item_id,
                state=BatchState.SUCCEEDED,
                attempts=attempts,
                result=materialized,
            )

    def _cancelled(self, checkpoint: BatchCheckpoint) -> BatchItemResult:
        if checkpoint.state in {BatchState.SUCCEEDED, BatchState.FAILED}:
            return _terminal_checkpoint_result(checkpoint)
        skipped = self._checkpoint_store.update(
            checkpoint.plan_id,
            checkpoint.item_id,
            expected_states=frozenset({checkpoint.state}),
            state=BatchState.SKIPPED,
            attempts=checkpoint.attempts,
        )
        return BatchItemResult(
            item_id=skipped.item_id,
            state=BatchState.SKIPPED,
            attempts=skipped.attempts,
            detail="cancelled",
        )

    def _record_failure(
        self,
        checkpoint: BatchCheckpoint,
        attempts: int,
        failure: BatchFailure,
    ) -> BatchItemResult:
        self._checkpoint_store.update(
            checkpoint.plan_id,
            checkpoint.item_id,
            expected_states=frozenset({BatchState.RUNNING}),
            state=BatchState.FAILED,
            attempts=attempts,
            failure=failure,
        )
        return BatchItemResult(
            item_id=checkpoint.item_id,
            state=BatchState.FAILED,
            attempts=attempts,
            failure=failure,
        )

    def _source_changed(self, checkpoint: BatchCheckpoint) -> BatchItemResult:
        if checkpoint.state is BatchState.SUCCEEDED:
            checkpoint = self._checkpoint_store.update(
                checkpoint.plan_id,
                checkpoint.item_id,
                expected_states=frozenset({BatchState.SUCCEEDED}),
                state=BatchState.PENDING,
                attempts=checkpoint.attempts,
            )
        self._checkpoint_store.update(
            checkpoint.plan_id,
            checkpoint.item_id,
            expected_states=frozenset({checkpoint.state}),
            state=BatchState.SOURCE_CHANGED,
            attempts=checkpoint.attempts,
            failure=_SOURCE_CHANGED_FAILURE,
        )
        return BatchItemResult(
            item_id=checkpoint.item_id,
            state=BatchState.SOURCE_CHANGED,
            attempts=checkpoint.attempts,
            failure=_SOURCE_CHANGED_FAILURE,
        )


def _terminal_checkpoint_result(checkpoint: BatchCheckpoint) -> BatchItemResult:
    return BatchItemResult(
        item_id=checkpoint.item_id,
        state=checkpoint.state,
        attempts=checkpoint.attempts,
        failure=checkpoint.failure,
    )


def _source_is_unchanged(item: BatchItem) -> bool:
    try:
        observed = _observe_regular_file(item.source_path, expected_snapshot=item.snapshot)
    except DirectoryDiscoveryError:
        return False
    return observed is not None and observed[0] == item.sha256 and observed[1] == item.snapshot
