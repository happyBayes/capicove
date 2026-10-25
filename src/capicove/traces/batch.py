"""Trace-side adapter for the batch-owned terminal completion observer port."""

from __future__ import annotations

from capicove.batch import BatchItemResult, BatchModality, PlannedBatchItem
from capicove.traces.content import TraceModality
from capicove.traces.events import BatchItemObserved
from capicove.traces.recorder import TraceSession


_MODALITIES = {
    BatchModality.AUDIO: TraceModality.AUDIO,
    BatchModality.IMAGE: TraceModality.IMAGE,
    BatchModality.VIDEO: TraceModality.VIDEO,
}


class TraceBatchObserver:
    """Append terminal batch facts without exposing source filesystem paths."""

    def __init__(self, session: TraceSession) -> None:
        self._session = session

    def on_item_completed(
        self,
        plan_id: str,
        planned: PlannedBatchItem,
        result: BatchItemResult,
        /,
    ) -> None:
        """Map one terminal item into a stable idempotent trace event."""
        detail = result.detail
        if detail is None and result.failure is not None:
            detail = result.failure.message
        payload = BatchItemObserved(
            plan_id=plan_id,
            item_id=planned.item.item_id,
            state=result.state,
            attempts=result.attempts,
            cache_hit=result.cache_hit,
            modality=_MODALITIES[planned.item.modality],
            media_type=planned.item.media_type,
            size_bytes=planned.item.size_bytes,
            source_sha256=planned.item.sha256,
            cache_key=planned.cache_key,
            detail=detail,
        )
        event_id = (
            f"batch.item:{plan_id}:{planned.item.item_id}:{result.attempts}:{result.state.value}"
        )
        self._session.record_batch_item(event_id=event_id, payload=payload)
