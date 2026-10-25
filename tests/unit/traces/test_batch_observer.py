"""Tests for mapping terminal batch facts into trace records."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from capicove.batch import (
    BatchFailure,
    BatchItem,
    BatchItemResult,
    BatchModality,
    BatchState,
    PlannedBatchItem,
    SourceSnapshot,
)
from capicove.traces import BatchItemObserved, TraceModality
from capicove.traces.batch import TraceBatchObserver


SHA256 = "a" * 64


class FakeSession:
    """Capture trace-side batch recording calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, BatchItemObserved]] = []

    def record_batch_item(self, *, event_id: str, payload: BatchItemObserved) -> object:
        self.calls.append((event_id, payload))
        return object()


def _planned(tmp_path: Path) -> PlannedBatchItem:
    source = tmp_path / "private" / "input.wav"
    return PlannedBatchItem(
        item=BatchItem(
            item_id=f"item:{SHA256}",
            modality=BatchModality.AUDIO,
            source_path=source,
            relative_path=PurePosixPath("input.wav"),
            media_type="audio/wav",
            sha256=SHA256,
            snapshot=SourceSnapshot(1, 2, 12, 3, 4),
        ),
        cache_key=f"cache:{SHA256}",
    )


@pytest.mark.parametrize(
    "result",
    [
        BatchItemResult(f"item:{SHA256}", BatchState.SUCCEEDED, 1, result={"text": "ok"}),
        BatchItemResult(
            f"item:{SHA256}",
            BatchState.SUCCEEDED,
            0,
            cache_hit=True,
            result={"text": "cached"},
        ),
        BatchItemResult(f"item:{SHA256}", BatchState.SKIPPED, 1, detail="no speech"),
        BatchItemResult(
            f"item:{SHA256}",
            BatchState.FAILED,
            1,
            failure=BatchFailure("INVALID", "invalid", False),
        ),
        BatchItemResult(
            f"item:{SHA256}",
            BatchState.SOURCE_CHANGED,
            0,
            failure=BatchFailure("SOURCE_CHANGED", "changed", False),
        ),
        BatchItemResult(f"item:{SHA256}", BatchState.SUCCEEDED, 3, result={"text": "retried"}),
    ],
)
def test_trace_batch_observer_maps_every_terminal_fact_without_source_path(
    tmp_path: Path,
    result: BatchItemResult,
) -> None:
    session = FakeSession()
    planned = _planned(tmp_path)
    observer = TraceBatchObserver(session)  # type: ignore[arg-type]

    observer.on_item_completed(f"plan:{SHA256}", planned, result)

    assert len(session.calls) == 1
    event_id, payload = session.calls[0]
    assert event_id == (
        f"batch.item:plan:{SHA256}:item:{SHA256}:{result.attempts}:{result.state.value}"
    )
    assert payload.plan_id == f"plan:{SHA256}"
    assert payload.item_id == f"item:{SHA256}"
    assert payload.state is result.state
    assert payload.attempts == result.attempts
    assert payload.cache_hit is result.cache_hit
    assert payload.modality is TraceModality.AUDIO
    assert payload.media_type == "audio/wav"
    assert payload.size_bytes == 12
    assert payload.source_sha256 == SHA256
    assert payload.cache_key == f"cache:{SHA256}"
    assert str(planned.item.source_path) not in repr(payload)
