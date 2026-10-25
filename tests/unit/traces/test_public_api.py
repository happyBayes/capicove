"""Tests for the deliberate trace package public API."""

from __future__ import annotations

import capicove.traces as traces


def test_trace_public_api_is_sorted_and_complete() -> None:
    assert traces.__all__ == sorted(traces.__all__)
    assert {
        "EpisodeCompleted",
        "EpisodeStarted",
        "MediaReference",
        "TraceRecord",
        "TraceRecordDraft",
        "TraceRecordType",
    } <= set(traces.__all__)
