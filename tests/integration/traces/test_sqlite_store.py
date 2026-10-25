"""Integration tests for the private append-only SQLite trace store."""

from __future__ import annotations

import stat
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from capicove.contracts import PrivacyClass, ToolStatus
from capicove.traces import (
    EpisodeCompleted,
    EpisodeFeedback,
    EpisodeStarted,
    EpisodeStatus,
    MediaContent,
    MessageObserved,
    MessageRole,
    TextContent,
    ToolCallObserved,
    ToolResultObserved,
    TraceConflictError,
    TraceError,
    TraceLifecycleError,
    TraceModality,
    TraceQuery,
    TraceRecordDraft,
    TraceStoreError,
)
from capicove.traces._adapters.sqlite import SqliteTraceStore
from capicove.traces.codec import trace_record_sha256


SHA256 = "a" * 64
BASE_TIME = datetime(2026, 8, 18, 12, tzinfo=UTC)


def _store(tmp_path: Path) -> SqliteTraceStore:
    return SqliteTraceStore(tmp_path / "traces.sqlite3")


def _start(
    episode_id: str,
    *,
    at: datetime = BASE_TIME,
    privacy: PrivacyClass = PrivacyClass.INTERNAL,
) -> TraceRecordDraft:
    return TraceRecordDraft(
        episode_id=episode_id,
        trace_id=f"trace-{episode_id}",
        occurred_at=at,
        payload=EpisodeStarted(
            goal="Test persistence.",
            workspace_id="workspace-1",
            actor_id="actor-1",
            environment_id="environment-1",
            environment_version=1,
            tool_catalog_sha256=SHA256,
            active_asset_versions={},
        ),
        privacy_class=privacy,
    )


def _message(
    episode_id: str,
    index: int,
    *,
    modality: TraceModality | None = None,
    privacy: PrivacyClass = PrivacyClass.INTERNAL,
) -> TraceRecordDraft:
    content = (
        (MediaContent(f"media-{index}"),)
        if modality is not None
        else (TextContent(f"message {index}"),)
    )
    return TraceRecordDraft(
        episode_id=episode_id,
        trace_id=f"trace-{episode_id}",
        occurred_at=BASE_TIME + timedelta(seconds=index),
        payload=MessageObserved(MessageRole.USER, content),
        modalities=() if modality is None else (modality,),
        privacy_class=privacy,
    )


def _tool_call(episode_id: str) -> TraceRecordDraft:
    return TraceRecordDraft(
        episode_id=episode_id,
        trace_id=f"trace-{episode_id}",
        occurred_at=BASE_TIME + timedelta(seconds=2),
        payload=ToolCallObserved("capicove.image.ocr", 1, {}, SHA256, 1),
        request_id="request-1",
        modalities=(TraceModality.IMAGE,),
    )


def _tool_result(episode_id: str, status: ToolStatus = ToolStatus.SUCCEEDED) -> TraceRecordDraft:
    return TraceRecordDraft(
        episode_id=episode_id,
        trace_id=f"trace-{episode_id}",
        occurred_at=BASE_TIME + timedelta(seconds=3),
        payload=ToolResultObserved(
            "capicove.image.ocr",
            1,
            status,
            SHA256,
            result={"text": "ok"} if status is ToolStatus.SUCCEEDED else None,
        ),
        request_id="request-1",
        modalities=(TraceModality.IMAGE,),
        error=None if status is ToolStatus.SUCCEEDED else TraceError("internal", "failed", False),
    )


def _complete(
    episode_id: str,
    status: EpisodeStatus = EpisodeStatus.SUCCEEDED,
    *,
    at: datetime = BASE_TIME + timedelta(seconds=4),
) -> TraceRecordDraft:
    return TraceRecordDraft(
        episode_id=episode_id,
        trace_id=f"trace-{episode_id}",
        occurred_at=at,
        payload=EpisodeCompleted(status),
    )


def _append_complete_episode(store: SqliteTraceStore, episode_id: str) -> None:
    drafts = (
        _start(episode_id),
        _message(episode_id, 1, modality=TraceModality.IMAGE),
        _tool_call(episode_id),
        _tool_result(episode_id),
        _complete(episode_id),
    )
    for index, draft in enumerate(drafts):
        store.append(draft, idempotency_key=f"event-{index}")


def test_store_persists_records_sequences_and_hash_chain_across_restart(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _append_complete_episode(store, "episode-1")
    first_read = store.get_episode("episode-1")

    reopened = SqliteTraceStore(tmp_path / "traces.sqlite3")
    second_read = reopened.get_episode("episode-1")

    assert second_read == first_read
    assert tuple(record.sequence for record in second_read) == (0, 1, 2, 3, 4)
    assert second_read[0].previous_record_sha256 is None
    assert tuple(record.previous_record_sha256 for record in second_read[1:]) == tuple(
        trace_record_sha256(previous) for previous in second_read[:-1]
    )


def test_store_uses_first_observation_time_for_idempotent_append_retries(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    start = _start("episode-1")
    first = store.append(start, idempotency_key="episode.started")
    assert store.append(start, idempotency_key="episode.started") == first

    retry = _start("episode-1", at=BASE_TIME + timedelta(seconds=1))
    assert store.append(retry, idempotency_key="episode.started") == first

    with pytest.raises(TraceConflictError):
        store.append(
            replace(
                retry,
                payload=replace(
                    cast(EpisodeStarted, retry.payload),
                    goal="Different persistence goal.",
                ),
            ),
            idempotency_key="episode.started",
        )

    store.append(_complete("episode-1"), idempotency_key="episode.completed")
    with pytest.raises(TraceLifecycleError):
        store.append(_message("episode-1", 5), idempotency_key="late-message")

    feedback = TraceRecordDraft(
        episode_id="episode-1",
        trace_id="trace-episode-1",
        occurred_at=BASE_TIME + timedelta(seconds=6),
        payload=EpisodeFeedback("quality", {"score": 1}, "reviewer"),
    )
    appended = store.append(feedback, idempotency_key="feedback-1")
    assert appended.sequence == 2


def test_store_creates_owner_only_state_directory_and_database(tmp_path: Path) -> None:
    state_directory = tmp_path / "nested" / "state"
    database = state_directory / "traces.sqlite3"

    store = SqliteTraceStore(database)
    store.append(_start("episode-1"), idempotency_key="episode.started")

    assert stat.S_IMODE(state_directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(database.stat().st_mode) == 0o600
    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{database}{suffix}")
        if sidecar.exists():
            assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600


def test_store_rejects_insecure_existing_state_directory(tmp_path: Path) -> None:
    state_directory = tmp_path / "shared"
    state_directory.mkdir(mode=0o755)
    state_directory.chmod(0o755)

    with pytest.raises(TraceStoreError, match="owner-only"):
        SqliteTraceStore(state_directory / "traces.sqlite3")


def test_concurrent_appends_assign_unique_contiguous_sequences(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append(_start("episode-1"), idempotency_key="episode.started")

    def append(index: int) -> int:
        return store.append(
            _message("episode-1", index), idempotency_key=f"message-{index}"
        ).sequence

    with ThreadPoolExecutor(max_workers=2) as executor:
        sequences = tuple(executor.map(append, (1, 2)))

    assert set(sequences) == {1, 2}
    assert tuple(record.sequence for record in store.get_episode("episode-1")) == (0, 1, 2)


def test_filters_select_whole_episodes_and_summarize_privacy_and_incomplete(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _append_complete_episode(store, "episode-image")

    store.append(
        _start("episode-audio", at=BASE_TIME + timedelta(days=1)),
        idempotency_key="episode.started",
    )
    store.append(
        _message("episode-audio", 1, modality=TraceModality.AUDIO),
        idempotency_key="message-1",
    )

    store.append(
        _start("episode-restricted", privacy=PrivacyClass.RESTRICTED),
        idempotency_key="episode.started",
    )
    store.append(
        _complete("episode-restricted", EpisodeStatus.FAILED),
        idempotency_key="episode.completed",
    )

    query = TraceQuery(
        modalities=frozenset({TraceModality.IMAGE}),
        tool_ids=frozenset({"capicove.image.ocr"}),
        tool_statuses=frozenset({ToolStatus.SUCCEEDED}),
        episode_statuses=frozenset({EpisodeStatus.SUCCEEDED}),
        started_after=BASE_TIME - timedelta(seconds=1),
        completed_before=BASE_TIME + timedelta(days=1),
    )
    with store.open_snapshot(query) as snapshot:
        records = tuple(snapshot.iter_records())
        assert snapshot.summary.selected_episode_count == 1
    assert {record.episode_id for record in records} == {"episode-image"}
    assert tuple(record.sequence for record in records) == (0, 1, 2, 3, 4)

    with store.open_snapshot(TraceQuery(include_incomplete=True)) as snapshot:
        assert snapshot.summary.selected_episode_count == 2
        assert snapshot.summary.selected_incomplete_episode_count == 1
        assert snapshot.summary.omitted_restricted_episode_count == 1

    with store.open_snapshot(
        TraceQuery(include_incomplete=True, include_restricted=True)
    ) as snapshot:
        assert snapshot.summary.selected_episode_count == 3


@pytest.mark.parametrize(
    "query",
    (
        TraceQuery(episode_ids=frozenset({"missing') OR 1=1 --"})),
        TraceQuery(tool_ids=frozenset({"missing') OR 1=1 --"})),
    ),
    ids=("episode_ids", "tool_ids"),
)
def test_sql_looking_filter_values_remain_bound_parameters(
    tmp_path: Path,
    query: TraceQuery,
) -> None:
    store = _store(tmp_path)
    _append_complete_episode(store, "episode-1")

    with store.open_snapshot(query) as snapshot:
        records = tuple(snapshot.iter_records())
        assert snapshot.summary.selected_episode_count == 0

    assert records == ()
    assert len(store.get_episode("episode-1")) == 5


def test_snapshot_summary_and_iterator_share_one_concurrent_read_view(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append(_start("episode-1"), idempotency_key="episode.started")

    with store.open_snapshot(TraceQuery(include_incomplete=True)) as snapshot:
        assert snapshot.summary.selected_episode_count == 1
        store.append(_message("episode-1", 1), idempotency_key="message-1")
        records = tuple(snapshot.iter_records())

    assert tuple(record.sequence for record in records) == (0,)
    with store.open_snapshot(TraceQuery(include_incomplete=True)) as snapshot:
        assert tuple(record.sequence for record in snapshot.iter_records()) == (0, 1)


def test_returned_records_cannot_mutate_persisted_state(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append(_start("episode-1"), idempotency_key="episode.started")
    record = store.get_episode("episode-1")[0]

    with pytest.raises(FrozenInstanceError):
        record.sequence = 10  # type: ignore[misc]
    assert store.get_episode("episode-1")[0].sequence == 0
