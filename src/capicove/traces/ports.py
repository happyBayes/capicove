"""Database-independent ports and query values for trace persistence."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from types import TracebackType
from typing import Protocol, Self

from capicove.contracts import ToolSpec, ToolStatus
from capicove.traces.content import MediaReference, TraceModality
from capicove.traces.events import EpisodeStatus
from capicove.traces.models import TraceRecord, TraceRecordDraft


def _validate_datetime(field_name: str, value: datetime | None) -> None:
    if value is not None and (not isinstance(value, datetime) or value.utcoffset() is None):
        raise ValueError(f"{field_name} must be timezone-aware")


def _validate_string_filter(field_name: str, value: frozenset[str] | None) -> None:
    if value is None:
        return
    if not isinstance(value, frozenset) or not value:
        raise ValueError(f"{field_name} must be a non-empty frozenset")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{field_name} must contain non-blank strings")


def _validate_enum_filter(
    field_name: str,
    value: frozenset[object] | None,
    enum_type: type[object],
) -> None:
    if value is None:
        return
    if not isinstance(value, frozenset) or not value:
        raise ValueError(f"{field_name} must be a non-empty frozenset")
    if any(not isinstance(item, enum_type) for item in value):
        raise ValueError(f"{field_name} contains an unsupported value")


class TraceStoreError(RuntimeError):
    """Report a stable trace persistence failure without adapter details."""


class TracePostCommitPrivacyError(TraceStoreError):
    """Report a failed privacy recheck after append committed; retry with the same key."""


class TraceConflictError(TraceStoreError):
    """Report reuse of one append identity for different content."""


class TraceLifecycleError(TraceStoreError):
    """Report an invalid append for the current Episode lifecycle."""


@dataclass(frozen=True, slots=True)
class TraceQuery:
    """Select complete Episodes for reading or export."""

    episode_ids: frozenset[str] | None = None
    started_after: datetime | None = None
    completed_before: datetime | None = None
    modalities: frozenset[TraceModality] | None = None
    tool_ids: frozenset[str] | None = None
    tool_statuses: frozenset[ToolStatus] | None = None
    episode_statuses: frozenset[EpisodeStatus] | None = None
    include_incomplete: bool = False
    include_restricted: bool = False

    def __post_init__(self) -> None:
        _validate_string_filter("episode_ids", self.episode_ids)
        _validate_datetime("started_after", self.started_after)
        _validate_datetime("completed_before", self.completed_before)
        if (
            self.started_after is not None
            and self.completed_before is not None
            and self.started_after > self.completed_before
        ):
            raise ValueError("query time bounds must be ordered")
        _validate_enum_filter("modalities", self.modalities, TraceModality)
        _validate_string_filter("tool_ids", self.tool_ids)
        _validate_enum_filter("tool_statuses", self.tool_statuses, ToolStatus)
        _validate_enum_filter("episode_statuses", self.episode_statuses, EpisodeStatus)
        if not isinstance(self.include_incomplete, bool):
            raise ValueError("include_incomplete must be a boolean")
        if not isinstance(self.include_restricted, bool):
            raise ValueError("include_restricted must be a boolean")


@dataclass(frozen=True, slots=True)
class TraceSelectionSummary:
    """Summarize the Episode selection represented by one fixed snapshot."""

    selected_episode_count: int
    selected_incomplete_episode_count: int
    omitted_restricted_episode_count: int

    def __post_init__(self) -> None:
        for field_name in (
            "selected_episode_count",
            "selected_incomplete_episode_count",
            "omitted_restricted_episode_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.selected_incomplete_episode_count > self.selected_episode_count:
            raise ValueError("incomplete Episode count cannot exceed selected Episode count")


DEFAULT_TRACE_QUERY = TraceQuery()


class TraceReadSnapshot(Protocol):
    """Provide one fixed, single-pass Episode selection."""

    @property
    def summary(self) -> TraceSelectionSummary: ...

    def iter_records(self) -> Iterator[TraceRecord]: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


class TraceReader(Protocol):
    """Open consistent snapshots of complete trace Episodes."""

    def open_snapshot(self, query: TraceQuery = DEFAULT_TRACE_QUERY, /) -> TraceReadSnapshot: ...


class TraceStore(TraceReader, Protocol):
    """Append immutable records and read complete Episodes."""

    def append(
        self,
        draft: TraceRecordDraft,
        /,
        *,
        idempotency_key: str,
    ) -> TraceRecord: ...

    def get_episode(self, episode_id: str, /) -> tuple[TraceRecord, ...]: ...


class TraceMediaResolver(Protocol):
    """Resolve a sensitive Tool argument to immutable media identity."""

    def resolve(
        self,
        spec: ToolSpec,
        field_path: str,
        value: object,
        /,
    ) -> MediaReference | None: ...
