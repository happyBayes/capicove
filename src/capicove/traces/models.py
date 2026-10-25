"""Versioned draft and persisted Agent operation trace envelopes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from capicove.contracts import PrivacyClass, ToolStatus
from capicove.traces.content import (
    MediaReference,
    TraceError,
    TraceLineage,
    TraceModality,
    TraceWarning,
)
from capicove.traces.events import (
    ToolCallObserved,
    ToolResultObserved,
    TracePayload,
    TraceRecordType,
    is_trace_payload,
    record_type_for_payload,
)


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class TraceRecordDraft:
    """Describe a validated record before store-assigned sequence and identity."""

    episode_id: str
    trace_id: str
    occurred_at: datetime
    payload: TracePayload
    modalities: tuple[TraceModality, ...] = ()
    step_id: str | None = None
    request_id: str | None = None
    parent_record_id: str | None = None
    media: tuple[MediaReference, ...] = ()
    lineage: TraceLineage | None = None
    warnings: tuple[TraceWarning, ...] = ()
    error: TraceError | None = None
    privacy_class: PrivacyClass = PrivacyClass.INTERNAL

    def __post_init__(self) -> None:
        _validate_envelope(self)

    @property
    def record_type(self) -> TraceRecordType:
        """Return the record type implied by the closed payload class."""
        return record_type_for_payload(self.payload)


@dataclass(frozen=True, slots=True)
class TraceRecord:
    """Describe one immutable persisted record in an Episode hash chain."""

    schema_version: int
    record_type: TraceRecordType
    record_version: int
    record_id: str
    episode_id: str
    trace_id: str
    sequence: int
    occurred_at: datetime
    payload: TracePayload
    modalities: tuple[TraceModality, ...] = ()
    step_id: str | None = None
    request_id: str | None = None
    parent_record_id: str | None = None
    media: tuple[MediaReference, ...] = ()
    lineage: TraceLineage | None = None
    warnings: tuple[TraceWarning, ...] = ()
    error: TraceError | None = None
    privacy_class: PrivacyClass = PrivacyClass.INTERNAL
    previous_record_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if self.record_version != 1:
            raise ValueError("record_version must be 1")
        if not isinstance(self.record_type, TraceRecordType):
            raise ValueError("record_type must be a TraceRecordType")
        _require_non_blank("record_id", self.record_id)
        _require_non_negative_int("sequence", self.sequence)
        _validate_envelope(self)
        if self.record_type is not record_type_for_payload(self.payload):
            raise ValueError("record_type must match payload")
        if self.sequence == 0:
            if self.previous_record_sha256 is not None:
                raise ValueError("previous_record_sha256 must be absent at sequence zero")
        elif (
            not isinstance(self.previous_record_sha256, str)
            or _SHA256_PATTERN.fullmatch(self.previous_record_sha256) is None
        ):
            raise ValueError("previous_record_sha256 must be a lowercase SHA-256 hash")


def _validate_envelope(value: TraceRecordDraft | TraceRecord) -> None:
    _require_non_blank("episode_id", value.episode_id)
    _require_non_blank("trace_id", value.trace_id)
    if not isinstance(value.occurred_at, datetime) or value.occurred_at.utcoffset() is None:
        raise ValueError("occurred_at must be timezone-aware")
    if not is_trace_payload(value.payload):
        raise ValueError("payload must be a supported trace payload")
    if len(set(value.modalities)) != len(value.modalities):
        raise ValueError("modalities must not contain duplicates")
    if not all(isinstance(modality, TraceModality) for modality in value.modalities):
        raise ValueError("modalities must contain only TraceModality values")
    for field_name in ("step_id", "request_id", "parent_record_id"):
        field_value = getattr(value, field_name)
        if field_value is not None:
            _require_non_blank(field_name, field_value)
    if isinstance(value.payload, (ToolCallObserved, ToolResultObserved)):
        _require_non_blank("request_id", value.request_id)
    if not all(isinstance(reference, MediaReference) for reference in value.media):
        raise ValueError("media must contain only MediaReference values")
    media_ids = tuple(reference.media_id for reference in value.media)
    if len(set(media_ids)) != len(media_ids):
        raise ValueError("media must not contain duplicate media IDs")
    if value.lineage is not None and not isinstance(value.lineage, TraceLineage):
        raise ValueError("lineage must be a TraceLineage")
    if not all(isinstance(warning, TraceWarning) for warning in value.warnings):
        raise ValueError("warnings must contain only TraceWarning values")
    if value.error is not None and not isinstance(value.error, TraceError):
        raise ValueError("error must be a TraceError")
    if not isinstance(value.privacy_class, PrivacyClass):
        raise ValueError("privacy_class must be a PrivacyClass")
    if isinstance(value.payload, ToolResultObserved):
        if value.payload.status is ToolStatus.SUCCEEDED and value.error is not None:
            raise ValueError("a successful Tool result must not contain an error")
        if value.payload.status is not ToolStatus.SUCCEEDED and value.error is None:
            raise ValueError("a failed or rejected Tool result must contain an error")


def _require_non_blank(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def _require_non_negative_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
