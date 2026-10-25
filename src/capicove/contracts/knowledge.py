"""Standard-library-only temporal graph contracts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import IntEnum
from typing import Literal

from capicove.contracts.evidence import EvidenceRef, EvidenceSpan


def text(value: str, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.encode()) > maximum:
        raise ValueError("invalid bounded text")
    return value


def utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() != timedelta(0):
        raise ValueError("time must be explicit UTC")
    return value.astimezone(UTC)


def micros(value: datetime) -> int:
    delta = utc(value) - datetime(1970, 1, 1, tzinfo=UTC)
    return (delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class TimeInterval:
    start: datetime | None = None
    end: datetime | None = None

    def __post_init__(self) -> None:
        if self.start is not None:
            utc(self.start)
        if self.end is not None:
            utc(self.end)
        if self.start is not None and self.end is not None and self.end <= self.start:
            raise ValueError("empty interval")

    def contains(self, instant: datetime) -> bool:
        utc(instant)
        return (self.start is None or self.start <= instant) and (
            self.end is None or instant < self.end
        )


class Authority(IntEnum):
    HEURISTIC = 1
    MODEL = 2
    PROCESSOR = 3
    SIGNED_SOURCE = 4
    USER = 5


@dataclass(frozen=True, slots=True)
class TypedLiteral:
    kind: Literal["string", "boolean", "int64", "decimal", "date", "datetime"]
    value: str | bool | int

    def __post_init__(self) -> None:
        value = self.value
        if self.kind == "string":
            text(value, 32768)  # type: ignore[arg-type]
        elif self.kind == "boolean":
            if type(value) is not bool:
                raise ValueError("invalid boolean")
        elif self.kind == "int64":
            if type(value) is not int or not -(2**63) <= value < 2**63:
                raise ValueError("invalid int64")
        elif self.kind == "decimal":
            if not isinstance(value, str) or len(value) > 512:
                raise ValueError("invalid decimal")
            try:
                number = Decimal(value)
            except InvalidOperation:
                raise ValueError("invalid decimal") from None
            if not number.is_finite() or abs(number.adjusted()) > 128:
                raise ValueError("invalid decimal")
            normalized = format(number, "f")
            if "." in normalized:
                normalized = normalized.rstrip("0").rstrip(".")
            if len(normalized) > 512:
                raise ValueError("invalid decimal")
            object.__setattr__(self, "value", "0" if number == 0 else normalized)
        elif self.kind == "date":
            if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
                raise ValueError("invalid date")
        elif self.kind == "datetime":
            if not isinstance(value, str):
                raise ValueError("invalid datetime")
            object.__setattr__(self, "value", utc(datetime.fromisoformat(value)).isoformat())
        else:
            raise ValueError("unknown literal type")


@dataclass(frozen=True, slots=True)
class EntityRef:
    entity_id: str

    def __post_init__(self) -> None:
        text(self.entity_id)


@dataclass(frozen=True, slots=True)
class Predicate:
    name: str
    subject_types: tuple[str, ...]
    object_kind: str
    object_types: tuple[str, ...]
    cardinality: Literal["one", "many"]
    temporal_kind: Literal["static", "interval"]

    def __post_init__(self) -> None:
        text(self.name)
        if (
            not isinstance(self.subject_types, tuple)
            or not isinstance(self.object_types, tuple)
            or not self.subject_types
            or not self.object_types
            or self.object_kind not in {"entity", "literal"}
            or self.cardinality not in {"one", "many"}
            or self.temporal_kind not in {"static", "interval"}
        ):
            raise ValueError("invalid predicate")
        for value in (*self.subject_types, *self.object_types):
            text(value)


@dataclass(frozen=True, slots=True)
class Entity:
    entity_id: str
    workspace_ref: str
    scope_ref: str
    entity_type: str
    display_name: str
    ontology_version: str


@dataclass(frozen=True, slots=True)
class MemoryEpisode:
    episode_id: str
    workspace_ref: str
    scope_ref: str
    episode_type: Literal["message", "document", "json", "fact_assertion"]
    evidence_refs: tuple[EvidenceRef, ...]
    content_sha256: str
    event_time: datetime
    recorded_at: datetime
    source_ref: str
    schema_version: int = 1
    content_length: int = 0

    def __post_init__(self) -> None:
        for value in (self.episode_id, self.workspace_ref, self.scope_ref, self.source_ref):
            text(value)
        if (
            self.episode_type not in {"message", "document", "json", "fact_assertion"}
            or not isinstance(self.evidence_refs, tuple)
            or any(not isinstance(ref, EvidenceRef) for ref in self.evidence_refs)
            or not self.evidence_refs
        ):
            raise ValueError("invalid Episode")
        if EvidenceRef(self.content_sha256) not in self.evidence_refs or self.schema_version != 1:
            raise ValueError("invalid Episode Evidence/version")
        utc(self.event_time)
        utc(self.recorded_at)
        if type(self.content_length) is not int or not 0 <= self.content_length <= 1048576:
            raise ValueError("invalid Episode content length")


@dataclass(frozen=True, slots=True)
class FactProposal:
    subject_ref: EntityRef
    predicate: str
    object: EntityRef | TypedLiteral
    validity: TimeInterval
    evidence: tuple[EvidenceSpan, ...]
    event_time: datetime
    authority: Authority = Authority.USER
    extraction_confidence: float | None = None

    def __post_init__(self) -> None:
        text(self.predicate)
        utc(self.event_time)
        if (
            not isinstance(self.subject_ref, EntityRef)
            or not isinstance(self.object, (EntityRef, TypedLiteral))
            or not isinstance(self.validity, TimeInterval)
            or not isinstance(self.authority, Authority)
            or not isinstance(self.evidence, tuple)
            or not self.evidence
            or any(not isinstance(e, EvidenceSpan) for e in self.evidence)
        ):
            raise ValueError("invalid fact proposal")
        confidence = self.extraction_confidence
        if confidence is not None and (
            type(confidence) not in (float, int)
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("invalid extraction confidence")


@dataclass(frozen=True, slots=True)
class TemporalFact:
    fact_id: str
    fact_version_id: str
    subject_ref: EntityRef
    predicate: str
    object: EntityRef | TypedLiteral
    validity: TimeInterval
    knowledge: TimeInterval
    evidence: tuple[EvidenceSpan, ...]
    assertion_hash: str
    conflict_key: str
    ontology_version: str
    authority: Authority
    extraction_confidence: float | None
    state: Literal["accepted", "disputed", "rejected"]
    parent_version_id: str | None = None
    superseding_version_id: str | None = None

    def __post_init__(self) -> None:
        for value in (
            self.fact_id,
            self.fact_version_id,
            self.predicate,
            self.conflict_key,
            self.ontology_version,
        ):
            text(value)
        EvidenceRef(self.assertion_hash)
        if (
            self.state not in {"accepted", "disputed", "rejected"}
            or not isinstance(self.subject_ref, EntityRef)
            or not isinstance(self.object, (EntityRef, TypedLiteral))
            or not isinstance(self.authority, Authority)
        ):
            raise ValueError("invalid temporal fact")
        if (
            not isinstance(self.validity, TimeInterval)
            or not isinstance(self.knowledge, TimeInterval)
            or self.knowledge.start is None
        ):
            raise ValueError("invalid fact intervals")
        if (
            not isinstance(self.evidence, tuple)
            or not self.evidence
            or any(not isinstance(span, EvidenceSpan) for span in self.evidence)
        ):
            raise ValueError("invalid fact provenance")
