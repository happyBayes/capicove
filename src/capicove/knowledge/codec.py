"""Canonical projection codecs; source payloads never enter these records."""

from datetime import UTC, datetime, timedelta
from typing import Any, Literal, cast

from capicove.contracts.evidence import EvidenceRef, EvidenceSpan
from capicove.contracts.knowledge import (
    Authority,
    EntityRef,
    MemoryEpisode,
    TemporalFact,
    TimeInterval,
    TypedLiteral,
    micros,
)


def timestamp(value: int | None) -> datetime | None:
    return (
        None if value is None else datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=value)
    )


def interval(value: TimeInterval) -> list[int | None]:
    return [
        None if value.start is None else micros(value.start),
        None if value.end is None else micros(value.end),
    ]


def decode_interval(value: list[int | None]) -> TimeInterval:
    return TimeInterval(timestamp(value[0]), timestamp(value[1]))


def encode_object(value: EntityRef | TypedLiteral) -> dict[str, Any]:
    if isinstance(value, EntityRef):
        return {"kind": "entity", "value": value.entity_id}
    return {"kind": value.kind, "value": value.value}


def decode_object(value: dict[str, Any]) -> EntityRef | TypedLiteral:
    if value["kind"] == "entity":
        return EntityRef(value["value"])
    return TypedLiteral(value["kind"], value["value"])


def encode_span(value: EvidenceSpan) -> dict[str, Any]:
    return {"digest": value.reference.content_sha256, "start": value.start, "end": value.end}


def decode_span(value: dict[str, Any]) -> EvidenceSpan:
    return EvidenceSpan(EvidenceRef(value["digest"]), value["start"], value["end"])


def decode_episode(body: dict[str, Any]) -> MemoryEpisode:
    return MemoryEpisode(
        body["id"],
        body["workspace"],
        body["scope"],
        body["episode_type"],
        tuple(EvidenceRef(d) for d in body["evidence_refs"]),
        body["content_sha256"],
        cast(datetime, timestamp(body["event_time"])),
        cast(datetime, timestamp(body["recorded"])),
        body["source_ref"],
        content_length=body["content_length"],
    )


def decode_fact(body: dict[str, Any]) -> TemporalFact:
    return TemporalFact(
        body["id"],
        body["version"],
        EntityRef(body["subject"]),
        body["predicate"],
        decode_object(body["object"]),
        decode_interval(body["validity"]),
        TimeInterval(timestamp(body["recorded"]), timestamp(body.get("recorded_to"))),
        tuple(decode_span(e) for e in body["evidence"]),
        body["assertion_hash"],
        body["conflict_key"],
        body["ontology_version"],
        Authority(body["authority"]),
        body.get("extraction_confidence"),
        cast(Literal["accepted", "disputed", "rejected"], body["state"]),
        body.get("parent_version_id"),
        body.get("superseding_version_id"),
    )
