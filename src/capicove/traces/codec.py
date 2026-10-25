"""Canonical version 1 encoding and strict decoding for trace records."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, TypeVar, cast

from capicove.batch import BatchState
from capicove.contracts import PrivacyClass, ToolStatus
from capicove.traces._json import FrozenJson, thaw_json
from capicove.traces.content import (
    AgentLineage,
    LineageAvailability,
    MediaContent,
    MediaLocator,
    MediaLocatorKind,
    MediaReference,
    MessageRole,
    TextContent,
    TraceError,
    TraceLineage,
    TraceModality,
    TraceWarning,
)
from capicove.traces.events import (
    ArtifactProduced,
    BatchItemObserved,
    EpisodeCompleted,
    EpisodeFeedback,
    EpisodeStarted,
    EpisodeStatus,
    MessageObserved,
    ToolCallObserved,
    ToolResultObserved,
    TracePayload,
    TraceRecordType,
)
from capicove.traces.models import TraceRecord, TraceRecordDraft


_TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")
_EnumT = TypeVar("_EnumT", bound=StrEnum)


def encode_trace_record(record: TraceRecord, /) -> dict[str, object]:
    """Encode one persisted trace record into its canonical mapping shape."""
    return {
        "schema_version": record.schema_version,
        "record_type": record.record_type.value,
        "record_version": record.record_version,
        "record_id": record.record_id,
        "episode_id": record.episode_id,
        "trace_id": record.trace_id,
        "sequence": record.sequence,
        "occurred_at": _encode_timestamp(record.occurred_at),
        "payload": _encode_payload(record.payload),
        "modalities": [modality.value for modality in record.modalities],
        "step_id": record.step_id,
        "request_id": record.request_id,
        "parent_record_id": record.parent_record_id,
        "media": [_encode_media(reference) for reference in record.media],
        "lineage": _encode_lineage(record.lineage),
        "warnings": [_encode_warning(warning) for warning in record.warnings],
        "error": _encode_error(record.error),
        "privacy_class": record.privacy_class.value,
        "previous_record_sha256": record.previous_record_sha256,
    }


def encode_trace_draft(draft: TraceRecordDraft, /) -> dict[str, object]:
    """Encode a draft deterministically for append idempotency checks."""
    return {
        "record_type": draft.record_type.value,
        "episode_id": draft.episode_id,
        "trace_id": draft.trace_id,
        "occurred_at": _encode_timestamp(draft.occurred_at),
        "payload": _encode_payload(draft.payload),
        "modalities": [modality.value for modality in draft.modalities],
        "step_id": draft.step_id,
        "request_id": draft.request_id,
        "parent_record_id": draft.parent_record_id,
        "media": [_encode_media(reference) for reference in draft.media],
        "lineage": _encode_lineage(draft.lineage),
        "warnings": [_encode_warning(warning) for warning in draft.warnings],
        "error": _encode_error(draft.error),
        "privacy_class": draft.privacy_class.value,
    }


def decode_trace_record(value: Mapping[str, object], /) -> TraceRecord:
    """Strictly decode one version 1 trace record without dynamic imports."""
    record = _object(value, "record")
    _exact_keys(
        record,
        {
            "schema_version",
            "record_type",
            "record_version",
            "record_id",
            "episode_id",
            "trace_id",
            "sequence",
            "occurred_at",
            "payload",
            "modalities",
            "step_id",
            "request_id",
            "parent_record_id",
            "media",
            "lineage",
            "warnings",
            "error",
            "privacy_class",
            "previous_record_sha256",
        },
        "record",
    )
    record_type = _enum(TraceRecordType, record["record_type"], "record_type")
    payload = _decode_payload(record_type, record["payload"])
    try:
        return TraceRecord(
            schema_version=_integer(record["schema_version"], "schema_version"),
            record_type=record_type,
            record_version=_integer(record["record_version"], "record_version"),
            record_id=_string(record["record_id"], "record_id"),
            episode_id=_string(record["episode_id"], "episode_id"),
            trace_id=_string(record["trace_id"], "trace_id"),
            sequence=_integer(record["sequence"], "sequence"),
            occurred_at=_decode_timestamp(record["occurred_at"]),
            payload=payload,
            modalities=tuple(
                _enum(TraceModality, item, "modalities item")
                for item in _array(record["modalities"], "modalities")
            ),
            step_id=_optional_string(record["step_id"], "step_id"),
            request_id=_optional_string(record["request_id"], "request_id"),
            parent_record_id=_optional_string(record["parent_record_id"], "parent_record_id"),
            media=tuple(_decode_media(item) for item in _array(record["media"], "media")),
            lineage=_decode_lineage(record["lineage"]),
            warnings=tuple(
                _decode_warning(item) for item in _array(record["warnings"], "warnings")
            ),
            error=_decode_error(record["error"]),
            privacy_class=_enum(PrivacyClass, record["privacy_class"], "privacy_class"),
            previous_record_sha256=_optional_string(
                record["previous_record_sha256"], "previous_record_sha256"
            ),
        )
    except (TypeError, KeyError) as error:
        raise ValueError("record has an invalid field type") from error


def canonical_json_bytes(value: Mapping[str, object], /) -> bytes:
    """Serialize one mapping with the version 1 canonical JSON parameters."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("value must be finite JSON") from error


def canonical_record_bytes(record: TraceRecord, /) -> bytes:
    """Return canonical JSON bytes for one persisted record without a line feed."""
    return canonical_json_bytes(encode_trace_record(record))


def trace_record_sha256(record: TraceRecord, /) -> str:
    """Return the lowercase SHA-256 digest of canonical record bytes."""
    return hashlib.sha256(canonical_record_bytes(record)).hexdigest()


def _encode_payload(payload: TracePayload) -> dict[str, object]:
    if isinstance(payload, EpisodeStarted):
        return {
            "goal": payload.goal,
            "workspace_id": payload.workspace_id,
            "actor_id": payload.actor_id,
            "environment_id": payload.environment_id,
            "environment_version": payload.environment_version,
            "tool_catalog_sha256": payload.tool_catalog_sha256,
            "active_asset_versions": thaw_json(payload.active_asset_versions),
            "agent_lineage": _encode_agent_lineage(payload.agent_lineage),
        }
    if isinstance(payload, MessageObserved):
        content: list[dict[str, object]] = []
        for block in payload.content:
            if isinstance(block, TextContent):
                content.append({"type": "text", "text": block.text})
            else:
                content.append({"type": "media", "media_id": block.media_id})
        return {"role": payload.role.value, "content": content}
    if isinstance(payload, ToolCallObserved):
        return {
            "tool_id": payload.tool_id,
            "tool_version": payload.tool_version,
            "arguments": thaw_json(cast(FrozenJson, payload.arguments)),
            "arguments_schema_sha256": payload.arguments_schema_sha256,
            "attempt": payload.attempt,
        }
    if isinstance(payload, ToolResultObserved):
        return {
            "tool_id": payload.tool_id,
            "tool_version": payload.tool_version,
            "status": payload.status.value,
            "result_schema_sha256": payload.result_schema_sha256,
            "result": (
                None if payload.result is None else thaw_json(cast(FrozenJson, payload.result))
            ),
            "evidence_refs": list(payload.evidence_refs),
            "metrics": thaw_json(payload.metrics),
        }
    if isinstance(payload, ArtifactProduced):
        return {
            "artifact_ref": payload.artifact_ref,
            "media_id": payload.media_id,
            "derived_from": list(payload.derived_from),
        }
    if isinstance(payload, BatchItemObserved):
        return {
            "plan_id": payload.plan_id,
            "item_id": payload.item_id,
            "state": payload.state.value,
            "attempts": payload.attempts,
            "cache_hit": payload.cache_hit,
            "modality": payload.modality.value,
            "media_type": payload.media_type,
            "size_bytes": payload.size_bytes,
            "source_sha256": payload.source_sha256,
            "cache_key": payload.cache_key,
            "detail": payload.detail,
        }
    if isinstance(payload, EpisodeFeedback):
        return {
            "kind": payload.kind,
            "value": thaw_json(cast(FrozenJson, payload.value)),
            "provenance": payload.provenance,
            "target_record_id": payload.target_record_id,
        }
    if isinstance(payload, EpisodeCompleted):
        return {"status": payload.status.value, "outcome": payload.outcome}
    raise ValueError("payload must be a supported trace payload")


def _decode_payload(record_type: TraceRecordType, value: object) -> TracePayload:
    payload = _object(value, "payload")
    try:
        if record_type is TraceRecordType.EPISODE_STARTED:
            _exact_keys(
                payload,
                {
                    "goal",
                    "workspace_id",
                    "actor_id",
                    "environment_id",
                    "environment_version",
                    "tool_catalog_sha256",
                    "active_asset_versions",
                    "agent_lineage",
                },
                "episode.started payload",
            )
            active_assets = _object(payload["active_asset_versions"], "active_asset_versions")
            return EpisodeStarted(
                goal=_string(payload["goal"], "goal"),
                workspace_id=_string(payload["workspace_id"], "workspace_id"),
                actor_id=_string(payload["actor_id"], "actor_id"),
                environment_id=_string(payload["environment_id"], "environment_id"),
                environment_version=_integer(payload["environment_version"], "environment_version"),
                tool_catalog_sha256=_string(payload["tool_catalog_sha256"], "tool_catalog_sha256"),
                active_asset_versions={
                    key: _integer(item, f"active_asset_versions.{key}")
                    for key, item in active_assets.items()
                },
                agent_lineage=_decode_agent_lineage(payload["agent_lineage"]),
            )
        if record_type is TraceRecordType.MESSAGE_OBSERVED:
            _exact_keys(payload, {"role", "content"}, "message.observed payload")
            content = tuple(_decode_content(item) for item in _array(payload["content"], "content"))
            return MessageObserved(
                role=_enum(MessageRole, payload["role"], "role"),
                content=content,
            )
        if record_type is TraceRecordType.TOOL_CALL:
            _exact_keys(
                payload,
                {
                    "tool_id",
                    "tool_version",
                    "arguments",
                    "arguments_schema_sha256",
                    "attempt",
                },
                "tool.call payload",
            )
            return ToolCallObserved(
                tool_id=_string(payload["tool_id"], "tool_id"),
                tool_version=_integer(payload["tool_version"], "tool_version"),
                arguments=_object(payload["arguments"], "arguments"),
                arguments_schema_sha256=_string(
                    payload["arguments_schema_sha256"], "arguments_schema_sha256"
                ),
                attempt=_integer(payload["attempt"], "attempt"),
            )
        if record_type is TraceRecordType.TOOL_RESULT:
            _exact_keys(
                payload,
                {
                    "tool_id",
                    "tool_version",
                    "status",
                    "result_schema_sha256",
                    "result",
                    "evidence_refs",
                    "metrics",
                },
                "tool.result payload",
            )
            result_value = payload["result"]
            return ToolResultObserved(
                tool_id=_string(payload["tool_id"], "tool_id"),
                tool_version=_integer(payload["tool_version"], "tool_version"),
                status=_enum(ToolStatus, payload["status"], "status"),
                result_schema_sha256=_string(
                    payload["result_schema_sha256"], "result_schema_sha256"
                ),
                result=None if result_value is None else _object(result_value, "result"),
                evidence_refs=tuple(
                    _string(item, "evidence_refs item")
                    for item in _array(payload["evidence_refs"], "evidence_refs")
                ),
                metrics={
                    key: _number(item, f"metrics.{key}")
                    for key, item in _object(payload["metrics"], "metrics").items()
                },
            )
        if record_type is TraceRecordType.ARTIFACT_PRODUCED:
            _exact_keys(payload, {"artifact_ref", "media_id", "derived_from"}, "artifact payload")
            return ArtifactProduced(
                artifact_ref=_string(payload["artifact_ref"], "artifact_ref"),
                media_id=_string(payload["media_id"], "media_id"),
                derived_from=tuple(
                    _string(item, "derived_from item")
                    for item in _array(payload["derived_from"], "derived_from")
                ),
            )
        if record_type is TraceRecordType.BATCH_ITEM:
            _exact_keys(
                payload,
                {
                    "plan_id",
                    "item_id",
                    "state",
                    "attempts",
                    "cache_hit",
                    "modality",
                    "media_type",
                    "size_bytes",
                    "source_sha256",
                    "cache_key",
                    "detail",
                },
                "batch.item payload",
            )
            return BatchItemObserved(
                plan_id=_string(payload["plan_id"], "plan_id"),
                item_id=_string(payload["item_id"], "item_id"),
                state=_enum(BatchState, payload["state"], "state"),
                attempts=_integer(payload["attempts"], "attempts"),
                cache_hit=_boolean(payload["cache_hit"], "cache_hit"),
                modality=_enum(TraceModality, payload["modality"], "modality"),
                media_type=_string(payload["media_type"], "media_type"),
                size_bytes=_integer(payload["size_bytes"], "size_bytes"),
                source_sha256=_string(payload["source_sha256"], "source_sha256"),
                cache_key=_string(payload["cache_key"], "cache_key"),
                detail=_optional_string(payload["detail"], "detail"),
            )
        if record_type is TraceRecordType.EPISODE_FEEDBACK:
            _exact_keys(
                payload,
                {"kind", "value", "provenance", "target_record_id"},
                "episode.feedback payload",
            )
            return EpisodeFeedback(
                kind=_string(payload["kind"], "kind"),
                value=_object(payload["value"], "value"),
                provenance=_string(payload["provenance"], "provenance"),
                target_record_id=_optional_string(payload["target_record_id"], "target_record_id"),
            )
        _exact_keys(payload, {"status", "outcome"}, "episode.completed payload")
        return EpisodeCompleted(
            status=_enum(EpisodeStatus, payload["status"], "status"),
            outcome=_optional_string(payload["outcome"], "outcome"),
        )
    except KeyError as error:
        raise ValueError("payload is missing a required field") from error


def _encode_media(reference: MediaReference) -> dict[str, object]:
    return {
        "media_id": reference.media_id,
        "modality": reference.modality.value,
        "media_type": reference.media_type,
        "sha256": reference.sha256,
        "size_bytes": reference.size_bytes,
        "evidence_ref": reference.evidence_ref,
        "locator": (
            None
            if reference.locator is None
            else {"kind": reference.locator.kind.value, "value": reference.locator.value}
        ),
        "start_seconds": reference.start_seconds,
        "end_seconds": reference.end_seconds,
        "frame_indices": list(reference.frame_indices),
        "derived_from": list(reference.derived_from),
    }


def _decode_media(value: object) -> MediaReference:
    media = _object(value, "media item")
    _exact_keys(
        media,
        {
            "media_id",
            "modality",
            "media_type",
            "sha256",
            "size_bytes",
            "evidence_ref",
            "locator",
            "start_seconds",
            "end_seconds",
            "frame_indices",
            "derived_from",
        },
        "media item",
    )
    locator_value = media["locator"]
    locator = None
    if locator_value is not None:
        locator_object = _object(locator_value, "locator")
        _exact_keys(locator_object, {"kind", "value"}, "locator")
        locator = MediaLocator(
            _enum(MediaLocatorKind, locator_object["kind"], "locator.kind"),
            _string(locator_object["value"], "locator.value"),
        )
    start_value = media["start_seconds"]
    end_value = media["end_seconds"]
    return MediaReference(
        media_id=_string(media["media_id"], "media_id"),
        modality=_enum(TraceModality, media["modality"], "modality"),
        media_type=_string(media["media_type"], "media_type"),
        sha256=_string(media["sha256"], "sha256"),
        size_bytes=_integer(media["size_bytes"], "size_bytes"),
        evidence_ref=_optional_string(media["evidence_ref"], "evidence_ref"),
        locator=locator,
        start_seconds=None if start_value is None else _number(start_value, "start_seconds"),
        end_seconds=None if end_value is None else _number(end_value, "end_seconds"),
        frame_indices=tuple(
            _integer(item, "frame_indices item")
            for item in _array(media["frame_indices"], "frame_indices")
        ),
        derived_from=tuple(
            _string(item, "derived_from item")
            for item in _array(media["derived_from"], "derived_from")
        ),
    )


def _encode_agent_lineage(value: AgentLineage | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "availability": value.availability.value,
        "agent_id": value.agent_id,
        "agent_version": value.agent_version,
        "model_id": value.model_id,
        "model_revision": value.model_revision,
        "generation_parameters_sha256": value.generation_parameters_sha256,
        "prompt_template_sha256": value.prompt_template_sha256,
    }


def _decode_agent_lineage(value: object) -> AgentLineage | None:
    if value is None:
        return None
    lineage = _object(value, "agent_lineage")
    _exact_keys(
        lineage,
        {
            "availability",
            "agent_id",
            "agent_version",
            "model_id",
            "model_revision",
            "generation_parameters_sha256",
            "prompt_template_sha256",
        },
        "agent_lineage",
    )
    return AgentLineage(
        availability=_enum(LineageAvailability, lineage["availability"], "availability"),
        agent_id=_optional_string(lineage["agent_id"], "agent_id"),
        agent_version=_optional_string(lineage["agent_version"], "agent_version"),
        model_id=_optional_string(lineage["model_id"], "model_id"),
        model_revision=_optional_string(lineage["model_revision"], "model_revision"),
        generation_parameters_sha256=_optional_string(
            lineage["generation_parameters_sha256"], "generation_parameters_sha256"
        ),
        prompt_template_sha256=_optional_string(
            lineage["prompt_template_sha256"], "prompt_template_sha256"
        ),
    )


def _encode_lineage(value: TraceLineage | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {"values": thaw_json(cast(FrozenJson, value.values))}


def _decode_lineage(value: object) -> TraceLineage | None:
    if value is None:
        return None
    lineage = _object(value, "lineage")
    _exact_keys(lineage, {"values"}, "lineage")
    return TraceLineage(_object(lineage["values"], "lineage.values"))


def _encode_warning(value: TraceWarning) -> dict[str, object]:
    return {"code": value.code, "message": value.message}


def _decode_warning(value: object) -> TraceWarning:
    warning = _object(value, "warning")
    _exact_keys(warning, {"code", "message"}, "warning")
    return TraceWarning(
        _string(warning["code"], "warning.code"),
        _string(warning["message"], "warning.message"),
    )


def _encode_error(value: TraceError | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {"code": value.code, "message": value.message, "retryable": value.retryable}


def _decode_error(value: object) -> TraceError | None:
    if value is None:
        return None
    error = _object(value, "error")
    _exact_keys(error, {"code", "message", "retryable"}, "error")
    return TraceError(
        _string(error["code"], "error.code"),
        _string(error["message"], "error.message"),
        _boolean(error["retryable"], "error.retryable"),
    )


def _decode_content(value: object) -> TextContent | MediaContent:
    block = _object(value, "content item")
    block_type = _string(block.get("type"), "content.type")
    if block_type == "text":
        _exact_keys(block, {"type", "text"}, "text content")
        return TextContent(_string(block["text"], "content.text"))
    if block_type == "media":
        _exact_keys(block, {"type", "media_id"}, "media content")
        return MediaContent(_string(block["media_id"], "content.media_id"))
    raise ValueError("content.type must be text or media")


def _encode_timestamp(value: datetime) -> str:
    if value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _decode_timestamp(value: object) -> datetime:
    text = _string(value, "occurred_at")
    if _TIMESTAMP_PATTERN.fullmatch(text) is None:
        raise ValueError("occurred_at must be a six-digit UTC timestamp")
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    except ValueError as error:
        raise ValueError("occurred_at must be a valid UTC timestamp") from error


def _object(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} must be a JSON object")
    return dict(value)


def _array(value: object, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a JSON array")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], field_name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{field_name} has unknown or missing fields")


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _string(value, field_name)


def _integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    number = float(value)
    if not (float("-inf") < number < float("inf")):
        raise ValueError(f"{field_name} must be finite")
    return number


def _boolean(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _enum(enum_type: type[_EnumT], value: object, field_name: str) -> _EnumT:
    text = _string(value, field_name)
    try:
        return enum_type(text)
    except ValueError as error:
        raise ValueError(f"{field_name} has an unsupported value") from error
