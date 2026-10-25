"""Typed payload contracts for observable Agent operation trace records."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias

from capicove.batch import BatchState
from capicove.contracts import ToolStatus
from capicove.traces._json import freeze_json
from capicove.traces.content import (
    AgentLineage,
    MediaContent,
    MessageRole,
    TextContent,
    TraceModality,
)


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TOOL_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)+$")


class TraceRecordType(StrEnum):
    """Discriminate the versioned trace payload stored in one record."""

    EPISODE_STARTED = "episode.started"
    MESSAGE_OBSERVED = "message.observed"
    TOOL_CALL = "tool.call"
    TOOL_RESULT = "tool.result"
    ARTIFACT_PRODUCED = "artifact.produced"
    BATCH_ITEM = "batch.item"
    EPISODE_FEEDBACK = "episode.feedback"
    EPISODE_COMPLETED = "episode.completed"


class EpisodeStatus(StrEnum):
    """Describe the terminal status of one recorded Episode."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True)
class EpisodeStarted:
    """Capture immutable context at the beginning of an Episode."""

    goal: str
    workspace_id: str
    actor_id: str
    environment_id: str
    environment_version: int
    tool_catalog_sha256: str
    active_asset_versions: Mapping[str, int]
    agent_lineage: AgentLineage | None = None

    def __post_init__(self) -> None:
        for field_name in ("goal", "workspace_id", "actor_id", "environment_id"):
            _require_non_blank(field_name, getattr(self, field_name))
        _require_positive_int("environment_version", self.environment_version)
        _require_sha256("tool_catalog_sha256", self.tool_catalog_sha256)
        frozen: dict[str, int] = {}
        for asset_id, version in self.active_asset_versions.items():
            _require_non_blank("active_asset_versions key", asset_id)
            _require_positive_int("active_asset_versions value", version)
            frozen[asset_id] = version
        object.__setattr__(
            self, "active_asset_versions", _freeze_object(frozen, "active_asset_versions")
        )
        if self.agent_lineage is not None and not isinstance(self.agent_lineage, AgentLineage):
            raise ValueError("agent_lineage must be an AgentLineage")


@dataclass(frozen=True, slots=True)
class MessageObserved:
    """Store visible message content in its original block order."""

    role: MessageRole
    content: tuple[TextContent | MediaContent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.role, MessageRole):
            raise ValueError("role must be a MessageRole")
        if not self.content:
            raise ValueError("content must not be empty")
        if not all(isinstance(block, (TextContent, MediaContent)) for block in self.content):
            raise ValueError("content must contain only text or media blocks")


@dataclass(frozen=True, slots=True)
class ToolCallObserved:
    """Store one sanitized observable Tool invocation."""

    tool_id: str
    tool_version: int
    arguments: Mapping[str, object]
    arguments_schema_sha256: str
    attempt: int

    def __post_init__(self) -> None:
        _require_tool_id(self.tool_id)
        _require_positive_int("tool_version", self.tool_version)
        object.__setattr__(self, "arguments", _freeze_object(self.arguments, "arguments"))
        _require_sha256("arguments_schema_sha256", self.arguments_schema_sha256)
        _require_positive_int("attempt", self.attempt)


@dataclass(frozen=True, slots=True)
class ToolResultObserved:
    """Store one canonical Tool result without transport-specific wrapping."""

    tool_id: str
    tool_version: int
    status: ToolStatus
    result_schema_sha256: str
    result: Mapping[str, object] | None = None
    evidence_refs: tuple[str, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_tool_id(self.tool_id)
        _require_positive_int("tool_version", self.tool_version)
        if not isinstance(self.status, ToolStatus):
            raise ValueError("status must be a ToolStatus")
        _require_sha256("result_schema_sha256", self.result_schema_sha256)
        if self.result is not None:
            object.__setattr__(self, "result", _freeze_object(self.result, "result"))
        _require_unique_non_blank("evidence_refs", self.evidence_refs)
        frozen_metrics: dict[str, float] = {}
        for key, value in self.metrics.items():
            _require_non_blank("metrics key", key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("metrics values must be finite numbers")
            if not math.isfinite(value):
                raise ValueError("metrics values must be finite numbers")
            frozen_metrics[key] = float(value)
        object.__setattr__(self, "metrics", _freeze_object(frozen_metrics, "metrics"))


@dataclass(frozen=True, slots=True)
class ArtifactProduced:
    """Link a produced artifact to its media identity and derivation parents."""

    artifact_ref: str
    media_id: str
    derived_from: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_blank("artifact_ref", self.artifact_ref)
        _require_non_blank("media_id", self.media_id)
        _require_unique_non_blank("derived_from", self.derived_from)


@dataclass(frozen=True, slots=True)
class BatchItemObserved:
    """Capture one terminal deterministic batch item outcome."""

    plan_id: str
    item_id: str
    state: BatchState
    attempts: int
    cache_hit: bool
    modality: TraceModality
    media_type: str
    size_bytes: int
    source_sha256: str
    cache_key: str
    detail: str | None = None

    def __post_init__(self) -> None:
        if re.fullmatch(r"plan:[0-9a-f]{64}", self.plan_id) is None:
            raise ValueError("plan_id must be a plan-prefixed lowercase SHA-256 hash")
        if re.fullmatch(r"item:[0-9a-f]{64}", self.item_id) is None:
            raise ValueError("item_id must be an item-prefixed lowercase SHA-256 hash")
        if not isinstance(self.state, BatchState) or self.state in {
            BatchState.PENDING,
            BatchState.RUNNING,
        }:
            raise ValueError("state must be a terminal BatchState")
        _require_non_negative_int("attempts", self.attempts)
        if not isinstance(self.cache_hit, bool):
            raise ValueError("cache_hit must be a boolean")
        if self.cache_hit and self.state is not BatchState.SUCCEEDED:
            raise ValueError("cache_hit requires a succeeded state")
        if not isinstance(self.modality, TraceModality):
            raise ValueError("modality must be a TraceModality")
        _require_non_blank("media_type", self.media_type)
        _require_non_negative_int("size_bytes", self.size_bytes)
        _require_sha256("source_sha256", self.source_sha256)
        if re.fullmatch(r"cache:[0-9a-f]{64}", self.cache_key) is None:
            raise ValueError("cache_key must be a cache-prefixed lowercase SHA-256 hash")
        if self.detail is not None:
            _require_non_blank("detail", self.detail)


@dataclass(frozen=True, slots=True)
class EpisodeFeedback:
    """Append a linked annotation without mutating an earlier record."""

    kind: str
    value: Mapping[str, object]
    provenance: str
    target_record_id: str | None = None

    def __post_init__(self) -> None:
        _require_non_blank("kind", self.kind)
        object.__setattr__(self, "value", _freeze_object(self.value, "value"))
        _require_non_blank("provenance", self.provenance)
        if self.target_record_id is not None:
            _require_non_blank("target_record_id", self.target_record_id)


@dataclass(frozen=True, slots=True)
class EpisodeCompleted:
    """Mark one Episode terminal without fabricating missing intermediate facts."""

    status: EpisodeStatus
    outcome: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, EpisodeStatus):
            raise ValueError("status must be an EpisodeStatus")
        if self.outcome is not None:
            _require_non_blank("outcome", self.outcome)


TracePayload: TypeAlias = (
    EpisodeStarted
    | MessageObserved
    | ToolCallObserved
    | ToolResultObserved
    | ArtifactProduced
    | BatchItemObserved
    | EpisodeFeedback
    | EpisodeCompleted
)


_PAYLOAD_TYPES = (
    EpisodeStarted,
    MessageObserved,
    ToolCallObserved,
    ToolResultObserved,
    ArtifactProduced,
    BatchItemObserved,
    EpisodeFeedback,
    EpisodeCompleted,
)


def record_type_for_payload(payload: TracePayload, /) -> TraceRecordType:
    """Return the exact record discriminator for a supported payload instance."""
    mappings = {
        EpisodeStarted: TraceRecordType.EPISODE_STARTED,
        MessageObserved: TraceRecordType.MESSAGE_OBSERVED,
        ToolCallObserved: TraceRecordType.TOOL_CALL,
        ToolResultObserved: TraceRecordType.TOOL_RESULT,
        ArtifactProduced: TraceRecordType.ARTIFACT_PRODUCED,
        BatchItemObserved: TraceRecordType.BATCH_ITEM,
        EpisodeFeedback: TraceRecordType.EPISODE_FEEDBACK,
        EpisodeCompleted: TraceRecordType.EPISODE_COMPLETED,
    }
    try:
        return mappings[type(payload)]
    except KeyError as error:
        raise ValueError("payload must be a supported trace payload") from error


def is_trace_payload(value: object) -> bool:
    """Return whether a value is one of the closed version 1 payload classes."""
    return isinstance(value, _PAYLOAD_TYPES)


def _freeze_object(value: Mapping[str, object], field_name: str) -> Mapping[str, object]:
    frozen = freeze_json(value, field_name=field_name)
    if not isinstance(frozen, Mapping):
        raise ValueError(f"{field_name} must be a JSON object")
    return frozen


def _require_non_blank(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def _require_positive_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _require_non_negative_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _require_sha256(field_name: str, value: object) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hash")


def _require_tool_id(value: object) -> None:
    if not isinstance(value, str) or _TOOL_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("tool_id must be a canonical dotted identifier")


def _require_unique_non_blank(field_name: str, values: tuple[str, ...]) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must contain unique values")
    for value in values:
        _require_non_blank(field_name, value)
