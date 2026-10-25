"""Immutable content, media, lineage, warning, and error trace values."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

from capicove.traces._json import freeze_json


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class TraceModality(StrEnum):
    """Identify a media modality present in one trace record."""

    AUDIO = "audio"
    IMAGE = "image"
    VIDEO = "video"


class MessageRole(StrEnum):
    """Identify the visible author of one observed message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class LineageAvailability(StrEnum):
    """State whether Agent lineage was captured or intentionally absent."""

    CAPTURED = "captured"
    NOT_OBSERVED = "not_observed"
    REDACTED = "redacted"


class MediaLocatorKind(StrEnum):
    """Describe how a media reference may be resolved locally."""

    ARTIFACT = "artifact"
    WORKSPACE_RELATIVE = "workspace_relative"
    BUNDLE = "bundle"


@dataclass(frozen=True, slots=True)
class MediaLocator:
    """Store a safe non-absolute media locator."""

    kind: MediaLocatorKind
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, MediaLocatorKind):
            raise ValueError("kind must be a MediaLocatorKind")
        _require_non_blank("locator", self.value)
        if "\\" in self.value:
            raise ValueError("locator must use POSIX separators")
        path = PurePosixPath(self.value)
        if path.is_absolute() or path == PurePosixPath(".") or ".." in path.parts:
            raise ValueError("locator must be a relative path without parent traversal")
        if path.as_posix() != self.value:
            raise ValueError("locator must be a normalized POSIX path")


@dataclass(frozen=True, slots=True)
class MediaReference:
    """Identify immutable media bytes and an optional temporal selection."""

    media_id: str
    modality: TraceModality
    media_type: str
    sha256: str
    size_bytes: int
    evidence_ref: str | None = None
    locator: MediaLocator | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None
    frame_indices: tuple[int, ...] = ()
    derived_from: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_blank("media_id", self.media_id)
        if not isinstance(self.modality, TraceModality):
            raise ValueError("modality must be a TraceModality")
        _require_non_blank("media_type", self.media_type)
        _require_sha256("sha256", self.sha256)
        _require_non_negative_int("size_bytes", self.size_bytes)
        if self.evidence_ref is not None:
            _require_non_blank("evidence_ref", self.evidence_ref)
        if self.locator is not None and not isinstance(self.locator, MediaLocator):
            raise ValueError("locator must be a MediaLocator")
        if (self.start_seconds is None) != (self.end_seconds is None):
            raise ValueError("time bounds must be supplied together")
        if self.start_seconds is not None and self.end_seconds is not None:
            for value in (self.start_seconds, self.end_seconds):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("time bounds must be finite numbers")
                if not math.isfinite(value) or value < 0:
                    raise ValueError("time bounds must be finite and non-negative")
            if self.end_seconds <= self.start_seconds:
                raise ValueError("time end must be greater than time start")
        previous = -1
        for frame_index in self.frame_indices:
            _require_non_negative_int("frame_indices", frame_index)
            if frame_index <= previous:
                raise ValueError("frame_indices must be strictly increasing")
            previous = frame_index
        _require_unique_non_blank("derived_from", self.derived_from)


@dataclass(frozen=True, slots=True)
class TextContent:
    """Store one visible text content block."""

    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")


@dataclass(frozen=True, slots=True)
class MediaContent:
    """Refer to one media identity within ordered message content."""

    media_id: str

    def __post_init__(self) -> None:
        _require_non_blank("media_id", self.media_id)


@dataclass(frozen=True, slots=True)
class AgentLineage:
    """Capture available Agent and model identity without private reasoning."""

    availability: LineageAvailability
    agent_id: str | None = None
    agent_version: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    generation_parameters_sha256: str | None = None
    prompt_template_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.availability, LineageAvailability):
            raise ValueError("availability must be a LineageAvailability")
        identity_fields = (
            "agent_id",
            "agent_version",
            "model_id",
            "model_revision",
        )
        for field_name in identity_fields:
            value = getattr(self, field_name)
            if value is not None:
                _require_non_blank(field_name, value)
        hash_fields = ("generation_parameters_sha256", "prompt_template_sha256")
        for field_name in hash_fields:
            value = getattr(self, field_name)
            if value is not None:
                _require_sha256(field_name, value)
        supplied = any(getattr(self, name) is not None for name in (*identity_fields, *hash_fields))
        if self.availability is LineageAvailability.CAPTURED and not supplied:
            raise ValueError("captured Agent lineage must include an identity")
        if self.availability is not LineageAvailability.CAPTURED and supplied:
            raise ValueError("uncaptured Agent lineage must not include identity fields")


@dataclass(frozen=True, slots=True)
class TraceLineage:
    """Store deeply immutable JSON lineage values."""

    values: Mapping[str, object]

    def __post_init__(self) -> None:
        frozen = freeze_json(self.values, field_name="values")
        if not isinstance(frozen, Mapping):
            raise ValueError("values must be a JSON object")
        object.__setattr__(self, "values", frozen)


@dataclass(frozen=True, slots=True)
class TraceWarning:
    """Describe one stable visible trace warning."""

    code: str
    message: str

    def __post_init__(self) -> None:
        _require_non_blank("code", self.code)
        _require_non_blank("message", self.message)


@dataclass(frozen=True, slots=True)
class TraceError:
    """Describe one stable visible trace error."""

    code: str
    message: str
    retryable: bool

    def __post_init__(self) -> None:
        _require_non_blank("code", self.code)
        _require_non_blank("message", self.message)
        if not isinstance(self.retryable, bool):
            raise ValueError("retryable must be a boolean")


def _require_non_blank(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def _require_sha256(field_name: str, value: object) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hash")


def _require_non_negative_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _require_unique_non_blank(field_name: str, values: tuple[str, ...]) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must contain unique values")
    for value in values:
        _require_non_blank(field_name, value)
