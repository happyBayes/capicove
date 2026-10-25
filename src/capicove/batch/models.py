"""Backend-independent values for deterministic batch processing."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ITEM_ID_PATTERN = re.compile(r"^item:[0-9a-f]{64}$")
_CACHE_KEY_PATTERN = re.compile(r"^cache:[0-9a-f]{64}$")
_PLAN_ID_PATTERN = re.compile(r"^plan:[0-9a-f]{64}$")


class BatchState(StrEnum):
    """Represent every persisted lifecycle state for one batch item."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    SOURCE_CHANGED = "source_changed"


class BatchModality(StrEnum):
    """Identify the media capability responsible for one batch item."""

    AUDIO = "audio"
    IMAGE = "image"
    VIDEO = "video"


class BatchCacheScope(StrEnum):
    """Define whether result reuse is content-wide or item-specific."""

    CONTENT = "content"
    ITEM = "item"


@dataclass(frozen=True, slots=True)
class BatchFailure:
    """Describe one bounded operation failure and its retry classification."""

    code: str
    message: str
    retryable: bool

    def __post_init__(self) -> None:
        _require_non_blank("code", self.code)
        _require_non_blank("message", self.message)
        if not isinstance(self.retryable, bool):
            raise ValueError("retryable must be a boolean")


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    """Capture filesystem identity used to detect source changes."""

    device: int
    inode: int
    size_bytes: int
    modified_ns: int
    changed_ns: int

    def __post_init__(self) -> None:
        for field_name in ("device", "inode", "size_bytes", "modified_ns", "changed_ns"):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must not be negative")


@dataclass(frozen=True, slots=True)
class BatchItem:
    """Describe one immutable source observation within a directory batch."""

    item_id: str
    modality: BatchModality
    source_path: Path
    relative_path: PurePosixPath
    media_type: str
    sha256: str
    snapshot: SourceSnapshot

    def __post_init__(self) -> None:
        if _ITEM_ID_PATTERN.fullmatch(self.item_id) is None:
            raise ValueError("item_id must be an item-prefixed lowercase SHA-256 hash")
        if not isinstance(self.modality, BatchModality):
            raise ValueError("modality must be a BatchModality")
        if not self.source_path.is_absolute():
            raise ValueError("source_path must be absolute")
        if (
            self.relative_path.is_absolute()
            or self.relative_path == PurePosixPath(".")
            or ".." in self.relative_path.parts
        ):
            raise ValueError("relative_path must remain below the discovery root")
        if not self.media_type.strip():
            raise ValueError("media_type must not be blank")
        _require_sha256("sha256", self.sha256)

    @property
    def size_bytes(self) -> int:
        """Return the snapshotted source size."""
        return self.snapshot.size_bytes


@dataclass(frozen=True, slots=True)
class OperationFingerprint:
    """Identify cache-affecting operation, parameter, and provider versions."""

    operation_id: str
    operation_version: str
    modalities: frozenset[BatchModality]
    parameters_sha256: str
    provider_fingerprint: str
    cache_scope: BatchCacheScope = BatchCacheScope.CONTENT

    def __post_init__(self) -> None:
        _require_non_blank("operation_id", self.operation_id)
        _require_non_blank("operation_version", self.operation_version)
        if not self.modalities:
            raise ValueError("modalities must not be empty")
        if not all(isinstance(modality, BatchModality) for modality in self.modalities):
            raise ValueError("modalities must contain only BatchModality values")
        _require_sha256("parameters_sha256", self.parameters_sha256)
        _require_sha256("provider_fingerprint", self.provider_fingerprint)
        if not isinstance(self.cache_scope, BatchCacheScope):
            raise ValueError("cache_scope must be a BatchCacheScope")


@dataclass(frozen=True, slots=True)
class PlannedBatchItem:
    """Bind one source observation to its content-based result cache key."""

    item: BatchItem
    cache_key: str

    def __post_init__(self) -> None:
        if _CACHE_KEY_PATTERN.fullmatch(self.cache_key) is None:
            raise ValueError("cache_key must be a cache-prefixed lowercase SHA-256 hash")


@dataclass(frozen=True, slots=True)
class BatchPlan:
    """Describe one deterministic operation plan without execution state."""

    plan_id: str
    operation: OperationFingerprint
    items: tuple[PlannedBatchItem, ...]

    def __post_init__(self) -> None:
        if _PLAN_ID_PATTERN.fullmatch(self.plan_id) is None:
            raise ValueError("plan_id must be a plan-prefixed lowercase SHA-256 hash")


@dataclass(frozen=True, slots=True)
class BatchItemResult:
    """Describe one terminal item outcome returned by the batch executor."""

    item_id: str
    state: BatchState
    attempts: int
    cache_hit: bool = False
    result: Mapping[str, object] | None = None
    failure: BatchFailure | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if _ITEM_ID_PATTERN.fullmatch(self.item_id) is None:
            raise ValueError("item_id must be an item-prefixed lowercase SHA-256 hash")
        if self.state in {BatchState.PENDING, BatchState.RUNNING}:
            raise ValueError("batch item result must be terminal")
        if self.attempts < 0:
            raise ValueError("attempts must not be negative")
        if self.state is BatchState.SUCCEEDED:
            if self.result is None:
                raise ValueError("succeeded result must include a payload")
            object.__setattr__(self, "result", MappingProxyType(dict(self.result)))
        elif self.result is not None:
            raise ValueError("non-succeeded result must not include a payload")
        failure_states = {BatchState.FAILED, BatchState.SOURCE_CHANGED}
        if self.state in failure_states and self.failure is None:
            raise ValueError("failed result must include failure details")
        if self.state not in failure_states and self.failure is not None:
            raise ValueError("failure details require a failed result state")
        if self.cache_hit and self.state is not BatchState.SUCCEEDED:
            raise ValueError("cache_hit requires a succeeded result")
        if self.detail is not None and not self.detail.strip():
            raise ValueError("detail must not be blank")


@dataclass(frozen=True, slots=True)
class BatchExecution:
    """Return deterministic terminal outcomes for one batch plan."""

    plan_id: str
    items: tuple[BatchItemResult, ...]

    def __post_init__(self) -> None:
        if _PLAN_ID_PATTERN.fullmatch(self.plan_id) is None:
            raise ValueError("plan_id must be a plan-prefixed lowercase SHA-256 hash")
        item_ids = tuple(item.item_id for item in self.items)
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("batch execution item_ids must be unique")


def _require_non_blank(field_name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def _require_sha256(field_name: str, value: str) -> None:
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hash")
