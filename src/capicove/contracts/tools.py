"""Canonical contracts for transport-neutral enterprise tools."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from capicove.contracts._validation import (
    freeze_mapping,
    require_aware_datetime,
    require_canonical_tool_id,
    require_non_blank,
    require_positive_float,
    require_positive_int,
)


class PrivacyClass(StrEnum):
    """Classify the sensitivity of tool inputs and outputs."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class SideEffect(StrEnum):
    """Describe the highest side-effect class of a tool invocation."""

    NONE = "none"
    READ = "read"
    WRITE = "write"
    EXTERNAL = "external"


class ToolStatus(StrEnum):
    """Describe the canonical outcome of a tool invocation."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"


class ToolErrorCode(StrEnum):
    """Classify tool failures consistently across transports."""

    INVALID_INPUT = "invalid_input"
    PERMISSION_DENIED = "permission_denied"
    POLICY_REJECTED = "policy_rejected"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    RETRYABLE = "retryable"
    INTERNAL = "internal"


@dataclass(frozen=True, slots=True)
class ToolResources:
    """Declare resources an execution environment must authorize."""

    filesystem_read: bool = False
    filesystem_write: bool = False
    subprocesses: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(set(self.subprocesses)) != len(self.subprocesses):
            raise ValueError("subprocesses must not contain duplicate names")
        for executable in self.subprocesses:
            if re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._+-]*", executable) is None:
                raise ValueError("subprocess must be a bare executable name")


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Describe one version of a transport-neutral Capicove tool."""

    tool_id: str
    contract_version: int
    description: str
    input_schema: Mapping[str, object]
    output_schema: Mapping[str, object]
    modalities: tuple[str, ...] = ()
    required_permissions: frozenset[str] = frozenset()
    privacy_class: PrivacyClass = PrivacyClass.INTERNAL
    side_effect: SideEffect = SideEffect.NONE
    network_access: bool = False
    deterministic: bool = True
    idempotent: bool = True
    timeout_seconds: float = 30.0
    redacted_fields: tuple[str, ...] = ()
    resources: ToolResources = ToolResources()

    def __post_init__(self) -> None:
        require_canonical_tool_id(self.tool_id)
        require_positive_int("contract_version", self.contract_version)
        require_non_blank("description", self.description)
        require_positive_float("timeout_seconds", self.timeout_seconds)
        object.__setattr__(self, "input_schema", freeze_mapping(self.input_schema))
        object.__setattr__(self, "output_schema", freeze_mapping(self.output_schema))

    @property
    def registry_key(self) -> tuple[str, int]:
        """Return the stable key used by a Tool Registry."""
        return self.tool_id, self.contract_version


@dataclass(frozen=True, slots=True)
class InvocationContext:
    """Carry policy-relevant values for one tool invocation."""

    request_id: str
    trace_id: str
    workspace_id: str
    actor_id: str
    environment_id: str
    deadline: datetime
    permissions: frozenset[str] = frozenset()
    privacy_class: PrivacyClass = PrivacyClass.INTERNAL

    def __post_init__(self) -> None:
        for field_name in (
            "request_id",
            "trace_id",
            "workspace_id",
            "actor_id",
            "environment_id",
        ):
            require_non_blank(field_name, getattr(self, field_name))
        require_aware_datetime("deadline", self.deadline)


@dataclass(frozen=True, slots=True)
class ToolError:
    """Describe a transport-independent tool failure."""

    code: ToolErrorCode
    message: str
    retryable: bool

    def __post_init__(self) -> None:
        require_non_blank("message", self.message)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Return a canonical, evidence-aware result envelope."""

    request_id: str
    trace_id: str
    tool_id: str
    tool_version: int
    status: ToolStatus
    result: Mapping[str, object] | None = None
    evidence_refs: tuple[str, ...] = ()
    error: ToolError | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        require_non_blank("request_id", self.request_id)
        require_non_blank("trace_id", self.trace_id)
        require_canonical_tool_id(self.tool_id)
        require_positive_int("tool_version", self.tool_version)
        require_aware_datetime("started_at", self.started_at)
        require_aware_datetime("completed_at", self.completed_at)
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must not be before started_at")
        if self.status is ToolStatus.SUCCEEDED and self.error is not None:
            raise ValueError("a successful result must not contain an error")
        if self.status is not ToolStatus.SUCCEEDED and self.error is None:
            raise ValueError("a failed or rejected result must contain an error")
        if self.result is not None:
            object.__setattr__(self, "result", freeze_mapping(self.result))
        object.__setattr__(self, "metrics", freeze_mapping(self.metrics))


@runtime_checkable
class ToolHandler(Protocol):
    """Handle one validated invocation without depending on a transport."""

    def __call__(
        self,
        context: InvocationContext,
        arguments: Mapping[str, object],
        /,
    ) -> ToolResult: ...
