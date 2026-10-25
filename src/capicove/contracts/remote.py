"""Immutable, transport-independent remote invocation authority contracts."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal


def _identifier(value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value.encode("utf-8")) > 2048
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("Identity values must be bounded, nonblank strings without controls")


def _epoch(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("Epochs must be nonnegative integers")


def _permissions(value: Iterable[str]) -> frozenset[str]:
    if isinstance(value, (str, bytes)):
        raise ValueError("Permissions must be a collection of identifiers")
    result = frozenset(value)
    for permission in result:
        _identifier(permission)
    return result


@dataclass(frozen=True, slots=True)
class PrincipalKey:
    """Identify a subject within an enrolled issuer, tenant and application."""

    issuer: str
    tenant_id: str
    application_id: str
    subject_kind: Literal["human", "service"]
    subject: str

    def __post_init__(self) -> None:
        for value in (self.issuer, self.tenant_id, self.application_id, self.subject):
            _identifier(value)
        if self.subject_kind not in ("human", "service"):
            raise ValueError("Subject kind must explicitly be human or service")


@dataclass(frozen=True, slots=True)
class AuthorityKey:
    """Partition ownership by the complete server-enrolled authority."""

    principal_key: PrincipalKey
    workspace_binding_id: str
    grant_id: str
    audience: str

    def __post_init__(self) -> None:
        if not isinstance(self.principal_key, PrincipalKey):
            raise ValueError("Authority requires a complete principal key")
        for value in (self.workspace_binding_id, self.grant_id, self.audience):
            _identifier(value)


@dataclass(frozen=True, slots=True)
class Principal:
    """Represent verified token facts without retaining a bearer credential."""

    key: PrincipalKey
    token_permissions: frozenset[str]
    expires_at: float
    audience: str

    def __post_init__(self) -> None:
        _identifier(self.audience)
        if not isinstance(self.key, PrincipalKey):
            raise ValueError("Principal requires a complete key")
        if (
            isinstance(self.expires_at, bool)
            or not math.isfinite(self.expires_at)
            or self.expires_at <= 0
        ):
            raise ValueError("Token expiry must be a finite positive timestamp")
        object.__setattr__(self, "token_permissions", _permissions(self.token_permissions))


@dataclass(frozen=True, slots=True)
class AuthorizationDecision:
    """Capture the immutable permission ceiling and revocation epochs."""

    authority: AuthorityKey
    permissions: frozenset[str]
    policy_epoch: int
    source_epoch: int

    def __post_init__(self) -> None:
        if not isinstance(self.authority, AuthorityKey):
            raise ValueError("Authorization requires a complete authority")
        object.__setattr__(self, "permissions", _permissions(self.permissions))
        _epoch(self.policy_epoch)
        _epoch(self.source_epoch)


@dataclass(frozen=True, slots=True)
class AuthorizedInvocation:
    """Bind one operation to an immutable decision and bounded deadline."""

    request_id: str
    trace_id: str
    operation_id: str
    decision: AuthorizationDecision
    deadline: float

    def __post_init__(self) -> None:
        for value in (self.request_id, self.trace_id, self.operation_id):
            _identifier(value)
        if not isinstance(self.decision, AuthorizationDecision):
            raise ValueError("Invocation requires an authorization decision")
        if (
            isinstance(self.deadline, bool)
            or not math.isfinite(self.deadline)
            or self.deadline <= 0
        ):
            raise ValueError("Deadline must be a finite positive timestamp")


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Opaque, owner-scoped handle for one immutable committed artifact.

    The SHA-256 is an integrity value only; it never locates content across owners.
    """

    artifact_id: str
    sha256: str
    size: int
    media_type: str
    version: int
    lineage: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.artifact_id, str)
            or not self.artifact_id.startswith("artifact:")
            or len(self.artifact_id) != 41
            or any(character not in "0123456789abcdef" for character in self.artifact_id[9:])
        ):
            raise ValueError("Artifact IDs are opaque 128-bit lowercase hex handles")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("Artifact digests are lowercase SHA-256 hex")
        if type(self.size) is not int or self.size < 1:
            raise ValueError("Artifact size must be a positive integer")
        _identifier(self.media_type)
        if type(self.version) is not int or self.version < 1:
            raise ValueError("Artifact version must be a positive integer")
        if not isinstance(self.lineage, tuple):
            raise ValueError("Artifact lineage must be a tuple of identifiers")
        for value in self.lineage:
            _identifier(value)


@dataclass(frozen=True, slots=True)
class OperationRef:
    """Durable handle for one admitted operation; never a claim of completion."""

    operation_id: str
    state: Literal[
        "accepted",
        "queued",
        "running",
        "cancellation_requested",
        "succeeded",
        "failed",
        "cancelled",
        "timed_out",
        "outcome_unknown",
    ]
    tool_id: str
    tool_version: int
    input_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.operation_id, str) or not self.operation_id.startswith("operation:"):
            raise ValueError("Operation IDs are opaque operation handles")
        _identifier(self.tool_id)
        if type(self.tool_version) is not int or self.tool_version < 1:
            raise ValueError("Tool versions are positive integers")
        if not isinstance(self.input_sha256, str) or len(self.input_sha256) != 64:
            raise ValueError("Operation input digests are SHA-256 hex")


@dataclass(frozen=True, slots=True)
class ApprovalRef:
    """Durable handle for one reviewed administrative plan."""

    approval_id: str
    state: Literal["pending", "approved", "reserved", "consumed", "void"]
    plan_sha256: str
    reviewer: str | None
    operation_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.approval_id, str) or not self.approval_id.startswith("approval:"):
            raise ValueError("Approval IDs are opaque approval handles")
        if not isinstance(self.plan_sha256, str) or len(self.plan_sha256) != 64:
            raise ValueError("Plan digests are SHA-256 hex")
