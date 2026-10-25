"""Backend-neutral values for governed data processing."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias, cast

from capicove.contracts.data import DatasetSnapshotRef, LineageEdge


_PLAN_ID = re.compile(r"^plan:[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^run:[A-Za-z0-9][A-Za-z0-9._:-]*$")
_OPERATOR_ID = re.compile(r"^capicove\.[a-z0-9][a-z0-9.-]*$")
_BINDING = re.compile(r"^[a-z][a-z0-9_-]*$")
_DATASET_ID = re.compile(r"^dataset:[A-Za-z0-9][A-Za-z0-9._-]*$")

FrozenJson: TypeAlias = (
    bool | int | float | str | tuple["FrozenJson", ...] | Mapping[str, "FrozenJson"] | None
)


class RunState(StrEnum):
    """Describe every durable processing-run state."""

    SUBMITTED = "submitted"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    """Declare one closed versioned logical data operator."""

    operator_id: str
    operator_version: int
    parameter_names: frozenset[str]
    deterministic: bool
    input_count: int = 1
    parameter_schema: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if _OPERATOR_ID.fullmatch(self.operator_id) is None:
            raise ValueError("operator_id must be a canonical capicove identifier")
        if (
            isinstance(self.operator_version, bool)
            or not isinstance(self.operator_version, int)
            or self.operator_version <= 0
        ):
            raise ValueError("operator_version must be a positive integer")
        names = frozenset(self.parameter_names)
        if any(_BINDING.fullmatch(name) is None for name in names):
            raise ValueError("parameter_names must be canonical identifiers")
        if not isinstance(self.deterministic, bool):
            raise ValueError("deterministic must be a boolean")
        if (
            isinstance(self.input_count, bool)
            or not isinstance(self.input_count, int)
            or not 1 <= self.input_count <= 8
        ):
            raise ValueError("input_count must be an integer from 1 through 8")
        frozen_schema = _freeze_json(self.parameter_schema, "parameter_schema")
        if not isinstance(frozen_schema, Mapping):
            raise ValueError("parameter_schema must be a JSON object")
        if frozen_schema:
            _validate_parameter_schema(frozen_schema, names)
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "parameter_schema", frozen_schema)

    @property
    def registry_key(self) -> tuple[str, int]:
        """Return the stable operator registry key."""
        return self.operator_id, self.operator_version


@dataclass(frozen=True, slots=True)
class LogicalNode:
    """Apply one registered operator to existing logical bindings."""

    node_id: str
    operator_id: str
    operator_version: int
    input_bindings: tuple[str, ...]
    output_binding: str
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        if _BINDING.fullmatch(self.node_id) is None:
            raise ValueError("node_id must be a canonical identifier")
        if _OPERATOR_ID.fullmatch(self.operator_id) is None:
            raise ValueError("operator_id must be a canonical capicove identifier")
        if (
            isinstance(self.operator_version, bool)
            or not isinstance(self.operator_version, int)
            or self.operator_version <= 0
        ):
            raise ValueError("operator_version must be a positive integer")
        bindings = tuple(self.input_bindings)
        if not bindings or any(_BINDING.fullmatch(binding) is None for binding in bindings):
            raise ValueError("input_bindings must contain canonical identifiers")
        if len(set(bindings)) != len(bindings):
            raise ValueError("input_bindings must be unique")
        if _BINDING.fullmatch(self.output_binding) is None:
            raise ValueError("output_binding must be a canonical identifier")
        frozen = _freeze_json(self.parameters, "parameters")
        if not isinstance(frozen, Mapping):  # pragma: no cover - caller type narrows this
            raise TypeError("parameters must be a mapping")
        object.__setattr__(self, "input_bindings", bindings)
        object.__setattr__(self, "parameters", frozen)


@dataclass(frozen=True, slots=True)
class ProcessingPlan:
    """Describe one validated deterministic logical processing graph."""

    plan_id: str
    input_bindings: Mapping[str, DatasetSnapshotRef]
    nodes: tuple[LogicalNode, ...]
    output_dataset_id: str
    processing_fingerprint: str

    def __post_init__(self) -> None:
        if _PLAN_ID.fullmatch(self.plan_id) is None:
            raise ValueError("plan_id must be a plan-prefixed lowercase SHA-256 hash")
        inputs = dict(self.input_bindings)
        if not inputs:
            raise ValueError("input_bindings must not be empty")
        if any(_BINDING.fullmatch(name) is None for name in inputs):
            raise ValueError("input binding names must be canonical identifiers")
        if any(not isinstance(value, DatasetSnapshotRef) for value in inputs.values()):
            raise TypeError("input_bindings must contain DatasetSnapshotRef values")
        nodes = tuple(self.nodes)
        if not nodes or any(not isinstance(node, LogicalNode) for node in nodes):
            raise ValueError("nodes must contain LogicalNode values")
        if _DATASET_ID.fullmatch(self.output_dataset_id) is None:
            raise ValueError("output_dataset_id must be canonical")
        if not re.fullmatch(r"[0-9a-f]{64}", self.processing_fingerprint):
            raise ValueError("processing_fingerprint must be a lowercase SHA-256 hash")
        object.__setattr__(self, "input_bindings", MappingProxyType(inputs))
        object.__setattr__(self, "nodes", nodes)
        expected_plan_id, expected_fingerprint = derive_processing_identity(
            inputs,
            nodes,
            self.output_dataset_id,
        )
        if self.processing_fingerprint != expected_fingerprint:
            raise ValueError("processing_fingerprint does not match logical plan content")
        if self.plan_id != expected_plan_id:
            raise ValueError("plan_id does not match logical plan content")


@dataclass(frozen=True, slots=True)
class ProcessingContext:
    """Carry runtime-owned values that do not affect the logical plan."""

    run_id: str
    output_root: Path
    requested_at: datetime

    def __post_init__(self) -> None:
        _require_run_id(self.run_id)
        if not isinstance(self.output_root, Path) or not self.output_root.is_absolute():
            raise ValueError("output_root must be an absolute Path")
        _require_aware("requested_at", self.requested_at)


@dataclass(frozen=True, slots=True)
class BackendRunRef:
    """Correlate one Capicove run with its backend-native identity."""

    run_id: str
    backend_id: str
    native_run_id: str

    def __post_init__(self) -> None:
        _require_run_id(self.run_id)
        _require_non_blank("backend_id", self.backend_id)
        _require_non_blank("native_run_id", self.native_run_id)


@dataclass(frozen=True, slots=True)
class ProcessingRun:
    """Represent one durable processing-run observation."""

    run_id: str
    plan_id: str
    backend_id: str
    state: RunState
    created_at: datetime
    updated_at: datetime
    native_run_id: str | None = None
    output_snapshot_id: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        _require_run_id(self.run_id)
        if _PLAN_ID.fullmatch(self.plan_id) is None:
            raise ValueError("plan_id must be a plan-prefixed lowercase SHA-256 hash")
        _require_non_blank("backend_id", self.backend_id)
        if not isinstance(self.state, RunState):
            raise ValueError("state must be a RunState")
        _require_aware("created_at", self.created_at)
        _require_aware("updated_at", self.updated_at)
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        if self.native_run_id is not None:
            _require_non_blank("native_run_id", self.native_run_id)
        if self.output_snapshot_id is not None and not re.fullmatch(
            r"snapshot:[0-9a-f]{64}", self.output_snapshot_id
        ):
            raise ValueError("output_snapshot_id must be a canonical snapshot ID")
        if self.state is RunState.SUCCEEDED and self.output_snapshot_id is None:
            raise ValueError("succeeded run requires output_snapshot_id")
        if self.state is not RunState.SUCCEEDED and self.output_snapshot_id is not None:
            raise ValueError("output_snapshot_id requires a succeeded run")
        if self.detail is not None:
            _require_non_blank("detail", self.detail)
            if len(self.detail) > 1000:
                raise ValueError("detail must not exceed 1000 characters")


@dataclass(frozen=True, slots=True)
class BackendRunStatus:
    """Report one backend run state without mutating durable storage."""

    reference: BackendRunRef
    state: RunState
    output_snapshot_id: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reference, BackendRunRef):
            raise TypeError("reference must be a BackendRunRef")
        if not isinstance(self.state, RunState):
            raise ValueError("state must be a RunState")
        if self.output_snapshot_id is not None and not re.fullmatch(
            r"snapshot:[0-9a-f]{64}", self.output_snapshot_id
        ):
            raise ValueError("output_snapshot_id must be a canonical snapshot ID")
        if self.state is RunState.SUCCEEDED and self.output_snapshot_id is None:
            raise ValueError("succeeded status requires output_snapshot_id")
        if self.detail is not None and (not self.detail.strip() or len(self.detail) > 1000):
            raise ValueError("detail must be non-blank and bounded")


@dataclass(frozen=True, slots=True)
class ProcessingExecution:
    """Return one terminal run with its output and lineage."""

    run: ProcessingRun
    output_snapshot: DatasetSnapshotRef | None
    lineage: tuple[LineageEdge, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.run, ProcessingRun):
            raise TypeError("run must be a ProcessingRun")
        lineage = tuple(self.lineage)
        if self.run.state is RunState.SUCCEEDED:
            if self.output_snapshot is None:
                raise ValueError("succeeded execution requires output_snapshot")
            if self.output_snapshot.snapshot_id != self.run.output_snapshot_id:
                raise ValueError("output snapshot must match the run")
        elif self.output_snapshot is not None or lineage:
            raise ValueError("unsuccessful execution must not publish output or lineage")
        object.__setattr__(self, "lineage", lineage)


def _freeze_json(value: object, field_name: str, *, depth: int = 0) -> FrozenJson:
    if depth > 32:
        raise ValueError(f"{field_name} exceeds maximum nesting")
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must contain finite numbers")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, FrozenJson] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{field_name} keys must be non-blank strings")
            frozen[key] = _freeze_json(item, field_name, depth=depth + 1)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, field_name, depth=depth + 1) for item in value)
    raise ValueError(f"{field_name} must contain JSON-compatible values")


def thaw_json(value: FrozenJson) -> object:
    """Return a mutable JSON-compatible representation for serialization."""
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def _validate_parameter_schema(
    schema: Mapping[str, FrozenJson], parameter_names: frozenset[str]
) -> None:
    """Validate the root contract for Agent-readable parameter metadata."""
    if schema.get("type") != "object":
        raise ValueError("parameter_schema must have an object root")
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        raise ValueError("parameter_schema must declare object properties")
    required = schema.get("required")
    if not isinstance(required, tuple) or any(not isinstance(name, str) for name in required):
        raise ValueError("parameter_schema must declare required parameter names")
    if schema.get("additionalProperties") is not False:
        raise ValueError("parameter_schema must reject additional properties")
    if frozenset(properties) != parameter_names or set(required) != parameter_names:
        raise ValueError("parameter_schema names must match parameter_names")
    if len(required) != len(parameter_names):
        raise ValueError("parameter_schema required names must be unique")


def derive_processing_identity(
    input_bindings: Mapping[str, DatasetSnapshotRef],
    nodes: tuple[LogicalNode, ...],
    output_dataset_id: str,
    /,
) -> tuple[str, str]:
    """Derive the canonical plan ID and processing fingerprint."""
    fingerprint_value = {
        "inputs": {name: snapshot.snapshot_id for name, snapshot in sorted(input_bindings.items())},
        "nodes": [_logical_node_value(node) for node in nodes],
    }
    processing_fingerprint = _digest(fingerprint_value)
    plan_id = "plan:" + _digest(
        {
            **fingerprint_value,
            "output_dataset_id": output_dataset_id,
            "processing_fingerprint": processing_fingerprint,
        }
    )
    return plan_id, processing_fingerprint


def _logical_node_value(node: LogicalNode) -> dict[str, object]:
    return {
        "input_bindings": list(node.input_bindings),
        "node_id": node.node_id,
        "operator_id": node.operator_id,
        "operator_version": node.operator_version,
        "output_binding": node.output_binding,
        "parameters": thaw_json(cast(FrozenJson, node.parameters)),
    }


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_run_id(value: str) -> None:
    if _RUN_ID.fullmatch(value) is None:
        raise ValueError("run_id must use the run: prefix")


def _require_aware(field_name: str, value: datetime) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _require_non_blank(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be blank")
