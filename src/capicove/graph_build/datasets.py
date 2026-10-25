"""Closed graph table schemas and recorded local materialization.

Rows use the canonical graph values. Resolved entities and relationships retain
the original Entity and RelationshipAssertion contracts; their role metadata
distinguishes the current analysis projection. Input order is canonicalized by
primary key and duplicate row identities are rejected, including equal rows.
Authorization and cross-table evidence checks remain the build host's responsibility.
"""

from __future__ import annotations

import math
import re
import types
from collections.abc import Iterable
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

from capicove.contracts.data import (
    ComplexSchemaType,
    ComplexTypeKind,
    DatasetSchema,
    DatasetSnapshotRef,
    LineageEdge,
    SchemaField,
    SchemaValueType,
)
from capicove.contracts.graph_build import (
    Community,
    CommunityMembership,
    CommunityReport,
    DiagnosticFinding,
    Entity,
    EntityAlignment,
    EntityMembership,
    EntityMention,
    GraphScope,
    RelationshipAggregate,
    RelationshipAssertion,
    RelationshipMention,
    ResolutionMembership,
    SourceSpan,
    StageCoverage,
    TextUnit,
)
from capicove.contracts.graph_build.snapshots import REQUIRED_STAGES
from capicove.graph_build.codec import fingerprint
from capicove.processing import LocalProcessingWorkspace, ProcessingRun, RunState


def _text(value: object) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError("expected nonblank text")


@dataclass(frozen=True, slots=True)
class DocumentRow:
    """Declared document availability, immutable representation and coverage.

    A missing representation is recorded explicitly for unavailable documents;
    provenance and coverage are supplied by the authorized representation stage.
    """

    document_ref: str
    scope: GraphScope
    source_sha256: str
    evidence_ref: str
    representation_ref: str | None
    media_type: str
    availability: str
    provenance: tuple[SourceSpan, ...]
    coverage: StageCoverage

    def __post_init__(self) -> None:
        for value in (self.document_ref, self.evidence_ref, self.media_type):
            _text(value)
        if not isinstance(self.scope, GraphScope) or not isinstance(self.coverage, StageCoverage):
            raise TypeError("document scope and coverage must be typed values")
        if not isinstance(self.source_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.source_sha256
        ):
            raise ValueError("source_sha256 must be lowercase SHA-256")
        if self.availability not in {"available", "unavailable"}:
            raise ValueError("unknown document availability")
        if self.representation_ref is not None:
            _text(self.representation_ref)
        if self.availability == "available" and self.representation_ref is None:
            raise ValueError("available documents require a representation")
        if not isinstance(self.provenance, tuple) or any(
            type(span) is not SourceSpan for span in self.provenance
        ):
            raise TypeError("provenance must contain SourceSpan values")


@dataclass(frozen=True, slots=True)
class OperationDiagnosticRow:
    """Bounded operational code without source text or provider payloads."""

    record_id: str
    stage_id: str
    unit_ref: str | None
    code: str
    severity: str

    def __post_init__(self) -> None:
        _text(self.record_id)
        if self.stage_id not in REQUIRED_STAGES | {"publication", "retrieval"}:
            raise ValueError("unknown diagnostic stage")
        if self.unit_ref is not None:
            _text(self.unit_ref)
        if (
            type(self.code) is not str
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_.:-]{0,127}", self.code) is None
        ):
            raise ValueError("diagnostic code must be a bounded identifier")
        if self.severity not in {"info", "warning", "error"}:
            raise ValueError("unknown diagnostic severity")


# Only these declared records enter persisted tables. No row kind is inferred
# from caller dictionaries, file metadata, class names or import paths.
_ROWS: dict[str, tuple[type[Any], tuple[str, ...]]] = {
    "documents": (DocumentRow, ("document_ref",)),
    "text_units": (TextUnit, ("text_unit_id",)),
    "entity_mentions": (EntityMention, ("mention_id",)),
    "relationship_mentions": (RelationshipMention, ("mention_id",)),
    "entities": (Entity, ("entity_id",)),
    "entity_memberships": (EntityMembership, ("entity_id", "mention_id")),
    "relationship_assertions": (RelationshipAssertion, ("assertion_id",)),
    "relationship_aggregates": (RelationshipAggregate, ("aggregate_id",)),
    "entity_alignments": (EntityAlignment, ("current_entity_id", "previous_entity_id")),
    "resolved_entities": (Entity, ("entity_id",)),
    "resolved_relationships": (RelationshipAssertion, ("assertion_id",)),
    "resolution_memberships": (ResolutionMembership, ("analysis_entity_id", "current_entity_id")),
    "diagnostic_findings": (DiagnosticFinding, ("finding_id",)),
    "communities": (Community, ("community_id",)),
    "community_memberships": (CommunityMembership, ("community_id", "entity_id")),
    "community_reports": (CommunityReport, ("report_id",)),
    "operation_diagnostics": (OperationDiagnosticRow, ("record_id",)),
}


def _nullable(annotation: Any) -> tuple[Any, bool]:
    if get_origin(annotation) is types.UnionType:
        args = get_args(annotation)
        if len(args) == 2 and type(None) in args:
            return next(arg for arg in args if arg is not type(None)), True
        raise TypeError("unsupported schema union")
    return annotation, False


# Resolved annotations are fixed once a frozen record class exists; resolving
# them per converted record dominated graph table decoding (RFC-0019).
_RECORD_FIELDS: dict[type[Any], tuple[tuple[str, Any], ...]] = {}


def _record_fields(cls: type[Any]) -> tuple[tuple[str, Any], ...]:
    resolved = _RECORD_FIELDS.get(cls)
    if resolved is None:
        hints = get_type_hints(cls)
        resolved = tuple((field.name, hints[field.name]) for field in fields(cls))
        _RECORD_FIELDS[cls] = resolved
    return resolved


def _field(name: str, annotation: Any) -> SchemaField:
    inner, nullable = _nullable(annotation)
    return SchemaField(name, _logical_type(inner), nullable)


def _logical_type(annotation: Any) -> SchemaValueType:
    scalars = {str: "string", bool: "boolean", int: "int64", float: "float64"}
    if annotation in scalars:
        return scalars[annotation]
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return "string"
    if get_origin(annotation) is tuple:
        args = get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            return ComplexSchemaType(
                ComplexTypeKind.ARRAY, element_type=_logical_type(args[0]), contains_null=False
            )
        if len(args) == 2:
            return ComplexSchemaType(
                ComplexTypeKind.STRUCT, fields=(_field("key", args[0]), _field("value", args[1]))
            )
        raise TypeError("unsupported tuple schema")
    if isinstance(annotation, type) and is_dataclass(annotation):
        return ComplexSchemaType(
            ComplexTypeKind.STRUCT,
            fields=tuple(_field(name, kind) for name, kind in _record_fields(annotation)),
        )
    raise TypeError("unsupported closed schema type")


def _convert(value: Any, annotation: Any, *, encode: bool) -> Any:
    kind, nullable = _nullable(annotation)
    if value is None:
        if nullable:
            return None
        raise TypeError("null is forbidden for this field")
    if kind in (str, bool, int, float):
        if type(value) is not kind:
            raise TypeError("scalar type mismatch")
        if kind is int and not -(2**63) <= value < 2**63:
            raise ValueError("integer exceeds int64")
        if kind is float and not math.isfinite(value):
            raise ValueError("nonfinite scalar")
        return value
    if isinstance(kind, type) and issubclass(kind, Enum):
        if encode:
            if type(value) is not kind:
                raise TypeError("enum type mismatch")
            return value.value
        if type(value) is not str:
            raise TypeError("enum must be a string")
        return kind(value)
    if get_origin(kind) is tuple:
        args = get_args(kind)
        if len(args) == 2 and args[1] is Ellipsis:
            if type(value) is not (tuple if encode else list):
                raise TypeError("array type mismatch")
            items = [_convert(item, args[0], encode=encode) for item in value]
            return items if encode else tuple(items)
        if encode:
            if type(value) is not tuple or len(value) != 2:
                raise TypeError("expected typed pair")
            return {
                name: _convert(item, inner, encode=True)
                for name, item, inner in zip(("key", "value"), value, args, strict=True)
            }
        if type(value) is not dict or set(value) != {"key", "value"}:
            raise ValueError("malformed pair fields")
        return tuple(
            _convert(value[name], inner, encode=False)
            for name, inner in zip(("key", "value"), args, strict=True)
        )
    record_fields = _record_fields(kind)
    if encode:
        if type(value) is not kind:
            raise TypeError("wrong row or nested record type")
        return {
            name: _convert(getattr(value, name), inner, encode=True)
            for name, inner in record_fields
        }
    if type(value) is not dict or set(value) != {name for name, _ in record_fields}:
        raise ValueError("missing or extra record fields")
    return kind(
        **{name: _convert(value[name], inner, encode=False) for name, inner in record_fields}
    )


@dataclass(frozen=True, slots=True)
class RoleCodec:
    """One closed role codec; canonical constructors revalidate every decode."""

    role: str
    row_type: type[Any]
    key_fields: tuple[str, ...]
    schema: DatasetSchema

    def encode(self, row: object) -> dict[str, Any]:
        """Validate exact logical types and canonical record invariants."""
        encoded: dict[str, Any] = _convert(row, self.row_type, encode=True)
        decoded = self.decode(encoded)
        if decoded != row:
            raise ValueError("noncanonical row")
        return encoded

    def decode(self, row: object) -> object:
        """Reject malformed structures before invoking canonical constructors."""
        return _convert(row, self.row_type, encode=False)

    def snapshot(self, rows: Iterable[object]) -> tuple[object, ...]:
        """Consume once, validate, reject duplicate identities, sort by key."""
        indexed: dict[tuple[str, ...], object] = {}
        for row in rows:
            validated = self.decode(self.encode(row))
            key = tuple(getattr(validated, field) for field in self.key_fields)
            if key in indexed:
                raise ValueError("duplicate graph row identity")
            indexed[key] = validated
        return tuple(indexed[key] for key in sorted(indexed))


@lru_cache(maxsize=17)
def codec_for_role(role: str) -> RoleCodec:
    """Return a codec selected only from the fixed 17-role registry."""
    if type(role) is not str or role not in _ROWS:
        raise ValueError("unknown graph table role")
    row_type, keys = _ROWS[role]
    schema_fields = tuple(_field(name, annotation) for name, annotation in _record_fields(row_type))
    version = (
        2
        if any(isinstance(field.logical_type, ComplexSchemaType) for field in schema_fields)
        else 1
    )
    return RoleCodec(role, row_type, keys, DatasetSchema(schema_fields, version))


def schema_for_role(role: str) -> DatasetSchema:
    """Return the full declared schema, including for a zero-row table."""
    return codec_for_role(role).schema


def publish_table(
    workspace: LocalProcessingWorkspace,
    *,
    scope: GraphScope,
    role: str,
    rows: Iterable[object],
    parent_refs: tuple[DatasetSnapshotRef, ...],
    processing_fingerprint: str,
) -> DatasetSnapshotRef:
    """Materialize recorded graph rows under one short public workspace lock.

    Consumes the iterable before locking. The caller has already authorized
    source access and run effectful stages. Identity binds exact scope, parents,
    schema, logical rows, processing fingerprint and actual serializer version.
    A plan-prefixed digest identifies this recorded materialization, without
    constructing a ProcessingPlan or claiming that upstream models are pure.
    Failed attempts get new run IDs; successful old requests remain discoverable
    through their immutable run even after the latest dataset version changes.
    """
    from capicove.graph_build._adapters.parquet import ParquetTableCodec

    if not isinstance(workspace, LocalProcessingWorkspace) or type(scope) is not GraphScope:
        raise TypeError("workspace and scope must be public typed values")
    if (
        type(processing_fingerprint) is not str
        or re.fullmatch(r"[0-9a-f]{64}", processing_fingerprint) is None
    ):
        raise ValueError("processing_fingerprint must be lowercase SHA-256")
    if type(parent_refs) is not tuple or any(
        type(ref) is not DatasetSnapshotRef for ref in parent_refs
    ):
        raise TypeError("parents must be a tuple of DatasetSnapshotRef values")
    parents = tuple(ref.snapshot_id for ref in parent_refs)
    if parents != tuple(sorted(set(parents))):
        raise ValueError("parents must be sorted and unique")
    role_codec = codec_for_role(role)
    ordered = role_codec.snapshot(rows)
    if any(getattr(row, "scope", scope) != scope for row in ordered):
        raise ValueError("row scope does not match publication scope")
    runtime = ParquetTableCodec()
    scope_identity = {"workspace_ref": scope.workspace_ref, "scope_ref": scope.scope_ref}
    identity = fingerprint(
        {
            "scope": scope_identity,
            "role": role,
            "schema": role_codec.schema.sha256,
            "rows": [role_codec.encode(row) for row in ordered],
            "parents": list(parents),
            "processing_fingerprint": processing_fingerprint,
            "serializer": runtime.serializer_identity,
        }
    )
    codec = ParquetTableCodec(publication_identity=identity)
    dataset_id = "dataset:graph-" + fingerprint({"scope": scope_identity, "role": role})
    plan_id = "plan:" + identity
    backend_id = "capicove.graph.materialization"

    def validate(path: Path) -> bool:
        entries = {child.name for child in path.iterdir()}
        if entries - {"part-00000.parquet", "manifest.json", ".capicove-pending.json"}:
            return False
        return codec.read(path / "part-00000.parquet", role) == ordered

    with workspace.execution_lock():
        for parent_ref in parent_refs:
            if workspace.get(parent_ref.snapshot_id) != parent_ref:
                raise ValueError("parent does not match current workspace catalog")
        attempt = 0
        while True:
            run_id = f"run:graph-{identity}-{attempt}"
            previous = workspace.get_run(run_id)
            if previous is None:
                break
            if previous.plan_id != plan_id or previous.backend_id != backend_id:
                raise ValueError("materialization run identity mismatch")
            if previous.state is RunState.SUCCEEDED:
                if previous.output_snapshot_id is None:
                    raise ValueError("successful materialization is missing its snapshot")
                ref = workspace.get(previous.output_snapshot_id)
                if (
                    ref is None
                    or ref.dataset_id != dataset_id
                    or ref.schema != role_codec.schema
                    or ref.processing_fingerprint != identity
                    or ref.parent_snapshot_ids != parents
                ):
                    raise ValueError("materialization snapshot metadata mismatch")
                verified = workspace.validate_snapshot(ref.snapshot_id, validator=validate)
                if verified != ref:
                    raise ValueError("materialization snapshot is unavailable")
                return ref
            if previous.state in {RunState.SUBMITTED, RunState.RUNNING}:
                workspace.transition(
                    run_id,
                    previous.state,
                    RunState.FAILED,
                    detail="RECOVERED_INCOMPLETE_MATERIALIZATION",
                )
            attempt += 1
        latest = workspace.latest(dataset_id)
        version = 1 if latest is None else latest.version + 1
        now = datetime.now(UTC)
        staging: Path | None = None
        snapshot: DatasetSnapshotRef | None = None
        native_run_id = f"graph-materialization:{identity}:{attempt}"
        try:
            # Persist recovery ownership before any durable run can be orphaned.
            staging = workspace.stage(run_id)
            workspace.create(
                ProcessingRun(run_id, plan_id, backend_id, RunState.SUBMITTED, now, now)
            )
            workspace.transition(
                run_id, RunState.SUBMITTED, RunState.RUNNING, native_run_id=native_run_id
            )
            codec.write(staging / "part-00000.parquet", role, ordered)
            snapshot = workspace.publish(
                run_id=run_id,
                dataset_id=dataset_id,
                version=version,
                staging_path=staging,
                schema=role_codec.schema,
                processing_fingerprint=identity,
                parent_snapshot_ids=parents,
                validator=validate,
            )
            lineage = tuple(
                LineageEdge(parent, snapshot.snapshot_id, run_id, ("capicove.graph.materialize",))
                for parent in parents
            )
            workspace.commit_success(run_id, snapshot, lineage, native_run_id=native_run_id)
            workspace.finalize(run_id, snapshot)
            return snapshot
        except Exception:
            recorded = workspace.get_run(run_id)
            # A commit can succeed before its caller observes an exception. Never
            # retract that committed version; startup will finalize its marker.
            if recorded is None or recorded.state is not RunState.SUCCEEDED:
                try:
                    if snapshot is not None:
                        workspace.retract(snapshot)
                    elif staging is not None:
                        workspace.discard(staging)
                finally:
                    if recorded is not None and recorded.state in {
                        RunState.SUBMITTED,
                        RunState.RUNNING,
                    }:
                        workspace.transition(
                            run_id,
                            recorded.state,
                            RunState.FAILED,
                            detail="GRAPH_MATERIALIZATION_FAILED",
                        )
            raise
