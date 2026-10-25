"""Closed version-1 evidence codec and bounded canonical JSON identity."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, fields
from typing import Any, TypeAlias

from capicove.contracts.data import (
    DataFormat,
    DatasetSchema,
    DatasetSnapshotRef,
    SchemaField,
    schema_value_type_from_json,
    schema_value_type_to_json,
)
from capicove.contracts.graph_build import (
    ChildStatementRef,
    Community,
    CommunityMembership,
    CommunityReport,
    DiagnosticFinding,
    Entity,
    EntityAlignment,
    EntityMembership,
    EntityMention,
    EvidenceBundle,
    GraphRetrievalBinding,
    GraphScope,
    GraphSnapshotManifest,
    GraphSnapshotRef,
    IdentityKey,
    InputSegment,
    PairDecision,
    RelationshipAggregate,
    RelationshipAssertion,
    RelationshipMention,
    RelativeSpan,
    ReportSections,
    ReportStatement,
    ResolutionMembership,
    SourceSpan,
    StageCoverage,
    StageResult,
    TableRef,
    TextUnit,
    ValidatedExtraction,
)
from capicove.contracts.retrieval import (
    CharacterRange,
    DistanceMetric,
    EmbeddingSpaceRef,
    QueryModality,
    TargetType,
    TemporalRange,
    TextChunkLocator,
    TextOrigin,
    VectorNormalization,
)


EvidenceValue: TypeAlias = GraphScope | SourceSpan | InputSegment | TextUnit
GraphValue: TypeAlias = (
    Entity
    | EntityAlignment
    | EntityMention
    | IdentityKey
    | PairDecision
    | RelativeSpan
    | RelationshipAggregate
    | RelationshipAssertion
    | RelationshipMention
    | ValidatedExtraction
    | ChildStatementRef
    | Community
    | CommunityMembership
    | CommunityReport
    | DiagnosticFinding
    | EntityMembership
    | EvidenceBundle
    | ReportSections
    | ReportStatement
    | ResolutionMembership
    | GraphRetrievalBinding
    | GraphSnapshotManifest
    | GraphSnapshotRef
    | StageCoverage
    | StageResult
    | TableRef
)
_GRAPH_TYPES = {
    cls.__name__: cls
    for cls in (
        ChildStatementRef,
        Community,
        CommunityMembership,
        CommunityReport,
        DiagnosticFinding,
        Entity,
        EntityAlignment,
        EntityMembership,
        EntityMention,
        EvidenceBundle,
        GraphRetrievalBinding,
        GraphSnapshotManifest,
        GraphSnapshotRef,
        IdentityKey,
        PairDecision,
        RelativeSpan,
        ReportSections,
        ReportStatement,
        ResolutionMembership,
        StageCoverage,
        StageResult,
        TableRef,
        RelationshipAggregate,
        RelationshipAssertion,
        RelationshipMention,
        ValidatedExtraction,
    )
}
_KINDS = {
    GraphScope: "scope",
    SourceSpan: "source_span",
    InputSegment: "segment",
    TextUnit: "text_unit",
}


def _validate_json(value: object, depth: int = 0, *, limit: int = 16) -> None:
    if depth > limit:
        raise ValueError(f"JSON nesting exceeds {limit} levels")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("JSON object keys must be strings")
            _validate_json(item, depth + 1, limit=limit)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_json(item, depth + 1, limit=limit)
        return
    raise TypeError("value is not a supported JSON value")


def canonical_bytes(value: object) -> bytes:
    """Encode JSON-compatible data without normalizing source Unicode text."""
    _validate_json(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def fingerprint(value: object) -> str:
    """Return the lowercase SHA-256 of a canonical JSON payload."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def evidence_to_json(value: EvidenceValue) -> str:
    """Serialize a supported evidence value in a closed versioned envelope."""
    if type(value) not in _KINDS:
        raise TypeError("unsupported evidence value")
    payload = asdict(value)
    if isinstance(value, TextUnit):
        payload["core_locator"]["origin"] = value.core_locator.origin.value
    return canonical_bytes(
        {"schema_version": 1, "kind": _KINDS[type(value)], "value": payload}
    ).decode("utf-8")


def _object(value: object, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("expected a JSON object")
    if set(value) != keys:
        raise ValueError("missing or unknown evidence fields")
    return value


def _record(value: object, cls: type[Any]) -> dict[str, Any]:
    return _object(value, {field.name for field in fields(cls)})


def _array(value: object) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError("expected a JSON array")
    return value


def _span(value: object) -> SourceSpan:
    return SourceSpan(**_record(value, SourceSpan))


def _segment(value: object) -> InputSegment:
    data = _record(value, InputSegment).copy()
    data["source_spans"] = tuple(_span(span) for span in _array(data["source_spans"]))
    return InputSegment(**data)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def evidence_from_json(payload: str) -> EvidenceValue:
    """Decode closed evidence records, validating every nested contract anew."""
    if not isinstance(payload, str):
        raise TypeError("payload must be a string")
    try:
        decoded = json.loads(payload, object_pairs_hook=_pairs)
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds parser bounds") from exc
    _validate_json(decoded)
    envelope = _object(decoded, {"schema_version", "kind", "value"})
    if type(envelope["schema_version"]) is not int or envelope["schema_version"] != 1:
        raise ValueError("unsupported evidence schema version")
    kind, value = envelope["kind"], envelope["value"]
    if kind == "scope":
        return GraphScope(**_record(value, GraphScope))
    if kind == "source_span":
        return _span(value)
    if kind == "segment":
        return _segment(value)
    if kind != "text_unit":
        raise ValueError("unsupported evidence kind")
    data = _record(value, TextUnit).copy()
    data["scope"] = GraphScope(**_record(data["scope"], GraphScope))
    data["segments"] = tuple(_segment(item) for item in _array(data["segments"]))
    data["section_path"] = tuple(_array(data["section_path"]))
    locator = _record(data["core_locator"], TextChunkLocator).copy()
    locator["origin"] = TextOrigin(locator["origin"])
    locator["character_range"] = CharacterRange(
        **_record(locator["character_range"], CharacterRange)
    )
    if locator["time_range"] is not None:
        locator["time_range"] = TemporalRange(**_record(locator["time_range"], TemporalRange))
    data["core_locator"] = TextChunkLocator(**locator)
    return TextUnit(**data)


def graph_to_json(value: GraphValue) -> str:
    """Encode a closed graph value without changing epistemic dimensions."""
    if type(value) not in _GRAPH_TYPES.values():
        raise TypeError("unsupported graph value")
    encoder = _manifest_canonical_bytes if type(value) is GraphSnapshotManifest else canonical_bytes
    return encoder(
        {
            "schema_version": 1,
            "kind": type(value).__name__,
            "value": _graph_payload(value),
        }
    ).decode("utf-8")


def _dataset_payload(value: DatasetSnapshotRef) -> dict[str, Any]:
    data = asdict(value)
    data["data_format"] = value.data_format.value
    data["schema"] = {
        "schema_version": value.schema.schema_version,
        "fields": [
            {
                "name": field.name,
                "logical_type": schema_value_type_to_json(field.logical_type),
                "nullable": field.nullable,
            }
            for field in value.schema.fields
        ],
    }
    return data


def _dataset_record(value: object) -> DatasetSnapshotRef:
    data = _record(value, DatasetSnapshotRef).copy()
    schema = _record(data["schema"], DatasetSchema).copy()
    schema_fields = []
    for item in _array(schema["fields"]):
        field = _record(item, SchemaField).copy()
        field["logical_type"] = schema_value_type_from_json(field["logical_type"])
        schema_fields.append(SchemaField(**field))
    data["schema"] = DatasetSchema(tuple(schema_fields), schema["schema_version"])
    data["data_format"] = DataFormat(data["data_format"])
    data["parent_snapshot_ids"] = tuple(_array(data["parent_snapshot_ids"]))
    return DatasetSnapshotRef(**data)


def _space_payload(value: EmbeddingSpaceRef) -> dict[str, Any]:
    data = asdict(value)
    data["normalization"] = value.normalization.value
    data["distance_metric"] = value.distance_metric.value
    data["query_modalities"] = sorted(item.value for item in value.query_modalities)
    data["target_modalities"] = sorted(item.value for item in value.target_modalities)
    return data


def _space_record(value: object) -> EmbeddingSpaceRef:
    data = _record(value, EmbeddingSpaceRef).copy()
    data["normalization"] = VectorNormalization(data["normalization"])
    data["distance_metric"] = DistanceMetric(data["distance_metric"])
    for field, enum in (("query_modalities", QueryModality), ("target_modalities", TargetType)):
        items = _array(data[field])
        converted = tuple(enum(item) for item in items)
        if len(set(converted)) != len(converted):
            raise ValueError("Duplicate embedding modality")
        data[field] = frozenset(converted)
    return EmbeddingSpaceRef(**data)


def _graph_payload(value: GraphValue) -> dict[str, Any]:
    data = asdict(value)
    if isinstance(value, TableRef):
        data["dataset_ref"] = _dataset_payload(value.dataset_ref)
    elif isinstance(value, GraphSnapshotManifest):
        data["input_snapshot_ref"] = _dataset_payload(value.input_snapshot_ref)
        data["tables"] = [_graph_payload(table) for table in value.tables]
    elif isinstance(value, GraphRetrievalBinding):
        data["embedding_spaces"] = [_space_payload(space) for space in value.embedding_spaces]
    return data


def manifest_bytes(manifest: GraphSnapshotManifest) -> bytes:
    """Encode semantic manifest content, excluding operational and retrieval state."""
    if type(manifest) is not GraphSnapshotManifest:
        raise TypeError("manifest must be GraphSnapshotManifest")
    return _manifest_canonical_bytes(_graph_payload(manifest))


def _manifest_canonical_bytes(value: object) -> bytes:
    """Bound the closed manifest's nested table schemas independently of evidence."""
    _validate_json(value, limit=32)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _graph_record(kind: str, value: object) -> Any:
    cls = _GRAPH_TYPES.get(kind)
    if cls is None:
        raise ValueError("unsupported graph record kind")
    data = _record(value, cls).copy()
    if cls is ReportSections:
        return ReportSections(
            **{
                key: tuple(_graph_record("ReportStatement", item) for item in _array(items))
                for key, items in data.items()
            }
        )
    if cls is CommunityReport:
        data["sections"] = _graph_record("ReportSections", data["sections"])
    if cls is TableRef:
        data["dataset_ref"] = _dataset_record(data["dataset_ref"])
    if cls is StageResult:
        data["coverage"] = _graph_record("StageCoverage", data["coverage"])
    if cls is GraphSnapshotManifest:
        data["input_snapshot_ref"] = _dataset_record(data["input_snapshot_ref"])
    if cls is GraphRetrievalBinding:
        data["embedding_spaces"] = tuple(
            _space_record(item) for item in _array(data["embedding_spaces"])
        )
    if "scope" in data:
        data["scope"] = GraphScope(**_record(data["scope"], GraphScope))
    if "source_spans" in data:
        data["source_spans"] = tuple(_span(item) for item in _array(data["source_spans"]))
    for field, child_kind in (
        ("identity_keys", "IdentityKey"),
        ("entities", "EntityMention"),
        ("relationships", "RelationshipMention"),
        ("child_statement_refs", "ChildStatementRef"),
        ("tables", "TableRef"),
        ("stages", "StageResult"),
    ):
        if field in data:
            data[field] = tuple(_graph_record(child_kind, item) for item in _array(data[field]))
    return cls(**data)


def graph_from_json(payload: str) -> GraphValue:
    """Decode known record kinds; arbitrary class imports are never permitted."""
    if not isinstance(payload, str):
        raise TypeError("payload must be text")
    try:
        decoded = json.loads(payload, object_pairs_hook=_pairs)
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds parser bounds") from exc
    envelope = _object(decoded, {"schema_version", "kind", "value"})
    _validate_json(decoded, limit=32 if envelope["kind"] == "GraphSnapshotManifest" else 16)
    if type(envelope["schema_version"]) is not int or envelope["schema_version"] != 1:
        raise ValueError("unsupported graph schema version")
    if not isinstance(envelope["kind"], str):
        raise TypeError("record kind must be a string")
    result: GraphValue = _graph_record(envelope["kind"], envelope["value"])
    return result
