"""Canonical immutable data snapshot and lineage contracts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import TypeAlias
from urllib.parse import unquote, urlsplit

from capicove.contracts._validation import (
    require_non_blank,
    require_positive_int,
    require_sha256,
)


_DATASET_ID = re.compile(r"^dataset:[A-Za-z0-9][A-Za-z0-9._-]*$")
_SNAPSHOT_ID = re.compile(r"^snapshot:[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^run:[A-Za-z0-9][A-Za-z0-9._:-]*$")
_LOGICAL_TYPES = frozenset(
    {"boolean", "int64", "float64", "string", "binary", "timestamp_utc", "json"}
)


class DataFormat(StrEnum):
    """Identify a version-1 persistent dataset representation."""

    JSONL = "jsonl"
    PARQUET = "parquet"


class ComplexTypeKind(StrEnum):
    """Identify one supported recursive dataset schema value type."""

    ARRAY = "array"
    MAP = "map"
    STRUCT = "struct"


@dataclass(frozen=True, slots=True)
class SchemaField:
    """Describe one ordered field in a canonical dataset schema."""

    name: str
    logical_type: SchemaValueType
    nullable: bool

    def __post_init__(self) -> None:
        require_non_blank("name", self.name)
        _validate_schema_value_type(self.logical_type, depth=1)
        _require_boolean("nullable", self.nullable)


@dataclass(frozen=True, slots=True)
class ComplexSchemaType:
    """Describe one immutable recursive dataset schema value type."""

    kind: ComplexTypeKind
    element_type: SchemaValueType | None = None
    contains_null: bool | None = None
    value_type: SchemaValueType | None = None
    value_contains_null: bool | None = None
    fields: tuple[SchemaField, ...] = ()

    def __post_init__(self) -> None:
        _validate_complex_schema_type(self, depth=1)


SchemaValueType: TypeAlias = str | ComplexSchemaType


@dataclass(frozen=True, slots=True)
class DatasetSchema:
    """Describe and hash one closed ordered dataset schema."""

    fields: tuple[SchemaField, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        fields = tuple(self.fields)
        if not fields:
            raise ValueError("fields must not be empty")
        if any(not isinstance(field, SchemaField) for field in fields):
            raise TypeError("fields must contain SchemaField values")
        names = tuple(field.name for field in fields)
        if len(set(names)) != len(names):
            raise ValueError("field names must be unique")
        if isinstance(self.schema_version, bool):
            raise ValueError("schema_version must be a positive integer")
        require_positive_int("schema_version", self.schema_version)
        if self.schema_version not in (1, 2):
            raise ValueError("schema_version is unsupported")
        has_complex_type = any(
            isinstance(field.logical_type, ComplexSchemaType) for field in fields
        )
        if has_complex_type != (self.schema_version == 2):
            raise ValueError("schema_version 2 is required if and only if a field is complex")
        object.__setattr__(self, "fields", fields)

    @property
    def sha256(self) -> str:
        """Return the canonical schema digest."""
        if self.schema_version == 1:
            value = {
                "fields": [
                    {
                        "logical_type": field.logical_type,
                        "name": field.name,
                        "nullable": field.nullable,
                    }
                    for field in self.fields
                ],
                "schema_version": self.schema_version,
            }
        else:
            value = {
                "fields": [
                    {
                        "logical_type": _schema_value_type_json(field.logical_type),
                        "name": field.name,
                        "nullable": field.nullable,
                    }
                    for field in self.fields
                ],
                "schema_version": self.schema_version,
            }
        payload = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _validate_schema_value_type(value: object, depth: int) -> None:
    """Validate one scalar or recursive canonical schema type."""
    if isinstance(value, str):
        if value not in _LOGICAL_TYPES:
            raise ValueError("logical_type is not supported")
        return
    if not isinstance(value, ComplexSchemaType):
        raise ValueError("logical_type must be a supported scalar or complex type")
    _validate_complex_schema_type(value, depth)


def _validate_complex_schema_type(value: ComplexSchemaType, depth: int) -> None:
    """Validate one complex schema type and its recursive children."""
    if depth > 16:
        raise ValueError("complex schema type recursion depth must not exceed 16")
    if not isinstance(value.kind, ComplexTypeKind):
        raise ValueError("kind must be a ComplexTypeKind")

    if value.kind is ComplexTypeKind.ARRAY:
        _require_schema_value_type("element_type", value.element_type, depth)
        _require_boolean("contains_null", value.contains_null)
        _require_inapplicable_none("value_type", value.value_type)
        _require_inapplicable_none("value_contains_null", value.value_contains_null)
        _require_inapplicable_fields(value.fields)
        return

    if value.kind is ComplexTypeKind.MAP:
        _require_inapplicable_none("element_type", value.element_type)
        _require_inapplicable_none("contains_null", value.contains_null)
        _require_schema_value_type("value_type", value.value_type, depth)
        _require_boolean("value_contains_null", value.value_contains_null)
        _require_inapplicable_fields(value.fields)
        return

    _require_inapplicable_none("element_type", value.element_type)
    _require_inapplicable_none("contains_null", value.contains_null)
    _require_inapplicable_none("value_type", value.value_type)
    _require_inapplicable_none("value_contains_null", value.value_contains_null)
    _validate_struct_fields(value.fields, depth)


def _require_schema_value_type(field_name: str, value: object, depth: int) -> None:
    """Require and validate one recursive schema value type."""
    if value is None:
        raise ValueError(f"{field_name} is required")
    _validate_schema_value_type(value, depth + 1)


def _require_boolean(field_name: str, value: object) -> None:
    """Require one strict boolean field."""
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")


def _require_inapplicable_none(field_name: str, value: object) -> None:
    """Reject a value that does not apply to the selected complex kind."""
    if value is not None:
        raise ValueError(f"{field_name} does not apply to this complex type")


def _require_inapplicable_fields(fields: object) -> None:
    """Reject struct fields on a non-struct complex type."""
    if fields != ():
        raise ValueError("fields do not apply to this complex type")


def _validate_struct_fields(fields: object, depth: int) -> None:
    """Validate an ordered nonempty tuple of unique nested schema fields."""
    if not isinstance(fields, tuple) or not fields:
        raise ValueError("fields must be a nonempty tuple of SchemaField values")
    if any(not isinstance(field, SchemaField) for field in fields):
        raise ValueError("fields must contain SchemaField values")
    names = tuple(field.name for field in fields)
    if len(set(names)) != len(names):
        raise ValueError("field names must be unique")
    for field in fields:
        _validate_schema_value_type(field.logical_type, depth + 1)


def schema_value_type_to_json(value: SchemaValueType) -> str | dict[str, object]:
    """Return one canonical JSON-compatible schema value type."""
    _validate_schema_value_type(value, depth=1)
    return _schema_value_type_json(value)


def schema_value_type_from_json(value: object) -> SchemaValueType:
    """Decode one closed canonical JSON-compatible schema value type."""
    return _schema_value_type_from_json(value, depth=1)


def _schema_value_type_from_json(value: object, depth: int) -> SchemaValueType:
    """Decode one recursive schema value type at a bounded depth."""
    if isinstance(value, str):
        _validate_schema_value_type(value, depth)
        return value
    if depth > 16:
        raise ValueError("complex schema type recursion depth must not exceed 16")
    if not isinstance(value, dict):
        raise ValueError("schema value type must be a string or object")

    kind = value.get("kind")
    if kind == ComplexTypeKind.ARRAY.value:
        _require_json_keys(value, {"contains_null", "element_type", "kind"})
        contains_null = value["contains_null"]
        if not isinstance(contains_null, bool):
            raise ValueError("contains_null must be a boolean")
        return ComplexSchemaType(
            kind=ComplexTypeKind.ARRAY,
            element_type=_schema_value_type_from_json(value["element_type"], depth + 1),
            contains_null=contains_null,
        )
    if kind == ComplexTypeKind.MAP.value:
        _require_json_keys(value, {"kind", "value_contains_null", "value_type"})
        value_contains_null = value["value_contains_null"]
        if not isinstance(value_contains_null, bool):
            raise ValueError("value_contains_null must be a boolean")
        return ComplexSchemaType(
            kind=ComplexTypeKind.MAP,
            value_type=_schema_value_type_from_json(value["value_type"], depth + 1),
            value_contains_null=value_contains_null,
        )
    if kind == ComplexTypeKind.STRUCT.value:
        _require_json_keys(value, {"fields", "kind"})
        fields = value["fields"]
        if not isinstance(fields, list):
            raise ValueError("fields must be a list")
        return ComplexSchemaType(
            kind=ComplexTypeKind.STRUCT,
            fields=tuple(_schema_field_from_json(field, depth) for field in fields),
        )
    raise ValueError("kind must be a supported complex schema type")


def _require_json_keys(value: dict[str, object], expected: set[str]) -> None:
    """Require a closed JSON object with exactly the expected keys."""
    if set(value) != expected:
        raise ValueError("schema value type object has unsupported fields")


def _schema_field_from_json(value: object, depth: int) -> SchemaField:
    """Decode one closed canonical JSON-compatible schema field."""
    if not isinstance(value, dict):
        raise ValueError("field must be an object")
    _require_json_keys(value, {"logical_type", "name", "nullable"})
    name = value["name"]
    nullable = value["nullable"]
    if not isinstance(name, str):
        raise ValueError("name must be a string")
    if not isinstance(nullable, bool):
        raise ValueError("nullable must be a boolean")
    return SchemaField(
        name,
        _schema_value_type_from_json(value["logical_type"], depth + 1),
        nullable,
    )


def _schema_value_type_json(value: SchemaValueType) -> str | dict[str, object]:
    """Return the canonical JSON-compatible representation of one schema type."""
    if isinstance(value, str):
        return value
    if value.kind is ComplexTypeKind.ARRAY:
        if value.element_type is None:
            raise ValueError("element_type is required")
        return {
            "contains_null": value.contains_null,
            "element_type": _schema_value_type_json(value.element_type),
            "kind": value.kind.value,
        }
    if value.kind is ComplexTypeKind.MAP:
        if value.value_type is None:
            raise ValueError("value_type is required")
        return {
            "kind": value.kind.value,
            "value_contains_null": value.value_contains_null,
            "value_type": _schema_value_type_json(value.value_type),
        }
    return {
        "fields": [
            {
                "logical_type": _schema_value_type_json(field.logical_type),
                "name": field.name,
                "nullable": field.nullable,
            }
            for field in value.fields
        ],
        "kind": value.kind.value,
    }


@dataclass(frozen=True, slots=True)
class DatasetSnapshotRef:
    """Reference one immutable, fully committed dataset version."""

    dataset_id: str
    version: int
    snapshot_id: str
    uri: str
    data_format: DataFormat
    schema: DatasetSchema
    manifest_sha256: str
    processing_fingerprint: str
    parent_snapshot_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if _DATASET_ID.fullmatch(self.dataset_id) is None:
            raise ValueError("dataset_id must use the dataset: prefix")
        if isinstance(self.version, bool):
            raise ValueError("version must be a positive integer")
        require_positive_int("version", self.version)
        _require_snapshot_id("snapshot_id", self.snapshot_id)
        _require_local_file_uri(self.uri)
        if not isinstance(self.data_format, DataFormat):
            raise ValueError("data_format must be a DataFormat")
        if not isinstance(self.schema, DatasetSchema):
            raise TypeError("schema must be a DatasetSchema")
        require_sha256("manifest_sha256", self.manifest_sha256)
        require_sha256("processing_fingerprint", self.processing_fingerprint)
        parents = tuple(self.parent_snapshot_ids)
        for parent in parents:
            _require_snapshot_id("parent_snapshot_id", parent)
        if len(set(parents)) != len(parents):
            raise ValueError("parent_snapshot_ids must be unique")
        if self.snapshot_id in parents:
            raise ValueError("snapshot_id must not be its own parent")
        object.__setattr__(self, "parent_snapshot_ids", parents)


@dataclass(frozen=True, slots=True)
class LineageEdge:
    """Link one immutable input snapshot to a derived output snapshot."""

    input_snapshot_id: str
    output_snapshot_id: str
    run_id: str
    operator_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_snapshot_id("input_snapshot_id", self.input_snapshot_id)
        _require_snapshot_id("output_snapshot_id", self.output_snapshot_id)
        if self.input_snapshot_id == self.output_snapshot_id:
            raise ValueError("input and output snapshot IDs must differ")
        if _RUN_ID.fullmatch(self.run_id) is None:
            raise ValueError("run_id must use the run: prefix")
        operator_ids = tuple(self.operator_ids)
        if not operator_ids:
            raise ValueError("operator_ids must not be empty")
        for operator_id in operator_ids:
            require_non_blank("operator_id", operator_id)
        if len(set(operator_ids)) != len(operator_ids):
            raise ValueError("operator_ids must be unique")
        object.__setattr__(self, "operator_ids", operator_ids)


def _require_snapshot_id(field_name: str, value: str) -> None:
    if _SNAPSHOT_ID.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a snapshot-prefixed lowercase SHA-256 hash")


def _require_local_file_uri(value: str) -> None:
    parsed = urlsplit(value)
    decoded = unquote(parsed.path)
    path = PurePosixPath(decoded)
    if (
        parsed.scheme != "file"
        or parsed.netloc
        or not decoded.startswith("/")
        or ".." in path.parts
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("uri must be an absolute canonical file URI")
