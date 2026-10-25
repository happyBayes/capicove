"""Explicit local JSONL and Parquet I/O for immutable dataset snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit

from capicove.contracts.data import (
    ComplexSchemaType,
    ComplexTypeKind,
    DataFormat,
    DatasetSchema,
    DatasetSnapshotRef,
    SchemaField,
    SchemaValueType,
)
from capicove.export import decode_trace_export_manifest


def spark_schema(schema: DatasetSchema) -> Any:
    """Convert a canonical schema to a Spark StructType."""
    from pyspark.sql.types import StructField, StructType

    return StructType([_spark_field(field, StructField) for field in schema.fields])


def _spark_field(field: SchemaField, struct_field_type: type[Any]) -> Any:
    """Convert one canonical field to a Spark StructField."""
    return struct_field_type(
        field.name,
        _spark_data_type(field.logical_type),
        field.nullable,
        _spark_metadata_for_logical_type(field.logical_type),
    )


def _spark_metadata_for_logical_type(schema_type: SchemaValueType) -> dict[str, object]:
    """Return bounded Spark metadata for one governed logical output type."""
    if schema_type == "json":
        return {"capicove.logical_type": "json"}
    if isinstance(schema_type, ComplexSchemaType) and _contains_json_type(schema_type):
        return {"capicove.logical_schema": _logical_schema_metadata(schema_type)}
    return {}


def _spark_data_type(schema_type: SchemaValueType) -> Any:
    """Convert one declared canonical type to its reviewed Spark data type."""
    from pyspark.sql.types import (
        ArrayType,
        BinaryType,
        BooleanType,
        DoubleType,
        LongType,
        MapType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    scalar_types = {
        "boolean": BooleanType,
        "int64": LongType,
        "float64": DoubleType,
        "string": StringType,
        "binary": BinaryType,
        "timestamp_utc": TimestampType,
        "json": StringType,
    }
    if isinstance(schema_type, str):
        return scalar_types[schema_type]()
    if schema_type.kind is ComplexTypeKind.ARRAY:
        if schema_type.element_type is None or schema_type.contains_null is None:
            raise ValueError("array schema type is invalid")
        return ArrayType(
            _spark_data_type(schema_type.element_type),
            containsNull=schema_type.contains_null,
        )
    if schema_type.kind is ComplexTypeKind.MAP:
        if schema_type.value_type is None or schema_type.value_contains_null is None:
            raise ValueError("map schema type is invalid")
        return MapType(
            StringType(),
            _spark_data_type(schema_type.value_type),
            valueContainsNull=schema_type.value_contains_null,
        )
    if schema_type.kind is ComplexTypeKind.STRUCT:
        return StructType([_spark_field(field, StructField) for field in schema_type.fields])
    raise ValueError("complex schema type is unsupported")


def dataset_schema_from_spark(value: Any) -> DatasetSchema:
    """Convert a reviewed Spark StructType to a canonical schema."""
    from pyspark.sql.types import (
        StringType,
        StructType,
    )

    if not isinstance(value, StructType):
        raise ValueError("Spark schema must be a StructType")
    fields = tuple(_canonical_field(field, StringType) for field in value.fields)
    schema_version = (
        2 if any(isinstance(field.logical_type, ComplexSchemaType) for field in fields) else 1
    )
    return DatasetSchema(fields, schema_version=schema_version)


def _canonical_schema_field_from_spark(spark_field: Any) -> SchemaField:
    """Validate and convert one Spark field, including governed metadata."""
    from pyspark.sql.types import StringType

    return _canonical_field(spark_field, StringType)


def _canonical_field(spark_field: Any, string_type: type[Any]) -> SchemaField:
    """Convert one reviewed Spark field to a canonical field."""
    logical_type = _canonical_data_type(spark_field.dataType)
    declared_type = spark_field.metadata.get("capicove.logical_type")
    declared_schema = spark_field.metadata.get("capicove.logical_schema")
    if declared_type is not None:
        if not isinstance(logical_type, str):
            raise ValueError(f"Spark field metadata conflicts for field: {spark_field.name}")
        if declared_type == "json":
            if not isinstance(spark_field.dataType, string_type):
                raise ValueError(f"Spark field metadata conflicts for field: {spark_field.name}")
            logical_type = "json"
        elif declared_type != logical_type:
            raise ValueError(f"Spark field metadata conflicts for field: {spark_field.name}")
    if declared_schema is not None:
        if declared_type is not None:
            raise ValueError(f"Spark field metadata conflicts for field: {spark_field.name}")
        try:
            marked_type = _logical_schema_from_metadata(declared_schema)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Spark logical schema metadata is invalid for field: {spark_field.name}"
            ) from error
        if (
            not isinstance(marked_type, ComplexSchemaType)
            or not _contains_json_type(marked_type)
            or not _spark_type_matches_schema_type(spark_field.dataType, marked_type)
            or not _nested_spark_metadata_matches_schema_type(spark_field.dataType, marked_type)
        ):
            raise ValueError(
                f"Spark logical schema metadata conflicts for field: {spark_field.name}"
            )
        logical_type = marked_type
    return SchemaField(spark_field.name, logical_type, spark_field.nullable)


def _contains_json_type(schema_type: ComplexSchemaType) -> bool:
    """Return whether a complex canonical type contains a JSON scalar."""
    if schema_type.kind is ComplexTypeKind.ARRAY:
        return schema_type.element_type == "json" or (
            isinstance(schema_type.element_type, ComplexSchemaType)
            and _contains_json_type(schema_type.element_type)
        )
    if schema_type.kind is ComplexTypeKind.MAP:
        return schema_type.value_type == "json" or (
            isinstance(schema_type.value_type, ComplexSchemaType)
            and _contains_json_type(schema_type.value_type)
        )
    return any(
        field.logical_type == "json"
        or (
            isinstance(field.logical_type, ComplexSchemaType)
            and _contains_json_type(field.logical_type)
        )
        for field in schema_type.fields
    )


def _logical_schema_metadata(schema_type: ComplexSchemaType) -> dict[str, object]:
    """Return a closed metadata marker for a complex type containing JSON."""
    if schema_type.kind is ComplexTypeKind.ARRAY:
        if schema_type.element_type is None:
            raise ValueError("array schema type is invalid")
        return {
            "kind": schema_type.kind.value,
            "element_type": _logical_type_metadata(schema_type.element_type),
            "contains_null": schema_type.contains_null,
        }
    if schema_type.kind is ComplexTypeKind.MAP:
        if schema_type.value_type is None:
            raise ValueError("map schema type is invalid")
        return {
            "kind": schema_type.kind.value,
            "value_type": _logical_type_metadata(schema_type.value_type),
            "value_contains_null": schema_type.value_contains_null,
        }
    return {
        "kind": schema_type.kind.value,
        "fields": [
            {
                "name": field.name,
                "logical_type": _logical_type_metadata(field.logical_type),
                "nullable": field.nullable,
            }
            for field in schema_type.fields
        ],
    }


def _logical_type_metadata(schema_type: SchemaValueType) -> object:
    """Return a closed metadata value for one canonical type."""
    if isinstance(schema_type, str):
        return schema_type
    return _logical_schema_metadata(schema_type)


def _logical_schema_from_metadata(value: object, depth: int = 1) -> ComplexSchemaType:
    """Parse and validate one closed complex logical-schema metadata marker."""
    if depth > 16 or not isinstance(value, dict):
        raise ValueError("logical schema metadata is invalid")
    kind_value = value.get("kind")
    if not isinstance(kind_value, str):
        raise ValueError("logical schema metadata kind is invalid")
    try:
        kind = ComplexTypeKind(kind_value)
    except ValueError as error:
        raise ValueError("logical schema metadata kind is invalid") from error
    if kind is ComplexTypeKind.ARRAY:
        if set(value) != {"kind", "element_type", "contains_null"}:
            raise ValueError("logical schema metadata array is invalid")
        return ComplexSchemaType(
            kind,
            element_type=_logical_type_from_metadata(value["element_type"], depth + 1),
            contains_null=value["contains_null"],
        )
    if kind is ComplexTypeKind.MAP:
        if set(value) != {"kind", "value_type", "value_contains_null"}:
            raise ValueError("logical schema metadata map is invalid")
        return ComplexSchemaType(
            kind,
            value_type=_logical_type_from_metadata(value["value_type"], depth + 1),
            value_contains_null=value["value_contains_null"],
        )
    if set(value) != {"kind", "fields"} or not isinstance(value["fields"], list):
        raise ValueError("logical schema metadata struct is invalid")
    fields: list[SchemaField] = []
    for field_value in value["fields"]:
        if not isinstance(field_value, dict) or set(field_value) != {
            "name",
            "logical_type",
            "nullable",
        }:
            raise ValueError("logical schema metadata field is invalid")
        fields.append(
            SchemaField(
                field_value["name"],
                _logical_type_from_metadata(field_value["logical_type"], depth + 1),
                field_value["nullable"],
            )
        )
    return ComplexSchemaType(kind, fields=tuple(fields))


def _logical_type_from_metadata(value: object, depth: int) -> SchemaValueType:
    """Parse and validate one closed scalar or complex logical metadata type."""
    if isinstance(value, str):
        SchemaField("metadata", value, True)
        return value
    return _logical_schema_from_metadata(value, depth)


def _spark_type_matches_schema_type(spark_type: Any, schema_type: SchemaValueType) -> bool:
    """Return whether a physical Spark type can represent one logical type marker."""
    from pyspark.sql.types import (
        ArrayType,
        BinaryType,
        BooleanType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        MapType,
        StringType,
        StructType,
        TimestampType,
    )

    if isinstance(schema_type, str):
        scalar_types: dict[str, tuple[type[Any], ...]] = {
            "boolean": (BooleanType,),
            "int64": (IntegerType, LongType),
            "float64": (FloatType, DoubleType),
            "string": (StringType,),
            "binary": (BinaryType,),
            "timestamp_utc": (TimestampType,),
            "json": (StringType,),
        }
        return isinstance(spark_type, scalar_types[schema_type])
    if schema_type.kind is ComplexTypeKind.ARRAY:
        return (
            isinstance(spark_type, ArrayType)
            and schema_type.element_type is not None
            and _spark_type_matches_schema_type(spark_type.elementType, schema_type.element_type)
        )
    if schema_type.kind is ComplexTypeKind.MAP:
        return (
            isinstance(spark_type, MapType)
            and isinstance(spark_type.keyType, StringType)
            and schema_type.value_type is not None
            and _spark_type_matches_schema_type(spark_type.valueType, schema_type.value_type)
        )
    return (
        isinstance(spark_type, StructType)
        and len(spark_type.fields) == len(schema_type.fields)
        and all(
            spark_field.name == schema_field.name
            and _spark_type_matches_schema_type(spark_field.dataType, schema_field.logical_type)
            for spark_field, schema_field in zip(spark_type.fields, schema_type.fields, strict=True)
        )
    )


def _nested_spark_metadata_matches_schema_type(
    spark_type: Any, schema_type: SchemaValueType
) -> bool:
    """Return whether nested field metadata agrees with an enclosing marker."""
    from pyspark.sql.types import ArrayType, MapType, StringType, StructType

    if isinstance(schema_type, str):
        return True
    if schema_type.kind is ComplexTypeKind.ARRAY:
        return (
            isinstance(spark_type, ArrayType)
            and schema_type.element_type is not None
            and _nested_spark_metadata_matches_schema_type(
                spark_type.elementType, schema_type.element_type
            )
        )
    if schema_type.kind is ComplexTypeKind.MAP:
        return (
            isinstance(spark_type, MapType)
            and schema_type.value_type is not None
            and _nested_spark_metadata_matches_schema_type(
                spark_type.valueType, schema_type.value_type
            )
        )
    if not isinstance(spark_type, StructType) or len(spark_type.fields) != len(schema_type.fields):
        return False
    for spark_field, schema_field in zip(spark_type.fields, schema_type.fields, strict=True):
        if spark_field.name != schema_field.name:
            return False
        metadata = spark_field.metadata
        if {
            "capicove.logical_type",
            "capicove.logical_schema",
        }.intersection(metadata):
            try:
                nested_type = _canonical_field(spark_field, StringType).logical_type
            except (TypeError, ValueError):
                return False
            if not _type_shapes_match(nested_type, schema_field.logical_type):
                return False
        if not _nested_spark_metadata_matches_schema_type(
            spark_field.dataType, schema_field.logical_type
        ):
            return False
    return True


def _canonical_data_type(spark_type: Any) -> SchemaValueType:
    """Convert one reviewed Spark data type to its canonical representation."""
    from pyspark.sql.types import (
        ArrayType,
        BinaryType,
        BooleanType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        MapType,
        StringType,
        StructType,
        TimestampType,
    )

    scalar_types: dict[type[Any], str] = {
        BooleanType: "boolean",
        IntegerType: "int64",
        LongType: "int64",
        FloatType: "float64",
        DoubleType: "float64",
        StringType: "string",
        BinaryType: "binary",
        TimestampType: "timestamp_utc",
    }
    if isinstance(spark_type, ArrayType):
        return ComplexSchemaType(
            ComplexTypeKind.ARRAY,
            element_type=_canonical_data_type(spark_type.elementType),
            contains_null=spark_type.containsNull,
        )
    if isinstance(spark_type, MapType):
        if not isinstance(spark_type.keyType, StringType):
            raise ValueError("Spark map key type must be string")
        return ComplexSchemaType(
            ComplexTypeKind.MAP,
            value_type=_canonical_data_type(spark_type.valueType),
            value_contains_null=spark_type.valueContainsNull,
        )
    if isinstance(spark_type, StructType):
        return ComplexSchemaType(
            ComplexTypeKind.STRUCT,
            fields=tuple(_canonical_field(field, StringType) for field in spark_type.fields),
        )
    try:
        return scalar_types[type(spark_type)]
    except KeyError as error:
        raise ValueError("Spark type is unsupported") from error


class SparkDatasetIO:
    """Read committed local snapshots and stage deterministic Parquet output."""

    def __init__(self, spark_session: Any) -> None:
        self._spark = spark_session

    def read(self, snapshot: DatasetSnapshotRef, /) -> Any:
        """Read one local committed snapshot using only its declared format."""
        path = _local_path(snapshot.uri)
        if snapshot.data_format is DataFormat.JSONL:
            sources = _jsonl_sources(path, snapshot)
            frame = (
                self._spark.read.schema(spark_schema(snapshot.schema))
                .option("mode", "FAILFAST")
                .option("multiLine", "false")
                .json(sources)
            )
            if not _compatible_frame(frame, snapshot.schema):
                raise ValueError("JSONL schema does not match the snapshot contract")
            return frame
        if snapshot.data_format is DataFormat.PARQUET:
            sources = [str(source) for source in _verified_parquet_sources(path, snapshot)]
            frame = self._spark.read.parquet(*sources)
            if not _compatible_frame(frame, snapshot.schema):
                raise ValueError("Parquet schema does not match the snapshot contract")
            return frame
        raise ValueError("dataset format is unsupported by the Spark MVP")

    def write_parquet(self, frame: Any, destination: Path, /) -> None:
        """Write one DataFrame to a new local Parquet staging directory."""
        if not destination.is_absolute():
            raise ValueError("Parquet destination must be absolute")
        if destination.exists():
            raise FileExistsError("Parquet destination already exists")
        frame.write.mode("errorifexists").parquet(str(destination))

    def validate_parquet(self, path: Path, expected_schema: DatasetSchema, /) -> bool:
        """Return whether staged Parquet is readable with the expected logical schema."""
        try:
            if not path.is_absolute() or not path.is_dir():
                return False
            frame = self._spark.read.parquet(str(path))
            return _compatible_frame(frame, expected_schema)
        except Exception:
            return False


def _local_path(uri: str) -> Path:
    parsed = urlsplit(uri)
    if parsed.scheme != "file" or parsed.netloc:
        raise ValueError("dataset URI must identify a local file")
    path = Path(unquote(parsed.path))
    if any(candidate.is_symlink() for candidate in (path, *path.parents)):
        raise ValueError("dataset URI must not identify a symbolic link")
    return path.resolve(strict=True)


def _jsonl_sources(path: Path, snapshot: DatasetSnapshotRef) -> str | list[str]:
    if path.is_file() and not path.is_symlink():
        if _file_sha256(path) != snapshot.snapshot_id.removeprefix("snapshot:"):
            raise ValueError("JSONL content hash does not match the snapshot contract")
        return str(path)
    if not path.is_dir() or path.is_symlink():
        raise ValueError("JSONL source must be a regular file or trace export directory")
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("JSONL directory must contain a regular trace manifest")
    manifest_bytes = _verified_manifest_bytes(manifest_path, snapshot)
    value = json.loads(manifest_bytes)
    if not isinstance(value, dict):
        raise ValueError("trace manifest must be an object")
    manifest = decode_trace_export_manifest(value)
    sources: list[str] = []
    for shard in manifest.shards:
        source = path / shard.path
        if not source.is_file() or source.is_symlink():
            raise ValueError("trace manifest shard must be a regular file")
        observed = source.stat()
        if observed.st_size != shard.size_bytes:
            raise ValueError("trace shard size does not match the manifest")
        if _file_sha256(source) != shard.sha256:
            raise ValueError("trace shard hash does not match the manifest")
        sources.append(str(source))
    if not sources:
        raise ValueError("trace export must contain at least one JSONL shard")
    return sources


def _verified_parquet_sources(
    path: Path,
    snapshot: DatasetSnapshotRef,
) -> tuple[Path, ...]:
    if not path.is_dir() or path.is_symlink():
        raise ValueError("Parquet snapshot must be a regular directory")
    manifest_path = path / "manifest.json"
    manifest_bytes = _verified_manifest_bytes(manifest_path, snapshot)
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Parquet manifest must be valid JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("Parquet manifest must be an object")
    expected_values = {
        "data_format": DataFormat.PARQUET.value,
        "dataset_id": snapshot.dataset_id,
        "parent_snapshot_ids": list(snapshot.parent_snapshot_ids),
        "processing_fingerprint": snapshot.processing_fingerprint,
        "schema_sha256": snapshot.schema.sha256,
        "version": snapshot.version,
    }
    if any(manifest.get(key) != value for key, value in expected_values.items()):
        raise ValueError("Parquet manifest does not match the snapshot contract")
    file_values = manifest.get("files")
    if not isinstance(file_values, list) or not file_values:
        raise ValueError("Parquet manifest must declare data parts")
    declared: list[Path] = []
    for value in file_values:
        if not isinstance(value, dict) or set(value) != {"path", "sha256", "size_bytes"}:
            raise ValueError("Parquet manifest file entry is invalid")
        relative_value = value["path"]
        if not isinstance(relative_value, str):
            raise ValueError("Parquet manifest path must be a string")
        relative = PurePosixPath(relative_value)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or len(relative.parts) < 2
            or relative.parts[0] != "data"
            or relative.suffix != ".parquet"
        ):
            raise ValueError("Parquet manifest part path is unsafe")
        part = path.joinpath(*relative.parts)
        if any(
            candidate.is_symlink() for candidate in (part, *part.parents[: len(relative.parts)])
        ):
            raise ValueError("Parquet manifest part path contains a symbolic link")
        if not part.is_file():
            raise ValueError("Parquet manifest part must be a regular file")
        size_bytes = value["size_bytes"]
        sha256 = value["sha256"]
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool):
            raise ValueError("Parquet manifest part size is invalid")
        if part.stat().st_size != size_bytes:
            raise ValueError("Parquet part size does not match the manifest")
        if not isinstance(sha256, str) or _file_sha256(part) != sha256:
            raise ValueError("Parquet part hash does not match the manifest")
        declared.append(part)
    data_path = path / "data"
    if not data_path.is_dir() or data_path.is_symlink():
        raise ValueError("Parquet snapshot data directory is invalid")
    observed = _observed_parquet_parts(data_path)
    if set(observed) != set(declared) or len(declared) != len(set(declared)):
        raise ValueError("Parquet data parts do not match the manifest")
    return tuple(declared)


def _observed_parquet_parts(data_path: Path) -> tuple[Path, ...]:
    """List exact regular Parquet files without following symbolic links."""
    observed: list[Path] = []
    for root_value, directory_names, file_names in os.walk(data_path, followlinks=False):
        root = Path(root_value)
        for name in directory_names:
            candidate = root / name
            if stat.S_ISLNK(candidate.lstat().st_mode):
                raise ValueError("Parquet data path contains a symbolic link")
        for name in file_names:
            candidate = root / name
            mode = candidate.lstat().st_mode
            if candidate.suffix != ".parquet":
                continue
            if stat.S_ISLNK(mode):
                raise ValueError("Parquet data path contains a symbolic link")
            if not stat.S_ISREG(mode):
                raise ValueError("Parquet data parts must be regular files")
            observed.append(candidate)
    return tuple(observed)


def _verified_manifest_bytes(path: Path, snapshot: DatasetSnapshotRef) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ValueError("snapshot manifest must be a regular file")
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("snapshot manifest exceeds the byte limit")
    payload = path.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != snapshot.manifest_sha256:
        raise ValueError("manifest hash does not match the snapshot contract")
    if snapshot.snapshot_id != f"snapshot:{observed}":
        raise ValueError("snapshot identity does not match the manifest")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _compatible_schema(observed: Any, expected: DatasetSchema) -> bool:
    observed_schema = dataset_schema_from_spark(observed)
    return observed_schema == expected


def _compatible_types(observed: Any, expected: DatasetSchema) -> bool:
    observed_schema = dataset_schema_from_spark(observed)
    return _field_shapes_match(observed_schema.fields, expected.fields)


def _field_shapes_match(
    observed_fields: tuple[SchemaField, ...], expected_fields: tuple[SchemaField, ...]
) -> bool:
    """Return whether fields have the same recursive names and data-type shapes."""
    return len(observed_fields) == len(expected_fields) and all(
        observed_field.name == expected_field.name
        and _type_shapes_match(observed_field.logical_type, expected_field.logical_type)
        for observed_field, expected_field in zip(observed_fields, expected_fields, strict=True)
    )


def _type_shapes_match(observed_type: SchemaValueType, expected_type: SchemaValueType) -> bool:
    """Return whether two canonical types match without physical nullability."""
    if isinstance(observed_type, str) or isinstance(expected_type, str):
        return observed_type == expected_type
    if observed_type.kind is not expected_type.kind:
        return False
    if observed_type.kind is ComplexTypeKind.ARRAY:
        return (
            observed_type.element_type is not None
            and expected_type.element_type is not None
            and _type_shapes_match(observed_type.element_type, expected_type.element_type)
        )
    if observed_type.kind is ComplexTypeKind.MAP:
        return (
            observed_type.value_type is not None
            and expected_type.value_type is not None
            and _type_shapes_match(observed_type.value_type, expected_type.value_type)
        )
    return _field_shapes_match(observed_type.fields, expected_type.fields)


def _compatible_frame(frame: Any, expected: DatasetSchema) -> bool:
    if not _compatible_types(frame.schema, expected):
        return False
    from pyspark.sql import functions

    null_expression = functions.lit(False)
    for field in expected.fields:
        quoted_name = f"`{field.name.replace('`', '``')}`"
        null_expression = null_expression | _null_violation(frame[quoted_name], field)
    return bool(frame.filter(null_expression).limit(1).count() == 0)


def _null_violation(column: Any, field: SchemaField) -> Any:
    """Build a reviewed Spark expression for a field's recursive null contract."""
    from pyspark.sql import functions

    violation = column.isNull() if not field.nullable else functions.lit(False)
    if isinstance(field.logical_type, str):
        return violation
    nested_violation = _nested_null_violation(column, field.logical_type)
    return violation | nested_violation


def _nested_null_violation(column: Any, schema_type: ComplexSchemaType) -> Any:
    """Build a null-safe reviewed Spark expression for nested field values."""
    from pyspark.sql import functions

    if schema_type.kind is ComplexTypeKind.ARRAY:
        if schema_type.element_type is None or schema_type.contains_null is None:
            raise ValueError("array schema type is invalid")
        element = SchemaField("element", schema_type.element_type, schema_type.contains_null)
        nested = functions.exists(column, lambda item: _null_violation(item, element))
    elif schema_type.kind is ComplexTypeKind.MAP:
        if schema_type.value_type is None or schema_type.value_contains_null is None:
            raise ValueError("map schema type is invalid")
        value = SchemaField("value", schema_type.value_type, schema_type.value_contains_null)
        nested = functions.exists(
            functions.map_values(column), lambda item: _null_violation(item, value)
        )
    elif schema_type.kind is ComplexTypeKind.STRUCT:
        nested = functions.lit(False)
        for child in schema_type.fields:
            nested = nested | _null_violation(column.getField(child.name), child)
    else:
        raise ValueError("complex schema type is unsupported")
    return functions.when(column.isNotNull(), nested).otherwise(functions.lit(False))
