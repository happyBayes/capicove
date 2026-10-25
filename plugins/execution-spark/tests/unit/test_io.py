"""Tests for canonical schema conversion in the Spark dataset adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from capicove_execution_spark.io import (
    SparkDatasetIO,
    _compatible_schema,
    _local_path,
    _verified_parquet_sources,
    dataset_schema_from_spark,
    spark_schema,
)

from capicove.contracts.data import (
    ComplexSchemaType,
    ComplexTypeKind,
    DataFormat,
    DatasetSchema,
    DatasetSnapshotRef,
    SchemaField,
)


def test_canonical_schema_round_trips_through_spark_types() -> None:
    schema = DatasetSchema(
        (
            SchemaField("enabled", "boolean", False),
            SchemaField("records", "int64", False),
            SchemaField("score", "float64", True),
            SchemaField("name", "string", False),
            SchemaField("payload", "binary", True),
            SchemaField("observed_at", "timestamp_utc", True),
            SchemaField("metadata", "json", True),
        )
    )

    assert dataset_schema_from_spark(spark_schema(schema)) == schema


def test_complex_canonical_schema_round_trips_through_spark_types() -> None:
    from pyspark.sql.types import ArrayType, MapType, StringType, StructType

    schema = DatasetSchema(
        (
            SchemaField(
                "tags",
                ComplexSchemaType(
                    ComplexTypeKind.ARRAY,
                    element_type="string",
                    contains_null=False,
                ),
                False,
            ),
            SchemaField(
                "attributes",
                ComplexSchemaType(
                    ComplexTypeKind.MAP,
                    value_type="string",
                    value_contains_null=True,
                ),
                True,
            ),
            SchemaField(
                "profile",
                ComplexSchemaType(
                    ComplexTypeKind.STRUCT,
                    fields=(
                        SchemaField("name", "string", False),
                        SchemaField("metadata", "json", True),
                    ),
                ),
                False,
            ),
        ),
        schema_version=2,
    )

    spark_value = spark_schema(schema)

    assert isinstance(spark_value["tags"].dataType, ArrayType)
    assert spark_value["tags"].dataType.containsNull is False
    assert isinstance(spark_value["attributes"].dataType, MapType)
    assert isinstance(spark_value["attributes"].dataType.keyType, StringType)
    assert spark_value["attributes"].dataType.valueContainsNull is True
    assert isinstance(spark_value["profile"].dataType, StructType)
    assert spark_value["profile"].dataType["name"].nullable is False
    assert spark_value["profile"].dataType["metadata"].metadata == {"capicove.logical_type": "json"}
    assert dataset_schema_from_spark(spark_value) == schema


def test_nested_json_types_round_trip_through_complex_spark_fields() -> None:
    schema = DatasetSchema(
        (
            SchemaField(
                "json_array",
                ComplexSchemaType(
                    ComplexTypeKind.ARRAY,
                    element_type="json",
                    contains_null=False,
                ),
                False,
            ),
            SchemaField(
                "json_map",
                ComplexSchemaType(
                    ComplexTypeKind.MAP,
                    value_type="json",
                    value_contains_null=False,
                ),
                False,
            ),
            SchemaField(
                "json_profile",
                ComplexSchemaType(
                    ComplexTypeKind.STRUCT,
                    fields=(SchemaField("metadata", "json", False),),
                ),
                False,
            ),
        ),
        schema_version=2,
    )

    spark_value = spark_schema(schema)

    assert dataset_schema_from_spark(spark_value) == schema


def test_legacy_scalar_metadata_round_trips_through_spark_schema() -> None:
    from pyspark.sql.types import LongType, StringType, StructField, StructType

    value = StructType(
        [
            StructField(
                "record_id",
                StringType(),
                nullable=False,
                metadata={"capicove.logical_type": "string"},
            ),
            StructField(
                "sequence",
                LongType(),
                nullable=False,
                metadata={"capicove.logical_type": "int64"},
            ),
        ]
    )

    assert dataset_schema_from_spark(value) == DatasetSchema(
        (
            SchemaField("record_id", "string", False),
            SchemaField("sequence", "int64", False),
        )
    )


@pytest.mark.parametrize(
    "metadata",
    (
        {"capicove.logical_schema": "invalid"},
        {
            "capicove.logical_schema": {
                "kind": "array",
                "element_type": "json",
                "contains_null": False,
            }
        },
    ),
    ids=("malformed", "physical-type-conflict"),
)
def test_complex_logical_schema_metadata_is_validated(metadata: dict[str, object]) -> None:
    from pyspark.sql.types import MapType, StringType, StructField, StructType

    value = StructType(
        [
            StructField(
                "json_map",
                MapType(StringType(), StringType(), valueContainsNull=False),
                nullable=False,
                metadata=metadata,
            )
        ]
    )

    with pytest.raises(ValueError, match="logical schema"):
        dataset_schema_from_spark(value)


def test_enclosing_logical_schema_rejects_conflicting_nested_metadata() -> None:
    from pyspark.sql.types import StringType, StructField, StructType

    value = StructType(
        [
            StructField(
                "profile",
                StructType(
                    [
                        StructField(
                            "a",
                            StringType(),
                            nullable=False,
                            metadata={"capicove.logical_type": "string"},
                        ),
                        StructField(
                            "b",
                            StringType(),
                            nullable=False,
                            metadata={"capicove.logical_type": "json"},
                        ),
                    ]
                ),
                nullable=False,
                metadata={
                    "capicove.logical_schema": {
                        "kind": "struct",
                        "fields": [
                            {
                                "name": "a",
                                "logical_type": "json",
                                "nullable": False,
                            },
                            {
                                "name": "b",
                                "logical_type": "string",
                                "nullable": False,
                            },
                        ],
                    }
                },
            )
        ]
    )

    with pytest.raises(ValueError, match="logical schema"):
        dataset_schema_from_spark(value)


def test_spark_schema_conversion_rejects_non_string_map_keys() -> None:
    from pyspark.sql.types import IntegerType, MapType, StringType, StructField, StructType

    value = StructType(
        [StructField("attributes", MapType(IntegerType(), StringType()), nullable=True)]
    )

    with pytest.raises(ValueError, match="map key"):
        dataset_schema_from_spark(value)


def test_schema_compatibility_rejects_nullability_drift() -> None:
    from pyspark.sql.types import StringType, StructField, StructType

    observed = StructType([StructField("record_id", StringType(), True)])
    expected = DatasetSchema((SchemaField("record_id", "string", False),))

    assert not _compatible_schema(observed, expected)


def test_changed_single_jsonl_is_rejected_before_spark(tmp_path: Path) -> None:
    source = tmp_path / "records.jsonl"
    payload = b'{"record_id":"record-1"}\n'
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    snapshot = DatasetSnapshotRef(
        dataset_id="dataset:records",
        version=1,
        snapshot_id=f"snapshot:{digest}",
        uri=source.resolve().as_uri(),
        data_format=DataFormat.JSONL,
        schema=DatasetSchema((SchemaField("record_id", "string", True),)),
        manifest_sha256=hashlib.sha256(b"manifest").hexdigest(),
        processing_fingerprint=hashlib.sha256(b"source").hexdigest(),
        parent_snapshot_ids=(),
    )
    source.write_bytes(payload + b'{"record_id":"record-2"}\n')

    with pytest.raises(ValueError, match="content hash"):
        SparkDatasetIO(None).read(snapshot)


def test_local_path_rejects_symbolic_parent_directory(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "records.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(source_root, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        _local_path((linked_root / source.name).as_uri())


def _parquet_snapshot(root: Path) -> DatasetSnapshotRef:
    data = root / "data"
    data.mkdir(parents=True)
    part = data / "part-00000.parquet"
    payload = b"PAR1-declared-PAR1"
    part.write_bytes(payload)
    schema = DatasetSchema((SchemaField("record_id", "string", False),))
    manifest = {
        "data_format": "parquet",
        "dataset_id": "dataset:verified-parts",
        "files": [
            {
                "path": "data/part-00000.parquet",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        ],
        "parent_snapshot_ids": [],
        "processing_fingerprint": "c" * 64,
        "schema_sha256": schema.sha256,
        "version": 1,
    }
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    )
    (root / "manifest.json").write_bytes(manifest_bytes)
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    return DatasetSnapshotRef(
        dataset_id="dataset:verified-parts",
        version=1,
        snapshot_id=f"snapshot:{digest}",
        uri=root.resolve().as_uri(),
        data_format=DataFormat.PARQUET,
        schema=schema,
        manifest_sha256=digest,
        processing_fingerprint="c" * 64,
        parent_snapshot_ids=(),
    )


@pytest.mark.parametrize("placement", ("extra", "nested", "symlink"))
def test_parquet_snapshot_rejects_undeclared_or_linked_data_parts(
    tmp_path: Path,
    placement: str,
) -> None:
    root = tmp_path / "snapshot"
    snapshot = _parquet_snapshot(root)
    data = root / "data"
    if placement == "extra":
        (data / "extra.parquet").write_bytes(b"PAR1-extra-PAR1")
    elif placement == "nested":
        nested = data / "nested"
        nested.mkdir()
        (nested / "extra.parquet").write_bytes(b"PAR1-extra-PAR1")
    else:
        outside = tmp_path / "outside.parquet"
        outside.write_bytes(b"PAR1-extra-PAR1")
        (data / "extra.parquet").symlink_to(outside)

    with pytest.raises(ValueError, match=r"parts|symbolic link"):
        _verified_parquet_sources(root, snapshot)
