"""Tests for immutable dataset snapshot and lineage contracts."""

from __future__ import annotations

import pytest

from capicove.contracts import ComplexSchemaType as PublicComplexSchemaType
from capicove.contracts import ComplexTypeKind as PublicComplexTypeKind
from capicove.contracts import SchemaValueType as PublicSchemaValueType
from capicove.contracts import schema_value_type_from_json, schema_value_type_to_json
from capicove.contracts.data import (
    ComplexSchemaType,
    ComplexTypeKind,
    DataFormat,
    DatasetSchema,
    DatasetSnapshotRef,
    LineageEdge,
    SchemaField,
    SchemaValueType,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def test_public_contracts_export_recursive_schema_types() -> None:
    assert PublicComplexSchemaType is ComplexSchemaType
    assert PublicComplexTypeKind is ComplexTypeKind
    assert PublicSchemaValueType is SchemaValueType


def _schema() -> DatasetSchema:
    return DatasetSchema((SchemaField("episode_id", "string", False),))


def _snapshot(**overrides: object) -> DatasetSnapshotRef:
    values: dict[str, object] = {
        "dataset_id": "dataset:agent-traces",
        "version": 1,
        "snapshot_id": f"snapshot:{SHA_A}",
        "uri": "file:///tmp/capicove/datasets/agent-traces/v1",
        "data_format": DataFormat.PARQUET,
        "schema": _schema(),
        "manifest_sha256": SHA_B,
        "processing_fingerprint": SHA_C,
        "parent_snapshot_ids": (),
    }
    values.update(overrides)
    return DatasetSnapshotRef(**values)  # type: ignore[arg-type]


def test_dataset_snapshot_requires_content_and_processing_identity() -> None:
    snapshot = _snapshot()

    assert snapshot.version == 1
    expected_sha256 = "95f2195d48c4715cf516f440b4d1e454e08fd5d526ddc15e8e2456fd04bd83d7"  # pragma: allowlist secret
    assert snapshot.schema.sha256 == expected_sha256


@pytest.mark.parametrize(
    "uri",
    (
        "relative/output",
        "s3://bucket/output",
        "file://relative/output",
        "file:///tmp/../secret",
    ),
)
def test_dataset_snapshot_rejects_noncanonical_local_uri(uri: str) -> None:
    with pytest.raises(ValueError, match="absolute canonical file URI"):
        _snapshot(uri=uri)


def test_schema_rejects_duplicate_field_names() -> None:
    with pytest.raises(ValueError, match="field names"):
        DatasetSchema((SchemaField("id", "string", False),) * 2)


def test_schema_rejects_unknown_logical_type() -> None:
    with pytest.raises(ValueError, match="logical_type"):
        SchemaField("value", "decimal", True)


def test_schema_supports_immutable_recursive_array_map_and_struct_types() -> None:
    profile = ComplexSchemaType(
        kind=ComplexTypeKind.STRUCT,
        fields=(
            SchemaField("name", "string", False),
            SchemaField(
                "scores",
                ComplexSchemaType(
                    kind=ComplexTypeKind.ARRAY,
                    element_type="float64",
                    contains_null=False,
                ),
                False,
            ),
        ),
    )
    attributes = ComplexSchemaType(
        kind=ComplexTypeKind.MAP,
        value_type="string",
        value_contains_null=True,
    )

    schema = DatasetSchema(
        (
            SchemaField("profile", profile, False),
            SchemaField("attributes", attributes, True),
        ),
        schema_version=2,
    )

    assert schema.schema_version == 2
    expected_sha256 = "71440f3ed5b3d40ecd1e6277957e7ae60040a3f16cc708af07673dc312250dac"  # pragma: allowlist secret
    assert schema.sha256 == expected_sha256


def test_schema_rejects_complex_type_in_version_1() -> None:
    tags = ComplexSchemaType(
        kind=ComplexTypeKind.ARRAY,
        element_type="string",
        contains_null=False,
    )

    with pytest.raises(ValueError, match="schema_version"):
        DatasetSchema((SchemaField("tags", tags, False),))


def test_schema_rejects_scalar_only_version_2() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        DatasetSchema((SchemaField("name", "string", False),), schema_version=2)


def test_struct_requires_nonempty_unique_tuple_of_schema_fields() -> None:
    with pytest.raises(ValueError, match="fields"):
        ComplexSchemaType(kind=ComplexTypeKind.STRUCT)

    with pytest.raises(ValueError, match="field names"):
        ComplexSchemaType(
            kind=ComplexTypeKind.STRUCT,
            fields=(SchemaField("name", "string", False),) * 2,
        )

    with pytest.raises(ValueError, match="fields"):
        ComplexSchemaType(
            kind=ComplexTypeKind.STRUCT,
            fields=[SchemaField("name", "string", False)],  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("complex_type", "message"),
    (
        (
            lambda: ComplexSchemaType(
                kind=ComplexTypeKind.ARRAY,
                element_type="string",
                contains_null=False,
                value_type="string",
            ),
            "value_type",
        ),
        (
            lambda: ComplexSchemaType(
                kind=ComplexTypeKind.MAP,
                value_type="string",
                value_contains_null=True,
                fields=(SchemaField("name", "string", False),),
            ),
            "fields",
        ),
        (
            lambda: ComplexSchemaType(
                kind=ComplexTypeKind.STRUCT,
                fields=(SchemaField("name", "string", False),),
                contains_null=False,
            ),
            "contains_null",
        ),
    ),
)
def test_complex_schema_type_rejects_inapplicable_fields(
    complex_type: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        complex_type()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("factory", "message"),
    (
        (
            lambda: ComplexSchemaType(
                kind=ComplexTypeKind.ARRAY,
                element_type="string",
                contains_null=1,  # type: ignore[arg-type]
            ),
            "contains_null",
        ),
        (
            lambda: ComplexSchemaType(
                kind=ComplexTypeKind.MAP,
                value_type="string",
                value_contains_null=0,  # type: ignore[arg-type]
            ),
            "value_contains_null",
        ),
        (
            lambda: SchemaField("name", "string", 1),  # type: ignore[arg-type]
            "nullable",
        ),
    ),
)
def test_schema_rejects_non_boolean_nullability_flags(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize(
    "factory",
    (
        lambda: ComplexSchemaType(kind="array"),  # type: ignore[arg-type]
        lambda: ComplexSchemaType(
            kind=ComplexTypeKind.ARRAY,
            element_type="decimal",
            contains_null=False,
        ),
    ),
)
def test_complex_schema_type_rejects_closed_kind_and_scalar_violations(
    factory: object,
) -> None:
    with pytest.raises(ValueError):
        factory()  # type: ignore[operator]


def test_schema_rejects_recursion_beyond_depth_16() -> None:
    value: str | ComplexSchemaType = "string"
    for _ in range(16):
        value = ComplexSchemaType(
            kind=ComplexTypeKind.ARRAY,
            element_type=value,
            contains_null=False,
        )

    DatasetSchema((SchemaField("value", value, False),), schema_version=2)

    with pytest.raises(ValueError, match="depth"):
        ComplexSchemaType(
            kind=ComplexTypeKind.ARRAY,
            element_type=value,
            contains_null=False,
        )


def test_scalar_version_1_schema_keeps_its_pre_change_digest() -> None:
    schema = DatasetSchema((SchemaField("record_id", "string", False),))
    expected_sha256 = "7c677736769395dc67af233c17ad4be0c107fbef82955a798a4cf89bc7a00a1e"  # pragma: allowlist secret

    assert schema.sha256 == expected_sha256


def test_schema_value_type_codec_round_trips_nested_canonical_values() -> None:
    value = ComplexSchemaType(
        kind=ComplexTypeKind.STRUCT,
        fields=(
            SchemaField(
                "metadata",
                ComplexSchemaType(
                    kind=ComplexTypeKind.MAP,
                    value_type=ComplexSchemaType(
                        kind=ComplexTypeKind.ARRAY,
                        element_type="json",
                        contains_null=True,
                    ),
                    value_contains_null=False,
                ),
                True,
            ),
        ),
    )
    expected = {
        "fields": [
            {
                "logical_type": {
                    "kind": "map",
                    "value_contains_null": False,
                    "value_type": {
                        "contains_null": True,
                        "element_type": "json",
                        "kind": "array",
                    },
                },
                "name": "metadata",
                "nullable": True,
            }
        ],
        "kind": "struct",
    }

    encoded = schema_value_type_to_json(value)

    assert encoded == expected
    assert schema_value_type_from_json(encoded) == value


@pytest.mark.parametrize(
    "value",
    (
        {
            "contains_null": False,
            "element_type": "string",
            "ignored": None,
            "kind": "array",
        },
        {"contains_null": 1, "element_type": "string", "kind": "array"},
        {"kind": "decimal", "value_type": "string", "value_contains_null": False},
        {"fields": [], "kind": "struct"},
    ),
)
def test_schema_value_type_codec_rejects_closed_malformed_input(value: object) -> None:
    with pytest.raises(ValueError):
        schema_value_type_from_json(value)


def test_schema_value_type_codec_rejects_recursion_beyond_depth_16() -> None:
    value: object = "string"
    for _ in range(17):
        value = {
            "contains_null": False,
            "element_type": value,
            "kind": "array",
        }

    with pytest.raises(ValueError, match="depth"):
        schema_value_type_from_json(value)


def test_schema_value_type_codec_accepts_16_complex_levels_ending_in_scalar() -> None:
    value: str | ComplexSchemaType = "string"
    for _ in range(16):
        value = ComplexSchemaType(
            kind=ComplexTypeKind.ARRAY,
            element_type=value,
            contains_null=False,
        )

    assert schema_value_type_from_json(schema_value_type_to_json(value)) == value


def test_snapshot_rejects_boolean_version() -> None:
    with pytest.raises(ValueError, match="version"):
        _snapshot(version=True)


def test_snapshot_rejects_duplicate_parents() -> None:
    parent = f"snapshot:{SHA_B}"
    with pytest.raises(ValueError, match="parent_snapshot_ids"):
        _snapshot(parent_snapshot_ids=(parent, parent))


def test_lineage_rejects_self_reference() -> None:
    with pytest.raises(ValueError, match="must differ"):
        LineageEdge(
            input_snapshot_id=f"snapshot:{SHA_A}",
            output_snapshot_id=f"snapshot:{SHA_A}",
            run_id="run:one",
            operator_ids=("capicove.data.project",),
        )


def test_lineage_freezes_unique_operator_ids() -> None:
    edge = LineageEdge(
        input_snapshot_id=f"snapshot:{SHA_A}",
        output_snapshot_id=f"snapshot:{SHA_B}",
        run_id="run:one",
        operator_ids=("capicove.data.project", "capicove.data.filter"),
    )

    assert edge.operator_ids == ("capicove.data.project", "capicove.data.filter")
