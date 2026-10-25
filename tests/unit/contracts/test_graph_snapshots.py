"""Closed snapshot table registry and immutable publication boundaries."""

import hashlib
import json
from dataclasses import replace
from typing import Any

import pytest

from capicove.contracts.data import (
    ComplexSchemaType,
    ComplexTypeKind,
    DataFormat,
    DatasetSchema,
    DatasetSnapshotRef,
    SchemaField,
)
from capicove.contracts.graph_build import (
    GraphRetrievalBinding,
    GraphScope,
    GraphSnapshotManifest,
    GraphSnapshotRef,
    StageCoverage,
    StageResult,
    TableRef,
)
from capicove.contracts.retrieval import (
    DistanceMetric,
    EmbeddingSpaceRef,
    QueryModality,
    TargetType,
    VectorNormalization,
)
from capicove.graph_build.codec import graph_from_json, graph_to_json, manifest_bytes
from capicove.graph_build.table_registry import (
    missing_table_roles,
    required_table_roles,
    validate_table_roles,
)


def test_complete_graph_requires_reports_and_alignment() -> None:
    roles = required_table_roles()
    assert len(roles) == 17
    assert missing_table_roles(roles - {"community_reports"}) == {"community_reports"}
    assert {"entity_alignments", "diagnostic_findings"} <= roles
    assert missing_table_roles(roles) == frozenset()


def test_empty_snapshot_still_requires_every_table() -> None:
    assert missing_table_roles(frozenset()) == required_table_roles()


@pytest.mark.parametrize("roles", [("documents", "documents"), ("unknown",)])
def test_registry_rejects_duplicate_and_unknown_roles(roles: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        validate_table_roles(roles)


def test_missing_check_does_not_silently_ignore_unknown_roles() -> None:
    with pytest.raises(ValueError):
        missing_table_roles(frozenset({"unknown"}))


def test_role_order_does_not_change_normalized_registry() -> None:
    assert validate_table_roles(("entities", "documents")) == (
        "documents",
        "entities",
    )


@pytest.mark.parametrize("roles", ["documents", (None,), ([],)])
def test_registry_rejects_malformed_role_containers(roles: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        validate_table_roles(roles)  # type: ignore[arg-type]


def dataset(name: str = "input") -> DatasetSnapshotRef:
    return DatasetSnapshotRef(
        f"dataset:{name}",
        1,
        f"snapshot:{'a' * 64}",
        f"file:///tmp/graph/{name}",
        DataFormat.PARQUET,
        DatasetSchema((SchemaField("id", "string", False),)),
        "b" * 64,
        "c" * 64,
        (),
    )


def manifest(**changes: Any) -> GraphSnapshotManifest:
    stage_units = (
        ("representations", "documents"),
        ("text_units", "codepoints"),
        ("extraction", "text_units"),
        ("identity_embeddings", "descriptors"),
        ("resolution", "mentions"),
        ("alignment", "entities"),
        ("aggregation", "assertions"),
        ("diagnostics", "bundles"),
        ("communities", "entities"),
        ("reports", "reports"),
    )
    value = GraphSnapshotManifest(
        1,
        GraphScope("workspace", "scope"),
        dataset(),
        None,
        "d" * 64,
        tuple(TableRef(role, dataset(role)) for role in sorted(required_table_roles())),
        tuple(
            StageResult(
                name,
                "succeeded",
                (f"artifact:{name}",),
                ("no_baseline",) if name == "alignment" else (),
                StageCoverage(unit, 0, 0, 0, 0, 0, 0),
            )
            for name, unit in stage_units
        ),
        "complete",
    )
    return replace(value, **changes)


def embedding_space() -> EmbeddingSpaceRef:
    return EmbeddingSpaceRef(
        "space",
        "local",
        "revision",
        "preprocessing-v1",
        2,
        VectorNormalization.L2,
        DistanceMetric.COSINE,
        frozenset({QueryModality.TEXT}),
        frozenset({TargetType.TEXT_CHUNK}),
    )


def test_manifest_round_trip_and_shuffled_tables_have_same_identity() -> None:
    value = manifest()
    reordered = replace(
        value, tables=tuple(reversed(value.tables)), stages=tuple(reversed(value.stages))
    )
    assert manifest_bytes(reordered) == manifest_bytes(value)
    assert graph_from_json(graph_to_json(value)) == value
    decoded = json.loads(manifest_bytes(value))
    assert decoded["tables"][0]["role"] == "communities"
    assert decoded["schema_version"] == 1
    assert "generation_refs" not in decoded


def test_dataset_content_changes_manifest_identity() -> None:
    value = manifest()
    first, *rest = value.tables
    changed = replace(first, dataset_ref=replace(first.dataset_ref, manifest_sha256="e" * 64))
    assert manifest_bytes(replace(value, tables=(changed, *rest))) != manifest_bytes(value)


@pytest.mark.parametrize("completeness", ["complete", "partial"])
def test_missing_empty_table_is_invalid_even_in_partial_snapshot(completeness: str) -> None:
    value = manifest()
    with pytest.raises(ValueError):
        replace(
            value,
            tables=tuple(t for t in value.tables if t.role != "community_reports"),
            completeness=completeness,
        )


def test_duplicate_table_role_is_not_hidden_by_normalization() -> None:
    value = manifest()
    with pytest.raises(ValueError):
        replace(value, tables=(*value.tables, value.tables[0]))


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"profile_fingerprint": "invalid"},
        {"completeness": "succeeded"},
        {"input_snapshot_ref": "input"},
        {"scope": "scope"},
    ],
)
def test_manifest_rejects_invalid_version_and_reference_types(changes: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        manifest(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"declared": True},
        {"accepted": -1},
        {"processed": 2},
        {"declared": 2},
        {"failed": 1},
        {"denominator": ""},
    ],
)
def test_coverage_rejects_invalid_counts_and_denominators(changes: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(StageCoverage("text_units", 1, 1, 1, 0, 0, 0), **changes)


@pytest.mark.parametrize("counts", [(2, 2, 1, 1, 0, 0), (2, 1, 1, 0, 1, 0), (2, 1, 1, 0, 0, 1)])
def test_successful_run_cannot_hide_incomplete_coverage(counts: tuple[int, ...]) -> None:
    value = manifest()
    stage = next(s for s in value.stages if s.stage_id == "reports")
    changed = replace(stage, coverage=StageCoverage("reports", *counts))
    stages = tuple(changed if s.stage_id == "reports" else s for s in value.stages)
    with pytest.raises(ValueError):
        replace(value, stages=stages)
    assert replace(value, stages=stages, completeness="partial").completeness == "partial"


def test_missing_or_failed_stage_cannot_be_complete() -> None:
    value = manifest()
    for stages in (
        value.stages[:-1],
        (
            replace(value.stages[0], status="failed", issue_codes=("unavailable",)),
            *value.stages[1:],
        ),
    ):
        with pytest.raises(ValueError):
            replace(value, stages=stages)
        assert replace(value, stages=stages, completeness="partial").completeness == "partial"


def test_first_build_no_baseline_is_successful_alignment() -> None:
    value = manifest()
    assert value.completeness == "complete"
    with pytest.raises(ValueError):
        replace(value, baseline_snapshot_id="previous")


def test_stage_roles_and_duplicates_are_closed() -> None:
    value = manifest()
    with pytest.raises(ValueError):
        replace(value.stages[0], stage_id="retrieval")
    with pytest.raises(ValueError):
        replace(value, stages=(*value.stages, value.stages[0]))


def test_nested_manifest_collections_are_defensively_frozen() -> None:
    value = manifest()
    tables = list(value.tables)
    stages = list(value.stages)
    frozen = replace(value, tables=tables, stages=stages)  # type: ignore[arg-type]
    tables.clear()
    stages.clear()
    assert len(frozen.tables) == 17
    assert len(frozen.stages) == 10


def test_nested_dataset_schema_uses_public_recursive_codec() -> None:
    schema = DatasetSchema(
        (
            SchemaField(
                "ids",
                ComplexSchemaType(
                    ComplexTypeKind.ARRAY,
                    element_type="string",
                    contains_null=False,
                ),
                False,
            ),
        ),
        schema_version=2,
    )
    value = TableRef("entities", replace(dataset(), schema=schema))
    assert graph_from_json(graph_to_json(value)) == value
    payload = json.loads(graph_to_json(value))
    payload["value"]["dataset_ref"]["schema"]["fields"][0]["logical_type"]["unknown"] = True
    with pytest.raises(ValueError):
        graph_from_json(json.dumps(payload))


def test_retrieval_binding_preserves_full_embedding_space_outside_graph_hash() -> None:
    value = manifest()
    digest = hashlib.sha256(manifest_bytes(value)).hexdigest()
    ref = GraphSnapshotRef("graph", digest)
    binding = GraphRetrievalBinding(
        "binding", ref.snapshot_id, ("generation",), (embedding_space(),), "f" * 64
    )
    for item in (ref, binding, value.stages[0], value.stages[0].coverage):
        assert graph_from_json(graph_to_json(item)) == item
    assert hashlib.sha256(manifest_bytes(value)).hexdigest() == digest
    with pytest.raises(ValueError):
        replace(binding, embedding_spaces=(embedding_space(), embedding_space()))
    with pytest.raises(ValueError):
        replace(binding, generation_refs=())
    with pytest.raises(ValueError):
        replace(binding, embedding_spaces=(replace(embedding_space(), schema_version=2),))


@pytest.mark.parametrize("field", ["schema_version", "unknown"])
def test_manifest_decoder_rejects_incompatible_or_unknown_fields(field: str) -> None:
    payload = json.loads(graph_to_json(manifest()))
    payload["value"][field] = 2
    with pytest.raises(ValueError):
        graph_from_json(json.dumps(payload))
