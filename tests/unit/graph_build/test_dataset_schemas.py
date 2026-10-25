"""Closed homogeneous graph dataset schemas and logical codecs."""

import copy
import subprocess
import sys
from dataclasses import fields
from typing import Any

import pytest

from capicove.contracts.data import ComplexSchemaType, ComplexTypeKind
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
    GraphScope,
    IdentityKey,
    InputSegment,
    RelationshipAggregate,
    RelationshipAssertion,
    RelationshipMention,
    ReportSections,
    ReportStatement,
    ResolutionMembership,
    SourceSpan,
    StageCoverage,
    TextUnit,
)
from capicove.contracts.retrieval import CharacterRange, TextChunkLocator, TextOrigin
from capicove.graph_build.table_registry import required_table_roles


def representative_rows() -> dict[str, object]:
    """Project-authored rows with nested Unicode evidence and citations."""
    from capicove.graph_build.datasets import DocumentRow, OperationDiagnosticRow

    scope = GraphScope("workspace:私有", "scope:学习")
    span = SourceSpan("evidence:中", "representation:1", 0, 3, "block:1", 0)
    entity = Entity(
        "e:1",
        scope,
        "PERSON",
        "学习🙂",
        ("名字",),
        ("m:1",),
        (IdentityKey("source", "中", True),),
        None,
        "subject:1",
    )
    assertion = RelationshipAssertion(
        "a:1",
        "e:1",
        "RELATED_TO",
        "e:2",
        "explicit",
        "negative",
        "speaker:1",
        (("condition_text", "条件🙂"),),
        ("evidence:中",),
        "support:1",
        None,
        (),
    )
    sections = ReportSections(
        **{
            field.name: (
                ReportStatement(
                    f"s:{index}",
                    "观察🙂",
                    "fact",
                    ("a:1",),
                    (),
                    (ChildStatementRef("child:1", "statement:1"),),
                ),
            )
            for index, field in enumerate(fields(ReportSections))
        }
    )
    return {
        "documents": DocumentRow(
            "document:1",
            scope,
            "a" * 64,
            "evidence:中",
            "representation:1",
            "text/plain",
            "available",
            (span,),
            StageCoverage("documents", 1, 1, 1, 0, 0, 0),
        ),
        "text_units": TextUnit(
            "unit:1",
            "document:1",
            scope,
            "a" * 64,
            "representation:1",
            TextChunkLocator(TextOrigin.OCR, CharacterRange(0, 3), page_number=1),
            (InputSegment("segment:1", "core", "学习🙂", (span,)),),
            ("章节",),
            0,
            3,
            "b" * 64,
        ),
        "entity_mentions": EntityMention(
            "m:1",
            "unit:1",
            "学习🙂",
            "PERSON",
            "说明",
            (span,),
            "core",
            0.9,
            entity.identity_keys,
            None,
        ),
        "relationship_mentions": RelationshipMention(
            "m:r",
            "unit:1",
            "m:1",
            "m:2",
            "RELATED_TO",
            "关联",
            (span,),
            "hypothesized",
            "positive",
            0.5,
            None,
            (("time_text", "今天"),),
        ),
        "entities": entity,
        "entity_memberships": EntityMembership("e:1", "m:1"),
        "relationship_assertions": assertion,
        "relationship_aggregates": RelationshipAggregate(
            "aggregate:1", ("a:1",), ("support:1",), ("a:2",)
        ),
        "entity_alignments": EntityAlignment(
            "e:1", "e:old", "ambiguous", (("name_similarity", 0.5),), ("m:1",), "c" * 64
        ),
        "resolved_entities": entity,
        "resolved_relationships": assertion,
        "resolution_memberships": ResolutionMembership("e:1", "e:1"),
        "diagnostic_findings": DiagnosticFinding(
            "f:1",
            "DIFFICULTY",
            "subject:1",
            "e:2",
            "inferred",
            ("a:1",),
            ("a:2",),
            "解释",
            ("限制",),
            "验证",
            "receipt:1",
        ),
        "communities": Community("community:1", None, 0, ("e:1", "e:2"), ("limit",)),
        "community_memberships": CommunityMembership("community:1", "e:1"),
        "community_reports": CommunityReport(
            "report:1", "community", "community:1", "标题", sections, "bundle:1", 2, "receipt:1"
        ),
        "operation_diagnostics": OperationDiagnosticRow(
            "diagnostic:1", "extraction", None, "EXTRACTION_REJECTED", "warning"
        ),
    }


def test_graph_tables_have_distinct_declared_schemas() -> None:
    from capicove.graph_build.datasets import schema_for_role

    assert schema_for_role("entities").sha256 != schema_for_role("community_reports").sha256


@pytest.mark.parametrize("role", sorted(required_table_roles()))
def test_every_role_has_closed_round_trip_and_full_empty_schema(role: str) -> None:
    from capicove.graph_build.datasets import codec_for_role, schema_for_role

    row = representative_rows()[role]
    codec = codec_for_role(role)
    encoded = codec.encode(row)
    assert codec.decode(encoded) == row
    assert set(encoded) == {field.name for field in schema_for_role(role).fields}
    with pytest.raises((TypeError, ValueError)):
        codec.decode({**encoded, "unknown": "value"})
    with pytest.raises((TypeError, ValueError)):
        codec.decode({})
    with pytest.raises((TypeError, ValueError)):
        codec.encode({"row": row})
    with pytest.raises((TypeError, ValueError)):
        codec.encode(representative_rows()["communities" if role != "communities" else "entities"])
    complex_fields = [
        f for f in schema_for_role(role).fields if isinstance(f.logical_type, ComplexSchemaType)
    ]
    assert schema_for_role(role).schema_version == (2 if complex_fields else 1)


def test_arrays_structs_nullable_fields_and_pair_values_are_typed() -> None:
    from capicove.graph_build.datasets import schema_for_role

    schema = {f.name: f for f in schema_for_role("entities").fields}
    assert schema["subject_ref"].nullable
    assert schema["scope"].logical_type.kind is ComplexTypeKind.STRUCT  # type: ignore[union-attr]
    aliases = schema["aliases"]
    assert not aliases.nullable
    assert isinstance(aliases.logical_type, ComplexSchemaType)
    assert aliases.logical_type.contains_null is False
    assert aliases.logical_type.element_type == "string"
    with pytest.raises(ValueError):
        schema_for_role("unknown")


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), "0.5"])
def test_decoders_reject_invalid_numbers(bad: object) -> None:
    from capicove.graph_build.datasets import codec_for_role

    codec = codec_for_role("entity_mentions")
    encoded = codec.encode(representative_rows()["entity_mentions"])
    encoded["extraction_confidence"] = bad
    with pytest.raises((TypeError, ValueError)):
        codec.decode(encoded)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda d: d["core_locator"].update(origin="unknown"),
        lambda d: d["core_locator"]["character_range"].update(start_char=True),
        lambda d: d["segments"][0].update(extra=1),
        lambda d: d["segments"].append(d["segments"][0]),
        lambda d: d.update(section_path=[None]),
    ],
)
def test_nested_malformed_records_fail_closed(mutator: Any) -> None:
    from capicove.graph_build.datasets import codec_for_role

    codec = codec_for_role("text_units")
    encoded = copy.deepcopy(codec.encode(representative_rows()["text_units"]))
    mutator(encoded)
    with pytest.raises((TypeError, ValueError)):
        codec.decode(encoded)


def test_duplicate_qualifiers_and_report_statement_ids_are_rejected() -> None:
    from capicove.graph_build.datasets import codec_for_role

    for role, field in (
        ("relationship_assertions", "qualifiers"),
        ("entity_alignments", "feature_scores"),
    ):
        codec = codec_for_role(role)
        encoded: Any = codec.encode(representative_rows()[role])
        encoded[field].append(encoded[field][0])
        with pytest.raises(ValueError):
            codec.decode(encoded)
    codec = codec_for_role("community_reports")
    encoded = codec.encode(representative_rows()["community_reports"])
    encoded["sections"]["strengths"] = encoded["sections"]["summary"]
    with pytest.raises(ValueError):
        codec.decode(encoded)


def test_dataset_imports_do_not_load_optional_runtime() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "import capicove.graph_build.datasets; "
            "import capicove.graph_build._adapters.parquet; "
            "assert 'pyarrow' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_record_field_introspection_runs_once_per_class(monkeypatch: pytest.MonkeyPatch) -> None:
    """Typed row conversion must not re-resolve annotations for every record."""
    import typing

    from capicove.graph_build import datasets
    from capicove.graph_build.datasets import DocumentRow

    calls: list[type[Any]] = []

    def counting(cls: type[Any]) -> dict[str, Any]:
        calls.append(cls)
        return typing.get_type_hints(cls)

    monkeypatch.setattr(datasets, "get_type_hints", counting)
    monkeypatch.setattr(datasets, "_RECORD_FIELDS", {})
    first = datasets._record_fields(DocumentRow)
    assert datasets._record_fields(DocumentRow) == first
    assert datasets._record_fields(DocumentRow) == first
    assert calls == [DocumentRow]
