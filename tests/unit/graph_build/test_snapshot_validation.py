"""Cross-table validation matrices over immutable authored rows, without storage."""

from dataclasses import replace
from typing import Any

import pytest

from capicove.contracts.graph_build import GraphSnapshotManifest, StageCoverage, TableRef
from capicove.contracts.graph_build.snapshots import TABLE_ROLES
from capicove.graph_build.snapshots import _validate_tables
from tests.graph_build_fixtures import _manifest_from_tables, _module, _rows
from tests.unit.contracts.test_graph_snapshots import dataset


def _manifest(*, rows: dict[str, tuple[Any, ...]], **changes: Any) -> GraphSnapshotManifest:
    return _manifest_from_tables(
        tuple(TableRef(role, dataset(role)) for role in sorted(TABLE_ROLES)),
        rows=rows,
        **changes,
    )


def validate(manifest: GraphSnapshotManifest, rows: dict[str, tuple[Any, ...]]) -> None:
    _validate_tables(manifest, {role: rows.get(role, ()) for role in TABLE_ROLES}, {})


def test_authored_rows_pass_cross_table_validation() -> None:
    rows = _rows()
    validate(_manifest(rows=rows), rows)


@pytest.mark.parametrize("partial", [False, True])
def test_review_successful_aggregation_covers_retained_relationship_mentions(partial: bool) -> None:
    module = _module()
    rows = _rows()
    for role in (
        "relationship_assertions",
        "relationship_aggregates",
        "resolved_relationships",
        "community_reports",
    ):
        rows[role] = ()
    manifest = _manifest(rows=rows, completeness="partial" if partial else "complete")
    if partial:
        manifest = replace(
            manifest,
            stages=tuple(
                replace(stage, status="partial", issue_codes=("incomplete_aggregation",))
                if stage.stage_id == "aggregation"
                else stage
                for stage in manifest.stages
            ),
        )
    if partial:
        validate(manifest, rows)
    else:
        with pytest.raises(module.GraphValidationError):
            validate(manifest, rows)


def test_review_relationship_mention_has_one_assertion_owner() -> None:
    from capicove.graph_build.aggregation import aggregate_assertions

    module = _module()
    rows = _rows()
    assertions = (
        *rows["relationship_assertions"],
        replace(rows["relationship_assertions"][0], assertion_id="assertion:duplicate-owner"),
    )
    rows["relationship_assertions"] = rows["resolved_relationships"] = assertions
    rows["relationship_aggregates"] = aggregate_assertions(assertions)
    with pytest.raises(module.GraphValidationError):
        validate(_manifest(rows=rows), rows)


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_review_successful_communities_cover_every_resolved_entity(
    empty: bool, partial: bool
) -> None:
    module = _module()
    rows = _rows()
    # Both retained entities are isolated; they must still belong to root communities.
    for role in (
        "relationship_mentions",
        "relationship_assertions",
        "relationship_aggregates",
        "resolved_relationships",
        "community_reports",
    ):
        rows[role] = ()
    rows["communities"] = (
        () if empty else (replace(rows["communities"][0], member_ids=("entity:A",)),)
    )
    rows["community_memberships"] = () if empty else (rows["community_memberships"][0],)
    manifest = _manifest(rows=rows, completeness="partial" if partial else "complete")
    if partial:
        manifest = replace(
            manifest,
            stages=tuple(
                replace(stage, status="partial", issue_codes=("incomplete_communities",))
                if stage.stage_id == "communities"
                else stage
                for stage in manifest.stages
            ),
        )
    if partial:
        validate(manifest, rows)
    else:
        with pytest.raises(module.GraphValidationError):
            validate(manifest, rows)


@pytest.mark.parametrize("missing", ["source", "target", "attribution"])
def test_review_partial_resolved_relationship_requires_its_own_view_endpoints(missing: str) -> None:
    from capicove.contracts.graph_build import EntityMembership
    from capicove.graph_build.aggregation import aggregate_assertions, bind_assertions

    module = _module()
    rows = _rows()
    missing_id = "entity:A" if missing == "source" else "entity:B"
    if missing == "attribution":
        speaker = replace(rows["entity_mentions"][0], mention_id="mention:C", name="C")
        rows["entity_mentions"] += (speaker,)
        rows["entities"] += (
            replace(
                rows["entities"][0],
                entity_id="entity:C",
                display_name="C",
                mention_ids=(speaker.mention_id,),
            ),
        )
        rows["entity_memberships"] += (EntityMembership("entity:C", speaker.mention_id),)
        rows["relationship_mentions"] = (
            replace(rows["relationship_mentions"][0], attribution_ref=speaker.mention_id),
        )
        assertions = bind_assertions(
            (*rows["entity_mentions"], *rows["relationship_mentions"]),
            rows["entity_memberships"],
            {},
        )
        rows["relationship_assertions"] = rows["resolved_relationships"] = assertions
        rows["relationship_aggregates"] = aggregate_assertions(assertions)
        missing_id = "entity:C"
    rows["resolved_entities"] = tuple(
        row for row in rows["entities"] if row.entity_id != missing_id
    )
    rows["resolution_memberships"] = tuple(
        row for row in rows["resolution_memberships"] if row.current_entity_id != missing_id
    )
    for role in ("communities", "community_memberships", "community_reports"):
        rows[role] = ()
    manifest = _manifest(rows=rows, completeness="partial")
    manifest = replace(
        manifest,
        stages=tuple(
            replace(stage, status="partial", issue_codes=("incomplete_alignment",))
            if stage.stage_id == "alignment"
            else stage
            for stage in manifest.stages
            if stage.stage_id not in {"communities", "reports"}
        ),
    )
    with pytest.raises(module.GraphValidationError):
        validate(manifest, rows)


@pytest.mark.parametrize("context_label", [False, True])
def test_review_entity_anchor_role_matches_actual_core_ownership(context_label: bool) -> None:
    from capicove.contracts.graph_build import InputSegment
    from capicove.contracts.retrieval import CharacterRange

    module = _module()
    rows = _rows()
    mention = rows["entity_mentions"][0]
    if context_label:
        rows["entity_mentions"] = (
            replace(mention, anchor_role="context_anchor"),
            *rows["entity_mentions"][1:],
        )
    else:
        span = mention.source_spans[0]
        context = replace(span, end_char=1)
        core = replace(span, start_char=1)
        unit = rows["text_units"][0]
        rows["text_units"] = (
            replace(
                unit,
                core_locator=replace(unit.core_locator, character_range=CharacterRange(1, 3)),
                segments=(
                    InputSegment("segment:context", "context", "A", (context,)),
                    InputSegment("segment:core", "core", " B", (core,)),
                ),
            ),
        )
        rows["entity_mentions"] = (
            replace(mention, source_spans=(context,)),
            replace(rows["entity_mentions"][1], source_spans=(core,)),
        )
        rows["relationship_mentions"] = (
            replace(rows["relationship_mentions"][0], source_spans=(core,)),
        )
    with pytest.raises(module.GraphValidationError):
        validate(_manifest(rows=rows), rows)


def test_relationship_predicate_requires_compatible_endpoint_types() -> None:
    from capicove.graph_build.aggregation import aggregate_assertions

    module = _module()
    rows = _rows()
    rows["relationship_mentions"] = (
        replace(rows["relationship_mentions"][0], predicate="HAS_SKILL"),
    )
    assertion = replace(rows["relationship_assertions"][0], predicate="HAS_SKILL")
    rows["relationship_assertions"] = rows["resolved_relationships"] = (assertion,)
    rows["relationship_aggregates"] = aggregate_assertions((assertion,))
    manifest = _manifest(rows=rows)
    with pytest.raises(module.GraphValidationError):
        validate(manifest, rows)


@pytest.mark.parametrize(
    "damage",
    [
        "unit_document",
        "unit_hash",
        "unit_evidence",
        "mention_unit",
        "mention_span",
        "relationship_endpoint",
        "membership",
        "entity_mentions",
        "assertion_endpoint",
        "assertion_evidence",
        "aggregate",
        "resolved",
        "resolution_membership",
        "community_member",
        "community_parent",
        "community_membership",
        "report_assertion",
        "report_community",
        "operation_unit",
        "finding_support",
        "coverage",
    ],
)
def test_cross_table_corruption_is_rejected(damage: str) -> None:
    from capicove.contracts.graph_build import DiagnosticFinding
    from capicove.graph_build.datasets import OperationDiagnosticRow

    module = _module()
    rows = _rows()
    role = ""
    change: dict[str, Any] = {}
    mapping: dict[str, tuple[str, dict[str, Any]]] = {
        "unit_document": ("text_units", {"document_ref": "missing"}),
        "unit_hash": ("text_units", {"source_sha256": "f" * 64}),
        "unit_evidence": ("text_units", {"representation_ref": "representation:foreign"}),
        "mention_unit": ("entity_mentions", {"text_unit_id": "missing"}),
        "mention_span": (
            "entity_mentions",
            {"source_spans": (replace(rows["entity_mentions"][0].source_spans[0], end_char=4),)},
        ),
        "relationship_endpoint": ("relationship_mentions", {"source_mention_id": "missing"}),
        "membership": ("entity_memberships", {"entity_id": "missing"}),
        "entity_mentions": ("entities", {"mention_ids": ("missing",)}),
        "assertion_endpoint": ("relationship_assertions", {"target_entity_id": "missing"}),
        "assertion_evidence": ("relationship_assertions", {"evidence_refs": ("foreign",)}),
        "aggregate": ("relationship_aggregates", {"assertion_ids": ("missing",)}),
        "resolved": ("resolved_entities", {"entity_id": "history:entity"}),
        "resolution_membership": ("resolution_memberships", {"current_entity_id": "missing"}),
        "community_member": ("communities", {"member_ids": ("missing",)}),
        "community_parent": ("communities", {"parent_id": "missing", "depth": 1}),
        "community_membership": ("community_memberships", {"entity_id": "missing"}),
        "report_community": ("community_reports", {"community_id": "missing"}),
    }
    if damage == "unit_evidence":
        unit = rows["text_units"][0]
        changed_span = replace(unit.segments[0].source_spans[0], evidence_ref="foreign")
        rows["text_units"] = (
            replace(unit, segments=(replace(unit.segments[0], source_spans=(changed_span,)),)),
        )
    elif damage in mapping:
        role, change = mapping[damage]
        rows[role] = (replace(rows[role][0], **change), *rows[role][1:])
    elif damage == "report_assertion":
        report = rows["community_reports"][0]
        rows["community_reports"] = (
            replace(
                report,
                sections=replace(
                    report.sections,
                    summary=(replace(report.sections.summary[0], assertion_refs=("missing",)),),
                ),
            ),
        )
    elif damage == "finding_support":
        rows["diagnostic_findings"] = (
            DiagnosticFinding(
                "finding:1",
                "ERROR_PATTERN",
                "entity:A",
                "entity:B",
                "inferred",
                ("missing",),
                (),
                "Pattern",
                (),
                None,
                "receipt:diagnostics",
            ),
        )
    elif damage == "operation_unit":
        rows["operation_diagnostics"] = (
            OperationDiagnosticRow("diag:1", "extraction", "missing", "bad_unit", "error"),
        )
    manifest = _manifest(rows=rows)
    if damage == "coverage":
        manifest = replace(
            manifest,
            stages=tuple(
                replace(stage, coverage=StageCoverage("documents", 0, 0, 0, 0, 0, 0))
                if stage.stage_id == "representations"
                else stage
                for stage in manifest.stages
            ),
        )
    with pytest.raises(module.GraphValidationError):
        validate(manifest, rows)


def test_assertion_cannot_rebind_existing_relation_to_different_endpoint() -> None:
    from capicove.graph_build.aggregation import aggregate_assertions

    module = _module()
    rows = _rows()
    changed = replace(rows["relationship_assertions"][0], target_entity_id="entity:A")
    rows["relationship_assertions"] = rows["resolved_relationships"] = (changed,)
    rows["relationship_aggregates"] = aggregate_assertions((changed,))
    manifest = _manifest(rows=rows)
    with pytest.raises(module.GraphValidationError):
        validate(manifest, rows)


@pytest.mark.parametrize(
    "damage", ["child_missing", "child_foreign", "child_fact", "fact_inference", "section_kind"]
)
def test_report_child_citations_and_epistemic_boundaries(damage: str) -> None:
    from capicove.contracts.graph_build import ChildStatementRef, ReportSections, ReportStatement
    from capicove.graph_build.aggregation import aggregate_assertions

    module = _module()
    rows = _rows()
    child = rows["community_reports"][0]
    root = replace(
        child,
        report_id="report:overview",
        report_kind="overview",
        community_id=None,
        sections=ReportSections(
            summary=(
                ReportStatement(
                    "root:1",
                    "Observed",
                    "fact",
                    (),
                    (),
                    (ChildStatementRef(child.report_id, "statement:1"),),
                ),
            )
        ),
    )
    if damage == "child_missing":
        root = replace(
            root,
            sections=ReportSections(
                summary=(
                    replace(
                        root.sections.summary[0],
                        child_statement_refs=(ChildStatementRef(child.report_id, "missing"),),
                    ),
                )
            ),
        )
    elif damage == "child_foreign":
        root = replace(root, report_kind="community", community_id=child.community_id)
    elif damage == "child_fact":
        child = replace(
            child,
            sections=ReportSections(
                summary=(replace(child.sections.summary[0], kind="hypothesis"),)
            ),
        )
    elif damage == "fact_inference":
        relation = replace(rows["relationship_mentions"][0], epistemic_status="hypothesized")
        assertion = replace(rows["relationship_assertions"][0], epistemic_status="hypothesized")
        rows["relationship_mentions"] = (relation,)
        rows["relationship_assertions"] = rows["resolved_relationships"] = (assertion,)
        rows["relationship_aggregates"] = aggregate_assertions((assertion,))
    else:
        child = replace(child, sections=ReportSections(limitations=child.sections.summary))
    rows["community_reports"] = (child, root)
    manifest = _manifest(rows=rows)
    with pytest.raises(module.GraphValidationError):
        validate(manifest, rows)
