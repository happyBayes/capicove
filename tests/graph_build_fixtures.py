"""Authored graph rows and reusable snapshot construction for test layers."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from capicove.contracts.graph_build import (
    GraphScope,
    GraphSnapshotManifest,
    StageCoverage,
    StageResult,
    TableRef,
)
from capicove.contracts.graph_build.snapshots import REQUIRED_STAGES, TABLE_ROLES
from capicove.graph_build.datasets import publish_table
from capicove.processing import LocalProcessingWorkspace


SCOPE = GraphScope("workspace:test", "scope:test")


STAGE_ROLES = {
    "representations": ("documents",),
    "text_units": ("text_units",),
    "extraction": ("entity_mentions", "relationship_mentions"),
    "identity_embeddings": (),
    "resolution": ("entities", "entity_memberships", "resolution_memberships"),
    "alignment": ("entity_alignments", "resolved_entities"),
    "aggregation": ("relationship_assertions", "relationship_aggregates", "resolved_relationships"),
    "diagnostics": ("diagnostic_findings",),
    "communities": ("communities", "community_memberships"),
    "reports": ("community_reports",),
}


STAGE_TABLES: dict[str, tuple[TableRef, ...]] = {}


STAGE_INPUTS: dict[str, Any] = {}


DENOMINATORS = dict(
    zip(
        (
            "representations",
            "text_units",
            "extraction",
            "identity_embeddings",
            "resolution",
            "alignment",
            "aggregation",
            "diagnostics",
            "communities",
            "reports",
        ),
        (
            "documents",
            "codepoints",
            "text_units",
            "descriptors",
            "mentions",
            "entities",
            "assertions",
            "bundles",
            "entities",
            "reports",
        ),
        strict=True,
    )
)


def _module() -> Any:
    assert importlib.util.find_spec("capicove.graph_build.snapshots") is not None, (
        "Task 20 snapshot catalog is not implemented"
    )
    return importlib.import_module("capicove.graph_build.snapshots")


class Gate:
    """Test host with explicit current policy and guarded monotonic revision."""

    revision = 0

    def __init__(self) -> None:
        self.denied: set[str] = set()
        self.denied_scopes: set[GraphScope] = set()

    def require_scope(self, scope: GraphScope) -> None:
        if scope in self.denied_scopes:
            raise PermissionError("scope denied")

    def require_access(self, evidence_refs: tuple[str, ...], scope: GraphScope) -> None:
        self.require_scope(scope)
        if self.denied.intersection(evidence_refs):
            raise PermissionError("source denied")

    @contextmanager
    def guard(self, expected_revision: int) -> Iterator[None]:
        if self.revision != expected_revision:
            raise PermissionError("policy changed")
        yield
        if self.revision != expected_revision:
            raise PermissionError("policy changed")


class Inputs:
    def __init__(self, workspace: LocalProcessingWorkspace) -> None:
        self.workspace = workspace

    def validate(self, ref: Any, scope: GraphScope) -> Any:
        from capicove.graph_build._adapters.parquet import ParquetTableCodec
        from capicove.graph_build.codec import fingerprint
        from capicove.graph_build.datasets import codec_for_role

        rows: tuple[Any, ...] = ()

        def validate(path: Path) -> bool:
            nonlocal rows
            rows = ParquetTableCodec(publication_identity=ref.processing_fingerprint).read(
                path / "part-00000.parquet", "documents"
            )
            return True

        assert self.workspace.validate_snapshot(ref.snapshot_id, validator=validate) == ref
        return _module().InputSnapshotInventory(
            ref,
            scope,
            tuple(row.document_ref for row in rows),
            tuple(sorted({row.evidence_ref for row in rows})),
            fingerprint([codec_for_role("documents").encode(row) for row in rows]),
        )


class Stages:
    def validate(self, stage: StageResult, scope: GraphScope, profile_fingerprint: str) -> Any:
        return _module().ValidatedStageArtifacts(
            stage,
            scope,
            STAGE_INPUTS[stage.artifact_refs[0]],
            profile_fingerprint,
            (),
            "c" * 64,
            STAGE_TABLES[stage.artifact_refs[0]]
            if stage.status in {"succeeded", "partial"}
            else (),
        )


def _catalog(path: Path, workspace: LocalProcessingWorkspace, gate: Gate) -> Any:
    return _module().GraphSnapshotCatalog(path, workspace, gate, Inputs(workspace), Stages())


def _coverage(stage: str, rows: dict[str, tuple[Any, ...]]) -> StageCoverage:
    role = {
        "representations": "documents",
        "extraction": "text_units",
        "resolution": "entity_mentions",
        "aggregation": "relationship_assertions",
        "communities": "resolved_entities",
        "reports": "community_reports",
    }.get(stage, "")
    count = len(rows.get(role, ()))
    return StageCoverage(DENOMINATORS[stage], count, count, count, 0, 0, 0)


def _manifest(workspace: LocalProcessingWorkspace, **changes: Any) -> GraphSnapshotManifest:

    scope = changes.pop("scope", SCOPE)
    rows = changes.pop("rows", {})
    tables = tuple(
        TableRef(
            role,
            publish_table(
                workspace,
                scope=scope,
                role=role,
                rows=rows.get(role, ()),
                parent_refs=(),
                processing_fingerprint="a" * 64,
            ),
        )
        for role in sorted(TABLE_ROLES)
    )
    return _manifest_from_tables(tables, scope=scope, rows=rows, **changes)


def _manifest_from_tables(
    tables: tuple[TableRef, ...],
    *,
    scope: GraphScope = SCOPE,
    rows: dict[str, tuple[Any, ...]],
    **changes: Any,
) -> GraphSnapshotManifest:
    from capicove.graph_build.codec import fingerprint

    artifact_refs = {}
    input_snapshot_ref = next(table.dataset_ref for table in tables if table.role == "documents")
    for stage, roles in STAGE_ROLES.items():
        committed = tuple(table for table in tables if table.role in roles)
        ref = "artifact:" + fingerprint(
            {
                "stage": stage,
                # Identical content in two private workspaces must not overwrite
                # a shared fixture's attestation with a different location.
                "input_snapshot": (input_snapshot_ref.snapshot_id, input_snapshot_ref.uri),
                "tables": [
                    (table.dataset_ref.snapshot_id, table.dataset_ref.uri) for table in committed
                ],
            }
        )
        STAGE_TABLES[ref] = committed
        STAGE_INPUTS[ref] = input_snapshot_ref
        artifact_refs[stage] = ref
    return GraphSnapshotManifest(
        1,
        scope,
        input_snapshot_ref,
        None,
        "a" * 64,
        tables,
        tuple(
            StageResult(
                stage,
                "succeeded",
                (artifact_refs[stage],),
                ("no_baseline",) if stage == "alignment" else (),
                _coverage(stage, rows),
            )
            for stage in sorted(REQUIRED_STAGES)
        ),
        changes.pop("completeness", "complete"),
        **changes,
    )


def _rows(scope: GraphScope = SCOPE) -> dict[str, tuple[Any, ...]]:
    from capicove.contracts.graph_build import (
        Community,
        CommunityMembership,
        CommunityReport,
        Entity,
        EntityMembership,
        EntityMention,
        InputSegment,
        RelationshipMention,
        ReportSections,
        ReportStatement,
        ResolutionMembership,
        SourceSpan,
        TextUnit,
    )
    from capicove.contracts.retrieval import CharacterRange, TextChunkLocator, TextOrigin
    from capicove.graph_build.aggregation import aggregate_assertions, bind_assertions
    from capicove.graph_build.datasets import DocumentRow

    span = SourceSpan("evidence:1", "representation:1", 0, 3)
    unit = TextUnit(
        "unit:1",
        "document:1",
        scope,
        "d" * 64,
        "representation:1",
        TextChunkLocator(TextOrigin.DOCUMENT, CharacterRange(0, 3)),
        (InputSegment("segment:1", "core", "A B", (span,)),),
        (),
        0,
        2,
        "e" * 64,
    )
    mentions = tuple(
        EntityMention(
            "mention:" + key,
            unit.text_unit_id,
            key,
            "PERSON",
            "Source person",
            (span,),
            "core",
            1.0,
            (),
            None,
        )
        for key in ("A", "B")
    )
    entities = tuple(
        Entity("entity:" + key, scope, "PERSON", key, (), ("mention:" + key,), (), None, None)
        for key in ("A", "B")
    )
    memberships = tuple(
        EntityMembership(entity.entity_id, entity.mention_ids[0]) for entity in entities
    )
    relationship = RelationshipMention(
        "relation:1",
        unit.text_unit_id,
        "mention:A",
        "mention:B",
        "RELATED_TO",
        "A relates to B",
        (span,),
        "explicit",
        "positive",
        1.0,
        None,
        (),
    )
    assertions = bind_assertions((*mentions, relationship), memberships, {})
    return {
        "documents": (
            DocumentRow(
                "document:1",
                scope,
                "d" * 64,
                "evidence:1",
                "representation:1",
                "text/plain",
                "available",
                (span,),
                StageCoverage("codepoints", 3, 3, 3, 0, 0, 0),
            ),
        ),
        "text_units": (unit,),
        "entity_mentions": mentions,
        "relationship_mentions": (relationship,),
        "entities": entities,
        "entity_memberships": memberships,
        "relationship_assertions": assertions,
        "relationship_aggregates": aggregate_assertions(assertions),
        "resolved_entities": entities,
        "resolved_relationships": assertions,
        "resolution_memberships": tuple(
            ResolutionMembership(row.entity_id, row.entity_id) for row in entities
        ),
        "communities": (Community("community:1", None, 0, ("entity:A", "entity:B"), ()),),
        "community_memberships": tuple(
            CommunityMembership("community:1", row.entity_id) for row in entities
        ),
        "community_reports": (
            CommunityReport(
                "report:1",
                "community",
                "community:1",
                "Observed relation",
                ReportSections(
                    summary=(
                        ReportStatement(
                            "statement:1",
                            "A relates to B",
                            "fact",
                            (assertions[0].assertion_id,),
                            (),
                            (),
                        ),
                    )
                ),
                "bundle:1",
                0,
                "receipt:1",
            ),
        ),
    }
