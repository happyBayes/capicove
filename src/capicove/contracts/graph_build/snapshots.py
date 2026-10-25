"""Canonical graph snapshots, independent of catalog and execution backends."""

from dataclasses import dataclass

from capicove.contracts.data import DatasetSnapshotRef
from capicove.contracts.graph_build._checks import choice, digest, refs, text, values
from capicove.contracts.graph_build.evidence import GraphScope, _integer
from capicove.contracts.retrieval import EmbeddingSpaceRef


TABLE_ROLES = frozenset(
    {
        "documents",
        "text_units",
        "entity_mentions",
        "relationship_mentions",
        "entities",
        "entity_memberships",
        "relationship_assertions",
        "relationship_aggregates",
        "entity_alignments",
        "resolved_entities",
        "resolved_relationships",
        "resolution_memberships",
        "diagnostic_findings",
        "communities",
        "community_memberships",
        "community_reports",
        "operation_diagnostics",
    }
)

# Publication and retrieval follow manifest construction and cannot enter its DAG.
REQUIRED_STAGES = frozenset(
    {
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
    }
)


@dataclass(frozen=True, slots=True)
class TableRef:
    """Bind one closed role to a homogeneous public dataset snapshot."""

    role: str
    dataset_ref: DatasetSnapshotRef

    def __post_init__(self) -> None:
        choice(self.role, TABLE_ROLES)
        if not isinstance(self.dataset_ref, DatasetSnapshotRef):
            raise TypeError("dataset_ref must be DatasetSnapshotRef")


@dataclass(frozen=True, slots=True)
class StageCoverage:
    """Partition one named population, never mix counts with different units.

    Processed items are accepted or rejected. Failed items did not reach a
    processing result; omitted items were not attempted. Hence declared equals
    processed + failed + omitted. Stage policies define the denominator name;
    ordinary identity abstentions count as accepted processing outcomes.
    """

    denominator: str
    declared: int
    processed: int
    accepted: int
    rejected: int
    failed: int
    omitted: int

    def __post_init__(self) -> None:
        text(self.denominator)
        for name in ("declared", "processed", "accepted", "rejected", "failed", "omitted"):
            _integer(name, getattr(self, name))
        if self.processed != self.accepted + self.rejected:
            raise ValueError("Processed coverage must equal accepted plus rejected")
        if self.declared != self.processed + self.failed + self.omitted:
            raise ValueError("Declared coverage must equal processed plus failed plus omitted")


@dataclass(frozen=True, slots=True)
class StageResult:
    """Immutable stage outcome; orchestration success alone does not imply coverage."""

    stage_id: str
    status: str
    artifact_refs: tuple[str, ...]
    issue_codes: tuple[str, ...]
    coverage: StageCoverage

    def __post_init__(self) -> None:
        choice(self.stage_id, REQUIRED_STAGES)
        choice(self.status, frozenset({"succeeded", "partial", "failed", "skipped", "cancelled"}))
        object.__setattr__(
            self, "artifact_refs", refs(self.artifact_refs, required=self.status == "succeeded")
        )
        object.__setattr__(
            self, "issue_codes", refs(self.issue_codes, required=self.status != "succeeded")
        )
        if not isinstance(self.coverage, StageCoverage):
            raise TypeError("coverage must be StageCoverage")
        if "no_baseline" in self.issue_codes and (
            self.stage_id != "alignment" or self.status != "succeeded"
        ):
            raise ValueError("no_baseline is a successful alignment outcome only")


@dataclass(frozen=True, slots=True)
class GraphSnapshotManifest:
    """Semantic identity only; storage integrity and authorization need readers.

    Even partial snapshots require all table roles, using valid empty datasets
    where necessary. A complete manifest also requires every preprocessing stage
    to succeed with complete coverage. Operational retries and timestamps, and
    post-publication retrieval bindings, are intentionally absent.
    """

    schema_version: int
    scope: GraphScope
    input_snapshot_ref: DatasetSnapshotRef
    baseline_snapshot_id: str | None
    profile_fingerprint: str
    tables: tuple[TableRef, ...]
    stages: tuple[StageResult, ...]
    completeness: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported graph manifest schema version")
        if not isinstance(self.scope, GraphScope):
            raise TypeError("scope must be GraphScope")
        if not isinstance(self.input_snapshot_ref, DatasetSnapshotRef):
            raise TypeError("input_snapshot_ref must be DatasetSnapshotRef")
        text(self.baseline_snapshot_id, optional=True)
        digest(self.profile_fingerprint)
        choice(self.completeness, frozenset({"complete", "partial"}))
        tables = values(self.tables, TableRef)
        roles = refs([table.role for table in tables])
        if frozenset(roles) != TABLE_ROLES:
            raise ValueError("Every graph table role is required, including empty tables")
        stages = values(self.stages, StageResult)
        stage_ids = refs([stage.stage_id for stage in stages])
        alignment = next((stage for stage in stages if stage.stage_id == "alignment"), None)
        if alignment is not None and "no_baseline" in alignment.issue_codes:
            if self.baseline_snapshot_id is not None or alignment.coverage.declared != 0:
                raise ValueError("no_baseline requires no pinned baseline and zero alignment pairs")
        if self.completeness == "complete":
            if frozenset(stage_ids) != REQUIRED_STAGES:
                raise ValueError("Complete graphs require every preprocessing stage")
            if any(
                stage.status != "succeeded" or stage.coverage.accepted != stage.coverage.declared
                for stage in stages
            ):
                raise ValueError("Complete graphs require successful stages with full coverage")
            if self.baseline_snapshot_id is None and (
                alignment is None or "no_baseline" not in alignment.issue_codes
            ):
                raise ValueError("First-build alignment must explicitly record no_baseline")
        object.__setattr__(self, "tables", tuple(sorted(tables, key=lambda item: item.role)))
        object.__setattr__(self, "stages", tuple(sorted(stages, key=lambda item: item.stage_id)))


@dataclass(frozen=True, slots=True)
class GraphSnapshotRef:
    """Opaque catalog identity plus the digest to verify on authorized reads."""

    snapshot_id: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        text(self.snapshot_id)
        digest(self.manifest_sha256)


@dataclass(frozen=True, slots=True)
class GraphRetrievalBinding:
    """Bind an already published graph to retrieval without changing its identity."""

    binding_id: str
    graph_snapshot_id: str
    generation_refs: tuple[str, ...]
    embedding_spaces: tuple[EmbeddingSpaceRef, ...]
    unit_manifest_sha256: str

    def __post_init__(self) -> None:
        text(self.binding_id)
        text(self.graph_snapshot_id)
        object.__setattr__(
            self, "generation_refs", tuple(sorted(refs(self.generation_refs, required=True)))
        )
        spaces = values(self.embedding_spaces, EmbeddingSpaceRef, required=True)
        refs([space.space_id for space in spaces])
        if any(
            type(space.schema_version) is not int or space.schema_version != 1 for space in spaces
        ):
            raise ValueError("Unsupported embedding space schema version")
        object.__setattr__(
            self, "embedding_spaces", tuple(sorted(spaces, key=lambda item: item.space_id))
        )
        digest(self.unit_manifest_sha256)
