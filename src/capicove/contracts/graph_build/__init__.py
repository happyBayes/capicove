"""Canonical graph evidence contracts, independent of optional runtimes."""

from capicove.contracts.graph_build.analysis import (
    ChildStatementRef,
    Community,
    CommunityMembership,
    CommunityReport,
    DiagnosticFinding,
    EvidenceBundle,
    ReportSections,
    ReportStatement,
)
from capicove.contracts.graph_build.evidence import GraphScope, InputSegment, SourceSpan, TextUnit
from capicove.contracts.graph_build.extraction import (
    EntityMention,
    RelationshipMention,
    RelativeSpan,
    ValidatedExtraction,
)
from capicove.contracts.graph_build.graph import (
    Entity,
    EntityAlignment,
    EntityMembership,
    IdentityKey,
    PairDecision,
    RelationshipAggregate,
    RelationshipAssertion,
    ResolutionMembership,
)
from capicove.contracts.graph_build.snapshots import (
    GraphRetrievalBinding,
    GraphSnapshotManifest,
    GraphSnapshotRef,
    StageCoverage,
    StageResult,
    TableRef,
)


__all__ = [
    "ChildStatementRef",
    "Community",
    "CommunityMembership",
    "CommunityReport",
    "DiagnosticFinding",
    "Entity",
    "EntityAlignment",
    "EntityMembership",
    "EntityMention",
    "EvidenceBundle",
    "GraphRetrievalBinding",
    "GraphScope",
    "GraphSnapshotManifest",
    "GraphSnapshotRef",
    "IdentityKey",
    "InputSegment",
    "PairDecision",
    "RelationshipAggregate",
    "RelationshipAssertion",
    "RelationshipMention",
    "RelativeSpan",
    "ReportSections",
    "ReportStatement",
    "ResolutionMembership",
    "SourceSpan",
    "StageCoverage",
    "StageResult",
    "TableRef",
    "TextUnit",
    "ValidatedExtraction",
]
