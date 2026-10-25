"""Evidence-backed graph building without implicit provider initialization."""

from typing import TYPE_CHECKING, Any

from capicove.graph_build.codec import (
    canonical_bytes,
    evidence_from_json,
    evidence_to_json,
    fingerprint,
    graph_from_json,
    graph_to_json,
    manifest_bytes,
)


if TYPE_CHECKING:
    from capicove.graph_build.evaluation import (
        EvaluationProfile,
        QualificationReport,
        evaluate_graph_run,
    )
    from capicove.graph_build.service import (
        DocumentSourceProvenance,
        GraphBuildRequest,
        GraphBuildResult,
        GraphBuildService,
        IdentityEvidence,
        IdentityEvidenceResolver,
        SourceProvenanceResolver,
    )


def __getattr__(name: str) -> Any:
    if name in {"EvaluationProfile", "QualificationReport", "evaluate_graph_run"}:
        from capicove.graph_build import evaluation

        return getattr(evaluation, name)
    if name in {
        "GraphBuildRequest",
        "GraphBuildResult",
        "GraphBuildService",
        "DocumentSourceProvenance",
        "IdentityEvidence",
        "IdentityEvidenceResolver",
        "SourceProvenanceResolver",
    }:
        from capicove.graph_build import service

        return getattr(service, name)
    raise AttributeError(name)


__all__ = [
    "DocumentSourceProvenance",
    "EvaluationProfile",
    "GraphBuildRequest",
    "GraphBuildResult",
    "GraphBuildService",
    "IdentityEvidence",
    "IdentityEvidenceResolver",
    "QualificationReport",
    "SourceProvenanceResolver",
    "canonical_bytes",
    "evaluate_graph_run",
    "evidence_from_json",
    "evidence_to_json",
    "fingerprint",
    "graph_from_json",
    "graph_to_json",
    "manifest_bytes",
]
