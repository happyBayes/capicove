"""Keep graph implementation instructions aligned with accepted Task 0R change control."""

from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[3]
_DOCUMENTS = (
    "docs/rfcs/document-graph-pipeline-contracts.md",
    "docs/adr/document-graph-snapshot-boundary.md",
    "docs/architecture/document-graph-contract-ledger.md",
)


@pytest.mark.parametrize("document", _DOCUMENTS)
def test_community_decision_has_no_obsolete_mandatory_leiden(document: str) -> None:
    text = " ".join((_ROOT / document).read_text(encoding="utf-8").split())
    obsolete = (
        "Hierarchical Leiden is bounded",
        "Apply bounded hierarchical Leiden",
        "optional Leiden runtime remains subject",
        "Real Bounded Hierarchical Leiden Adapter",
        "igraph/leidenalg",
        "leidenalg.find_partition",
        "Real Parquet/Leiden integration",
        "real Parquet and Leiden integration",
        "leidenalg.readthedocs.io",
    )
    contradictions = [phrase for phrase in obsolete if phrase in text]
    assert not contradictions, f"{document}: obsolete mandatory Leiden decision: {contradictions}"

    required = (
        "CommunityPartitioner",
        "CommunityPartitionerRegistry",
        "PartitionProfile",
        "LouvainPartitionProfile(resolution, threshold, max_level, seed, version)",
        "networkx>=3.6,<4",
        "BSD-3-Clause",
        "CommunityProfile",
    )
    missing = [term for term in required if term not in text]
    assert not missing, f"{document}: missing accepted community contract: {missing}"


@pytest.mark.parametrize("document", _DOCUMENTS[:2])
def test_community_revision_preserves_accepted_unnumbered_decisions(document: str) -> None:
    text = (_ROOT / document).read_text(encoding="utf-8")
    assert "- Status: Accepted\n" in text
    assert "- Number: Awaiting maintainer assignment\n" in text
