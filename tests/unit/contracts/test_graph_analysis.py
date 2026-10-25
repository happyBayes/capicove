"""Analysis contracts preserve provenance and separate facts from synthesis."""

import json
from dataclasses import replace
from typing import Any

import pytest

from capicove.contracts.graph_build import (
    ChildStatementRef,
    Community,
    CommunityMembership,
    CommunityReport,
    DiagnosticFinding,
    EntityMembership,
    EvidenceBundle,
    ReportSections,
    ReportStatement,
    ResolutionMembership,
)
from capicove.graph_build.codec import graph_from_json, graph_to_json


def finding(**changes: Any) -> DiagnosticFinding:
    value = DiagnosticFinding(
        "finding",
        "CAUSE_HYPOTHESIS",
        "person",
        "skill",
        "hypothesized",
        ("assertion",),
        ("counter",),
        "A possible explanation.",
        ("Only two observations.",),
        "Check an independent example.",
        "receipt",
    )
    return replace(value, **changes)


def report(**changes: Any) -> CommunityReport:
    value = CommunityReport(
        "report",
        "community",
        "community",
        "Learning review",
        ReportSections(
            summary=(ReportStatement("s", "Observed practice.", "fact", ("a",), (), ()),)
        ),
        "bundle",
        0,
        "receipt",
    )
    return replace(value, **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"epistemic_status": "explicit"},
        {"epistemic_status": "disputed"},
        {"support_refs": ()},
        {"derivation_ref": None},
        {"diagnostic_type": "PERSON"},
        {"support_refs": ("a", "a")},
        {"limitations": (" ",)},
    ],
)
def test_generated_finding_rejects_fact_promotion_or_missing_provenance(
    changes: dict[str, Any],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        finding(**changes)


def test_finding_round_trip_retains_counters_and_hypothesis_label() -> None:
    value = finding()
    restored = graph_from_json(graph_to_json(value))
    assert restored == value
    assert isinstance(restored, DiagnosticFinding)
    assert restored.epistemic_status == "hypothesized"
    assert restored.counter_refs == ("counter",)


@pytest.mark.parametrize("kind", ["fact", "hypothesis"])
def test_evidence_bearing_statement_cannot_be_uncited(kind: str) -> None:
    with pytest.raises(ValueError):
        ReportStatement("s", "A claim.", kind, (), (), ())


def test_finding_alone_cannot_be_cited_as_an_explicit_fact() -> None:
    with pytest.raises(ValueError):
        ReportStatement("s", "A claim.", "fact", (), ("finding",), ())


def test_parent_report_preserves_namespaced_child_citations() -> None:
    statement = ReportStatement(
        "local", "Possible cause.", "hypothesis", (), (), (ChildStatementRef("child", "local"),)
    )
    value = report(sections=ReportSections(possible_causes=(statement,)))
    assert graph_from_json(graph_to_json(value)) == value
    with pytest.raises(ValueError):
        replace(value, report_id="child")


@pytest.mark.parametrize(
    "changes",
    [
        {"community_id": None},
        {"report_kind": "other"},
        {"report_kind": "overview", "community_id": "community"},
        {"receipt_ref": ""},
        {"omitted_count": -1},
        {"omitted_count": True},
    ],
)
def test_report_rejects_invalid_scope_or_coverage(changes: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        report(**changes)


def test_overview_has_no_fabricated_community() -> None:
    value = report(report_kind="overview", community_id=None)
    assert graph_from_json(graph_to_json(value)) == value


def test_sections_freeze_input_and_report_rejects_duplicate_statement_ids() -> None:
    statements = [ReportStatement("s", "Insufficient evidence.", "limitation", (), (), ())]
    sections = ReportSections(limitations=statements)  # type: ignore[arg-type]
    statements.clear()
    assert len(sections.limitations) == 1
    with pytest.raises(ValueError):
        report(sections=replace(sections, summary=sections.limitations))


def test_report_codec_requires_all_ten_sections_and_rejects_unknown_keys() -> None:
    payload = json.loads(graph_to_json(report()))
    del payload["value"]["sections"]["counter_evidence"]
    with pytest.raises(ValueError):
        graph_from_json(json.dumps(payload))
    payload = json.loads(graph_to_json(report()))
    payload["value"]["sections"]["provider_metadata"] = {}
    with pytest.raises(ValueError):
        graph_from_json(json.dumps(payload))


@pytest.mark.parametrize(
    "changes",
    [
        {"depth": True},
        {"depth": -1},
        {"parent_id": "c"},
        {"depth": 1},
        {"parent_id": "p"},
        {"member_ids": ()},
        {"member_ids": ("e", "e")},
    ],
)
def test_community_rejects_invalid_hierarchy(changes: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(Community("c", None, 0, ("e",), ()), **changes)


def test_analysis_and_membership_values_have_closed_round_trips() -> None:
    values = (
        EvidenceBundle("bundle", "person", "skill", ("a",), ("b",), ("event",), 2),
        Community("child", "parent", 1, ("e",), ("oversized_leaf",)),
        EntityMembership("e", "m"),
        ResolutionMembership("resolved", "current"),
        CommunityMembership("c", "e"),
        ChildStatementRef("child", "s"),
        ReportSections(),
        ReportStatement("s", "Try another example.", "recommendation", (), ("finding",), ()),
    )
    for value in values:
        assert graph_from_json(graph_to_json(value)) == value


def test_bundle_does_not_count_repeated_event_ids_as_independent() -> None:
    with pytest.raises(ValueError):
        EvidenceBundle("b", "person", "skill", ("a",), (), ("event", "event"), 0)


@pytest.mark.parametrize(
    "cls", [EntityMembership, ResolutionMembership, CommunityMembership, ChildStatementRef]
)
def test_membership_and_child_refs_require_nonblank_endpoints(cls: Any) -> None:
    with pytest.raises(ValueError):
        cls("valid", "")
