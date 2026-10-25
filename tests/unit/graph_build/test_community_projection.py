"""Source-only association projection, support accounting, and replay identity."""

import importlib
import math
from dataclasses import FrozenInstanceError, asdict, replace
from itertools import permutations
from typing import Any

import pytest

from capicove.contracts.graph_build import (
    DiagnosticFinding,
    Entity,
    GraphScope,
    RelationshipAssertion,
)
from capicove.graph_build.profiles import CommunityProfile


SCOPE = GraphScope("workspace", "scope")


def entity(ref: str, kind: str = "PERSON") -> Entity:
    return Entity(ref, SCOPE, kind, ref, (), ("mention:" + ref,), (), None, None)


ENTITIES = (entity("owner"), entity("other"), entity("事件", "WORK_EVENT"), entity("🧠", "SKILL"))


def assertion(ref: str = "a", group: str = "support:1", **changes: Any) -> RelationshipAssertion:
    return replace(
        RelationshipAssertion(
            ref,
            "owner",
            "HAS_SKILL",
            "🧠",
            "explicit",
            "positive",
            None,
            (),
            ("evidence:" + ref,),
            group,
            None,
            (),
        ),
        **changes,
    )


def project(rows: tuple[RelationshipAssertion, ...], **kwargs: Any) -> Any:
    module = importlib.import_module("capicove.graph_build.communities")
    return module.community_projection(
        rows,
        kwargs.pop("entities", ENTITIES),
        kwargs.pop("owner_entity_id", None),
        kwargs.pop("profile", CommunityProfile()),
        **kwargs,
    )


def test_support_weight_saturates_at_profile_limit() -> None:
    module = importlib.import_module("capicove.graph_build.communities")
    assert module.edge_weight(0) == 1.0
    assert module.edge_weight(2) == 1.0 + math.log1p(2)
    assert module.edge_weight(200) == module.edge_weight(20)


@pytest.mark.parametrize(
    "count,cap", [(True, 20), (-1, 20), (1.0, 20), (1, True), (1, 0), (1, -1), (1, 2.0)]
)
def test_weight_rejects_noninteger_or_out_of_range_counts(count: Any, cap: Any) -> None:
    module = importlib.import_module("capicove.graph_build.communities")
    with pytest.raises((TypeError, ValueError)):
        module.edge_weight(count, cap)


def test_support_deduplication_combines_directions_and_predicates_and_caps_weight() -> None:
    rows = (
        assertion("a"),
        assertion("b", source_entity_id="🧠", target_entity_id="owner"),
        assertion("c", "support:2", predicate="RELATED_TO"),
        assertion("d", "support:3"),
    )
    result = project(rows, profile=CommunityProfile(support_cap=2))
    assert result.node_ids == ("other", "owner", "事件", "🧠")
    assert len(result.edges) == 1
    edge = result.edges[0]
    assert (edge.source_id, edge.target_id) == ("owner", "🧠")
    assert edge.assertion_ids == ("a", "b", "c", "d")
    assert edge.support_group_ids == ("support:1", "support:2", "support:3")
    assert edge.weight == 1.0 + math.log1p(2)
    assert result.diagnostics.total_assertions == result.diagnostics.included_assertions == 4


@pytest.mark.parametrize(
    "groups,expected_count",
    [
        (("unknown-support:x",), 0),
        (("composite-support:x",), 0),
        (("unknown-support:x", "composite-support:y"), 0),
        (("unknown-support:x", "composite-support:y", "event:known"), 1),
    ],
)
def test_uncertain_support_never_contributes_a_base_weight(
    groups: tuple[str, ...], expected_count: int
) -> None:
    result = project(tuple(assertion(str(i), group) for i, group in enumerate(groups)))
    assert len(result.edges) == expected_count
    assert result.diagnostics.included_assertions == expected_count
    assert (
        dict(result.diagnostics.excluded_counts)["unknown_independent_support"]
        == len(groups) - expected_count
    )
    if expected_count:
        assert result.edges[0].weight == 1 + math.log1p(1)
        assert result.edges[0].assertion_ids == ("2",)


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"epistemic_status": "inferred"}, "not_explicit"),
        ({"epistemic_status": "hypothesized"}, "not_explicit"),
        ({"epistemic_status": "disputed", "counter_assertion_ids": ("counter",)}, "not_explicit"),
        ({"polarity": "negative"}, "not_positive"),
        ({"derivation_ref": "source:derived"}, "generated_source"),
        ({"target_entity_id": "owner"}, "self_loop"),
    ],
)
def test_projection_excludes_ineligible_assertions(changes: dict[str, Any], reason: str) -> None:
    result = project((assertion(**changes),))
    assert result.edges == ()
    assert result.diagnostics.total_assertions == 1
    assert result.diagnostics.included_assertions == 0
    assert result.diagnostics.excluded_counts == ((reason, 1),)


@pytest.mark.parametrize(
    "prefix",
    [
        "report:",
        "finding:",
        "diagnostic-finding:",
        "generated:",
        "generated_report:",
        "generated-report:",
        "community-report:",
        "hypothesis:",
    ],
)
@pytest.mark.parametrize("field", ["assertion_id", "evidence_refs"])
def test_known_generated_reference_prefixes_are_excluded(prefix: str, field: str) -> None:
    value: Any = prefix + "1"
    if field == "evidence_refs":
        value = ("source:1", value)
    result = project((assertion(**{field: value}),))
    assert result.edges == ()
    assert result.diagnostics.excluded_counts == (("generated_source", 1),)


def test_owner_exclusion_is_exact_and_preserves_other_owner_facts() -> None:
    rows = (
        assertion("owner-event", predicate="PARTICIPATED_IN", target_entity_id="事件"),
        assertion(
            "other-event",
            predicate="PARTICIPATED_IN",
            source_entity_id="other",
            target_entity_id="事件",
        ),
        assertion("skill"),
    )
    assert len(project(rows).edges) == 3
    result = project(rows, owner_entity_id="owner")
    assert {ref for edge in result.edges for ref in edge.assertion_ids} == {"other-event", "skill"}
    assert result.diagnostics.excluded_counts == (("owner_participation", 1),)


def test_projection_is_immutable_and_order_independent_and_retains_isolates() -> None:
    rows = (
        assertion("b"),
        assertion("a", "support:2"),
        assertion("loop", target_entity_id="owner"),
    )
    before = tuple(asdict(row) for row in rows)
    expected = project(rows)
    for row_order in permutations(rows):
        for entity_order in permutations(ENTITIES):
            assert project(row_order, entities=entity_order) == expected
    assert tuple(asdict(row) for row in rows) == before
    assert expected.node_ids == ("other", "owner", "事件", "🧠")
    for obj, field in (
        (expected, "edges"),
        (expected.edges[0], "weight"),
        (expected.diagnostics, "excluded_counts"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(obj, field, None)
    empty = project(())
    assert empty.node_ids == expected.node_ids
    assert empty.edges == ()
    assert project((), entities=()).node_ids == ()


def test_fingerprint_binds_evidence_profile_and_exclusion_details() -> None:
    original = project((assertion(),))
    variants = (
        project((assertion(evidence_refs=("source:new",)),)),
        project((assertion(predicate="RELATED_TO"),)),
        project((assertion(),), profile=CommunityProfile(support_cap=19)),
        project((assertion(), assertion("excluded", polarity="negative"))),
        project((assertion(),), entities=(*ENTITIES, entity("new"))),
    )
    assert all(item.projection_fingerprint != original.projection_fingerprint for item in variants)
    left = project((assertion(evidence_refs=("source:a", "source:b")),))
    right = project((assertion(evidence_refs=("source:b", "source:a")),))
    assert left.projection_fingerprint == right.projection_fingerprint


def test_exclusion_accounting_assigns_one_stable_reason_per_assertion() -> None:
    rows = (
        assertion(),
        assertion("negative", polarity="negative"),
        assertion("multi", polarity="negative", epistemic_status="inferred"),
        assertion("unknown", "unknown-support:1"),
    )
    result = project(rows)
    assert result.diagnostics.total_assertions == 4
    assert result.diagnostics.included_assertions == 1
    assert result.diagnostics.excluded_counts == (
        ("not_explicit", 1),
        ("not_positive", 1),
        ("unknown_independent_support", 1),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"entities": (*ENTITIES, ENTITIES[0])},
        {"entities": (ENTITIES[0], replace(ENTITIES[-1], scope=GraphScope("other", "scope")))},
        {"entities": ENTITIES[:-1]},
        {"owner_entity_id": "missing"},
        {"owner_entity_id": "🧠"},
        {"owner_entity_id": True},
        {"profile": {}},
        {"entities": ("text",)},
    ],
)
def test_invalid_inventory_scope_endpoint_owner_and_profile_fail_closed(
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        project((assertion(),), **kwargs)


def test_duplicate_and_nonassertion_inputs_fail_closed_before_filtering() -> None:
    with pytest.raises(ValueError):
        project((assertion(), assertion(polarity="negative")))
    with pytest.raises(TypeError):
        project(("generated diagnosis",))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        project((assertion(target_entity_id="missing", polarity="negative"),))


def test_diagnostic_findings_are_not_assertions() -> None:
    finding = DiagnosticFinding(
        "finding:1",
        "ERROR_PATTERN",
        "owner",
        "🧠",
        "inferred",
        ("a",),
        (),
        "A generated claim",
        (),
        None,
        "derivation:1",
    )
    with pytest.raises(TypeError):
        project((finding,))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "changes",
    [
        {"source_id": " "},
        {"source_id": "🧠", "target_id": "owner"},
        {"target_id": "owner"},
        {"weight": True},
        {"weight": float("inf")},
        {"weight": float("nan")},
        {"weight": 0},
        {"weight": -1},
        {"assertion_ids": ["a"]},
        {"assertion_ids": ()},
        {"assertion_ids": ("a", "a")},
        {"assertion_ids": ("b", "a")},
        {"assertion_ids": ("",)},
        {"support_group_ids": ["support:1"]},
        {"support_group_ids": ()},
        {"support_group_ids": ("support:2", "support:1")},
        {"support_group_ids": ("unknown-support:x",)},
    ],
)
def test_weighted_edge_constructor_rejects_mutable_or_noncanonical_values(
    changes: dict[str, Any],
) -> None:
    edge = project((assertion(),)).edges[0]
    with pytest.raises((TypeError, ValueError)):
        replace(edge, **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"total_assertions": True},
        {"included_assertions": -1},
        {"included_assertions": 2},
        {"excluded_counts": []},
        {"excluded_counts": (("not_explicit", True),)},
        {"excluded_counts": (("not_explicit", -1),)},
        {"excluded_counts": (("not_explicit", 0),)},
        {"excluded_counts": (("not_explicit", 1), ("not_explicit", 1)), "total_assertions": 3},
        {"excluded_counts": (("not_positive", 1), ("not_explicit", 1)), "total_assertions": 3},
        {"excluded_counts": (("", 1),), "total_assertions": 2},
        {"excluded_counts": (["not_explicit", 1],), "total_assertions": 2},
    ],
)
def test_diagnostic_constructor_rejects_invalid_or_inconsistent_counts(
    changes: dict[str, Any],
) -> None:
    diagnostics = project((assertion(),)).diagnostics
    with pytest.raises((TypeError, ValueError)):
        replace(diagnostics, **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"node_ids": ["owner", "🧠"]},
        {"node_ids": ("owner", "owner", "🧠")},
        {"node_ids": ("🧠", "owner")},
        {"node_ids": ("owner",)},
        {"node_ids": ("", "owner", "🧠")},
        {"edges": []},
        {"edges": ("edge",)},
        {"diagnostics": {}},
        {"projection_fingerprint": "invalid"},
        {"projection_fingerprint": "A" * 64},
        {"projection_fingerprint": None},
    ],
)
def test_graph_constructor_rejects_mutable_or_noncanonical_values(changes: dict[str, Any]) -> None:
    graph = project((assertion(),))
    with pytest.raises((TypeError, ValueError)):
        replace(graph, **changes)


def test_graph_constructor_rejects_duplicate_or_unsorted_edges() -> None:
    graph = project((assertion(), assertion("b", source_entity_id="other")))
    with pytest.raises(ValueError):
        replace(graph, edges=(graph.edges[0], graph.edges[0]))
    with pytest.raises(ValueError):
        replace(graph, edges=tuple(reversed(graph.edges)))


def test_graph_constructor_rejects_assertion_reuse_and_false_included_counts() -> None:
    graph = project((assertion(), assertion("b", source_entity_id="other")))
    reused = replace(graph.edges[1], assertion_ids=graph.edges[0].assertion_ids)
    with pytest.raises(ValueError):
        replace(graph, edges=(graph.edges[0], reused))
    with pytest.raises(ValueError):
        replace(
            graph, diagnostics=replace(graph.diagnostics, total_assertions=0, included_assertions=0)
        )


@pytest.mark.parametrize("excluded_by", ["negative", "generated"])
@pytest.mark.parametrize(
    "changes",
    [
        {"assertion_id": "bad:\ud800"},
        {"source_entity_id": "bad:\ud800"},
        {"target_entity_id": "bad:\ud800"},
        {"attribution_ref": "bad:\ud800"},
        {"qualifiers": (("time_text", "bad:\ud800"),)},
        {"evidence_refs": ("bad:\ud800",)},
        {"support_group_id": "bad:\ud800"},
        {"derivation_ref": "bad:\ud800"},
        {"counter_assertion_ids": ("bad:\ud800",)},
    ],
)
def test_excluded_assertion_text_is_validated_before_filtering(
    excluded_by: str,
    changes: dict[str, Any],
) -> None:
    row = (
        assertion(polarity="negative")
        if excluded_by == "negative"
        else assertion(derivation_ref="generated:1")
    )
    row = replace(row, **changes)
    original = asdict(row)
    with pytest.raises(ValueError, match="valid UTF-8"):
        project((row,))
    assert asdict(row) == original


def test_edge_cannot_claim_more_support_groups_than_source_assertions() -> None:
    edge = project((assertion(),)).edges[0]
    with pytest.raises(ValueError):
        replace(edge, support_group_ids=("support:1", "support:2"))
