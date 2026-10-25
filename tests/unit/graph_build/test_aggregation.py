"""Authored fixtures protect claim semantics, occurrence identity, and evidence."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from itertools import permutations
from typing import Any

import pytest

from capicove.contracts.graph_build import (
    EntityMembership,
    EntityMention,
    GraphScope,
    RelationshipAssertion,
    RelationshipMention,
    RelativeSpan,
    SourceSpan,
)
from capicove.graph_build.aggregation import (
    aggregate_assertions,
    assertion_group_key,
    bind_assertions,
    source_occurrence_ref,
)
from capicove.graph_build.extraction import bind_segment_span
from capicove.graph_build.profiles import ChunkingProfile
from capicove.graph_build.representations import text_representation
from capicove.graph_build.text_units import compose_text_units


MEMBERSHIPS = (EntityMembership("person", "p"), EntityMembership("skill", "s"))


def mention(ref: str = "r", **changes: object) -> RelationshipMention:
    value = RelationshipMention(
        ref,
        "unit",
        "p",
        "s",
        "HAS_SKILL",
        "能力 中文 🧠",
        (SourceSpan("source:1", "representation:1", 2, 8),),
        "explicit",
        "positive",
        0.9,
        None,
        (),
    )
    return replace(value, **changes)  # type: ignore[arg-type]


def assertion(ref: str = "a", **changes: object) -> RelationshipAssertion:
    value = RelationshipAssertion(
        ref,
        "person",
        "HAS_SKILL",
        "skill",
        "explicit",
        "positive",
        None,
        (),
        ("evidence:" + ref,),
        "event:1",
        None,
        (),
    )
    return replace(value, **changes)  # type: ignore[arg-type]


def person(**changes: object) -> EntityMention:
    value = EntityMention(
        "p",
        "unit",
        "小明 🧠",
        "PERSON",
        "Source speaker",
        (SourceSpan("speaker", "representation:1", 0, 2),),
        "core",
        1.0,
        (),
        None,
    )
    return replace(value, **changes)  # type: ignore[arg-type]


def test_polarity_and_hypothesis_do_not_disappear_in_aggregation() -> None:
    base = ("a", "HAS_SKILL", "b")
    keys = {
        assertion_group_key(*base, "explicit", "positive", None, ()),
        assertion_group_key(*base, "explicit", "negative", None, ()),
        assertion_group_key(*base, "hypothesized", "positive", None, ()),
    }
    assert len(keys) == 3


@pytest.mark.parametrize(
    "index,value",
    [
        (0, "other"),
        (1, "RELATED_TO"),
        (2, "other"),
        (3, "inferred"),
        (4, "negative"),
        (5, "speaker"),
        (6, (("time_text", "yesterday"),)),
    ],
)
def test_every_semantic_field_changes_group_key(index: int, value: object) -> None:
    args: list[object] = ["a", "HAS_SKILL", "b", "explicit", "positive", None, ()]
    original = assertion_group_key(*args)  # type: ignore[arg-type]
    args[index] = value
    assert assertion_group_key(*args) != original  # type: ignore[arg-type]


def test_direction_and_exact_unicode_qualifiers_are_preserved() -> None:
    args = ("explicit", "positive", None, (("time_text", "🧠"), ("condition_text", "中文")))
    key = assertion_group_key("p", "HAS_SKILL", "s", *args)
    assert key != assertion_group_key("s", "HAS_SKILL", "p", *args)
    assert key == assertion_group_key(
        "p",
        "HAS_SKILL",
        "s",
        "explicit",
        "positive",
        None,
        (("condition_text", "中文"), ("time_text", "🧠")),
    )
    assert assertion_group_key(
        "p", "HAS_SKILL", "s", "explicit", "positive", None, (("condition_text", "é"),)
    ) != assertion_group_key(
        "p", "HAS_SKILL", "s", "explicit", "positive", None, (("condition_text", "e\u0301"),)
    )


def test_binding_preserves_semantics_exact_evidence_and_lineage() -> None:
    relation = mention(
        attribution_ref="p",
        polarity="negative",
        qualifiers=(("time_text", "中文"), ("condition_text", "🧠")),
    )
    (result,) = bind_assertions(
        (relation, person()),
        MEMBERSHIPS,
        {source_occurrence_ref(relation.source_spans[0]): "event:1"},
    )
    assert (result.source_entity_id, result.predicate, result.target_entity_id) == (
        "person",
        "HAS_SKILL",
        "skill",
    )
    assert (result.epistemic_status, result.polarity, result.attribution_ref) == (
        "explicit",
        "negative",
        "person",
    )
    assert result.qualifiers == (("condition_text", "🧠"), ("time_text", "中文"))
    assert set(result.evidence_refs) == {"source:1", "r"}
    assert result.support_group_id == "event:1"
    assert result.derivation_ref is None
    assert relation.attribution_ref == "p"


def test_unattributed_claim_never_defaults_to_owner() -> None:
    (result,) = bind_assertions((mention(), person()), MEMBERSHIPS, {})
    assert result.attribution_ref is None


def test_regenerated_mentions_chunks_retries_deduplicate_without_losing_evidence() -> None:
    first = mention()
    copy = mention(
        "retry",
        text_unit_id="new-chunk",
        extraction_confidence=0.3,
        source_spans=(SourceSpan("regenerated-evidence", "representation:1", 2, 8),),
    )
    rows = bind_assertions((first, copy), MEMBERSHIPS, {})
    assert len(rows) == 1
    assert set(rows[0].evidence_refs) == {"source:1", "regenerated-evidence", "r", "retry"}
    assert bind_assertions((copy,), MEMBERSHIPS, {})[0].assertion_id == rows[0].assertion_id
    assert bind_assertions((first,), MEMBERSHIPS, {})[0].assertion_id == rows[0].assertion_id


def test_same_source_across_snapshots_keeps_id_and_unknown_support() -> None:
    (first,) = bind_assertions((mention(),), MEMBERSHIPS, {})
    (later,) = bind_assertions(
        (mention("new-snapshot", text_unit_id="snapshot:2"),), MEMBERSHIPS, {}
    )
    assert first.assertion_id == later.assertion_id
    assert first.support_group_id == later.support_group_id


def test_distinct_source_ranges_retain_assertions_but_not_invented_independence() -> None:
    rows = bind_assertions(
        (
            mention(),
            mention("other", source_spans=(SourceSpan("source:1", "representation:1", 20, 26),)),
        ),
        MEMBERSHIPS,
        {},
    )
    assert len(rows) == 2
    (aggregate,) = aggregate_assertions(rows)
    assert len(aggregate.support_group_ids) == 1
    assert aggregate.support_group_ids[0].startswith("unknown-support:")


def test_different_source_hashes_refs_and_representations_do_not_prove_independence() -> None:
    rows = bind_assertions(
        (
            mention(),
            mention(
                "copy",
                source_spans=(SourceSpan("sha256:" + "b" * 64, "representation:copy", 2, 8),),
            ),
        ),
        MEMBERSHIPS,
        {},
    )
    assert len(rows) == 2
    (aggregate,) = aggregate_assertions(rows)
    assert len(aggregate.support_group_ids) == 1


def test_explicit_independent_events_are_counted_and_copies_share_support() -> None:
    items = (
        mention(),
        mention("copy", source_spans=(SourceSpan("copy-source", "copy-representation", 2, 8),)),
        mention("second", source_spans=(SourceSpan("second-source", "second-rep", 2, 8),)),
    )
    rows = bind_assertions(
        items,
        MEMBERSHIPS,
        {
            source_occurrence_ref(items[0].source_spans[0]): "event:1",
            source_occurrence_ref(items[1].source_spans[0]): "event:1",
            source_occurrence_ref(items[2].source_spans[0]): "event:2",
        },
    )
    (aggregate,) = aggregate_assertions(rows)
    assert len(aggregate.assertion_ids) == 3
    assert aggregate.support_group_ids == ("event:1", "event:2")


def test_multispan_unknown_is_conservative_and_all_evidence_is_retained() -> None:
    spans = (
        SourceSpan("e1", "rep", 0, 3, "block:1", 0),
        SourceSpan("e2", "rep", 6, 10, "block:2", 1),
    )
    (result,) = bind_assertions(
        (mention(source_spans=spans),), MEMBERSHIPS, {source_occurrence_ref(spans[0]): "event"}
    )
    assert result.support_group_id.startswith("unknown-support:")
    assert set(result.evidence_refs) == {"e1", "e2", "r"}
    (composite,) = bind_assertions(
        (mention(source_spans=spans),),
        MEMBERSHIPS,
        {source_occurrence_ref(spans[0]): "event:1", source_occurrence_ref(spans[1]): "event:2"},
    )
    assert composite.support_group_id.startswith("composite-support:")
    assert len(aggregate_assertions((composite,))[0].support_group_ids) == 1


def test_legacy_evidence_alias_assignments_cannot_promote_duplicate_occurrence_support() -> None:
    copy = mention("copy", source_spans=(SourceSpan("e2", "representation:1", 2, 8),))
    (row,) = bind_assertions(
        (mention(), copy), MEMBERSHIPS, {"source:1": "event:1", "e2": "event:2"}
    )
    assert row.support_group_id.startswith("unknown-support:")


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_mention_id", "missing"),
        ("target_mention_id", "missing"),
        ("attribution_ref", "missing"),
        ("attribution_ref", "s"),
    ],
)
def test_missing_endpoints_and_unverified_attribution_fail(field: str, value: str) -> None:
    with pytest.raises(ValueError):
        bind_assertions((mention(**{field: value}), person()), MEMBERSHIPS, {})


def test_attribution_requires_person_mention_and_its_membership() -> None:
    for items, members in (
        ((mention(attribution_ref="p"),), MEMBERSHIPS),
        ((mention(attribution_ref="p"), person(entity_type="PROJECT")), MEMBERSHIPS),
        ((mention(attribution_ref="speaker"), person(mention_id="speaker")), MEMBERSHIPS),
    ):
        with pytest.raises(ValueError, match="attribution"):
            bind_assertions(items, members, {})


@pytest.mark.parametrize(
    "members",
    [
        (*MEMBERSHIPS, MEMBERSHIPS[0]),
        (*MEMBERSHIPS, EntityMembership("other", "p")),
    ],
)
def test_duplicate_membership_mentions_are_rejected(members: tuple[EntityMembership, ...]) -> None:
    with pytest.raises(ValueError, match="membership"):
        bind_assertions((mention(),), members, {})


def test_duplicate_mention_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        bind_assertions((mention(), mention()), MEMBERSHIPS, {})


def test_disputed_binding_translates_explicit_counters_after_ids_are_computed() -> None:
    items = (mention(), mention("dispute", epistemic_status="disputed", polarity="negative"))
    rows = bind_assertions(items, MEMBERSHIPS, {}, counter_mentions={"dispute": ("r",)})
    positive = next(r for r in rows if r.epistemic_status == "explicit")
    disputed = next(r for r in rows if r.epistemic_status == "disputed")
    assert disputed.counter_assertion_ids == (positive.assertion_id,)
    assert positive.counter_assertion_ids == ()
    assert len(aggregate_assertions(rows)) == 2


def test_explicit_non_disputed_counter_evidence_is_allowed_but_never_inferred() -> None:
    items = (mention(), mention("negative", polarity="negative"))
    plain = bind_assertions(items, MEMBERSHIPS, {})
    assert all(not row.counter_assertion_ids for row in plain)
    linked = bind_assertions(items, MEMBERSHIPS, {}, counter_mentions={"negative": ("r",)})
    assert sum(bool(row.counter_assertion_ids) for row in linked) == 1


@pytest.mark.parametrize(
    "mapping",
    [{}, {"dispute": ()}, {"alien": ("r",)}, {"dispute": ("alien",)}, {"dispute": ("dispute",)}],
)
def test_disputes_fail_closed_on_missing_alien_or_self_counter(
    mapping: dict[str, tuple[str, ...]],
) -> None:
    with pytest.raises(ValueError):
        bind_assertions(
            (mention(), mention("dispute", epistemic_status="disputed")),
            MEMBERSHIPS,
            {},
            counter_mentions=mapping,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"source_mention_id": "s", "target_mention_id": "p"},
        {"predicate": "RELATED_TO"},
        {"attribution_ref": "p"},
        {"qualifiers": (("time_text", "later"),)},
        {"qualifiers": (("occurrence_ref", "other"),)},
    ],
)
def test_counter_relevance_requires_identical_directed_claim_and_qualifiers(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="counter"):
        bind_assertions(
            (mention(), mention("dispute", epistemic_status="disputed", **changes), person()),
            MEMBERSHIPS,
            {},
            counter_mentions={"dispute": ("r",)},
        )


def test_duplicate_mentions_cannot_turn_into_self_counters_after_binding() -> None:
    with pytest.raises(ValueError, match="counter"):
        bind_assertions(
            (mention(), mention("retry")), MEMBERSHIPS, {}, counter_mentions={"retry": ("r",)}
        )


def test_aggregation_preserves_all_rows_support_and_cross_group_counters_unchanged() -> None:
    rows = (
        assertion("z"),
        assertion("b", support_group_id="event:2"),
        assertion("n", polarity="negative", counter_assertion_ids=("z",)),
        assertion("d", epistemic_status="disputed", counter_assertion_ids=("n",)),
    )
    before = tuple(rows)
    result = aggregate_assertions(rows)
    assert len(result) == 3
    positive = next(group for group in result if "z" in group.assertion_ids)
    assert positive.assertion_ids == ("b", "z")
    assert positive.support_group_ids == ("event:1", "event:2")
    assert {ref for group in result for ref in group.counter_assertion_ids} == {"n", "z"}
    assert rows == before
    assert all(row is original for row, original in zip(rows, before, strict=True))


@pytest.mark.parametrize(
    "changes",
    [
        {"source_entity_id": "alien"},
        {"target_entity_id": "alien"},
        {"predicate": "RELATED_TO"},
        {"attribution_ref": "speaker"},
        {"qualifiers": (("time_text", "later"),)},
    ],
)
def test_aggregation_rejects_irrelevant_counter_targets(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="counter"):
        aggregate_assertions(
            (assertion("target", **changes), assertion("d", counter_assertion_ids=("target",)))
        )


def test_aggregation_rejects_missing_counters_and_conflicting_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="counter"):
        aggregate_assertions((assertion(counter_assertion_ids=("alien",)),))
    with pytest.raises(ValueError, match="assertion"):
        aggregate_assertions((assertion(), assertion(polarity="negative")))
    assert aggregate_assertions((assertion(), assertion())) == aggregate_assertions((assertion(),))


def test_input_permutations_preserve_binding_and_aggregation() -> None:
    items = (
        mention(),
        mention("n", polarity="negative"),
        mention("d", epistemic_status="disputed"),
    )
    expected = bind_assertions(items, MEMBERSHIPS, {}, counter_mentions={"d": ("n", "r")})
    for ordered in permutations(items):
        actual = bind_assertions(
            ordered, tuple(reversed(MEMBERSHIPS)), {}, counter_mentions={"d": ("r", "n")}
        )
        assert actual == expected
        assert aggregate_assertions(actual) == aggregate_assertions(tuple(reversed(expected)))


@pytest.mark.parametrize(
    "args",
    [
        ([], MEMBERSHIPS, {}),
        ((object(),), MEMBERSHIPS, {}),
        ((mention(),), list(MEMBERSHIPS), {}),
        ((mention(),), (object(),), {}),
        ((mention(),), MEMBERSHIPS, []),
        ((mention(),), MEMBERSHIPS, {1: "event"}),
        ((mention(),), MEMBERSHIPS, {"source:1": 2}),
    ],
)
def test_binding_validates_input_shapes(args: tuple[object, ...]) -> None:
    with pytest.raises(TypeError):
        bind_assertions(*args)  # type: ignore[arg-type]


@pytest.mark.parametrize("mapping", [[], {"r": ["x"]}, {1: ("r",)}, {"r": (1,)}])
def test_counter_mapping_has_strict_shape(mapping: object) -> None:
    with pytest.raises(TypeError):
        bind_assertions((mention(),), MEMBERSHIPS, {}, counter_mentions=mapping)  # type: ignore[arg-type]


def test_empty_binding_and_aggregation_are_valid() -> None:
    assert bind_assertions((), (), {}) == ()
    assert aggregate_assertions(()) == ()
    with pytest.raises(TypeError):
        aggregate_assertions([])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        aggregate_assertions((object(),))  # type: ignore[arg-type]


def test_cross_kind_duplicate_mention_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        bind_assertions((mention("p"), person()), MEMBERSHIPS, {})


def test_multi_span_order_and_duplicate_evidence_do_not_change_identity() -> None:
    first = SourceSpan("e1", "rep", 0, 3, "block:1", 0)
    second = SourceSpan("e2", "rep", 6, 10, "block:2", 1)
    (original,) = bind_assertions((mention(source_spans=(first, second)),), MEMBERSHIPS, {})
    (reordered,) = bind_assertions((mention(source_spans=(second, first, first)),), MEMBERSHIPS, {})
    assert reordered == original
    (changed,) = bind_assertions(
        (mention(source_spans=(first, replace(second, page_index=2))),), MEMBERSHIPS, {}
    )
    assert changed.assertion_id != original.assertion_id
    assert changed.support_group_id == original.support_group_id


def test_generated_and_build_ids_are_not_used_as_support_mapping_keys() -> None:
    (result,) = bind_assertions(
        (mention(),),
        MEMBERSHIPS,
        {
            "r": "event:1",
            "unit": "event:2",
            "representation:1": "event:3",
            "build:1": "event:4",
        },
    )
    assert result.support_group_id.startswith("unknown-support:")


def test_generated_derivation_is_preserved_and_changes_assertion_identity() -> None:
    (source,) = bind_assertions((mention(),), MEMBERSHIPS, {})
    (generated,) = bind_assertions(
        (mention(),),
        MEMBERSHIPS,
        {},
        derivation_refs={"r": "community-report:derived"},
    )
    assert generated.derivation_ref == "community-report:derived"
    assert generated.assertion_id != source.assertion_id


def test_duplicate_occurrence_keeps_source_and_generated_lineage_separate() -> None:
    rows = bind_assertions(
        (mention(), mention("generated-copy")),
        MEMBERSHIPS,
        {},
        derivation_refs={"generated-copy": "community-report:derived"},
    )
    assert len(rows) == 2
    assert {row.derivation_ref for row in rows} == {None, "community-report:derived"}


@pytest.mark.parametrize(
    "mapping",
    [[], {"alien": "report:1"}, {"r": ""}, {"r": 1}],
)
def test_derivation_mapping_is_closed_and_typed(mapping: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        bind_assertions(
            (mention(),),
            MEMBERSHIPS,
            {},
            derivation_refs=mapping,  # type: ignore[arg-type]
        )


def test_negative_feedback_preserves_its_predicate_and_attributed_source() -> None:
    feedback = mention(predicate="EVALUATES", polarity="negative", attribution_ref="p")
    (bound,) = bind_assertions((feedback, person()), MEMBERSHIPS, {})
    assert bound.predicate == "EVALUATES"
    assert bound.polarity == "negative"
    assert bound.attribution_ref == "person"


def test_distinct_times_neither_infer_nor_allow_an_unqualified_counter_link() -> None:
    earlier = mention(qualifiers=(("time_text", "Monday"),))
    later = mention("later", polarity="negative", qualifiers=(("time_text", "Tuesday"),))
    unlinked = bind_assertions((earlier, later), MEMBERSHIPS, {})
    assert len(aggregate_assertions(unlinked)) == 2
    assert all(not row.counter_assertion_ids for row in unlinked)
    with pytest.raises(ValueError, match="counter"):
        bind_assertions((earlier, later), MEMBERSHIPS, {}, counter_mentions={"later": ("r",)})


def test_duplicate_bound_claims_union_counter_evidence_without_mutating_mentions() -> None:
    first = mention("d1", epistemic_status="disputed")
    second = replace(first, mention_id="d2", text_unit_id="copy")
    targets = (mention("p"), mention("n", polarity="negative"))
    rows = bind_assertions(
        (first, second, *targets), MEMBERSHIPS, {}, counter_mentions={"d1": ("p",), "d2": ("n",)}
    )
    disputed = next(row for row in rows if row.epistemic_status == "disputed")
    assert set(disputed.counter_assertion_ids) == {
        row.assertion_id for row in rows if row.epistemic_status == "explicit"
    }
    assert set(disputed.evidence_refs) == {"d1", "d2", "source:1"}
    assert first.mention_id == "d1" and second.mention_id == "d2"


@pytest.mark.parametrize(
    "changes",
    [
        {"source_entity_id": "other"},
        {"target_entity_id": "other"},
        {"predicate": "RELATED_TO"},
        {"epistemic_status": "hypothesized"},
        {"polarity": "negative"},
        {"attribution_ref": "speaker"},
        {"qualifiers": (("raw_predicate", "来源关系"),)},
    ],
)
def test_aggregation_never_combines_different_semantic_fields(changes: dict[str, object]) -> None:
    groups = aggregate_assertions((assertion("base"), assertion("variant", **changes)))
    assert len(groups) == 2
    assert {group.assertion_ids for group in groups} == {("base",), ("variant",)}


@pytest.mark.parametrize(
    "mapping",
    [
        {"source:1": ""},
        {"": "event"},
        {"source:1": "unknown-support:pretend"},
        {"source:1": "composite-support:pretend"},
    ],
)
def test_invalid_or_reserved_support_ids_cannot_impersonate_independence(
    mapping: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        bind_assertions((mention(),), MEMBERSHIPS, mapping)


def test_duplicate_counter_refs_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate counter"):
        bind_assertions(
            (mention(), mention("other", polarity="negative")),
            MEMBERSHIPS,
            {},
            counter_mentions={"other": ("r", "r")},
        )


@pytest.mark.parametrize(
    "index,value",
    [
        (0, ""),
        (1, "OPEN_PREDICATE"),
        (3, "proven"),
        (4, "neutral"),
        (5, ""),
        (6, (("time_text", "a"), ("time_text", "b"))),
    ],
)
def test_group_key_rejects_invalid_semantics(index: int, value: object) -> None:
    args: list[object] = ["a", "HAS_SKILL", "b", "explicit", "positive", None, ()]
    args[index] = value
    with pytest.raises(ValueError):
        assertion_group_key(*args)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [[], ("time_text",), (("time_text",),), ((1, "a"),)])
def test_group_key_validates_qualifier_shape(value: object) -> None:
    with pytest.raises(TypeError):
        assertion_group_key("a", "HAS_SKILL", "b", "explicit", "positive", None, value)  # type: ignore[arg-type]


class SingleReadMapping(Mapping[str, Any]):
    """Caller mapping that exposes one snapshot and rejects all later reads."""

    def __init__(self, entries: tuple[tuple[str, Any], ...]) -> None:
        self.entries = entries
        self.read = False

    def items(self) -> tuple[tuple[str, Any], ...]:  # type: ignore[override]
        if self.read:
            raise AssertionError("caller mapping was reread")
        self.read = True
        return self.entries

    def __getitem__(self, key: str) -> Any:
        raise AssertionError("caller mapping lookup bypassed its validated snapshot")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("caller mapping was iterated after its snapshot")

    def __len__(self) -> int:
        return len(self.entries)


@dataclass(frozen=True)
class CharacterTokenizer:
    fingerprint: str = "a" * 64

    def count(self, text: str) -> int:
        return len(text)


def test_extraction_ranges_with_shared_evidence_ref_accept_distinct_explicit_support() -> None:
    rep = text_representation("doc", "Alice PythonAlice Python", GraphScope("w", "s"))
    units = compose_text_units(
        rep,
        CharacterTokenizer(),
        ChunkingProfile(target_tokens=12, max_tokens=12, context_tokens=0),
    )
    assert len(units) == 2
    spans = tuple(
        bind_segment_span(segment, RelativeSpan(segment.segment_id, 0, 12, "Alice Python"))[0]
        for unit in units
        for segment in unit.segments
        if segment.role == "core"
    )
    assert spans[0].evidence_ref == spans[1].evidence_ref
    rows = bind_assertions(
        tuple(mention(f"r:{index}", source_spans=(span,)) for index, span in enumerate(spans)),
        MEMBERSHIPS,
        {
            source_occurrence_ref(spans[0]): "event:Monday",
            source_occurrence_ref(spans[1]): "event:Tuesday",
        },
    )
    (aggregate,) = aggregate_assertions(rows)
    assert len(aggregate.assertion_ids) == 2
    assert aggregate.support_group_ids == ("event:Monday", "event:Tuesday")


def test_occurrence_refs_ignore_regenerated_evidence_ids_but_bind_every_coordinate() -> None:
    span = SourceSpan("e1", "representation", 5, 12, "block", 2)
    ref = source_occurrence_ref(span)
    assert ref.startswith("source-occurrence:")
    assert ref == source_occurrence_ref(replace(span, evidence_ref="regenerated"))
    for changes in (
        {"representation_ref": "other"},
        {"start_char": 6},
        {"end_char": 13},
        {"block_id": "other"},
        {"page_index": 3},
        {"block_id": None, "page_index": None},
    ):
        assert ref != source_occurrence_ref(replace(span, **changes))
    with pytest.raises(TypeError):
        source_occurrence_ref("representation")  # type: ignore[arg-type]


def test_partial_coordinate_support_stays_unknown_across_regenerated_duplicate_bindings() -> None:
    first = SourceSpan("e1", "rep", 0, 4)
    second = SourceSpan("e2", "rep", 8, 12)
    items = (
        mention("original", source_spans=(first, second)),
        mention(
            "copy",
            source_spans=(replace(first, evidence_ref="e3"), replace(second, evidence_ref="e4")),
        ),
    )
    partial = {source_occurrence_ref(first): "event:1"}
    (row,) = bind_assertions(items, MEMBERSHIPS, partial)
    assert row.support_group_id.startswith("unknown-support:")
    assert set(row.evidence_refs) == {"e1", "e2", "e3", "e4", "original", "copy"}
    assert bind_assertions(tuple(reversed(items)), MEMBERSHIPS, partial) == (row,)
    complete = {**partial, source_occurrence_ref(second): "event:1"}
    (established,) = bind_assertions(items, MEMBERSHIPS, complete)
    assert established.support_group_id == "event:1"
    (composite,) = bind_assertions(
        items, MEMBERSHIPS, {**partial, source_occurrence_ref(second): "event:2"}
    )
    assert composite.support_group_id.startswith("composite-support:")


def test_custom_mapping_cannot_supply_conflicting_duplicate_occurrence_assignments() -> None:
    first = SourceSpan("e1", "rep", 0, 4)
    second = SourceSpan("e2", "rep", 8, 12)
    items = (
        mention("original", source_spans=(first, second)),
        mention("copy", source_spans=(replace(first, evidence_ref="e3"), second)),
    )
    mapping = SingleReadMapping(
        ((source_occurrence_ref(first), "event:1"), (source_occurrence_ref(first), "event:2"))
    )
    with pytest.raises(ValueError, match=r"duplicate|conflicting"):
        bind_assertions(items, MEMBERSHIPS, mapping)


def test_caller_support_and_counter_mappings_are_read_only_once() -> None:
    source = mention()
    negative = mention("negative", polarity="negative")
    support = SingleReadMapping(((source_occurrence_ref(source.source_spans[0]), "event:1"),))
    counters = SingleReadMapping((("negative", ("r",)),))
    rows = bind_assertions((source, negative), MEMBERSHIPS, support, counter_mentions=counters)
    assert {row.support_group_id for row in rows} == {"event:1"}
    assert sum(bool(row.counter_assertion_ids) for row in rows) == 1


def test_invalid_values_in_mapping_snapshots_fail_validation() -> None:
    ref = source_occurrence_ref(mention().source_spans[0])
    for value in ("unknown-support:injected", "composite-support:injected", ""):
        with pytest.raises(ValueError):
            bind_assertions((mention(),), MEMBERSHIPS, SingleReadMapping(((ref, value),)))
    with pytest.raises(TypeError):
        bind_assertions(
            (mention(),), MEMBERSHIPS, {}, counter_mentions=SingleReadMapping((("r", ["other"]),))
        )


def test_conflicting_duplicate_counter_snapshot_keys_are_rejected() -> None:
    mapping = SingleReadMapping((("r", ("first",)), ("r", ("second",))))
    with pytest.raises(ValueError, match="duplicate"):
        bind_assertions((mention(),), MEMBERSHIPS, {}, counter_mentions=mapping)
