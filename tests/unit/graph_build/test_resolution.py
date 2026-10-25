"""Snapshot-local identity preserves evidence and blocks transitive false merges."""

from dataclasses import FrozenInstanceError, replace
from itertools import combinations, permutations

import pytest

from capicove.contracts.graph_build import (
    EntityMention,
    GraphScope,
    IdentityKey,
    PairDecision,
    SourceSpan,
)
from capicove.graph_build.candidates import IdentityContext, pair_features
from capicove.graph_build.profiles import ResolutionProfile
from capicove.graph_build.resolution import ResolutionResult, can_merge, resolve_entities


SCOPE = GraphScope("workspace", "scope")
PROFILE = ResolutionProfile()


def mention(mention_id: str, name: str = "Alex", **changes: object) -> EntityMention:
    value = EntityMention(
        mention_id,
        f"unit:{mention_id}",
        name,
        "PERSON",
        "Sourced description",
        (SourceSpan(f"source:{mention_id}", f"representation:{mention_id}", 0, 4),),
        "core",
        0.9,
        (),
        None,
    )
    return replace(value, **changes)  # type: ignore[arg-type]


def decision(left: str, right: str, **changes: object) -> PairDecision:
    value = PairDecision(
        left,
        right,
        "matched",
        ("explicit_alias_evidence",),
        (),
        (left, right, "source:explicit-identity"),
    )
    return replace(value, **changes)  # type: ignore[arg-type]


def resolve(
    items: tuple[EntityMention, ...], pairs: tuple[PairDecision, ...] = ()
) -> ResolutionResult:
    return resolve_entities(items, pairs, PROFILE, scope=SCOPE)


def groups(result: ResolutionResult) -> set[tuple[str, ...]]:
    return {item.mention_ids for item in result.entities}


def test_pair_match_cannot_hide_cluster_conflict() -> None:
    blocked = frozenset({frozenset({"a", "c"})})
    assert can_merge(frozenset({"a"}), frozenset({"b"}), blocked)
    assert not can_merge(frozenset({"a", "b"}), frozenset({"c"}), blocked)


def test_transitive_chain_checks_all_accumulated_distinct_decisions() -> None:
    pairs = (decision("a", "b"), decision("b", "c"), decision("a", "c", decision="distinct"))
    result = resolve(tuple(mention(i) for i in "abc"), pairs)
    assert groups(result) == {("a", "b"), ("c",)}
    assert set(result.decisions) == set(pairs)
    assert {item.mention_id for item in result.memberships} == {"a", "b", "c"}
    assert "hard_cluster_conflict" in result.issue_codes


@pytest.mark.parametrize(
    "reason", ["conflicting_authoritative_identity", "cannot_link", "conflicting_occurrence"]
)
def test_persisted_hard_conflict_cannot_be_overridden_by_adjudication(reason: str) -> None:
    pairs = (decision("a", "b"), decision("b", "c"), decision("a", "c", reason_codes=(reason,)))
    assert groups(resolve(tuple(mention(i) for i in "abc"), pairs)) == {("a", "b"), ("c",)}


def test_incompatible_type_blocks_even_a_strong_matched_edge() -> None:
    result = resolve((mention("a"), mention("b", entity_type="PROJECT")), (decision("a", "b"),))
    assert groups(result) == {("a",), ("b",)}


def test_unequal_or_partial_authoritative_inventories_do_not_invent_conflicts() -> None:
    work = IdentityKey("email", "alex@work.example", True)
    home = IdentityKey("email", "alex@home.example", True)
    items = (
        mention("a", identity_keys=(work,)),
        mention("b", identity_keys=(home,)),
        mention("c", identity_keys=(work, home)),
    )
    pairs = (
        decision("a", "c", reason_codes=("authoritative_identity_key",)),
        decision("b", "c", reason_codes=("authoritative_identity_key",)),
    )
    result = resolve(items, pairs)
    assert groups(result) == {("a", "b", "c")}
    assert set(result.entities[0].identity_keys) == {work, home}


@pytest.mark.parametrize("authoritative", [False, True])
def test_shared_keys_require_the_persisted_strong_rule(authoritative: bool) -> None:
    key = IdentityKey("directory", "42", authoritative)
    items = (mention("a", identity_keys=(key,)), mention("b", identity_keys=(key,)))
    assert groups(resolve(items)) == {("a",), ("b",)}
    result = resolve(items, (decision("a", "b", reason_codes=("authoritative_identity_key",)),))
    assert groups(result) == ({("a", "b")} if authoritative else {("a",), ("b",)})


@pytest.mark.parametrize("description", ["Practice on Monday", "Practice; date unknown"])
def test_event_wording_dates_aliases_and_extracted_occurrence_keys_do_not_prove_identity(
    description: str,
) -> None:
    key = IdentityKey("capicove:occurrence_ref", "explicit:occurrence", False)
    items = tuple(
        mention(
            i,
            "practice",
            entity_type="PRACTICE_EVENT",
            description=description,
            identity_keys=(key,),
        )
        for i in "ab"
    )
    result = resolve(items, (decision("a", "b"),))
    assert groups(result) == {("a",), ("b",)}
    assert "insufficient_occurrence_evidence" in result.issue_codes


def test_explicit_duplicate_event_evidence_can_merge_unknown_dates_without_inventing_time() -> None:
    items = tuple(
        mention(i, "practice", entity_type="PRACTICE_EVENT", attribution_ref="narrator")
        for i in "ab"
    )
    result = resolve(items, (decision("a", "b", reason_codes=("explicit_occurrence_identity",)),))
    assert groups(result) == {("a", "b")}
    assert result.entities[0].occurrence_ref is None
    assert result.entities[0].subject_ref is None


def test_explicit_occurrence_conflict_is_checked_across_cluster() -> None:
    items = tuple(
        mention(
            i,
            entity_type="WORK_EVENT",
            identity_keys=(IdentityKey("capicove:occurrence_ref", ref, False),) if ref else (),
        )
        for i, ref in (("a", "occurrence:1"), ("b", None), ("c", "occurrence:2"))
    )
    pairs = tuple(
        decision(a, b, reason_codes=("explicit_occurrence_identity",))
        for a, b in (("a", "b"), ("b", "c"))
    )
    assert groups(resolve(items, pairs)) == {("a", "b"), ("c",)}


@pytest.mark.parametrize("context", ["different", "missing", "same"])
def test_diagnostic_subject_is_explicit_and_never_attribution(context: str) -> None:
    first = mention(
        "a",
        entity_type="DIFFICULTY",
        attribution_ref="owner",
        identity_keys=(IdentityKey("capicove:subject_ref", "person:1", False),),
    )
    second = mention(
        "b",
        entity_type="DIFFICULTY",
        attribution_ref="owner",
        identity_keys=()
        if context == "missing"
        else (
            IdentityKey(
                "capicove:subject_ref", "person:1" if context == "same" else "person:2", False
            ),
        ),
    )
    result = resolve((first, second), (decision("a", "b"),))
    assert groups(result) == ({("a", "b")} if context == "same" else {("a",), ("b",)})
    assert (
        next(item for item in result.entities if "a" in item.mention_ids).subject_ref == "person:1"
    )


@pytest.mark.parametrize(
    "reason,evidence",
    [
        ("normalized_name_or_alias", ("a", "b", "source:alias")),
        ("explicit_alias_evidence", ()),
        ("explicit_alias_evidence", ("a", "b")),
        ("explicit_alias_evidence", ("foreign:source",)),
    ],
)
def test_same_words_fuzzy_matched_and_unbound_reasons_abstain(
    reason: str, evidence: tuple[str, ...]
) -> None:
    pair = decision(
        "a",
        "b",
        reason_codes=(reason,),
        evidence_refs=evidence,
        features=(("name_similarity", 1.0), ("embedding_similarity", 1.0)),
    )
    assert groups(resolve((mention("a"), mention("b")), (pair,))) == {("a",), ("b",)}


def test_qualification_reference_alone_does_not_supply_fuzzy_policy() -> None:
    pair = decision("a", "b", reason_codes=("fuzzy_match",), features=(("name_similarity", 1.0),))
    result = resolve_entities(
        (mention("a"), mention("b")),
        (pair,),
        replace(PROFILE, qualification_ref="evaluation:1"),
        scope=SCOPE,
    )
    assert groups(result) == {("a",), ("b",)}


def test_alias_evidence_merges_and_preserves_all_raw_names_and_memberships() -> None:
    result = resolve((mention("b", "A. Smith"), mention("a", "Alex Smith")), (decision("a", "b"),))
    assert groups(result) == {("a", "b")}
    assert {result.entities[0].display_name, *result.entities[0].aliases} == {
        "Alex Smith",
        "A. Smith",
    }
    assert {(item.entity_id, item.mention_id) for item in result.memberships} == {
        (result.entities[0].entity_id, "a"),
        (result.entities[0].entity_id, "b"),
    }


def test_deduplicated_source_frequency_prevents_context_copies_outvoting_independent_evidence() -> (
    None
):
    first = mention("a", "Alex")
    items = (
        first,
        replace(first, mention_id="b", anchor_role="context_anchor"),
        replace(first, mention_id="c"),
        mention("d", "Alexander"),
        mention("e", "Alexander"),
    )
    pairs = tuple(decision("a", i) for i in "bcde")
    result = resolve(items, pairs)
    assert result.entities[0].display_name == "Alexander"
    assert result.entities[0].aliases == ("Alex",)


def test_display_name_quality_prefers_core_then_confidence_before_lexical_ties() -> None:
    items = (
        mention("a", "A", anchor_role="context_anchor"),
        mention("b", "B", extraction_confidence=0.1),
        mention("c", "C", extraction_confidence=0.8),
    )
    assert resolve(items, (decision("a", "b"), decision("a", "c"))).entities[0].display_name == "C"


def test_all_input_permutations_and_regenerated_decisions_are_identical() -> None:
    items = tuple(mention(i, i.upper()) for i in "abc")
    pairs = (decision("a", "b"), decision("b", "c"), decision("a", "c", decision="distinct"))
    expected = resolve(items, pairs)
    for ordered_items in permutations(items):
        for ordered_pairs in permutations(pairs):
            assert resolve(ordered_items, ordered_pairs) == expected
    assert resolve(items, tuple(replace(pair) for pair in pairs)) == expected


def test_ids_change_with_grouping_scope_and_closed_profile_fingerprint() -> None:
    items = (mention("a"), mention("b"))
    separate = resolve(items)
    merged = resolve(items, (decision("a", "b"),))
    assert {item.entity_id for item in separate.entities}.isdisjoint(
        item.entity_id for item in merged.entities
    )
    changed = resolve_entities(
        items, (), replace(PROFILE, qualification_ref="evaluation:2"), scope=SCOPE
    )
    other_scope = resolve_entities(items, (), PROFILE, scope=GraphScope("workspace", "other"))
    assert separate.resolution_fingerprint != changed.resolution_fingerprint
    assert separate.entities[0].entity_id != changed.entities[0].entity_id
    assert separate.entities[0].entity_id != other_scope.entities[0].entity_id


def test_empty_singleton_and_ambiguous_results_preserve_immutable_inputs() -> None:
    assert resolve(()).entities == ()
    items = (mention("a"), mention("b"))
    pair = decision("a", "b", decision="ambiguous")
    result = resolve(items, (pair,))
    assert groups(result) == {("a",), ("b",)}
    assert result.decisions == (pair,)
    with pytest.raises(FrozenInstanceError):
        result.entities = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    "items,pairs,error",
    [
        ((mention("a"), mention("a")), (), "unique"),
        ((mention("a"),), (decision("a", "missing"),), "endpoint"),
        ((mention("a"), mention("b")), (decision("a", "b"), decision("b", "a")), "duplicate"),
    ],
)
def test_input_identity_and_decision_integrity(
    items: tuple[EntityMention, ...], pairs: tuple[PairDecision, ...], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        resolve(items, pairs)


def test_pool_limit_is_enforced_before_grouping() -> None:
    with pytest.raises(ValueError, match="identity_pool_limit"):
        resolve_entities(
            (mention("a"), mention("b")), (), ResolutionProfile(identity_pool_limit=1), scope=SCOPE
        )


@pytest.mark.parametrize(
    "namespace,entity_type",
    [("capicove:subject_ref", "DIFFICULTY"), ("capicove:occurrence_ref", "WORK_EVENT")],
)
def test_contradictory_explicit_single_mention_context_is_not_silently_dropped(
    namespace: str, entity_type: str
) -> None:
    item = mention(
        "a",
        entity_type=entity_type,
        identity_keys=(
            IdentityKey(namespace, "first", False),
            IdentityKey(namespace, "second", False),
        ),
    )
    with pytest.raises(ValueError, match="conflicting explicit"):
        resolve((item,))


def test_domain_compatibility_is_checked_across_all_cluster_members() -> None:
    shared = IdentityKey("directory", "42", True)
    items = tuple(
        mention(
            i,
            entity_type="CONCEPT",
            identity_keys=(shared, *(IdentityKey("domain", domain, False) for domain in domains)),
        )
        for i, domains in (("a", ("physics",)), ("b", ("physics", "finance")), ("c", ("finance",)))
    )
    pairs = tuple(
        decision(a, b, reason_codes=("authoritative_identity_key",))
        for a, b in (("a", "b"), ("b", "c"))
    )
    assert groups(resolve(items, pairs)) == {("a", "b"), ("c",)}


@pytest.mark.parametrize(
    "items,pairs,profile,scope",
    [
        ([], (), PROFILE, SCOPE),
        ((), [], PROFILE, SCOPE),
        (("not-a-mention",), (), PROFILE, SCOPE),
        ((), ("not-a-decision",), PROFILE, SCOPE),
        ((), (), None, SCOPE),
        ((), (), PROFILE, None),
    ],
)
def test_invalid_input_types_are_rejected(
    items: object, pairs: object, profile: object, scope: object
) -> None:
    with pytest.raises(TypeError):
        resolve_entities(items, pairs, profile, scope=scope)  # type: ignore[arg-type]


def test_different_evidence_ids_and_units_for_the_same_source_range_do_not_add_name_votes() -> None:
    first = mention("a", "Alex")
    repeated = replace(
        first,
        mention_id="b",
        text_unit_id="different",
        source_spans=(replace(first.source_spans[0], evidence_ref="regenerated-evidence"),),
    )
    items = (first, repeated, mention("c", "Alexander"), mention("d", "Alexander"))
    result = resolve(items, tuple(decision("a", i) for i in "bcd"))
    assert result.entities[0].display_name == "Alexander"


@pytest.mark.parametrize("namespace", ["domain", "capicove:subject_ref", "capicove:occurrence_ref"])
def test_reserved_context_keys_cannot_supply_identity_authority(namespace: str) -> None:
    key = IdentityKey(namespace, "context:1", True)
    items = tuple(mention(i, identity_keys=(key,)) for i in "ab")
    assert groups(
        resolve(items, (decision("a", "b", reason_codes=("authoritative_identity_key",)),))
    ) == {("a",), ("b",)}


def test_dense_matches_do_not_recheck_entire_already_merged_clusters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import capicove.graph_build.resolution as module

    actual = module.can_merge
    calls = 0

    def checked(
        left: frozenset[str], right: frozenset[str], blocked: frozenset[frozenset[str]]
    ) -> bool:
        nonlocal calls
        calls += 1
        return actual(left, right, blocked)

    monkeypatch.setattr(module, "can_merge", checked)
    pairs = tuple(decision(a, b) for a, b in combinations("abcd", 2))
    result = resolve(tuple(mention(i) for i in "abcd"), pairs)
    assert groups(result) == {("a", "b", "c", "d")}
    assert result.decisions == pairs
    assert calls == 3


@pytest.mark.parametrize(
    "entity_type,reason",
    [("PERSON", "explicit_alias_evidence"), ("WORK_EVENT", "explicit_occurrence_identity")],
)
@pytest.mark.parametrize("extra_evidence", [(), ("separate:identity-evidence",)])
@pytest.mark.parametrize("cite_mention_ids", [False, True])
def test_endpoint_source_citations_are_not_separate_identity_evidence(
    entity_type: str, reason: str, extra_evidence: tuple[str, ...], cite_mention_ids: bool
) -> None:
    items = tuple(
        mention(
            i,
            entity_type=entity_type,
            source_spans=(
                SourceSpan(f"source:{i}", f"representation:{i}", 0, 4),
                SourceSpan(f"source:extra-{i}", f"representation:{i}", 8, 12),
            ),
        )
        for i in "ab"
    )
    endpoint_citations = ("a", "b") if cite_mention_ids else ()
    pair = decision(
        "a",
        "b",
        reason_codes=(reason,),
        evidence_refs=(
            *endpoint_citations,
            "source:a",
            "source:b",
            "source:extra-a",
            "source:extra-b",
            *extra_evidence,
        ),
    )
    result = resolve(items, (pair,))
    assert groups(result) == ({("a", "b")} if extra_evidence else {("a",), ("b",)})


def test_cited_common_explicit_occurrence_reference_can_support_event_identity() -> None:
    key = IdentityKey("capicove:occurrence_ref", "occurrence:shared", False)
    items = tuple(mention(i, entity_type="WORK_EVENT", identity_keys=(key,)) for i in "ab")
    pair = decision(
        "a",
        "b",
        reason_codes=("explicit_occurrence_identity",),
        evidence_refs=("source:a", "source:b", "occurrence:shared"),
    )
    result = resolve(items, (pair,))
    assert groups(result) == {("a", "b")}
    assert result.entities[0].occurrence_ref == "occurrence:shared"


@pytest.mark.parametrize("namespace", ["domain", "capicove:subject_ref", "capicove:occurrence_ref"])
@pytest.mark.parametrize("entity_type", ["PERSON", "CONCEPT", "DIFFICULTY", "WORK_EVENT"])
def test_emitted_context_keys_cannot_regain_authority_in_candidate_scoring(
    namespace: str, entity_type: str
) -> None:
    key = IdentityKey(namespace, "context:shared", True)
    items = tuple(mention(i, entity_type=entity_type, identity_keys=(key,)) for i in "ab")
    result = resolve(items)
    left, right = result.entities
    assert left.identity_keys == right.identity_keys == (replace(key, authoritative=False),)
    scored = pair_features(left, right, IdentityContext(SCOPE))
    assert "authoritative_identity_key" not in scored.reason_codes
    if entity_type == "PERSON":
        assert scored.decision == "ambiguous"
    assert all(item.identity_keys == (key,) for item in items)


def test_regenerated_semantically_identical_decisions_have_canonical_result_values() -> None:
    items = tuple(mention(i) for i in "abc")
    canonical = (
        decision(
            "a",
            "b",
            reason_codes=("explicit_alias_evidence", "normalized_name_or_alias"),
            features=(("embedding_similarity", 0.7), ("name_similarity", 1.0)),
            evidence_refs=("a", "b", "source:alias"),
        ),
        decision(
            "a",
            "c",
            decision="distinct",
            reason_codes=("cannot_link", "conflicting_authoritative_identity"),
            evidence_refs=("a", "c", "source:distinct"),
        ),
        decision(
            "b",
            "c",
            reason_codes=("explicit_alias_evidence", "normalized_name_or_alias"),
            features=(("embedding_similarity", 0.8), ("name_similarity", 0.9)),
            evidence_refs=("b", "c", "source:alias"),
        ),
    )
    regenerated = tuple(
        replace(
            pair,
            left_id=pair.right_id,
            right_id=pair.left_id,
            reason_codes=tuple(reversed(pair.reason_codes)),
            features=tuple(reversed(pair.features)),
            evidence_refs=tuple(reversed(pair.evidence_refs)),
        )
        for pair in reversed(canonical)
    )
    expected = resolve(items, canonical)
    actual = resolve(tuple(reversed(items)), regenerated)
    assert actual == expected
    assert actual.decisions == canonical
    assert groups(actual) == {("a", "b"), ("c",)}
    assert actual.issue_codes == ("hard_cluster_conflict",)
    assert regenerated[0].left_id == "c"
    assert regenerated[0].reason_codes == ("normalized_name_or_alias", "explicit_alias_evidence")
