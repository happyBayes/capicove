"""Bounded identity candidates preserve evidence and abstain on fuzzy matches."""

# Fullwidth source strings intentionally exercise Unicode candidate normalization.
# ruff: noqa: RUF001

from collections.abc import Iterable, Iterator
from dataclasses import replace
from itertools import permutations

import pytest

from capicove.contracts.graph_build import Entity, GraphScope, IdentityKey
from capicove.graph_build.candidates import (
    IdentityContext,
    IdentityStageLimitError,
    candidate_batches,
    cross_identity_candidates,
    identity_candidates,
    normalize_name,
    pair_features,
)
from capicove.graph_build.profiles import ResolutionProfile


SCOPE = GraphScope("workspace", "scope")


def entity(entity_id: str, name: str = "Alex", **changes: object) -> Entity:
    value = Entity(entity_id, SCOPE, "PERSON", name, (), (f"mention:{entity_id}",), (), None, None)
    return replace(value, **changes)  # type: ignore[arg-type]


def candidates(*items: Entity) -> tuple[tuple[str, str], ...]:
    return identity_candidates(tuple(items), tuple((1.0, 0.0) for _ in items), ResolutionProfile())


def test_cross_snapshot_top_k_cannot_be_crowded_out_by_same_side_neighbors() -> None:
    current = (entity("current:a", "one"), entity("current:c", "two"))
    baseline = (entity("baseline:b", "three"), entity("baseline:d", "four"))

    assert cross_identity_candidates(
        current,
        ((1.0, 0.0), (1.0, 0.0)),
        baseline,
        ((0.0, 1.0), (0.0, 1.0)),
        ResolutionProfile(top_k=1),
    ) == (
        ("current:a", "baseline:b"),
        ("current:a", "baseline:d"),
        ("current:c", "baseline:b"),
    )


def test_cross_snapshot_top_k_unions_nearest_neighbors_from_both_sides() -> None:
    current = (entity("current:a", "one"), entity("current:c", "two"))
    baseline = (entity("baseline:b", "three"), entity("baseline:d", "four"))

    assert cross_identity_candidates(
        current,
        ((1.0, 0.0), (0.0, 1.0)),
        baseline,
        ((0.6, 0.8), (0.0, 1.0)),
        ResolutionProfile(top_k=1),
    ) == (
        ("current:a", "baseline:b"),
        ("current:c", "baseline:b"),
        ("current:c", "baseline:d"),
    )


def test_cross_snapshot_exact_recall_is_not_limited_by_embedding_top_k() -> None:
    shared_key = (IdentityKey("directory", "42", True),)
    current = (
        entity("current:key", "key current", identity_keys=shared_key),
        entity("current:alias", "alias current", aliases=("Shared Alias",)),
    )
    baseline = (
        entity("baseline:key", "key baseline", identity_keys=shared_key),
        entity("baseline:name", "ＳＨＡＲＥＤ　ＡＬＩＡＳ"),
    )

    assert cross_identity_candidates(
        current,
        ((1.0, 0.0), (0.0, 1.0)),
        baseline,
        ((1.0, 0.0), (1.0, 0.0)),
        ResolutionProfile(top_k=1),
    ) == (
        ("current:alias", "baseline:key"),
        ("current:alias", "baseline:name"),
        ("current:key", "baseline:key"),
        ("current:key", "baseline:name"),
    )


def test_cross_snapshot_ties_are_stable_under_independent_input_permutations() -> None:
    current = (
        (entity("current:b", "second"), (1.0, 0.0)),
        (entity("current:a", "first"), (1.0, 0.0)),
    )
    baseline = (
        (entity("baseline:d", "fourth"), (0.0, 1.0)),
        (entity("baseline:c", "third"), (0.0, -1.0)),
    )
    for current_order in permutations(current):
        for baseline_order in permutations(baseline):
            assert cross_identity_candidates(
                tuple(item[0] for item in current_order),
                tuple(item[1] for item in current_order),
                tuple(item[0] for item in baseline_order),
                tuple(item[1] for item in baseline_order),
                ResolutionProfile(top_k=1, pair_batch_size=1),
            ) == (
                ("current:a", "baseline:c"),
                ("current:a", "baseline:d"),
                ("current:b", "baseline:c"),
            )


def test_cross_snapshot_constraints_and_invalid_inputs_fail_closed() -> None:
    current = (entity("current:a"),)
    different_type = (entity("baseline:a", entity_type="PROJECT"),)
    assert (
        cross_identity_candidates(
            current, ((1.0,),), different_type, ((1.0,),), ResolutionProfile()
        )
        == ()
    )

    with pytest.raises(ValueError, match="scope"):
        cross_identity_candidates(
            current,
            ((1.0,),),
            (entity("baseline:a", scope=GraphScope("other", "scope")),),
            ((1.0,),),
            ResolutionProfile(),
        )
    with pytest.raises(ValueError, match="unique"):
        cross_identity_candidates(
            current, ((1.0,),), (entity("current:a"),), ((1.0,),), ResolutionProfile()
        )
    with pytest.raises(ValueError, match="dimensions"):
        cross_identity_candidates(
            current, ((1.0,),), (entity("baseline:a"),), ((1.0, 0.0),), ResolutionProfile()
        )
    with pytest.raises(ValueError, match="finite"):
        cross_identity_candidates(
            current,
            ((float("nan"),),),
            (entity("baseline:a"),),
            ((1.0,),),
            ResolutionProfile(),
        )


def test_cross_snapshot_pool_and_pair_work_respect_profile_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import capicove.graph_build.candidates as module

    real_batches = module.candidate_batches
    batch_sizes: list[int] = []

    def recording_batches(
        pairs: Iterable[tuple[int, int]], profile: ResolutionProfile
    ) -> Iterator[tuple[tuple[int, int], ...]]:
        for batch in real_batches(pairs, profile):
            batch_sizes.append(len(batch))
            yield batch

    monkeypatch.setattr(module, "candidate_batches", recording_batches)
    current = tuple(entity(f"current:{index:02d}", f"current {index}") for index in range(5))
    baseline = tuple(entity(f"baseline:{index:02d}", f"baseline {index}") for index in range(5))
    cross_identity_candidates(
        current,
        ((1.0,),) * 5,
        baseline,
        ((1.0,),) * 5,
        ResolutionProfile(top_k=1, pair_batch_size=7),
    )
    assert max(batch_sizes) == 7
    assert all(size <= 7 for size in batch_sizes)

    with pytest.raises(IdentityStageLimitError, match="identity_pool_limit"):
        cross_identity_candidates(
            current[:1],
            ((1.0,),),
            baseline[:1],
            ((1.0,),),
            ResolutionProfile(identity_pool_limit=1),
        )


def test_name_normalization_is_only_a_candidate_key() -> None:
    assert normalize_name("  Ｐｙｔｈｏｎ\t学习  ") == "python 学习"


def test_same_name_people_are_candidates_but_not_proven_identity() -> None:
    left, right = entity("b", "  Ａｌｅｘ "), entity("a", "alex")
    assert candidates(left, right) == (("a", "b"),)
    decision = pair_features(left, right, IdentityContext(SCOPE))
    assert (decision.left_id, decision.right_id, decision.decision) == ("a", "b", "ambiguous")
    assert dict(decision.features)["name_similarity"] == 1.0
    assert left.display_name == "  Ａｌｅｘ "


def test_alias_candidates_are_not_limited_by_dense_top_k() -> None:
    items = (
        entity("a", "first", aliases=("Shared",)),
        entity("b", "second", aliases=("ＳＨＡＲＥＤ",)),
        entity("c", "third", aliases=("shared",)),
    )
    assert identity_candidates(items, ((1.0,), (1.0,), (1.0,)), ResolutionProfile(top_k=1)) == (
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    )
    assert items[1].aliases == ("ＳＨＡＲＥＤ",)
    assert pair_features(items[0], items[1], IdentityContext(SCOPE)).decision == "ambiguous"


@pytest.mark.parametrize(
    "field,changes",
    [
        ("type", {"entity_type": "PROJECT"}),
        ("domain", {"identity_keys": (IdentityKey("domain", "finance", False),)}),
    ],
)
def test_type_and_domain_constraints_precede_all_lanes(
    field: str, changes: dict[str, object]
) -> None:
    left = entity(
        "a", entity_type="CONCEPT", identity_keys=(IdentityKey("domain", "physics", False),)
    )
    right = replace(left, entity_id="b", **changes)  # type: ignore[arg-type]
    assert candidates(left, right) == ()
    assert pair_features(left, right, IdentityContext(SCOPE)).decision == "distinct"


def test_scope_is_validated_before_candidate_work_and_context_cannot_cross_scope() -> None:
    left, right = entity("a"), entity("b", scope=GraphScope("other", "scope"))
    with pytest.raises(ValueError, match="scope"):
        candidates(left, right)
    with pytest.raises(ValueError, match="scope"):
        pair_features(left, right, IdentityContext(SCOPE))
    with pytest.raises(ValueError, match="scope"):
        pair_features(left, entity("c"), IdentityContext(GraphScope("workspace", "other")))


def test_authoritative_keys_match_only_exact_namespace_and_value() -> None:
    left = entity("a", identity_keys=(IdentityKey("directory:a", "42", True),))
    same = entity("b", "different", identity_keys=left.identity_keys)
    other = entity("c", identity_keys=(IdentityKey("directory:b", "42", True),))
    different = entity("d", identity_keys=(IdentityKey("directory:a", "43", True),))
    assert pair_features(left, same, IdentityContext(SCOPE)).decision == "matched"
    assert pair_features(left, other, IdentityContext(SCOPE)).decision == "ambiguous"
    assert pair_features(left, different, IdentityContext(SCOPE)).decision == "ambiguous"
    assert candidates(left, different) == (("a", "d"),)


def test_untrusted_keys_and_owner_subject_do_not_establish_person_identity() -> None:
    left = entity(
        "a", "I", subject_ref="owner", identity_keys=(IdentityKey("directory", "42", False),)
    )
    right = entity("b", "I", subject_ref="owner", identity_keys=left.identity_keys)
    assert pair_features(left, right, IdentityContext(SCOPE)).decision == "ambiguous"


def test_strong_key_lane_preserves_candidates_outside_top_k() -> None:
    items = tuple(
        entity(name, name, identity_keys=(IdentityKey("directory", "42", True),)) for name in "abc"
    )
    assert identity_candidates(items, ((1.0,), (1.0,), (1.0,)), ResolutionProfile(top_k=1)) == (
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    )


def test_top_k_ties_and_input_reordering_use_stable_ids_and_aligned_vectors() -> None:
    aligned = (
        (entity("a", "first"), (1.0, 0.0)),
        (entity("b", "second"), (0.0, 1.0)),
        (entity("c", "third"), (-1.0, 0.0)),
        (entity("d", "fourth"), (0.0, -1.0)),
    )
    for ordered in permutations(aligned):
        assert identity_candidates(
            tuple(item[0] for item in ordered),
            tuple(item[1] for item in ordered),
            ResolutionProfile(top_k=1, pair_batch_size=2),
        ) == (
            ("a", "b"),
            ("a", "d"),
            ("b", "c"),
        )


@pytest.mark.parametrize(
    "vectors",
    [
        (),
        ((1.0,),),
        ((1.0,), (1.0, 2.0)),
        ((), ()),
        ((0.0, 0.0), (1.0, 0.0)),
        ((float("nan"),), (1.0,)),
        ((float("inf"),), (1.0,)),
        ((True,), (1.0,)),
    ],
)
def test_invalid_aligned_vectors_fail_before_work(vectors: tuple[tuple[float, ...], ...]) -> None:
    with pytest.raises((ValueError, TypeError)):
        identity_candidates((entity("a"), entity("b")), vectors, ResolutionProfile())


def test_duplicate_ids_are_rejected_and_empty_or_singleton_pools_are_supported() -> None:
    with pytest.raises(ValueError, match="unique"):
        candidates(entity("a"), entity("a"))
    assert identity_candidates((), (), ResolutionProfile()) == ()
    assert candidates(entity("a")) == ()


def test_pool_and_pair_batch_limits_have_absolute_caps_but_preserve_smaller_policy() -> None:
    with pytest.raises(ValueError, match="10000"):
        ResolutionProfile(identity_pool_limit=10001)
    with pytest.raises(ValueError, match="2000"):
        ResolutionProfile(pair_batch_size=2001)
    profile = ResolutionProfile(identity_pool_limit=1)
    with pytest.raises(IdentityStageLimitError, match="identity_pool_limit"):
        identity_candidates((entity("a"), entity("b")), ((1.0,), (1.0,)), profile)
    # No comparisons are necessary for a valid 10,000-entity pool of distinct domains.
    items = tuple(
        entity(str(i), entity_type="CONCEPT", identity_keys=(IdentityKey("domain", str(i), False),))
        for i in range(10000)
    )
    assert (
        identity_candidates(items, ((1.0,),) * 10000, ResolutionProfile(identity_pool_limit=10000))
        == ()
    )


def test_pair_batches_are_lazy_bounded_and_preserve_the_last_partial_batch() -> None:
    consumed: list[int] = []

    def pairs() -> Iterator[tuple[str, str]]:
        for i in range(4001):
            consumed.append(i)
            yield (str(i), str(i + 1))

    batches = candidate_batches(pairs(), ResolutionProfile())
    assert consumed == []
    assert len(next(batches)) == 2000
    assert len(consumed) == 2000
    assert tuple(map(len, batches)) == (2000, 1)


def test_repeated_event_wording_never_merges_different_or_unknown_occurrences() -> None:
    left = entity("a", "practice", entity_type="PRACTICE_EVENT", occurrence_ref="Monday")
    right = entity("b", "practice", entity_type="PRACTICE_EVENT", occurrence_ref="Tuesday")
    assert candidates(left, right) == ()
    assert pair_features(left, right, IdentityContext(SCOPE)).decision == "distinct"
    unknown = replace(right, occurrence_ref=None)
    assert pair_features(left, unknown, IdentityContext(SCOPE)).decision == "ambiguous"
    same = replace(right, occurrence_ref="Monday")
    assert pair_features(left, same, IdentityContext(SCOPE)).decision == "matched"


def test_explicit_alias_and_occurrence_evidence_stay_scoped_and_cited() -> None:
    left, right = entity("a"), entity("b", "other")
    context = IdentityContext(SCOPE, explicit_aliases=(("a", "b", "alias-source"),))
    decision = pair_features(left, right, context)
    assert decision.decision == "matched"
    assert "alias-source" in decision.evidence_refs
    event_left, event_right = (
        replace(left, entity_type="WORK_EVENT"),
        replace(right, entity_type="WORK_EVENT"),
    )
    assert pair_features(event_left, event_right, context).decision == "ambiguous"
    context = IdentityContext(SCOPE, occurrence_evidence=(("a", "b", "event-source"),))
    decision = pair_features(event_left, event_right, context)
    assert decision.decision == "matched"
    assert "event-source" in decision.evidence_refs


def test_cosine_and_neighbors_are_separate_features_and_absence_is_not_distinct() -> None:
    context = IdentityContext(
        SCOPE,
        vector_space="identity-v1",
        vectors=(("a", (1.0, 0.0)), ("b", (-1.0, 0.0))),
        neighbor_ids=(("a", ("x", "y")), ("b", ("x", "z"))),
    )
    decision = pair_features(entity("a", "first"), entity("b", "second"), context)
    assert decision.decision == "ambiguous"
    assert dict(decision.features)["embedding_similarity"] == 0.0
    assert "uncalibrated_affine_cosine" in decision.reason_codes
    assert dict(decision.features)["shared_context_score"] == pytest.approx(1 / 3)
    assert "probability" not in dict(decision.features)
    assert (
        pair_features(entity("a", "first"), entity("b", "second"), IdentityContext(SCOPE)).decision
        == "ambiguous"
    )


def test_context_rejects_unpinned_invalid_or_duplicate_vectors() -> None:
    with pytest.raises(ValueError, match="space"):
        IdentityContext(SCOPE, vectors=(("a", (1.0,)),))
    with pytest.raises(ValueError, match="unique"):
        IdentityContext(SCOPE, vector_space="space", vectors=(("a", (1.0,)), ("a", (1.0,))))
    with pytest.raises(ValueError, match="vector"):
        IdentityContext(SCOPE, vector_space="space", vectors=(("a", (0.0,)),))


def test_cosine_handles_finite_extreme_magnitudes_without_overflow_or_underflow() -> None:
    context = IdentityContext(
        SCOPE, vector_space="space", vectors=(("a", (1e308, 1e308)), ("b", (1e-308, 1e-308)))
    )
    assert dict(pair_features(entity("a"), entity("b"), context).features)[
        "embedding_similarity"
    ] == pytest.approx(1.0)


def test_missing_domain_or_subject_is_not_evidence_of_distinct_identity() -> None:
    known = entity(
        "a", entity_type="CONCEPT", identity_keys=(IdentityKey("domain", "physics", False),)
    )
    unknown = entity("b", entity_type="CONCEPT")
    assert pair_features(known, unknown, IdentityContext(SCOPE)).decision == "ambiguous"
    known = entity("a", entity_type="DIFFICULTY", subject_ref="person:a")
    unknown = entity("b", entity_type="DIFFICULTY")
    assert pair_features(known, unknown, IdentityContext(SCOPE)).decision == "ambiguous"


def test_batch_limits_bound_real_comparisons_and_all_lanes(monkeypatch: pytest.MonkeyPatch) -> None:
    import capicove.graph_build.candidates as module

    real_batches, real_cosine = module.candidate_batches, module._cosine
    batch_sizes: list[int] = []
    comparisons = 0

    def recording_batches(
        pairs: Iterable[tuple[int, int]], profile: ResolutionProfile
    ) -> Iterator[tuple[tuple[int, int], ...]]:
        for batch in real_batches(pairs, profile):
            batch_sizes.append(len(batch))
            yield batch

    def recording_cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        nonlocal comparisons
        comparisons += 1
        return real_cosine(left, right)

    monkeypatch.setattr(module, "candidate_batches", recording_batches)
    monkeypatch.setattr(module, "_cosine", recording_cosine)
    items = (
        *(entity(f"person:{i:03d}") for i in range(65)),
        entity("project", entity_type="PROJECT"),
    )
    result = candidates(*items)
    assert len(result) == 2080
    assert comparisons == 2080
    assert max(batch_sizes) == 2000
    assert all(size <= 2000 for size in batch_sizes)
    assert all("project" not in pair for pair in result)


def test_validation_precedes_any_pair_work(monkeypatch: pytest.MonkeyPatch) -> None:
    import capicove.graph_build.candidates as module

    def forbidden(*args: object) -> None:
        pytest.fail("invalid pool reached pair work")

    monkeypatch.setattr(module, "candidate_batches", forbidden)
    with pytest.raises(ValueError, match="nonzero"):
        identity_candidates((entity("a"), entity("b")), ((1.0,), (0.0,)), ResolutionProfile())
    with pytest.raises(IdentityStageLimitError):
        identity_candidates(
            (entity("a"), entity("b")), ((1.0,), (0.0,)), ResolutionProfile(identity_pool_limit=1)
        )


def test_partial_key_inventories_preserve_shared_identity_and_occurrence_conflicts_remain_distinct() -> (
    None
):
    left = entity("a", identity_keys=(IdentityKey("directory", "42", True),))
    right = entity("b", identity_keys=(*left.identity_keys, IdentityKey("directory", "43", True)))
    context = IdentityContext(SCOPE, explicit_aliases=(("a", "b", "alias"),))
    assert pair_features(left, right, context).decision == "matched"
    first = entity("a", entity_type="WORK_EVENT", occurrence_ref="Monday")
    second = entity("b", entity_type="WORK_EVENT", occurrence_ref="Tuesday")
    context = IdentityContext(SCOPE, occurrence_evidence=(("a", "b", "same-event"),))
    assert pair_features(first, second, context).decision == "distinct"


@pytest.mark.parametrize("vectors", [((1.0,),), ((1.0,), (1.0,))])
def test_input_order_is_positional_and_mutable_outer_containers_are_rejected(
    vectors: tuple[tuple[float, ...], ...],
) -> None:
    with pytest.raises(TypeError, match="tuples"):
        identity_candidates([entity("a")], vectors, ResolutionProfile())  # type: ignore[arg-type]


def test_pair_constraints_run_before_cosine_feature_work(monkeypatch: pytest.MonkeyPatch) -> None:
    import capicove.graph_build.candidates as module

    def forbidden(*args: object) -> None:
        pytest.fail("incompatible pair reached cosine feature work")

    monkeypatch.setattr(module, "_cosine", forbidden)
    context = IdentityContext(SCOPE, vector_space="space", vectors=(("a", (1.0,)), ("b", (1.0,))))
    decision = pair_features(entity("a"), entity("b", entity_type="PROJECT"), context)
    assert decision.decision == "distinct"
    assert "embedding_similarity" not in dict(decision.features)


@pytest.mark.parametrize("entity_type", ["SKILL", "CONCEPT"])
@pytest.mark.parametrize("lane", ["key", "name", "alias", "embedding"])
def test_overlapping_domains_remain_candidates_and_require_strong_identity(
    entity_type: str, lane: str
) -> None:
    shared_key = (IdentityKey("directory", "42", True),) if lane == "key" else ()
    left = entity(
        "b",
        "shared" if lane == "name" else "first",
        entity_type=entity_type,
        identity_keys=(IdentityKey("domain", "physics", False), *shared_key),
    )
    right = entity(
        "c",
        "shared" if lane == "name" else "second",
        entity_type=entity_type,
        aliases=("first",) if lane == "alias" else (),
        identity_keys=(
            IdentityKey("domain", "physics", False),
            IdentityKey("domain", "math", False),
            *shared_key,
        ),
    )
    if lane == "embedding":
        assert candidates(left, right) == (("b", "c"),)
    else:
        # This lower-ID equal-cosine neighbor wins each top-1; only the selected
        # key/name/alias lane can recover the b,c pair outside dense top-k.
        other = entity(
            "a",
            "unrelated",
            entity_type=entity_type,
            identity_keys=(IdentityKey("domain", "physics", False),),
        )
        assert identity_candidates(
            (left, other, right), ((1.0,),) * 3, ResolutionProfile(top_k=1)
        ) == (
            ("a", "b"),
            ("a", "c"),
            ("b", "c"),
        )
    context = IdentityContext(SCOPE, explicit_aliases=(("b", "c", "source:alias"),))
    assert pair_features(left, right, context).decision == (
        "matched" if lane == "key" else "ambiguous"
    )


def test_work_and_home_identity_keys_are_not_exclusive_identifiers() -> None:
    work = entity("a", identity_keys=(IdentityKey("email", "alex@work.example", True),))
    home = entity("b", identity_keys=(IdentityKey("email", "alex@home.example", True),))
    combined = entity("c", identity_keys=(*work.identity_keys, *home.identity_keys))
    assert candidates(work, home, combined) == (("a", "b"), ("a", "c"), ("b", "c"))
    for left, right, expected in (
        (work, home, "ambiguous"),
        (work, combined, "matched"),
        (home, combined, "matched"),
    ):
        decision = pair_features(left, right, IdentityContext(SCOPE))
        assert decision.decision == expected
        assert "conflicting_authoritative_identity" not in decision.reason_codes
    assert (
        pair_features(work, replace(work, entity_id="d"), IdentityContext(SCOPE)).decision
        == "matched"
    )


@pytest.mark.parametrize("entity_type", ["SKILL", "CONCEPT", "DIFFICULTY"])
@pytest.mark.parametrize("evidence", ["key", "alias"])
@pytest.mark.parametrize("one_known", [False, True])
def test_required_domain_and_subject_context_cannot_be_replaced_by_identity_evidence(
    entity_type: str, evidence: str, one_known: bool
) -> None:
    keys = (IdentityKey("directory", "42", True),) if evidence == "key" else ()
    left = entity("a", entity_type=entity_type, identity_keys=keys)
    right = entity("b", entity_type=entity_type, identity_keys=keys)
    if one_known:
        if entity_type == "DIFFICULTY":
            left = replace(left, subject_ref="person:a")
        else:
            left = replace(left, identity_keys=(*keys, IdentityKey("domain", "physics", False)))
    context = IdentityContext(
        SCOPE, explicit_aliases=(("a", "b", "source:alias"),) if evidence == "alias" else ()
    )
    decision = pair_features(left, right, context)
    assert decision.decision == "ambiguous"
    assert "insufficient_domain_or_subject_context" in decision.reason_codes


@pytest.mark.parametrize("entity_type", ["SKILL", "CONCEPT", "DIFFICULTY"])
def test_equal_present_required_context_allows_strong_identity(entity_type: str) -> None:
    keys: tuple[IdentityKey, ...] = (IdentityKey("directory", "42", True),)
    if entity_type != "DIFFICULTY":
        keys = (*keys, IdentityKey("domain", "physics", False))
    left = entity(
        "a",
        entity_type=entity_type,
        identity_keys=keys,
        subject_ref="person:a" if entity_type == "DIFFICULTY" else None,
    )
    assert (
        pair_features(left, replace(left, entity_id="b"), IdentityContext(SCOPE)).decision
        == "matched"
    )


def test_known_subject_conflict_overrides_shared_keys_and_explicit_alias() -> None:
    left = entity(
        "a",
        entity_type="DIFFICULTY",
        subject_ref="person:a",
        identity_keys=(IdentityKey("directory", "42", True),),
    )
    right = replace(left, entity_id="b", subject_ref="person:b")
    assert candidates(left, right) == ()
    context = IdentityContext(SCOPE, explicit_aliases=(("a", "b", "source:alias"),))
    assert pair_features(left, right, context).decision == "distinct"


def test_overlapping_domain_postings_score_each_pair_once_and_keep_global_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import capicove.graph_build.candidates as module

    actual_cosine = module._cosine
    comparisons = 0

    def count_cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        nonlocal comparisons
        comparisons += 1
        return actual_cosine(left, right)

    monkeypatch.setattr(module, "_cosine", count_cosine)
    items = tuple(
        entity(
            name,
            name,
            entity_type="CONCEPT",
            identity_keys=(
                IdentityKey("domain", "physics", False),
                IdentityKey("domain", "math", False),
            ),
        )
        for name in "abcd"
    )
    result = identity_candidates(
        items, ((1.0,),) * 4, ResolutionProfile(top_k=1, pair_batch_size=2)
    )
    assert result == (("a", "b"), ("a", "c"), ("a", "d"))
    assert comparisons == 6


def test_disjoint_explicit_domains_override_shared_authoritative_identity_and_alias() -> None:
    left = entity(
        "a",
        entity_type="CONCEPT",
        identity_keys=(
            IdentityKey("domain", "physics", False),
            IdentityKey("directory", "42", True),
        ),
    )
    right = entity(
        "b",
        entity_type="CONCEPT",
        identity_keys=(
            IdentityKey("domain", "finance", False),
            IdentityKey("directory", "42", True),
        ),
    )
    assert candidates(left, right) == ()
    context = IdentityContext(SCOPE, explicit_aliases=(("a", "b", "source:alias"),))
    assert pair_features(left, right, context).decision == "distinct"
