"""Cross-snapshot correspondence never restores historical entities or identity."""

import hashlib
from dataclasses import FrozenInstanceError, replace
from itertools import permutations
from typing import Any

import pytest

from capicove.contracts.data import DataFormat, DatasetSchema, DatasetSnapshotRef, SchemaField
from capicove.contracts.graph_build import (
    Entity,
    GraphScope,
    GraphSnapshotManifest,
    GraphSnapshotRef,
    IdentityKey,
    PairDecision,
    StageCoverage,
    StageResult,
    TableRef,
)
from capicove.contracts.graph_build.snapshots import REQUIRED_STAGES, TABLE_ROLES
from capicove.graph_build.alignment import LoadedBaseline, align_entities, align_with_baseline
from capicove.graph_build.codec import manifest_bytes
from capicove.graph_build.profiles import graph_build_profile_fingerprint
from tests.unit.graph_build.test_profiles import profile


SCOPE = GraphScope("workspace", "scope")
RESOLVER = "a" * 64


def entity(identifier: str, **changes: Any) -> Entity:
    value = Entity(
        identifier,
        SCOPE,
        "PERSON",
        "Ada",
        (),
        (f"mention:{identifier}",),
        (IdentityKey("source:person", "ada", True),),
        None,
        None,
    )
    return replace(value, **changes)


def decision(left: str, right: str, **changes: Any) -> PairDecision:
    value = PairDecision(
        left,
        right,
        "matched",
        ("authoritative_identity_key",),
        (("name_similarity", 1.0),),
        (f"mention:{left}", f"mention:{right}"),
    )
    return replace(value, **changes)


@pytest.mark.parametrize("current", [(), (entity("now"),)])
def test_first_build_retains_current_identity(current: tuple[Entity, ...]) -> None:
    result = align_entities(current, None, (), RESOLVER)
    assert result.baseline_status == "no_baseline"
    assert result.alignments == ()
    assert result.resolved_entities == current
    assert [(m.analysis_entity_id, m.current_entity_id) for m in result.resolution_memberships] == [
        (item.entity_id, item.entity_id) for item in current
    ]
    with pytest.raises(FrozenInstanceError):
        result.baseline_status = "changed"  # type: ignore[misc]


def test_retains_all_assessed_pairs_without_copying_previous_entities() -> None:
    current = (entity("now"),)
    previous = tuple(entity(name) for name in ("old1", "old2", "old3", "disappeared"))
    decisions = (
        decision("now", "old1"),
        decision("now", "old2", decision="distinct", reason_codes=("cannot_link",)),
        decision("now", "old3", decision="ambiguous"),
    )
    result = align_entities(current, previous, decisions, RESOLVER)
    assert [(a.previous_entity_id, a.decision) for a in result.alignments] == [
        ("old1", "matched"),
        ("old2", "distinct"),
        ("old3", "ambiguous"),
    ]
    assert result.baseline_status == "aligned"
    assert result.resolved_entities == current
    assert [(m.analysis_entity_id, m.current_entity_id) for m in result.resolution_memberships] == [
        ("now", "now"),
    ]
    assert all(a.resolver_fingerprint == RESOLVER for a in result.alignments)


def test_deleted_document_and_empty_current_do_not_restore_old_entities() -> None:
    result = align_entities((), (entity("old"),), (), RESOLVER)
    assert result.resolved_entities == ()
    assert result.resolution_memberships == ()
    assert result.alignments == ()


@pytest.mark.parametrize("pairs", [(("n1", "p1"), ("n2", "p1")), (("n1", "p1"), ("n1", "p2"))])
def test_competing_strong_matches_all_abstain(pairs: tuple[tuple[str, str], ...]) -> None:
    result = align_entities(
        (entity("n1"), entity("n2")),
        (entity("p1"), entity("p2")),
        tuple(decision(*pair) for pair in pairs),
        RESOLVER,
    )
    assert [a.decision for a in result.alignments] == ["ambiguous", "ambiguous"]
    assert "conflicting_correspondence" in result.issue_codes
    assert len(result.resolved_entities) == len(result.resolution_memberships) == 2


@pytest.mark.parametrize(
    "reason",
    [
        "cannot_link",
        "conflicting_authoritative_identity",
        "incompatible_entity_type",
        "incompatible_domain_or_subject",
        "conflicting_occurrence",
    ],
)
def test_hard_conflict_overrides_claimed_match(reason: str) -> None:
    result = align_entities(
        (entity("n"),),
        (entity("p"),),
        (decision("n", "p", reason_codes=(reason, "authoritative_identity_key")),),
        RESOLVER,
    )
    assert result.alignments[0].decision == "ambiguous"
    assert "hard_alignment_conflict" in result.issue_codes


@pytest.mark.parametrize(
    "changes",
    [
        {"reason_codes": ("low_similarity",), "features": (("embedding_similarity", 0.0),)},
        {"reason_codes": ("cannot_link",), "evidence_refs": ()},
        {"reason_codes": ("cannot_link",), "evidence_refs": ("mention:n",)},
        {"reason_codes": ("cannot_link",), "evidence_refs": ("mention:p",)},
        {"reason_codes": ("endpoint_citations",)},
        {"reason_codes": ("authoritative_identity_key",)},
    ],
)
def test_unsupported_distinct_abstains_without_losing_assessed_content(
    changes: dict[str, object],
) -> None:
    pair = decision("n", "p", decision="distinct", **changes)
    result = align_entities((entity("n"),), (entity("p"),), (pair,), RESOLVER)
    assert result.alignments[0].decision == "ambiguous"
    assert result.issue_codes == ("insufficient_distinct_evidence",)
    assert result.alignments[0].feature_scores == pair.features
    assert result.alignments[0].evidence_refs == tuple(sorted(pair.evidence_refs))


@pytest.mark.parametrize(
    "reason",
    [
        "cannot_link",
        "conflicting_authoritative_identity",
        "incompatible_entity_type",
        "incompatible_domain_or_subject",
        "conflicting_occurrence",
    ],
)
def test_bound_assessed_hard_conflict_remains_distinct(reason: str) -> None:
    pair = decision("n", "p", decision="distinct", reason_codes=(reason,))
    result = align_entities((entity("n"),), (entity("p"),), (pair,), RESOLVER)
    assert result.alignments[0].decision == "distinct"
    assert result.issue_codes == ()


def test_distinct_validation_is_deterministic_with_competing_matches() -> None:
    current, previous = (entity("n1"), entity("n2")), (entity("p1"), entity("p2"))
    pairs = (
        decision("n1", "p1"),
        decision("n2", "p1"),
        decision("n1", "p2", decision="distinct", reason_codes=("cannot_link",)),
        decision("n2", "p2", decision="distinct", reason_codes=("low_similarity",)),
    )
    expected = align_entities(current, previous, pairs, RESOLVER)
    assert [item.decision for item in expected.alignments] == [
        "ambiguous",
        "distinct",
        "ambiguous",
        "ambiguous",
    ]
    assert expected.issue_codes == ("conflicting_correspondence", "insufficient_distinct_evidence")
    for ordered in permutations(pairs):
        reversed_pairs = tuple(
            replace(
                pair,
                left_id=pair.right_id,
                right_id=pair.left_id,
                reason_codes=pair.reason_codes[::-1],
                evidence_refs=pair.evidence_refs[::-1],
            )
            for pair in ordered
        )
        assert align_entities(current[::-1], previous[::-1], reversed_pairs, RESOLVER) == expected


@pytest.mark.parametrize(
    "changes",
    [
        {"reason_codes": ("unqualified_identity_evidence",)},
        {"evidence_refs": ("unrelated",)},
        {"reason_codes": ("explicit_alias_evidence",)},
    ],
)
def test_weak_or_unbound_match_abstains(changes: dict[str, object]) -> None:
    result = align_entities(
        (entity("n"),), (entity("p"),), (decision("n", "p", **changes),), RESOLVER
    )
    assert result.alignments[0].decision == "ambiguous"
    assert "insufficient_identity_evidence" in result.issue_codes


def test_explicit_alias_requires_separate_bound_evidence() -> None:
    result = align_entities(
        (entity("n", identity_keys=()),),
        (entity("p", identity_keys=()),),
        (
            decision(
                "n",
                "p",
                reason_codes=("explicit_alias_evidence",),
                evidence_refs=("mention:n", "mention:p", "alias:source"),
            ),
        ),
        RESOLVER,
    )
    assert result.alignments[0].decision == "matched"


def test_reordered_inputs_and_reversed_endpoints_are_deterministic() -> None:
    current = (entity("n1"), entity("n2"))
    previous = (entity("p1"), entity("p2"))
    pairs = (decision("n1", "p1"), decision("n2", "p2"))
    expected = align_entities(current, previous, pairs, RESOLVER)
    for first in permutations(current):
        for second in permutations(previous):
            for ordered in permutations(pairs):
                reversed_pairs = tuple(
                    replace(
                        p,
                        left_id=p.right_id,
                        right_id=p.left_id,
                        evidence_refs=p.evidence_refs[::-1],
                    )
                    for p in ordered
                )
                assert align_entities(first, second, reversed_pairs, RESOLVER) == expected


def test_unchanged_content_keeps_local_identity_without_synthetic_memory_ids() -> None:
    current = (entity("unchanged"),)
    first = align_entities(current, None, (), RESOLVER)
    next_build = align_entities(current, current, (), RESOLVER)
    assert next_build.resolved_entities == first.resolved_entities
    assert next_build.resolution_memberships == first.resolution_memberships
    assert [
        (a.current_entity_id, a.previous_entity_id, a.decision) for a in next_build.alignments
    ] == [("unchanged", "unchanged", "matched")]


@pytest.mark.parametrize("pair", [("new", "unchanged"), ("unchanged", "old")])
def test_exact_unchanged_identity_reserves_both_endpoints(pair: tuple[str, str]) -> None:
    unchanged = entity("unchanged")
    result = align_entities(
        (entity("new"), unchanged),
        (entity("old"), unchanged),
        (decision(*pair),),
        RESOLVER,
    )
    statuses = {(a.current_entity_id, a.previous_entity_id): a.decision for a in result.alignments}
    assert statuses[("unchanged", "unchanged")] == "matched"
    assert statuses[pair] == "ambiguous"
    assert "conflicting_correspondence" in result.issue_codes


@pytest.mark.parametrize(
    "current,previous,pairs",
    [
        ((entity("n"), entity("n")), (), ()),
        ((), (entity("p"), entity("p")), ()),
        ((entity("n"),), (entity("p", scope=GraphScope("other", "scope")),), ()),
        ((entity("n"), entity("n2")), (), (decision("n", "n2"),)),
        ((), (entity("p"), entity("p2")), (decision("p", "p2"),)),
        ((entity("n"),), (entity("p"),), (decision("n", "missing"),)),
        ((entity("n"),), None, (decision("n", "p"),)),
        ((entity("n"),), (entity("p"),), (decision("n", "p"), decision("p", "n"))),
        ((entity("same"),), (entity("same", display_name="Changed"),), ()),
    ],
)
def test_invalid_identity_inputs_fail_closed(
    current: tuple[Entity, ...],
    previous: tuple[Entity, ...] | None,
    pairs: tuple[PairDecision, ...],
) -> None:
    with pytest.raises(ValueError):
        align_entities(current, previous, pairs, RESOLVER)


@pytest.mark.parametrize("digest", ["", "A" * 64, "a" * 63, "wrong"])
def test_invalid_resolver_digest_rejected_even_without_entities(digest: str) -> None:
    with pytest.raises(ValueError):
        align_entities((), None, (), digest)


def baseline() -> LoadedBaseline:
    dataset = DatasetSnapshotRef(
        "dataset:baseline",
        1,
        f"snapshot:{'a' * 64}",
        "file:///baseline",
        DataFormat.PARQUET,
        DatasetSchema((SchemaField("id", "string", False),)),
        "b" * 64,
        "c" * 64,
        (),
    )
    manifest = GraphSnapshotManifest(
        1,
        SCOPE,
        dataset,
        None,
        graph_build_profile_fingerprint(profile()),
        tuple(TableRef(role, dataset) for role in sorted(TABLE_ROLES)),
        tuple(
            StageResult(
                stage,
                "succeeded",
                (f"artifact:{stage}",),
                ("no_baseline",) if stage == "alignment" else (),
                StageCoverage("entities", 0, 0, 0, 0, 0, 0),
            )
            for stage in sorted(REQUIRED_STAGES)
        ),
        "complete",
    )
    return LoadedBaseline(manifest, profile(), RESOLVER, (entity("old"),))


def reference(loaded: LoadedBaseline) -> GraphSnapshotRef:
    return GraphSnapshotRef(
        "graph:pinned", hashlib.sha256(manifest_bytes(loaded.manifest)).hexdigest()
    )


class Gate:
    def __init__(self, denied: str = "") -> None:
        self.denied = denied
        self.calls: list[str] = []

    def require_scope(self, scope: GraphScope) -> None:
        assert scope == SCOPE
        self.calls.append("scope")
        if self.denied == "scope":
            raise PermissionError("scope denied")

    def require_access(self, evidence_refs: tuple[str, ...], scope: GraphScope) -> None:
        assert evidence_refs == ("source:old",)
        assert scope == SCOPE
        self.calls.append("access")
        if self.denied == "access":
            raise PermissionError("snapshot denied")


class Reader:
    def __init__(self, loaded: LoadedBaseline, gate: Gate, ref: GraphSnapshotRef) -> None:
        self.loaded, self.gate, self.ref = loaded, gate, ref
        self.calls = 0

    def read(self, ref: GraphSnapshotRef, scope: GraphScope) -> LoadedBaseline:
        assert ref == self.ref and scope == SCOPE
        assert self.gate.calls == ["scope"]
        self.calls += 1
        if self.calls > 1:
            raise AssertionError("baseline read again after pinning")
        # This trusted reader owns persisted source lineage; snapshot IDs cannot
        # stand in for source evidence when authorizing the read.
        self.gate.require_access(("source:old",), scope)
        return self.loaded


@pytest.mark.parametrize("denied", ["scope", "access"])
def test_unauthorized_baseline_fails_at_scope_or_persisted_lineage_boundary(denied: str) -> None:
    loaded, gate = baseline(), Gate(denied)
    reader = Reader(loaded, gate, reference(loaded))
    with pytest.raises(PermissionError):
        align_with_baseline(
            (),
            (),
            RESOLVER,
            baseline=reader.ref,
            scope=SCOPE,
            profile=profile(),
            access_gate=gate,
            reader=reader,
        )
    assert reader.calls == (0 if denied == "scope" else 1)


def test_baseline_pin_is_read_once_for_entire_build() -> None:
    loaded, gate = baseline(), Gate()
    reader = Reader(loaded, gate, reference(loaded))
    result = align_with_baseline(
        (entity("n1"), entity("n2")),
        (decision("n1", "old"), decision("n2", "old")),
        RESOLVER,
        baseline=reader.ref,
        scope=SCOPE,
        profile=profile(),
        access_gate=gate,
        reader=reader,
    )
    assert reader.calls == 1
    assert [a.decision for a in result.alignments] == ["ambiguous", "ambiguous"]


def test_first_build_authorizes_scope_without_baseline_reads() -> None:
    loaded, gate = baseline(), Gate()
    reader = Reader(loaded, gate, reference(loaded))
    result = align_with_baseline(
        (entity("n"),),
        (),
        RESOLVER,
        baseline=None,
        scope=SCOPE,
        profile=profile(),
        access_gate=gate,
        reader=reader,
    )
    assert result.baseline_status == "no_baseline"
    assert gate.calls == ["scope"] and reader.calls == 0


@pytest.mark.parametrize(
    "failure",
    [
        "partial",
        "profile",
        "ontology",
        "schema",
        "scope",
        "digest",
        "resolver",
        "entity_scope",
        "profile_digest",
    ],
)
def test_incompatible_or_tampered_baseline_fails_closed(failure: str) -> None:
    loaded, gate = baseline(), Gate()
    expected_schema = 1
    if failure == "partial":
        loaded = replace(loaded, manifest=replace(loaded.manifest, completeness="partial"))
    elif failure in {"profile", "ontology"}:
        changed = (
            replace(loaded.profile, profile_id="different")
            if failure == "profile"
            else replace(loaded.profile, ontology_version="other-ontology")
        )
        loaded = replace(
            loaded,
            profile=changed,
            manifest=replace(
                loaded.manifest, profile_fingerprint=graph_build_profile_fingerprint(changed)
            ),
        )
    elif failure == "schema":
        expected_schema = 2
    elif failure == "scope":
        loaded = replace(
            loaded, manifest=replace(loaded.manifest, scope=GraphScope("other", "scope"))
        )
    elif failure == "resolver":
        loaded = replace(loaded, resolver_fingerprint="d" * 64)
    elif failure == "entity_scope":
        loaded = replace(loaded, entities=(entity("old", scope=GraphScope("other", "scope")),))
    elif failure == "profile_digest":
        loaded = replace(loaded, manifest=replace(loaded.manifest, profile_fingerprint="f" * 64))
    ref = reference(loaded)
    if failure == "digest":
        ref = replace(ref, manifest_sha256="f" * 64)
    reader = Reader(loaded, gate, ref)
    with pytest.raises(ValueError):
        align_with_baseline(
            (),
            (),
            RESOLVER,
            baseline=ref,
            scope=SCOPE,
            profile=profile(),
            access_gate=gate,
            reader=reader,
            schema_version=expected_schema,
        )


@pytest.mark.parametrize(
    "kind,first,second",
    [
        ("PERSON", None, None),
        ("SKILL", None, None),
        ("DIFFICULTY", None, None),
        ("GENERAL_EVENT", "occurrence:1", "occurrence:2"),
    ],
)
def test_type_context_and_reserved_keys_cannot_authorize_match(
    kind: str,
    first: str | None,
    second: str | None,
) -> None:
    reserved = (IdentityKey("capicove:subject_ref", "subject", True),)
    current = entity("n", entity_type=kind, identity_keys=reserved, occurrence_ref=first)
    previous = entity("p", entity_type=kind, identity_keys=reserved, occurrence_ref=second)
    result = align_entities((current,), (previous,), (decision("n", "p"),), RESOLVER)
    assert result.alignments[0].decision == "ambiguous"


def test_explicit_occurrence_evidence_matches_events() -> None:
    result = align_entities(
        (entity("n", entity_type="GENERAL_EVENT"),),
        (entity("p", entity_type="GENERAL_EVENT"),),
        (
            decision(
                "n",
                "p",
                reason_codes=("explicit_occurrence_identity",),
                evidence_refs=("mention:n", "mention:p", "occurrence:source"),
            ),
        ),
        RESOLVER,
    )
    assert result.alignments[0].decision == "matched"


@pytest.mark.parametrize(
    "current,previous,pairs",
    [
        ([], None, ()),
        ((), [], ()),
        (("entity",), None, ()),
        ((), None, []),
        ((), None, ("decision",)),
    ],
)
def test_mutable_or_wrong_typed_inputs_rejected(
    current: object, previous: object, pairs: object
) -> None:
    with pytest.raises(TypeError):
        align_entities(current, previous, pairs, RESOLVER)  # type: ignore[arg-type]
