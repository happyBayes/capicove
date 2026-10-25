"""Personal evidence, conservative recurrence, and closed synthesis boundaries."""

import json
from dataclasses import asdict, replace
from importlib.resources import files
from itertools import permutations
from pathlib import Path
from typing import Any

import pytest

from capicove.contracts.graph_build import Entity, EvidenceBundle, GraphScope, RelationshipAssertion
from capicove.graph_build.diagnostics import (
    DiagnosticProfile,
    diagnostic_candidates,
    repeat_pattern_eligible,
    validate_findings,
)


SCOPE = GraphScope("workspace", "scope")
PROFILE = DiagnosticProfile("person", 32, 20)


def entity(ref: str, kind: str) -> Entity:
    return Entity(ref, SCOPE, kind, ref, (), ("mention:" + ref,), (), None, None)


ENTITIES = (
    entity("person", "PERSON"),
    entity("other", "PERSON"),
    entity("event:1", "WORK_EVENT"),
    entity("event:2", "WORK_EVENT"),
    entity("event:3", "WORK_EVENT"),
    entity("error", "ERROR_PATTERN"),
    entity("difficulty", "DIFFICULTY"),
    entity("gap", "KNOWLEDGE_GAP"),
    entity("cause", "CAUSE_HYPOTHESIS"),
    entity("assumption", "FALSE_ASSUMPTION"),
    entity("intervention", "INTERVENTION"),
    entity("feedback:1", "FEEDBACK_EVENT"),
    entity("feedback:2", "FEEDBACK_EVENT"),
    entity("outcome", "OUTCOME_EVENT"),
    entity("skill", "SKILL"),
    entity("book", "RESOURCE"),
    entity("concept", "CONCEPT"),
)


def assertion(
    ref: str, source: str, predicate: str, target: str, group: str = "support:1", **changes: Any
) -> RelationshipAssertion:
    return replace(
        RelationshipAssertion(
            ref,
            source,
            predicate,
            target,
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


def personal(
    index: int, target: str = "error", predicate: str = "EXHIBITS"
) -> tuple[RelationshipAssertion, ...]:
    return (
        assertion(
            f"participation:{index}",
            "person",
            "PARTICIPATED_IN",
            f"event:{index}",
            f"support:{index}",
        ),
        assertion(f"observation:{index}", f"event:{index}", predicate, target, f"support:{index}"),
    )


def bundle() -> EvidenceBundle:
    return diagnostic_candidates(personal(1) + personal(2), ENTITIES, PROFILE)[0]


def record(evidence: EvidenceBundle, **changes: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "local_id": "finding:1",
        "diagnostic_type": "ERROR_PATTERN",
        "candidate_kind": "error_pattern",
        "subject_ref": evidence.subject_ref,
        "target_ref": evidence.target_ref,
        "epistemic_status": "hypothesized",
        "support_refs": list(evidence.assertion_ids),
        "counter_refs": list(evidence.counter_assertion_ids),
        "explanation": "The supplied occurrences warrant checking for a repeated error.",
        "limitations": ["eligibility_not_diagnosis"],
        "verification_step": "Check the next independently observed attempt.",
    }
    value.update(changes)
    return value


def payload(*records: dict[str, Any]) -> bytes:
    return json.dumps({"version": 1, "findings": records}).encode()


def test_reprints_are_not_repeated_personal_failures() -> None:
    assert not repeat_pattern_eligible(("event:1", "event:1"), True, "source")
    assert repeat_pattern_eligible(("event:1", "event:2"), True, "source")
    assert not repeat_pattern_eligible(("event:1", "event:2"), True, "generated_report")
    assert not repeat_pattern_eligible(("event:1", "event:2"), False, "source")


def test_two_independent_personal_errors_retain_complete_paths() -> None:
    evidence = bundle()
    assert evidence.bundle_id.startswith("diagnostic-bundle:v1:error_pattern:")
    assert evidence.assertion_ids == (
        "observation:1",
        "observation:2",
        "participation:1",
        "participation:2",
    )
    assert evidence.independent_event_ids == ("event:1", "event:2")
    assert evidence.subject_ref == "person"
    assert evidence.target_ref == "error"
    assert evidence.omitted_count == 0


@pytest.mark.parametrize("rows", [(), personal(1), (personal(1)[1], personal(2)[1])])
def test_absent_attribution_or_single_error_abstains(
    rows: tuple[RelationshipAssertion, ...],
) -> None:
    assert diagnostic_candidates(rows, ENTITIES, PROFILE) == ()


def test_book_pronouns_owner_and_unrelated_quoted_opinion_do_not_attribute_performance() -> None:
    rows = (
        assertion("book-claim", "book", "ABOUT_CONCEPT", "concept"),
        assertion("quote", "other", "PARTICIPATED_IN", "event:1", attribution_ref="other"),
        personal(1)[1],
        personal(2)[1],
    )
    named = tuple(
        replace(e, display_name="I", subject_ref="person") if e.entity_type.endswith("EVENT") else e
        for e in ENTITIES
    )
    assert diagnostic_candidates(rows, named, PROFILE) == ()


@pytest.mark.parametrize("group", ["support:1", "unknown-support:x", "composite-support:x"])
def test_reprints_and_uncertain_groups_never_establish_independence(group: str) -> None:
    rows = personal(1) + tuple(replace(row, support_group_id=group) for row in personal(2))
    assert diagnostic_candidates(rows, ENTITIES, PROFILE) == ()


def test_duplicate_observations_of_one_event_are_not_repeat_events() -> None:
    rows = (
        *personal(1),
        replace(personal(1)[1], assertion_id="copy", support_group_id="support:2"),
    )
    assert diagnostic_candidates(rows, ENTITIES, PROFILE) == ()


@pytest.mark.parametrize(
    "changes",
    [
        {"derivation_ref": "receipt:old"},
        {"evidence_refs": ("report:old",)},
        {"epistemic_status": "inferred"},
        {"polarity": "negative"},
    ],
)
def test_generated_or_unestablished_participation_is_not_personal(changes: dict[str, Any]) -> None:
    rows = tuple(
        replace(row, **changes) if row.predicate == "PARTICIPATED_IN" else row
        for row in personal(1) + personal(2)
    )
    assert diagnostic_candidates(rows, ENTITIES, PROFILE) == ()


def test_generated_profile_cannot_feed_source_candidates() -> None:
    assert (
        diagnostic_candidates(
            personal(1) + personal(2), ENTITIES, replace(PROFILE, source_kind="generated_report")
        )
        == ()
    )


@pytest.mark.parametrize("target", ["difficulty", "gap"])
def test_single_explicit_personal_difficulty_is_only_an_investigation(target: str) -> None:
    (evidence,) = diagnostic_candidates(personal(1, target, "REVEALS"), ENTITIES, PROFILE)
    assert evidence.target_ref == target
    assert evidence.independent_event_ids == ("event:1",)


def test_counter_closure_is_bidirectional_and_budget_atomic() -> None:
    rows = personal(1) + personal(2)
    counter = replace(
        rows[1],
        assertion_id="counter",
        polarity="negative",
        counter_assertion_ids=("observation:1",),
    )
    (evidence,) = diagnostic_candidates((*rows, counter), ENTITIES, PROFILE)
    assert evidence.counter_assertion_ids == ("counter",)
    assert "observation:1" in evidence.assertion_ids
    assert (
        diagnostic_candidates(
            (*rows, counter), ENTITIES, replace(PROFILE, max_assertions_per_bundle=4)
        )
        == ()
    )


def test_budget_omissions_preserve_paths_and_change_identity() -> None:
    rows = personal(1) + personal(2) + personal(3)
    (complete,) = diagnostic_candidates(rows, ENTITIES, PROFILE)
    (limited,) = diagnostic_candidates(
        rows, ENTITIES, replace(PROFILE, max_assertions_per_bundle=4)
    )
    assert limited.omitted_count == 2
    assert limited.assertion_ids == bundle().assertion_ids
    assert limited.bundle_id != complete.bundle_id
    assert limited.bundle_id != bundle().bundle_id
    assert diagnostic_candidates(rows, ENTITIES, replace(PROFILE, max_bundles=0)) == ()


def test_permutations_preserve_complete_bundle_identity() -> None:
    rows = personal(1) + personal(2)
    expected = (bundle(),)
    for ordering in permutations(rows):
        assert diagnostic_candidates(ordering, tuple(reversed(ENTITIES)), PROFILE) == expected


def feedback() -> tuple[RelationshipAssertion, ...]:
    return tuple(
        row
        for i in (1, 2)
        for row in (
            assertion(f"fp:{i}", "person", "PARTICIPATED_IN", f"feedback:{i}", f"support:{i}"),
            assertion(
                f"fb:{i}",
                f"feedback:{i}",
                "EVALUATES",
                "skill",
                f"support:{i}",
                attribution_ref="other",
                polarity="negative" if i == 2 else "positive",
            ),
        )
    )


def test_recurring_feedback_retains_opposite_attributed_evidence() -> None:
    (evidence,) = diagnostic_candidates(feedback(), ENTITIES, PROFILE)
    assert evidence.target_ref == "skill"
    assert evidence.counter_assertion_ids == ()
    assert evidence.assertion_ids == ("fb:1", "fb:2", "fp:1", "fp:2")
    assert evidence.independent_event_ids == ("feedback:1", "feedback:2")


@pytest.mark.parametrize(
    "predicate,target,kind",
    [
        ("MAY_BE_CAUSED_BY", "cause", "cause_hypothesis"),
        ("BASED_ON", "assumption", "false_assumption"),
    ],
)
def test_causal_candidates_require_supplied_personal_path(
    predicate: str, target: str, kind: str
) -> None:
    link = assertion("link", "difficulty", predicate, target)
    assert diagnostic_candidates((link,), ENTITIES, PROFILE) == ()
    bundles = diagnostic_candidates(
        (*personal(1, "difficulty", "REVEALS"), link), ENTITIES, PROFILE
    )
    evidence = next(item for item in bundles if item.target_ref == target)
    assert f":{kind}:" in evidence.bundle_id
    assert set(evidence.assertion_ids) == {"participation:1", "observation:1", "link"}


def intervention_rows() -> tuple[RelationshipAssertion, ...]:
    return (
        *personal(1, "difficulty", "REVEALS"),
        assertion("addresses", "intervention", "ADDRESSES", "difficulty"),
    )


def test_intervention_never_implies_execution_effectiveness_or_failure() -> None:
    bundles = diagnostic_candidates(intervention_rows(), ENTITIES, PROFILE)
    evidence = next(item for item in bundles if item.target_ref == "intervention")
    assert ":intervention_recommendation:" in evidence.bundle_id
    accepted = record(
        evidence,
        diagnostic_type="INTERVENTION",
        candidate_kind="intervention_recommendation",
        limitations=["eligibility_not_diagnosis", "not_executed_or_verified"],
    )
    (finding,) = validate_findings(payload(accepted), evidence, "receipt:new")
    assert finding.epistemic_status == "hypothesized"
    changes: dict[str, Any]
    for changes in (
        {"epistemic_status": "inferred"},
        {"limitations": []},
        {"verification_step": None},
        {"candidate_kind": "intervention_outcome"},
        {"executed": True},
        {"effective": True},
    ):
        with pytest.raises(ValueError):
            validate_findings(payload({**accepted, **changes}), evidence, "receipt:new")


def test_outcome_requires_explicit_personal_evaluation_path() -> None:
    evaluation = assertion("effect", "outcome", "EVALUATES_EFFECT_OF", "intervention")
    rows = (*intervention_rows(), evaluation)
    assert not any(
        ":intervention_outcome:" in b.bundle_id
        for b in diagnostic_candidates(rows, ENTITIES, PROFILE)
    )
    bundles = diagnostic_candidates(
        (*rows, assertion("op", "person", "PARTICIPATED_IN", "outcome")), ENTITIES, PROFILE
    )
    evidence = next(b for b in bundles if ":intervention_outcome:" in b.bundle_id)
    assert set(evidence.assertion_ids) >= {"op", "effect"}
    value = record(
        evidence,
        diagnostic_type="INTERVENTION",
        candidate_kind="intervention_outcome",
        limitations=["eligibility_not_diagnosis", "association_not_causation"],
    )
    assert validate_findings(payload(value), evidence, "receipt:new")


@pytest.mark.parametrize(
    "mutation",
    ["duplicate_entity", "duplicate_assertion", "endpoint", "ontology", "counter", "attribution"],
)
def test_invalid_graph_inputs_fail_closed(mutation: str) -> None:
    rows = personal(1) + personal(2)
    entities: tuple[Entity, ...] = ENTITIES
    if mutation == "duplicate_entity":
        entities += (ENTITIES[0],)
    elif mutation == "duplicate_assertion":
        rows += (rows[0],)
    elif mutation == "endpoint":
        rows = (replace(rows[0], target_entity_id="missing"), *rows[1:])
    elif mutation == "ontology":
        rows = (replace(rows[0], target_entity_id="skill"), *rows[1:])
    elif mutation == "counter":
        rows = (replace(rows[0], counter_assertion_ids=("missing",)), *rows[1:])
    else:
        rows = (replace(rows[0], attribution_ref="book"), *rows[1:])
    with pytest.raises(ValueError):
        diagnostic_candidates(rows, entities, PROFILE)


def test_findings_bind_receipt_content_and_bundle_and_preserve_sources() -> None:
    evidence = bundle()
    encoded = payload(record(evidence))
    (result,) = validate_findings(encoded, evidence, "receipt:new")
    assert result.derivation_ref == "receipt:new"
    assert result.finding_id.startswith("diagnostic-finding:")
    assert result.support_refs == evidence.assertion_ids
    assert validate_findings(encoded, evidence, "receipt:new") == (result,)
    assert validate_findings(encoded, evidence, "receipt:other")[0].finding_id != result.finding_id
    assert (
        validate_findings(
            payload(record(evidence, explanation="Different explanation.")), evidence, "receipt:new"
        )[0].finding_id
        != result.finding_id
    )
    assert validate_findings(payload(), evidence, "receipt:new") == ()


@pytest.mark.parametrize(
    "changes",
    [
        {"subject_ref": "other"},
        {"target_ref": "gap"},
        {"support_refs": []},
        {"support_refs": ["alien"]},
        {"counter_refs": ["alien"]},
        {"diagnostic_type": "CAUSE_HYPOTHESIS"},
        {"diagnostic_type": "FALSE_ASSUMPTION"},
        {"epistemic_status": "explicit"},
        {"epistemic_status": "disputed"},
        {"candidate_kind": "intervention_outcome"},
        {"explanation": " "},
        {"extra": 1},
        {"support_refs": ["observation:1", "observation:1"]},
        {"limitations": []},
        {"explanation": "x" * 4097},
    ],
)
def test_invalid_findings_are_rejected(changes: dict[str, Any]) -> None:
    evidence = bundle()
    with pytest.raises(ValueError):
        validate_findings(payload(record(evidence, **changes)), evidence, "receipt:new")


@pytest.mark.parametrize(
    "encoded",
    [
        b"bad",
        b"\xff",
        b"[]",
        b"{}",
        b'{"version":1,"version":1,"findings":[]}',
        b'{"version":true,"findings":[]}',
        b" " * 65537,
    ],
)
def test_malformed_oversized_and_duplicate_json_is_rejected(encoded: bytes) -> None:
    with pytest.raises(ValueError):
        validate_findings(encoded, bundle(), "receipt:new")


def test_duplicate_finding_ids_and_missing_counter_evidence_are_rejected() -> None:
    evidence = bundle()
    value = record(evidence)
    with pytest.raises(ValueError):
        validate_findings(payload(value, value), evidence, "receipt:new")
    rows = personal(1) + personal(2)
    counter = replace(
        rows[1],
        assertion_id="counter",
        polarity="negative",
        counter_assertion_ids=("observation:1",),
    )
    (evidence,) = diagnostic_candidates((*rows, counter), ENTITIES, PROFILE)
    with pytest.raises(ValueError):
        validate_findings(payload(record(evidence, counter_refs=[])), evidence, "receipt:new")


def test_omissions_must_be_disclosed_in_findings() -> None:
    (evidence,) = diagnostic_candidates(
        personal(1) + personal(2) + personal(3),
        ENTITIES,
        replace(PROFILE, max_assertions_per_bundle=4),
    )
    with pytest.raises(ValueError):
        validate_findings(payload(record(evidence)), evidence, "receipt:new")
    value = record(evidence, limitations=["eligibility_not_diagnosis", "omitted_assertions:2"])
    assert validate_findings(payload(value), evidence, "receipt:new")


@pytest.mark.parametrize("receipt", ["", " ", "observation:1", "person"])
def test_invalid_or_self_receipt_is_rejected(receipt: str) -> None:
    evidence = bundle()
    with pytest.raises(ValueError):
        validate_findings(payload(record(evidence)), evidence, receipt)


def test_generated_or_forged_bundle_references_are_rejected() -> None:
    for evidence in (
        replace(bundle(), assertion_ids=("report:old",)),
        replace(bundle(), bundle_id="opaque"),
    ):
        with pytest.raises(ValueError):
            validate_findings(payload(record(evidence)), evidence, "receipt:new")


@pytest.mark.parametrize("field", ["support_refs", "counter_refs", "limitations"])
def test_nested_values_in_reference_arrays_fail_as_validation_errors(field: str) -> None:
    evidence = bundle()
    with pytest.raises(ValueError):
        validate_findings(payload(record(evidence, **{field: [{}]})), evidence, "receipt:new")


def test_response_local_id_cannot_self_cite_or_impersonate_host_identity() -> None:
    evidence = bundle()
    for ref in (*evidence.assertion_ids, evidence.bundle_id, evidence.subject_ref, "receipt:new"):
        with pytest.raises(ValueError):
            validate_findings(payload(record(evidence, local_id=ref)), evidence, "receipt:new")


def test_unknown_independence_abstains_for_single_event_candidates() -> None:
    rows = tuple(
        replace(row, support_group_id="unknown-support:x")
        for row in personal(1, "difficulty", "REVEALS")
    )
    assert diagnostic_candidates(rows, ENTITIES, PROFILE) == ()


@pytest.mark.parametrize(
    "changes",
    [
        {"subject_ref": " "},
        {"max_assertions_per_bundle": 0},
        {"max_assertions_per_bundle": 129},
        {"max_assertions_per_bundle": True},
        {"max_bundles": -1},
        {"max_bundles": 129},
        {"max_bundles": True},
        {"version": 2},
        {"version": True},
    ],
)
def test_invalid_profiles_cannot_expand_budget_or_infer_subject(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        replace(PROFILE, **changes)


def test_declared_generated_counters_force_omission_instead_of_silent_loss() -> None:
    rows = personal(1) + personal(2)
    counter = replace(
        rows[1],
        assertion_id="counter",
        polarity="negative",
        derivation_ref="receipt:report",
        counter_assertion_ids=("observation:1",),
    )
    assert diagnostic_candidates((*rows, counter), ENTITIES, PROFILE) == ()


def test_opposite_polarity_does_not_create_counter_links() -> None:
    rows = personal(1) + personal(2)
    counter = replace(rows[1], assertion_id="counter", polarity="negative")
    (evidence,) = diagnostic_candidates((*rows, counter), ENTITIES, PROFILE)
    assert evidence.counter_assertion_ids == ()
    assert counter.counter_assertion_ids == ()


def test_counter_cycles_fit_as_one_atomic_closure() -> None:
    participation, observation = personal(1, "difficulty", "REVEALS")
    observation = replace(observation, counter_assertion_ids=("counter",))
    counter = replace(
        observation,
        assertion_id="counter",
        polarity="negative",
        counter_assertion_ids=(observation.assertion_id,),
    )
    (evidence,) = diagnostic_candidates((participation, observation, counter), ENTITIES, PROFILE)
    assert evidence.assertion_ids == ("participation:1",)
    assert evidence.counter_assertion_ids == ("counter", "observation:1")


def test_positive_explicit_counter_keeps_its_declared_counter_role() -> None:
    rows = feedback()
    counter = replace(rows[1], assertion_id="counter", counter_assertion_ids=("fb:1",))
    (evidence,) = diagnostic_candidates((*rows, counter), ENTITIES, PROFILE)
    assert "counter" in evidence.counter_assertion_ids
    assert "counter" not in evidence.assertion_ids


def test_schema_is_closed_and_bounds_real_provider_records() -> None:
    from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

    schema = json.loads(
        files("capicove.graph_build").joinpath("schemas/diagnostics-v1.json").read_bytes()
    )
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    value = json.loads(payload(record(bundle())))
    assert validator.is_valid(value)
    for mutation in (
        {**value, "extra": True},
        {"version": 1, "findings": value["findings"] * 33},
        {"version": 1, "findings": [{**value["findings"][0], "executed": True}]},
    ):
        assert not validator.is_valid(mutation)
        with pytest.raises(ValueError):
            validate_findings(json.dumps(mutation).encode(), bundle(), "receipt:new")


def test_scripted_task8_runner_validates_bounded_diagnostic_payload(tmp_path: Path) -> None:
    from capicove.graph_build.codec import canonical_bytes, fingerprint
    from capicove.graph_build.inference import ValidationResult
    from capicove.graph_build.receipts import receipt_key
    from tests.unit.graph_build.test_inference import SOURCES, Harness, raw

    evidence = bundle()
    encoded = payload(record(evidence))
    harness = Harness(tmp_path, [raw(encoded)])
    schema = json.loads(
        files("capicove.graph_build").joinpath("schemas/diagnostics-v1.json").read_bytes()
    )
    wire = json.loads(harness.request.canonical_payload)
    wire.update(
        schema=schema,
        schema_version="diagnostics-v1",
        segments=(
            (
                "evidence",
                "source",
                json.dumps(
                    {
                        "bundle": asdict(evidence),
                        "assertions": [asdict(row) for row in personal(1) + personal(2)],
                    }
                ),
            ),
        ),
    )
    wire["prompt"] = (
        "Investigate only the supplied source assertions; return bounded diagnostic JSON."
    )
    harness.request = replace(
        harness.request,
        schema_version="diagnostics-v1",
        schema_payload=canonical_bytes(schema),
        canonical_payload=canonical_bytes(wire),
        request_fingerprint=fingerprint(wire),
        prompt_text=wire["prompt"],
        input_segments=wire["segments"],
    )
    expected_receipt = receipt_key(SCOPE, harness.request.role, harness.request.request_fingerprint)

    def validator(data: bytes) -> ValidationResult:
        findings = validate_findings(data, evidence, expected_receipt)
        assert findings[0].support_refs == evidence.assertion_ids
        return ValidationResult(data)

    receipt = harness.runner.run(SCOPE, harness.request, SOURCES, validator)
    assert receipt.status == "succeeded"
    assert receipt.call_count == 1
    assert receipt.repair_count == 0
    assert harness.artifacts.read(SCOPE, receipt.result_artifact_ref) == encoded


@pytest.mark.parametrize("duplicate_index", [0, 1])
def test_redundant_observation_or_participation_cannot_crowd_out_repeat_event(
    duplicate_index: int,
) -> None:
    rows = (
        *personal(1),
        *personal(2),
        replace(personal(1)[duplicate_index], assertion_id="alternate"),
    )
    profile = replace(PROFILE, max_assertions_per_bundle=4)
    (evidence,) = diagnostic_candidates(rows, ENTITIES, profile)
    assert evidence.independent_event_ids == ("event:1", "event:2")
    assert len(evidence.assertion_ids) == 4
    assert evidence.omitted_count == 1
    for permutation in permutations(rows):
        assert diagnostic_candidates(permutation, tuple(reversed(ENTITIES)), profile) == (evidence,)


def test_minimal_counter_closed_seed_precedes_redundant_observation() -> None:
    rows = personal(1) + personal(2)
    alternate = replace(rows[1], assertion_id="alternate")
    counter = replace(
        rows[1],
        assertion_id="counter",
        polarity="negative",
        counter_assertion_ids=("observation:1",),
    )
    profile = replace(PROFILE, max_assertions_per_bundle=4)
    (evidence,) = diagnostic_candidates((*rows, alternate, counter), ENTITIES, profile)
    assert evidence.assertion_ids == (
        "alternate",
        "observation:2",
        "participation:1",
        "participation:2",
    )
    assert evidence.counter_assertion_ids == ()
    assert evidence.omitted_count == 2
    assert evidence.independent_event_ids == ("event:1", "event:2")
    for ordering in ((*rows, alternate, counter), (counter, alternate, *reversed(rows))):
        assert diagnostic_candidates(ordering, ENTITIES, profile) == (evidence,)


def test_repeat_seed_uses_alternate_group_when_smallest_paths_share_support() -> None:
    rows = (*personal(1), *tuple(replace(row, support_group_id="support:1") for row in personal(2)))
    alternate = replace(rows[1], assertion_id="alternate", support_group_id="support:2")
    (evidence,) = diagnostic_candidates(
        (*rows, alternate), ENTITIES, replace(PROFILE, max_assertions_per_bundle=4)
    )
    assert evidence.independent_event_ids == ("event:1", "event:2")
    assert "alternate" in evidence.assertion_ids
    assert "observation:1" not in evidence.assertion_ids


@pytest.mark.parametrize("uncertainty", ["unknown-support:x", "composite-support:x"])
def test_uncertain_path_is_omitted_atomically_even_when_other_path_is_known(
    uncertainty: str,
) -> None:
    rows = (
        *personal(1, "difficulty", "REVEALS"),
        *tuple(
            replace(row, support_group_id=uncertainty)
            for row in personal(2, "difficulty", "REVEALS")
        ),
    )
    (evidence,) = diagnostic_candidates(rows, ENTITIES, PROFILE)
    assert evidence.assertion_ids == ("observation:1", "participation:1")
    assert evidence.independent_event_ids == ("event:1",)
    assert evidence.omitted_count == 2
    value = record(evidence, candidate_kind="difficulty", diagnostic_type="DIFFICULTY")
    with pytest.raises(ValueError):
        validate_findings(payload(value), evidence, "receipt:new")
    value["limitations"].append("omitted_assertions:2")
    assert validate_findings(payload(value), evidence, "receipt:new")
    value["support_refs"].append("observation:2")
    with pytest.raises(ValueError):
        validate_findings(payload(value), evidence, "receipt:new")


@pytest.mark.parametrize("position", ["extension", "counter"])
def test_uncertain_extension_or_counter_rejects_complete_path(position: str) -> None:
    rows = personal(1, "difficulty", "REVEALS")
    extension = assertion(
        "cause-link", "difficulty", "MAY_BE_CAUSED_BY", "cause", "unknown-support:x"
    )
    if position == "counter":
        extension = replace(
            rows[1],
            assertion_id="counter",
            polarity="negative",
            support_group_id="unknown-support:x",
            counter_assertion_ids=(rows[1].assertion_id,),
        )
    evidence = diagnostic_candidates((*rows, extension), ENTITIES, PROFILE)
    assert not any(
        item.target_ref == ("cause" if position == "extension" else "difficulty")
        for item in evidence
    )


@pytest.mark.parametrize(
    "target,predicate", [("error", "EXHIBITS"), ("difficulty", "REVEALS"), ("gap", "REVEALS")]
)
def test_shared_personal_event_does_not_transfer_another_subjects_diagnostic(
    target: str, predicate: str
) -> None:
    rows = (
        *personal(1, target, predicate),
        *personal(2, target, predicate),
        assertion("other-participation", "other", "PARTICIPATED_IN", "event:1"),
    )
    nodes = tuple(
        replace(node, subject_ref="other") if node.entity_id == target else node
        for node in ENTITIES
    )
    assert diagnostic_candidates(rows, nodes, PROFILE) == ()


@pytest.mark.parametrize("conflicting_ref", ["difficulty", "cause", "assumption", "intervention"])
def test_subject_conflicts_anywhere_in_diagnostic_extension_abstain(conflicting_ref: str) -> None:
    rows = (
        *intervention_rows(),
        assertion("cause-link", "difficulty", "MAY_BE_CAUSED_BY", "cause"),
        assertion("premise-link", "difficulty", "BASED_ON", "assumption"),
        assertion("assumption-remedy", "intervention", "ADDRESSES", "assumption"),
    )
    nodes = tuple(
        replace(node, subject_ref="other") if node.entity_id == conflicting_ref else node
        for node in ENTITIES
    )
    bundles = diagnostic_candidates(rows, nodes, PROFILE)
    if conflicting_ref == "difficulty":
        assert bundles == ()
    else:
        assert all(item.target_ref != conflicting_ref for item in bundles)
    if conflicting_ref == "assumption":
        intervention = next(item for item in bundles if item.target_ref == "intervention")
        assert "premise-link" not in intervention.assertion_ids
        assert "assumption-remedy" not in intervention.assertion_ids
        assert intervention.omitted_count == 2


def test_outcome_intervention_subject_conflict_is_rejected() -> None:
    rows = (
        assertion("op", "person", "PARTICIPATED_IN", "outcome"),
        assertion("effect", "outcome", "EVALUATES_EFFECT_OF", "intervention"),
    )
    nodes = tuple(
        replace(node, subject_ref="other") if node.entity_id == "intervention" else node
        for node in ENTITIES
    )
    assert diagnostic_candidates(rows, nodes, PROFILE) == ()
    assert diagnostic_candidates(rows, ENTITIES, PROFILE)[0].target_ref == "intervention"
