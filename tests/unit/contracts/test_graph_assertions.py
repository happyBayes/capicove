"""Protect graph assertion provenance and closed extraction records."""

from dataclasses import replace
from typing import Any

import pytest

from capicove.contracts.graph_build import (
    Entity,
    EntityAlignment,
    EntityMention,
    GraphScope,
    IdentityKey,
    PairDecision,
    RelationshipAggregate,
    RelationshipAssertion,
    RelationshipMention,
    RelativeSpan,
    SourceSpan,
    ValidatedExtraction,
)
from capicove.graph_build.codec import graph_from_json, graph_to_json


def mention() -> EntityMention:
    return EntityMention(
        "m",
        "u",
        "Python",
        "SKILL",
        "Named skill",
        (SourceSpan("e", "r", 0, 6),),
        "core",
        0.8,
        (),
        None,
    )


def assertion(**changes: Any) -> RelationshipAssertion:
    value = RelationshipAssertion(
        "a",
        "person",
        "HAS_SKILL",
        "skill",
        "explicit",
        "negative",
        "speaker",
        (),
        ("e",),
        "support",
        None,
        (),
    )
    return replace(value, **changes)


def test_negative_attributed_assertion_survives_round_trip() -> None:
    value = assertion()
    restored = graph_from_json(graph_to_json(value))
    assert restored == value
    assert isinstance(restored, RelationshipAssertion)
    assert restored.polarity == "negative"
    assert restored.attribution_ref == "speaker"


@pytest.mark.parametrize("confidence", [True, -0.1, 1.1, float("nan"), float("inf"), "0.5"])
def test_confidence_rejects_invalid_values(confidence: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(mention(), extraction_confidence=confidence)


@pytest.mark.parametrize(
    "changes",
    [
        {"epistemic_status": "proven"},
        {"polarity": "neutral"},
        {"attribution_ref": ""},
        {"qualifiers": (("tool", "run"),)},
        {"qualifiers": (("time_text", "Monday"), ("time_text", "Tuesday"))},
        {"epistemic_status": "disputed"},
        {"evidence_refs": ()},
    ],
)
def test_assertion_rejects_lossy_or_open_fields(changes: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        assertion(**changes)


def test_dispute_links_target_without_mutating_original() -> None:
    original = assertion()
    dispute = assertion(assertion_id="b", epistemic_status="disputed", counter_assertion_ids=("a",))
    assert original.epistemic_status == "explicit"
    assert graph_from_json(graph_to_json(dispute)) == dispute


def test_source_stated_hypothesis_does_not_fabricate_a_derivation_receipt() -> None:
    value = assertion(epistemic_status="hypothesized", derivation_ref=None)
    assert graph_from_json(graph_to_json(value)) == value


def test_relative_span_preserves_unicode_and_rejects_wrong_quote_length() -> None:
    assert graph_from_json(graph_to_json(RelativeSpan("core", 0, 2, "学习"))) == RelativeSpan(
        "core", 0, 2, "学习"
    )
    with pytest.raises(ValueError):
        RelativeSpan("core", 0, 2, "学")


def test_closed_features_are_sorted_and_not_truth_decisions() -> None:
    value = PairDecision(
        "a",
        "b",
        "ambiguous",
        ("insufficient_evidence",),
        (("shared_context_score", 0.5), ("name_similarity", 1.0)),
        (),
    )
    assert value.features == (("name_similarity", 1.0), ("shared_context_score", 0.5))
    assert graph_from_json(graph_to_json(value)) == value
    for features in [
        (("truth", 1.0),),
        (("name_similarity", float("nan")),),
        (("name_similarity", True),),
    ]:
        with pytest.raises((TypeError, ValueError)):
            replace(value, features=features)


def test_entity_identity_keys_and_aggregate_round_trip() -> None:
    key = IdentityKey("authored", "person:one", True)
    entity = Entity(
        "entity", GraphScope("w", "s"), "PERSON", "Author", ("Writer",), ("m",), (key,), None, None
    )
    alignment = EntityAlignment("current", "previous", "ambiguous", (), (), "a" * 64)
    aggregate = RelationshipAggregate("aggregate", ("a",), ("support",), ())
    for value in (key, entity, alignment, aggregate, mention()):
        assert graph_from_json(graph_to_json(value)) == value
    with pytest.raises((TypeError, ValueError)):
        IdentityKey("n", "v", 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        replace(entity, entity_type="UNKNOWN")


def test_validated_extraction_enforces_endpoint_and_unit_integrity() -> None:
    source = replace(mention(), mention_id="person", entity_type="PERSON")
    relation = RelationshipMention(
        "rel",
        "u",
        "person",
        "m",
        "HAS_SKILL",
        "Reported",
        (SourceSpan("e", "r", 0, 6),),
        "explicit",
        "negative",
        0.9,
        "author",
        (("time_text", "Monday"),),
    )
    value = ValidatedExtraction((source, mention()), (relation,), 0, (), "b" * 64)
    assert graph_from_json(graph_to_json(value)) == value
    for entities in [
        (mention(),),
        (source, source),
        (source, replace(mention(), text_unit_id="other")),
    ]:
        with pytest.raises(ValueError):
            replace(value, entities=entities)
    with pytest.raises(ValueError):
        replace(value, rejected_count=1)


def test_graph_decoder_rejects_unknown_nested_fields() -> None:
    import json

    payload = json.loads(graph_to_json(mention()))
    payload["value"]["source_spans"][0]["command"] = "run"
    with pytest.raises(ValueError):
        graph_from_json(json.dumps(payload))
