"""Exercise the personal-growth relationship endpoint policy."""

import pytest

from capicove.graph_build.ontology import relation_allowed


EVENTS = (
    "LEARNING_EVENT",
    "PRACTICE_EVENT",
    "WORK_EVENT",
    "FEEDBACK_EVENT",
    "OUTCOME_EVENT",
    "GENERAL_EVENT",
)


@pytest.mark.parametrize(
    "source,predicate,target",
    [
        ("PERSON", "HAS_SKILL", "SKILL"),
        ("TASK", "REQUIRES_SKILL", "SKILL"),
        ("PROJECT", "REQUIRES_SKILL", "SKILL"),
        ("RESOURCE", "ABOUT_CONCEPT", "CONCEPT"),
        ("TASK", "PART_OF_PROJECT", "PROJECT"),
        ("DIFFICULTY", "MAY_BE_CAUSED_BY", "CAUSE_HYPOTHESIS"),
        ("ERROR_PATTERN", "MAY_BE_CAUSED_BY", "CAUSE_HYPOTHESIS"),
        ("KNOWLEDGE_GAP", "MAY_BE_CAUSED_BY", "CAUSE_HYPOTHESIS"),
        ("DIFFICULTY", "BASED_ON", "FALSE_ASSUMPTION"),
        ("ERROR_PATTERN", "BASED_ON", "FALSE_ASSUMPTION"),
        ("TASK", "USES_RESOURCE", "RESOURCE"),
        ("PROJECT", "USES_RESOURCE", "RESOURCE"),
        ("TASK", "APPLIES_CONCEPT", "CONCEPT"),
        ("FEEDBACK_EVENT", "EVALUATES", "TASK"),
        ("OUTCOME_EVENT", "EVALUATES", "PROJECT"),
        ("FEEDBACK_EVENT", "EVALUATES", "SKILL"),
        ("TASK", "PRODUCES", "RESOURCE"),
        ("PROJECT", "PRODUCES", "OUTCOME_EVENT"),
        ("DIFFICULTY", "CONSTRAINED_BY", "CONSTRAINT"),
        ("TASK", "CONSTRAINED_BY", "CONSTRAINT"),
        ("PROJECT", "CONSTRAINED_BY", "CONSTRAINT"),
        ("INTERVENTION", "ADDRESSES", "DIFFICULTY"),
        ("INTERVENTION", "ADDRESSES", "ERROR_PATTERN"),
        ("INTERVENTION", "ADDRESSES", "FALSE_ASSUMPTION"),
        ("INTERVENTION", "TARGETS", "SKILL"),
        ("INTERVENTION", "TARGETS", "CONCEPT"),
        ("INTERVENTION", "TARGETS", "KNOWLEDGE_GAP"),
        ("OUTCOME_EVENT", "EVALUATES_EFFECT_OF", "INTERVENTION"),
    ],
)
def test_accepts_declared_endpoint_families(source: str, predicate: str, target: str) -> None:
    assert relation_allowed(source, predicate, target)


@pytest.mark.parametrize("event", EVENTS)
def test_each_event_supports_event_relationships(event: str) -> None:
    for predicate, target in (
        ("PRACTICES_SKILL", "SKILL"),
        ("OCCURS_IN_PROJECT", "PROJECT"),
        ("USES_RESOURCE", "RESOURCE"),
        ("APPLIES_CONCEPT", "CONCEPT"),
        ("PRODUCES", "OUTCOME_EVENT"),
        ("PRODUCES", "RESOURCE"),
        ("REVEALS", "DIFFICULTY"),
        ("REVEALS", "KNOWLEDGE_GAP"),
        ("EXHIBITS", "ERROR_PATTERN"),
        ("CONSTRAINED_BY", "CONSTRAINT"),
    ):
        assert relation_allowed(event, predicate, target)
    assert relation_allowed("PERSON", "PARTICIPATED_IN", event)


@pytest.mark.parametrize(
    "source,predicate,target",
    [
        ("RESOURCE", "HAS_SKILL", "SKILL"),
        ("PERSON", "HAS_SKILL", "CONCEPT"),
        ("SKILL", "HAS_SKILL", "PERSON"),
        ("PERSON", "EXHIBITS", "ERROR_PATTERN"),
        ("ERROR_PATTERN", "MAY_BE_CAUSED_BY", "PERSON"),
        ("ERROR_PATTERN", "BASED_ON", "CAUSE_HYPOTHESIS"),
        ("WORK_EVENT", "EVALUATES_EFFECT_OF", "INTERVENTION"),
        ("PERSON", "PARTICIPATED_IN", "TASK"),
        ("RESOURCE", "APPLIES_CONCEPT", "CONCEPT"),
        ("WORK_EVENT", "EVALUATES", "PERSON"),
        ("UNKNOWN", "RELATED_TO", "SKILL"),
        ("PERSON", "RELATED_TO", "UNKNOWN"),
        ("PERSON", "CAUSES", "DIFFICULTY"),
        ("person", "HAS_SKILL", "SKILL"),
    ],
)
def test_rejects_unsupported_semantics(source: str, predicate: str, target: str) -> None:
    assert not relation_allowed(source, predicate, target)


def test_related_to_does_not_require_or_infer_a_causal_direction() -> None:
    assert relation_allowed("PERSON", "RELATED_TO", "DIFFICULTY")
    assert relation_allowed("DIFFICULTY", "RELATED_TO", "PERSON")
    assert not relation_allowed("PERSON", "MAY_BE_CAUSED_BY", "DIFFICULTY")
