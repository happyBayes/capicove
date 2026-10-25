"""Closed personal-growth-v1 endpoint policy, without inference or model calls."""

from types import MappingProxyType

from capicove.contracts.graph_build.vocabulary import ENTITY_TYPES, EVENTS


RELATION_ENDPOINTS = MappingProxyType(
    {
        "HAS_SKILL": (frozenset({"PERSON"}), frozenset({"SKILL"})),
        "REQUIRES_SKILL": (frozenset({"TASK", "PROJECT"}), frozenset({"SKILL"})),
        "ABOUT_CONCEPT": (frozenset({"RESOURCE"}), frozenset({"CONCEPT"})),
        "PART_OF_PROJECT": (frozenset({"TASK"}), frozenset({"PROJECT"})),
        "PRACTICES_SKILL": (EVENTS, frozenset({"SKILL"})),
        "OCCURS_IN_PROJECT": (EVENTS, frozenset({"PROJECT"})),
        "PARTICIPATED_IN": (frozenset({"PERSON"}), EVENTS),
        "USES_RESOURCE": (EVENTS | {"TASK", "PROJECT"}, frozenset({"RESOURCE"})),
        "APPLIES_CONCEPT": (EVENTS | {"TASK"}, frozenset({"CONCEPT"})),
        "EVALUATES": (
            frozenset({"FEEDBACK_EVENT", "OUTCOME_EVENT"}),
            frozenset({"TASK", "PROJECT", "SKILL"}),
        ),
        "PRODUCES": (EVENTS | {"TASK", "PROJECT"}, frozenset({"OUTCOME_EVENT", "RESOURCE"})),
        "REVEALS": (EVENTS, frozenset({"DIFFICULTY", "KNOWLEDGE_GAP"})),
        "EXHIBITS": (EVENTS, frozenset({"ERROR_PATTERN"})),
        "MAY_BE_CAUSED_BY": (
            frozenset({"DIFFICULTY", "ERROR_PATTERN", "KNOWLEDGE_GAP"}),
            frozenset({"CAUSE_HYPOTHESIS"}),
        ),
        "BASED_ON": (frozenset({"DIFFICULTY", "ERROR_PATTERN"}), frozenset({"FALSE_ASSUMPTION"})),
        "CONSTRAINED_BY": (EVENTS | {"DIFFICULTY", "TASK", "PROJECT"}, frozenset({"CONSTRAINT"})),
        "ADDRESSES": (
            frozenset({"INTERVENTION"}),
            frozenset({"DIFFICULTY", "ERROR_PATTERN", "FALSE_ASSUMPTION"}),
        ),
        "TARGETS": (frozenset({"INTERVENTION"}), frozenset({"SKILL", "CONCEPT", "KNOWLEDGE_GAP"})),
        "EVALUATES_EFFECT_OF": (frozenset({"OUTCOME_EVENT"}), frozenset({"INTERVENTION"})),
        "RELATED_TO": (ENTITY_TYPES, ENTITY_TYPES),
    }
)


def relation_allowed(source_type: str, predicate: str, target_type: str) -> bool:
    """Check endpoint compatibility, never evidence truth or personal attribution."""
    endpoints = RELATION_ENDPOINTS.get(predicate)
    return endpoints is not None and source_type in endpoints[0] and target_type in endpoints[1]
