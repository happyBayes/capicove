"""Canonical personal-growth-v1 vocabulary, independent of graph strategies."""

BASE_OBJECTS = frozenset({"PERSON", "SKILL", "CONCEPT", "PROJECT", "TASK", "RESOURCE"})
EVENTS = frozenset(
    {
        "LEARNING_EVENT",
        "PRACTICE_EVENT",
        "WORK_EVENT",
        "FEEDBACK_EVENT",
        "OUTCOME_EVENT",
        "GENERAL_EVENT",
    }
)
DIAGNOSTICS = frozenset(
    {
        "DIFFICULTY",
        "KNOWLEDGE_GAP",
        "ERROR_PATTERN",
        "CAUSE_HYPOTHESIS",
        "FALSE_ASSUMPTION",
        "CONSTRAINT",
        "INTERVENTION",
    }
)
ENTITY_TYPES = BASE_OBJECTS | EVENTS | DIAGNOSTICS

PREDICATES = frozenset(
    {
        "HAS_SKILL",
        "REQUIRES_SKILL",
        "ABOUT_CONCEPT",
        "PART_OF_PROJECT",
        "PRACTICES_SKILL",
        "OCCURS_IN_PROJECT",
        "PARTICIPATED_IN",
        "USES_RESOURCE",
        "APPLIES_CONCEPT",
        "EVALUATES",
        "PRODUCES",
        "REVEALS",
        "EXHIBITS",
        "MAY_BE_CAUSED_BY",
        "BASED_ON",
        "CONSTRAINED_BY",
        "ADDRESSES",
        "TARGETS",
        "EVALUATES_EFFECT_OF",
        "RELATED_TO",
    }
)
