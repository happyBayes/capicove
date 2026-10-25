"""Canonical graph identity and assertion values."""

from dataclasses import dataclass

from capicove.contracts.graph_build._checks import (
    choice,
    digest,
    features,
    qualifiers,
    refs,
    text,
    values,
)
from capicove.contracts.graph_build.evidence import GraphScope
from capicove.contracts.graph_build.vocabulary import ENTITY_TYPES, PREDICATES


@dataclass(frozen=True, slots=True)
class IdentityKey:
    """A namespaced identity key, never inferred authoritative by a model."""

    namespace: str
    value: str
    authoritative: bool

    def __post_init__(self) -> None:
        text(self.namespace)
        text(self.value)
        if type(self.authoritative) is not bool:
            raise TypeError("authoritative must be boolean")


@dataclass(frozen=True, slots=True)
class Entity:
    """A snapshot-local entity retaining mentions and identity evidence."""

    entity_id: str
    scope: GraphScope
    entity_type: str
    display_name: str
    aliases: tuple[str, ...]
    mention_ids: tuple[str, ...]
    identity_keys: tuple[IdentityKey, ...]
    occurrence_ref: str | None
    subject_ref: str | None

    def __post_init__(self) -> None:
        text(self.entity_id)
        if not isinstance(self.scope, GraphScope):
            raise TypeError("scope must be GraphScope")
        choice(self.entity_type, ENTITY_TYPES)
        text(self.display_name)
        object.__setattr__(self, "aliases", refs(self.aliases))
        object.__setattr__(self, "mention_ids", refs(self.mention_ids, required=True))
        object.__setattr__(self, "identity_keys", values(self.identity_keys, IdentityKey))
        text(self.occurrence_ref, optional=True)
        text(self.subject_ref, optional=True)


@dataclass(frozen=True, slots=True)
class PairDecision:
    """An assessed pair with scores that do not imply truth probability."""

    left_id: str
    right_id: str
    decision: str
    reason_codes: tuple[str, ...]
    features: tuple[tuple[str, float], ...]
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        text(self.left_id)
        text(self.right_id)
        if self.left_id == self.right_id:
            raise ValueError("pair endpoints must differ")
        choice(self.decision, frozenset({"matched", "distinct", "ambiguous"}))
        object.__setattr__(self, "reason_codes", refs(self.reason_codes, required=True))
        object.__setattr__(self, "features", features(self.features))
        object.__setattr__(self, "evidence_refs", refs(self.evidence_refs))


@dataclass(frozen=True, slots=True)
class EntityAlignment:
    """Correspondence to a pinned baseline, not canonical memory identity."""

    current_entity_id: str
    previous_entity_id: str
    decision: str
    feature_scores: tuple[tuple[str, float], ...]
    evidence_refs: tuple[str, ...]
    resolver_fingerprint: str

    def __post_init__(self) -> None:
        text(self.current_entity_id)
        text(self.previous_entity_id)
        choice(self.decision, frozenset({"matched", "distinct", "ambiguous"}))
        object.__setattr__(self, "feature_scores", features(self.feature_scores))
        object.__setattr__(self, "evidence_refs", refs(self.evidence_refs))
        digest(self.resolver_fingerprint)


@dataclass(frozen=True, slots=True)
class RelationshipAssertion:
    """An immutable attributed claim, distinct from verified truth."""

    assertion_id: str
    source_entity_id: str
    predicate: str
    target_entity_id: str
    epistemic_status: str
    polarity: str
    attribution_ref: str | None
    qualifiers: tuple[tuple[str, str], ...]
    evidence_refs: tuple[str, ...]
    support_group_id: str
    derivation_ref: str | None
    counter_assertion_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        text(self.assertion_id)
        text(self.source_entity_id)
        text(self.target_entity_id)
        choice(self.predicate, PREDICATES)
        choice(
            self.epistemic_status, frozenset({"explicit", "inferred", "hypothesized", "disputed"})
        )
        choice(self.polarity, frozenset({"positive", "negative"}))
        text(self.attribution_ref, optional=True)
        text(self.derivation_ref, optional=True)
        text(self.support_group_id)
        object.__setattr__(self, "qualifiers", qualifiers(self.qualifiers))
        object.__setattr__(self, "evidence_refs", refs(self.evidence_refs, required=True))
        object.__setattr__(self, "counter_assertion_ids", refs(self.counter_assertion_ids))
        if self.assertion_id in self.counter_assertion_ids:
            raise ValueError("assertion cannot counter itself")
        if self.epistemic_status == "disputed" and not self.counter_assertion_ids:
            raise ValueError("dispute must reference its target")


@dataclass(frozen=True, slots=True)
class RelationshipAggregate:
    """Reference compatible assertions without erasing their provenance."""

    aggregate_id: str
    assertion_ids: tuple[str, ...]
    support_group_ids: tuple[str, ...]
    counter_assertion_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        text(self.aggregate_id)
        object.__setattr__(self, "assertion_ids", refs(self.assertion_ids, required=True))
        object.__setattr__(self, "support_group_ids", refs(self.support_group_ids, required=True))
        object.__setattr__(self, "counter_assertion_ids", refs(self.counter_assertion_ids))


@dataclass(frozen=True, slots=True)
class EntityMembership:
    """Associate a snapshot-local entity with a retained mention."""

    entity_id: str
    mention_id: str

    def __post_init__(self) -> None:
        text(self.entity_id)
        text(self.mention_id)


@dataclass(frozen=True, slots=True)
class ResolutionMembership:
    """Preserve local identity behind the resolved analysis view."""

    analysis_entity_id: str
    current_entity_id: str

    def __post_init__(self) -> None:
        text(self.analysis_entity_id)
        text(self.current_entity_id)
