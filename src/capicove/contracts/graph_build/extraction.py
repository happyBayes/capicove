"""Validated extraction values; model-relative coordinates stay separate."""

from dataclasses import dataclass

from capicove.contracts.graph_build._checks import (
    choice,
    digest,
    qualifiers,
    refs,
    score,
    text,
    values,
)
from capicove.contracts.graph_build.evidence import SourceSpan
from capicove.contracts.graph_build.graph import IdentityKey
from capicove.contracts.graph_build.vocabulary import ENTITY_TYPES, PREDICATES
from capicove.contracts.retrieval import CharacterRange


@dataclass(frozen=True, slots=True)
class RelativeSpan:
    """Quote a segment-relative range before occurrence binding."""

    segment_id: str
    start_char: int
    end_char: int
    quote: str

    def __post_init__(self) -> None:
        text(self.segment_id)
        CharacterRange(self.start_char, self.end_char)
        if not isinstance(self.quote, str):
            raise TypeError("quote must be a string")
        if len(self.quote) != self.end_char - self.start_char:
            raise ValueError("quote length must match code-point range")


@dataclass(frozen=True, slots=True)
class EntityMention:
    """An occurrence-bound entity; attribution never defaults to the owner."""

    mention_id: str
    text_unit_id: str
    name: str
    entity_type: str
    description: str
    source_spans: tuple[SourceSpan, ...]
    anchor_role: str
    extraction_confidence: float
    identity_keys: tuple[IdentityKey, ...]
    attribution_ref: str | None

    def __post_init__(self) -> None:
        text(self.mention_id)
        text(self.text_unit_id)
        text(self.name)
        text(self.description)
        choice(self.entity_type, ENTITY_TYPES)
        choice(self.anchor_role, frozenset({"core", "context_anchor"}))
        object.__setattr__(self, "extraction_confidence", score(self.extraction_confidence))
        object.__setattr__(
            self, "source_spans", values(self.source_spans, SourceSpan, required=True)
        )
        object.__setattr__(self, "identity_keys", values(self.identity_keys, IdentityKey))
        text(self.attribution_ref, optional=True)


@dataclass(frozen=True, slots=True)
class RelationshipMention:
    """A source-stated relationship, retaining status, polarity, and attribution."""

    mention_id: str
    text_unit_id: str
    source_mention_id: str
    target_mention_id: str
    predicate: str
    description: str
    source_spans: tuple[SourceSpan, ...]
    epistemic_status: str
    polarity: str
    extraction_confidence: float
    attribution_ref: str | None
    qualifiers: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for value in (
            self.mention_id,
            self.text_unit_id,
            self.source_mention_id,
            self.target_mention_id,
            self.description,
        ):
            text(value)
        choice(self.predicate, PREDICATES)
        choice(
            self.epistemic_status, frozenset({"explicit", "inferred", "hypothesized", "disputed"})
        )
        choice(self.polarity, frozenset({"positive", "negative"}))
        object.__setattr__(self, "extraction_confidence", score(self.extraction_confidence))
        object.__setattr__(
            self, "source_spans", values(self.source_spans, SourceSpan, required=True)
        )
        text(self.attribution_ref, optional=True)
        object.__setattr__(self, "qualifiers", qualifiers(self.qualifiers))


@dataclass(frozen=True, slots=True)
class ValidatedExtraction:
    """A bounded extraction result whose local reference graph is intact.

    Evidence binding, ontology endpoints, and authorization remain host validation
    responsibilities; this value cannot independently read original sources.
    """

    entities: tuple[EntityMention, ...]
    relationships: tuple[RelationshipMention, ...]
    rejected_count: int
    issue_codes: tuple[str, ...]
    result_fingerprint: str

    def __post_init__(self) -> None:
        entities = values(self.entities, EntityMention)
        relationships = values(self.relationships, RelationshipMention)
        object.__setattr__(self, "entities", entities)
        object.__setattr__(self, "relationships", relationships)
        object.__setattr__(self, "issue_codes", refs(self.issue_codes))
        if type(self.rejected_count) is not int or self.rejected_count < 0:
            raise ValueError("rejected_count must be a nonnegative integer")
        if self.rejected_count and not self.issue_codes:
            raise ValueError("rejections must have issue codes")
        digest(self.result_fingerprint)
        refs([entity.mention_id for entity in entities])
        refs([relation.mention_id for relation in relationships])
        units = {entity.text_unit_id for entity in entities} | {
            r.text_unit_id for r in relationships
        }
        if len(units) > 1:
            raise ValueError("extraction must belong to one text unit")
        ids = {entity.mention_id for entity in entities}
        if ids & {relation.mention_id for relation in relationships}:
            raise ValueError("accepted mention IDs must be unique")
        for relation in relationships:
            if relation.source_mention_id not in ids or relation.target_mention_id not in ids:
                raise ValueError("relationship endpoint is missing")
