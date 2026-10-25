"""Untrusted named extraction candidates; identity decisions belong to knowledge."""

import math
from dataclasses import dataclass

from capicove.contracts.evidence import EvidenceSpan
from capicove.contracts.knowledge import TimeInterval, TypedLiteral, text


@dataclass(frozen=True, slots=True)
class EntityMention:
    name: str
    entity_type: str

    def __post_init__(self) -> None:
        text(self.name)
        text(self.entity_type)


@dataclass(frozen=True, slots=True)
class ExtractedFact:
    subject: EntityMention
    predicate: str
    object: EntityMention | TypedLiteral
    validity: TimeInterval
    evidence: tuple[EvidenceSpan, ...]
    confidence: float | None = None

    def __post_init__(self) -> None:
        text(self.predicate)
        if (
            not isinstance(self.subject, EntityMention)
            or not isinstance(self.object, (EntityMention, TypedLiteral))
            or not isinstance(self.validity, TimeInterval)
        ):
            raise ValueError("invalid named extraction candidate")
        if (
            not isinstance(self.evidence, tuple)
            or not self.evidence
            or any(not isinstance(span, EvidenceSpan) for span in self.evidence)
        ):
            raise ValueError("invalid candidate evidence")
        if self.confidence is not None and (
            type(self.confidence) not in (int, float)
            or not math.isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("invalid candidate confidence")
