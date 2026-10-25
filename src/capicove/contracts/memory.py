"""Immutable memory admission, query, and cited-context contracts."""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from capicove.contracts.evidence import EvidenceSpan
from capicove.contracts.knowledge import TimeInterval, text, utc


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    memory_id: str
    workspace_ref: str
    scope_ref: str
    target_ref: str
    kind: Literal["episodic", "semantic"]
    sensitivity: Literal["public", "internal", "confidential", "restricted"]
    retention: Literal["session", "standard", "pinned"]
    retention_deadline: datetime | None
    importance: float
    admission_reason: str
    admitted_at: datetime
    state: Literal["admitted", "archived", "superseded", "erased"] = "admitted"

    def __post_init__(self) -> None:
        for value in (
            self.memory_id,
            self.workspace_ref,
            self.scope_ref,
            self.target_ref,
            self.admission_reason,
        ):
            text(value)
        if (
            self.kind not in {"episodic", "semantic"}
            or self.sensitivity not in {"public", "internal", "confidential", "restricted"}
            or self.retention not in {"session", "standard", "pinned"}
            or self.state not in {"admitted", "archived", "superseded", "erased"}
        ):
            raise ValueError("invalid memory policy")
        if (
            type(self.importance) not in (int, float)
            or not math.isfinite(self.importance)
            or not 0 <= self.importance <= 1
        ):
            raise ValueError("invalid importance")
        utc(self.admitted_at)
        if self.retention_deadline is not None:
            utc(self.retention_deadline)
            if self.retention_deadline <= self.admitted_at:
                raise ValueError("invalid retention deadline")


@dataclass(frozen=True, slots=True)
class MemoryQuery:
    workspace_ref: str
    scope_ref: str
    text: str = ""
    valid_at: datetime | None = None
    known_at: datetime | None = None
    types: tuple[str, ...] = ("episodic", "semantic")
    states: tuple[str, ...] = ("accepted",)
    center_entities: tuple[str, ...] = ()
    top_k: int = 10
    retrieval_profile_ref: str = "rrf-v1"
    data_policy_ref: str = "default"
    min_revision: int | None = None

    def __post_init__(self) -> None:
        text(self.workspace_ref)
        text(self.scope_ref)
        if not isinstance(self.text, str) or len(self.text.encode()) > 8192:
            raise ValueError("invalid query")
        if type(self.top_k) is not int or not 1 <= self.top_k <= 100:
            raise ValueError("invalid top_k")
        if (
            not self.types
            or any(t not in {"episodic", "semantic"} for t in self.types)
            or not self.states
            or any(s not in {"accepted", "disputed", "rejected"} for s in self.states)
        ):
            raise ValueError("invalid filters")
        if not all(
            isinstance(value, tuple) for value in (self.types, self.states, self.center_entities)
        ):
            raise ValueError("query collections must be immutable")
        if any(
            len(set(values)) != len(values)
            for values in (self.types, self.states, self.center_entities)
        ):
            raise ValueError("duplicate query filters")
        if len(self.center_entities) > 100:
            raise ValueError("too many center entities")
        for value in self.center_entities:
            text(value)
        if self.retrieval_profile_ref != "rrf-v1" or self.data_policy_ref != "default":
            raise ValueError("unknown query policy")
        for instant in (self.valid_at, self.known_at):
            if instant is not None:
                utc(instant)
        if self.min_revision is not None and (
            type(self.min_revision) is not int or self.min_revision < 0
        ):
            raise ValueError("invalid revision")


@dataclass(frozen=True, slots=True)
class MemoryHit:
    target_ref: str
    kind: str
    score: float
    text: str
    citations: tuple[EvidenceSpan, ...]
    validity: TimeInterval
    knowledge: TimeInterval
    assertion_hash: str | None = None
    subject_ref: str | None = None
    predicate: str | None = None

    def __post_init__(self) -> None:
        text(self.target_ref)
        if (
            self.kind not in {"episodic", "semantic"}
            or type(self.score) not in (int, float)
            or not math.isfinite(self.score)
            or not 0 <= self.score <= 1
        ):
            raise ValueError("invalid memory hit")
        text(self.text, 1024 * 1024)
        if (
            not isinstance(self.citations, tuple)
            or not self.citations
            or any(not isinstance(span, EvidenceSpan) for span in self.citations)
        ):
            raise ValueError("invalid hit citations")
        if not isinstance(self.validity, TimeInterval) or not isinstance(
            self.knowledge, TimeInterval
        ):
            raise ValueError("invalid hit time intervals")


@dataclass(frozen=True, slots=True)
class MemoryRecall:
    hits: tuple[MemoryHit, ...]
    committed_revision: int
    warnings: tuple[str, ...] = ()
    degraded: bool = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.hits, tuple)
            or any(not isinstance(hit, MemoryHit) for hit in self.hits)
            or not isinstance(self.warnings, tuple)
        ):
            raise ValueError("invalid memory result")
        object.__setattr__(self, "degraded", bool(self.warnings))


@dataclass(frozen=True, slots=True)
class MemoryContext:
    hits: tuple[MemoryHit, ...]
    characters: int
    estimated_tokens: int
    warnings: tuple[str, ...] = ()
    degraded: bool = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.hits, tuple)
            or any(not isinstance(hit, MemoryHit) for hit in self.hits)
            or not isinstance(self.warnings, tuple)
        ):
            raise ValueError("invalid memory result")
        object.__setattr__(self, "degraded", bool(self.warnings))
