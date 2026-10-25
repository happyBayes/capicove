"""Provider-neutral repository/extraction surfaces consumed by knowledge."""

from collections.abc import Callable
from datetime import datetime
from typing import Any, Protocol

from capicove.contracts.evidence import EvidenceSpan
from capicove.contracts.knowledge import FactProposal, MemoryEpisode
from capicove.contracts.memory import MemoryQuery
from capicove.knowledge.models import ExtractedFact


class TemporalMemoryRepository(Protocol):
    def head(self, workspace: str) -> tuple[int, datetime]: ...
    def replay_receipt(
        self, workspace: str, key: str, request: object
    ) -> dict[str, Any] | None: ...
    def records(
        self, workspace: str, scope: str, kind: str, identity: str | None = None
    ) -> list[dict[str, Any]]: ...
    def facts(
        self,
        workspace: str,
        scope: str,
        *,
        slot: str | None = None,
        fact: str | None = None,
        assertion: str | None = None,
        current: bool = False,
    ) -> list[dict[str, Any]]: ...
    def commit(
        self,
        *,
        workspace: str,
        scope: str,
        key: str,
        request: object,
        expected_revision: int,
        recorded: datetime,
        events: list[tuple[str, dict[str, Any]]],
        receipt: dict[str, Any],
        indexes: list[tuple[str, str, str]] | None = None,
        job: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...
    def lexical(self, workspace: str, scope: str, query: str) -> tuple[str, ...]: ...
    def rebuild(self, read_text: Callable[[str, str, EvidenceSpan], str]) -> None: ...
    def candidates(
        self, query: MemoryQuery, *, lane: str, identities: tuple[str, ...] = ()
    ) -> list[dict[str, Any]]: ...
    def claim_job(
        self,
        workspace: str,
        scope: str,
        identity: str,
        *,
        now: datetime,
        lease_seconds: int,
        force: bool = False,
    ) -> dict[str, Any] | None: ...
    def entity_assertions_for_merge(
        self, workspace: str, scope: str, entity_id: str
    ) -> list[dict[str, Any]]: ...
    def memory_policy_for(
        self, workspace: str, scope: str, target: str
    ) -> dict[str, Any] | None: ...
    def checkpoint(self) -> None: ...
    def doctor(self) -> dict[str, Any]: ...
    def due_jobs(
        self, workspace: str, scope: str, *, now: datetime, limit: int
    ) -> tuple[str, ...]: ...
    def similar_entities(
        self, workspace: str, scope: str, name: str, entity_type: str
    ) -> list[dict[str, Any]]: ...


class EntityFactExtractor(Protocol):
    extractor_id: str
    revision: str

    def extract(
        self, episode: MemoryEpisode, read_text: Callable[[int, int], str]
    ) -> tuple[FactProposal | ExtractedFact, ...]: ...
