"""Bounded five-lane temporal search; optional provider scores never bypass filters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from capicove.contracts.evidence import EvidenceSpan
from capicove.contracts.knowledge import TimeInterval
from capicove.contracts.memory import MemoryHit, MemoryQuery, MemoryRecall
from capicove.contracts.rank_fusion import normalized_rrf
from capicove.knowledge.codec import decode_episode, decode_fact
from capicove.knowledge.errors import OptionalMemoryLaneUnavailable
from capicove.knowledge.ports import TemporalMemoryRepository
from capicove.source import TextEvidenceStore


class MemorySearchPlanner:
    def __init__(
        self,
        repository: TemporalMemoryRepository,
        evidence: TextEvidenceStore,
        dense: Callable[[MemoryQuery], tuple[str, ...]] | None = None,
    ) -> None:
        self.repository, self.evidence, self.dense = repository, evidence, dense

    def recall(self, query: MemoryQuery) -> MemoryRecall:
        now = datetime.now(UTC)
        query = replace(query, valid_at=query.valid_at or now, known_at=query.known_at or now)
        workspace, scope = query.workspace_ref, query.scope_ref
        revision, _ = self.repository.head(workspace)
        lanes: list[tuple[float, list[dict[str, Any]]]] = []
        warnings: list[str] = []
        if query.text.strip():
            lanes.append((1.0, self.repository.candidates(query, lane="lexical")))
        elif not query.center_entities:
            warnings.append("browse_without_query")
        if self.dense is not None:
            try:
                identities = tuple(dict.fromkeys(self.dense(query)))[:100]
            except (OptionalMemoryLaneUnavailable, TimeoutError, ConnectionError):
                warnings.append("embedding_unavailable")
            else:
                accepted = {
                    b["id"]: b
                    for b in self.repository.candidates(query, lane="ids", identities=identities)
                }
                lanes.append((0.8, [accepted[i] for i in identities if i in accepted]))
        visited = set(query.center_entities)
        frontier = tuple(query.center_entities)
        graph: dict[str, dict[str, Any]] = {}
        for _ in range(2):
            if not frontier:
                break
            neighbors = self.repository.candidates(query, lane="graph", identities=frontier)
            next_frontier: list[str] = []
            for body in neighbors:
                graph.setdefault(body["id"], body)
                nodes = [body["subject"]]
                if body["object"]["kind"] == "entity":
                    nodes.append(body["object"]["value"])
                for node in nodes:
                    if node not in visited and len(visited) < 1000:
                        visited.add(node)
                        next_frontier.append(node)
            frontier = tuple(next_frontier)
        if query.center_entities:
            lanes.append((0.7, list(graph.values())[:100]))
        relevant = {body["id"] for _, bodies in lanes for body in bodies}
        for lane in ("recency", "utility"):
            bodies = self.repository.candidates(query, lane=lane)
            # Recency/utility rerank relevance; they cannot invent a query match.
            if query.text.strip():
                bodies = [body for body in bodies if body["id"] in relevant]
            lanes.append((0.3, bodies))
        representatives = {b["id"]: b for _, bodies in lanes for b in bodies}
        scores = normalized_rrf(
            (weight, tuple((b["id"], rank) for rank, b in enumerate(bodies, 1)))
            for weight, bodies in lanes
        )

        def start(identity: str) -> int:
            body = representatives[identity]
            value = body.get("validity", [body.get("event_time")])[0]
            return value if isinstance(value, int) else -(2**63)

        ordered = sorted(scores, key=lambda i: (-scores[i], -start(i), i))[: query.top_k]
        citations_by_id = {}
        for identity in ordered:
            body = representatives[identity]
            citations_by_id[identity] = (
                decode_fact(body).evidence
                if body["target_kind"] == "semantic"
                else (
                    EvidenceSpan(decode_episode(body).evidence_refs[0], 0, body["content_length"]),
                )
            )
            bounded = []
            remaining = 32768
            for span in citations_by_id[identity]:
                if remaining <= 0:
                    break
                size = min(span.end - span.start, remaining)
                bounded.append(EvidenceSpan(span.reference, span.start, span.start + size))
                remaining -= size + 1
            citations_by_id[identity] = tuple(bounded)
        spans = tuple(
            dict.fromkeys(span for identity in ordered for span in citations_by_id[identity])
        )
        values = self.evidence.read_spans(workspace=workspace, scope=scope, spans=spans)
        cited = dict(zip(spans, values, strict=True))
        hits: list[MemoryHit] = []
        for identity in ordered:
            body = representatives[identity]
            if body["target_kind"] == "semantic":
                fact = decode_fact(body)
                citations = citations_by_id[identity]
                content = "\n".join(cited[span] for span in citations)
                hit = MemoryHit(
                    identity,
                    "semantic",
                    scores[identity],
                    content,
                    citations,
                    fact.validity,
                    fact.knowledge,
                    fact.assertion_hash,
                    fact.subject_ref.entity_id,
                    fact.predicate,
                )
            else:
                episode = decode_episode(body)
                span = citations_by_id[identity][0]
                content = cited[span]
                hit = MemoryHit(
                    identity,
                    "episodic",
                    scores[identity],
                    content,
                    (span,),
                    TimeInterval(episode.event_time),
                    TimeInterval(episode.recorded_at),
                )
            hits.append(hit)
        return MemoryRecall(tuple(hits), revision, tuple(warnings))
