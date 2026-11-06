"""Experimental GraphRAG local and global queries over one verified graph snapshot.

Status: experimental preview (RFC-0035, Proposed). Nothing here changes how graphs
are built, published or activated.

``GraphQueryIndex.open`` reads one published snapshot through the catalog's full
revalidation, recomputes its manifest digest, verifies every committed text-unit
segment against the authorized source representation, and keeps the resolved
analysis view in memory. Results are pinned to the snapshot ID, the manifest
SHA-256, the graph-build profile fingerprint and the query profile fingerprint.

* Local mode matches query terms to resolved entity names and aliases, expands a
  bounded neighborhood over resolved relationships, and returns the original
  source sentences behind the selected relationships as exact-quote citations.
* Global mode ranks community reports by term overlap and returns the top reports
  with their statements, each linked to the exact source quotes of the
  assertions it cites (following child-report statements).

Derived entity and report text is navigation context, never evidence; only
source quotes are citable. Answer generation is opt-in through an explicit
answerer callable, and every cited quote is validated as an exact substring of a
returned citation.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from capicove.contracts.graph_build import (
    CommunityReport,
    GraphScope,
    GraphSnapshotRef,
    RelationshipAssertion,
    SourceSpan,
)
from capicove.graph_build.codec import canonical_bytes, fingerprint, manifest_bytes
from capicove.graph_build.ports import AuthorizedRepresentationReader
from capicove.graph_build.snapshots import GraphSnapshotCatalog, SnapshotAccessGate
from capicove.processing import LocalProcessingWorkspace


GRAPH_QUERY_PROFILE_SCHEMA = "capicove.graph-query-profile.v1"
GRAPH_QUERY_RESULT_SCHEMA = "capicove.graphrag-result.v1"
_TABLES = (
    "documents",
    "text_units",
    "entity_mentions",
    "relationship_mentions",
    "resolved_entities",
    "resolved_relationships",
    "communities",
    "community_reports",
)
_TOKEN = re.compile(r"[a-z0-9]+|[㐀-鿿]")
_STOPWORDS = frozenset(
    "a an and are as at be by did do does for from has have how in is it of on or that the "
    "this to was were what when where which who whom whose why with".split()
)
_PROFILE_ID = re.compile(r"^[a-z][a-z0-9.-]{0,63}$")
Answerer = Callable[[str, list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]]


class GraphQueryError(ValueError):
    """Report a stable, path-free graph query failure."""


def query_terms(text: str) -> frozenset[str]:
    """Return casefolded content terms with a light deterministic suffix strip."""
    terms = set()
    for token in _TOKEN.findall(text.casefold()):
        if token in _STOPWORDS:
            continue
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        else:
            for suffix in ("ing", "ed", "es", "s"):
                if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                    token = token[: -len(suffix)]
                    break
        terms.add(token)
    return frozenset(terms)


def _bounded(name: str, value: object, maximum: int) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise GraphQueryError(f"{name} must be an integer from 1 to {maximum}")


@dataclass(frozen=True, slots=True)
class GraphQueryProfile:
    """Versioned bounds for local and global graph queries."""

    profile_id: str = "graphrag-default-v1"
    version: int = 1
    max_seed_entities: int = 5
    min_entity_match: float = 0.5
    max_hops: int = 2
    max_neighborhood_entities: int = 48
    max_relationships: int = 96
    max_evidence: int = 20
    max_reports: int = 5
    max_child_depth: int = 3
    bridge_same_name: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not _PROFILE_ID.fullmatch(self.profile_id):
            raise GraphQueryError("profile_id must be a lowercase identifier")
        if type(self.version) is not int or self.version != 1:
            raise GraphQueryError("version must be 1")
        _bounded("max_seed_entities", self.max_seed_entities, 64)
        if type(self.min_entity_match) not in (int, float) or not 0 < self.min_entity_match <= 1:
            raise GraphQueryError("min_entity_match must be in (0, 1]")
        object.__setattr__(self, "min_entity_match", float(self.min_entity_match))
        _bounded("max_hops", self.max_hops, 4)
        _bounded("max_neighborhood_entities", self.max_neighborhood_entities, 1_000)
        _bounded("max_relationships", self.max_relationships, 2_000)
        _bounded("max_evidence", self.max_evidence, 100)
        _bounded("max_reports", self.max_reports, 50)
        _bounded("max_child_depth", self.max_child_depth, 8)
        if type(self.bridge_same_name) is not bool:
            raise GraphQueryError("bridge_same_name must be a boolean")

    def to_mapping(self) -> dict[str, Any]:
        """Return the closed profile document."""
        return {
            "schema": GRAPH_QUERY_PROFILE_SCHEMA,
            **{item.name: getattr(self, item.name) for item in fields(self)},
        }

    @classmethod
    def from_mapping(cls, raw: object) -> GraphQueryProfile:
        """Decode a closed profile document."""
        names = {item.name for item in fields(cls)}
        if not isinstance(raw, Mapping) or set(raw) != names | {"schema"}:
            raise GraphQueryError("graph query profile must contain exactly the documented fields")
        if raw["schema"] != GRAPH_QUERY_PROFILE_SCHEMA:
            raise GraphQueryError("unsupported graph query profile schema")
        return cls(**{name: raw[name] for name in names})

    @property
    def fingerprint(self) -> str:
        """Return the SHA-256 of the canonical profile document."""
        return hashlib.sha256(canonical_bytes(self.to_mapping())).hexdigest()

    def identity(self) -> dict[str, Any]:
        """Describe this profile for result pinning."""
        return {"id": self.profile_id, "version": self.version, "fingerprint": self.fingerprint}


def load_graph_query_profile(path: Path) -> GraphQueryProfile:
    """Read a bounded profile file that rejects duplicate keys."""
    payload = Path(path).read_bytes()
    if len(payload) > 65_536:
        raise GraphQueryError("graph query profile exceeds 64 KiB")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise GraphQueryError("graph query profile contains a duplicate key")
            result[key] = value
        return result

    try:
        raw = json.loads(payload, object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise GraphQueryError("graph query profile is not valid JSON") from None
    try:
        return GraphQueryProfile.from_mapping(raw)
    except TypeError:
        raise GraphQueryError("graph query profile fields have invalid types") from None


@dataclass(frozen=True, slots=True)
class GraphCitation:
    """One exact quote of original source text, located by half-open offsets."""

    citation_id: str
    document_ref: str
    evidence_ref: str
    representation_ref: str
    text_unit_id: str
    start_char: int
    end_char: int
    quote: str
    page_index: int | None

    def to_mapping(self) -> dict[str, Any]:
        """Return the JSON citation."""
        return {item.name: getattr(self, item.name) for item in fields(self)}


class GraphQueryIndex:
    """Immutable, verified in-memory view of one graph snapshot."""

    def __init__(
        self,
        pin: dict[str, Any],
        scope: GraphScope,
        tables: Mapping[str, tuple[Any, ...]],
        texts: Mapping[str, str],
    ) -> None:
        self.pin = pin
        self.scope = scope
        self._tables = tables
        self._texts = texts
        self.entities = {row.entity_id: row for row in tables["resolved_entities"]}
        self.relationships = {row.assertion_id: row for row in tables["resolved_relationships"]}
        self.mentions = {
            row.mention_id: row
            for role in ("entity_mentions", "relationship_mentions")
            for row in tables[role]
        }
        self.units = {row.text_unit_id: row for row in tables["text_units"]}
        self.documents = {row.representation_ref: row for row in tables["documents"]}
        self.reports = {row.report_id: row for row in tables["community_reports"]}
        self.communities = {row.community_id: row for row in tables["communities"]}
        adjacency: dict[str, list[RelationshipAssertion]] = {}
        for row in sorted(self.relationships.values(), key=lambda item: item.assertion_id):
            adjacency.setdefault(row.source_entity_id, []).append(row)
            adjacency.setdefault(row.target_entity_id, []).append(row)
        self._adjacency = adjacency
        groups: dict[tuple[str, str], list[str]] = {}
        for entity in sorted(self.entities.values(), key=lambda item: item.entity_id):
            groups.setdefault((entity.display_name.casefold(), entity.entity_type), []).append(
                entity.entity_id
            )
        self._same_name = {
            entity_id: tuple(group) for group in groups.values() for entity_id in group
        }

    @classmethod
    def open(
        cls,
        catalog: GraphSnapshotCatalog,
        workspace: LocalProcessingWorkspace,
        gate: SnapshotAccessGate,
        sources: AuthorizedRepresentationReader,
        graph_ref: GraphSnapshotRef,
        scope: GraphScope,
    ) -> GraphQueryIndex:
        """Verify the snapshot, its tables and every committed source segment."""
        from capicove.graph_build._adapters.parquet import ParquetTableCodec

        gate.require_scope(scope)
        manifest = catalog.get(graph_ref.snapshot_id, scope)
        digest = hashlib.sha256(manifest_bytes(manifest)).hexdigest()
        if digest != graph_ref.manifest_sha256:
            raise GraphQueryError("graph manifest digest mismatch")
        tables: dict[str, tuple[Any, ...]] = {}
        for table in manifest.tables:
            if table.role not in _TABLES:
                continue

            def read(path: Path, table: Any = table) -> bool:
                tables[table.role] = ParquetTableCodec(
                    publication_identity=table.dataset_ref.processing_fingerprint
                ).read(path / "part-00000.parquet", table.role)
                return True

            if workspace.validate_snapshot(table.dataset_ref.snapshot_id, validator=read) != (
                table.dataset_ref
            ):
                raise GraphQueryError("graph table commitment mismatch")
        if set(tables) != set(_TABLES):
            raise GraphQueryError("graph snapshot lacks a required table")
        texts: dict[str, str] = {}
        for document in tables["documents"]:
            gate.require_access((document.evidence_ref,), scope)
            representation = sources.read(document.document_ref, scope)
            if (
                representation.scope != scope
                or representation.document_ref != document.document_ref
                or representation.source_sha256 != document.source_sha256
                or representation.representation_ref != document.representation_ref
            ):
                raise GraphQueryError("source representation commitment mismatch")
            texts[document.representation_ref] = representation.text
        for unit in tables["text_units"]:
            for segment in unit.segments:
                exact = "".join(
                    texts[span.representation_ref][span.start_char : span.end_char]
                    for span in segment.source_spans
                )
                if exact != segment.text:
                    raise GraphQueryError("source text differs from committed text unit")
        pin = {
            "snapshot_id": graph_ref.snapshot_id,
            "manifest_sha256": digest,
            "graph_profile_fingerprint": manifest.profile_fingerprint,
            "scope": {"workspace_ref": scope.workspace_ref, "scope_ref": scope.scope_ref},
        }
        return cls(pin, scope, tables, texts)

    def citations_for(self, graph_id: str) -> tuple[GraphCitation, ...]:
        """Resolve a mention or assertion ID to the exact source quotes behind it."""
        if graph_id in self.mentions:
            mention = self.mentions[graph_id]
            return tuple(
                self._citation(span, mention.text_unit_id) for span in mention.source_spans
            )
        if graph_id in self.relationships:
            result: list[GraphCitation] = []
            for ref in self.relationships[graph_id].evidence_refs:
                if ref in self.mentions:
                    result.extend(self.citations_for(ref))
            return _unique(result)
        if graph_id in self.entities:
            result = []
            for mention_id in self.entities[graph_id].mention_ids:
                result.extend(self.citations_for(mention_id))
            return _unique(result)
        raise GraphQueryError("graph ID has no source lineage in this snapshot")

    def _citation(self, span: SourceSpan, text_unit_id: str) -> GraphCitation:
        text = self._texts.get(span.representation_ref)
        document = self.documents.get(span.representation_ref)
        if (
            text is None
            or document is None
            or not 0 <= span.start_char < span.end_char <= len(text)
        ):
            raise GraphQueryError("citation span is outside its source representation")
        quote = text[span.start_char : span.end_char]
        identity = fingerprint(
            [
                self.pin["snapshot_id"],
                span.evidence_ref,
                span.representation_ref,
                span.start_char,
                span.end_char,
            ]
        )
        return GraphCitation(
            "graphcite:" + identity,
            document.document_ref,
            span.evidence_ref,
            span.representation_ref,
            text_unit_id,
            span.start_char,
            span.end_char,
            quote,
            span.page_index,
        )

    def _names(self, entity_id: str) -> str:
        entity = self.entities.get(entity_id)
        return "entity" if entity is None else entity.display_name

    def _meaning(self, row: RelationshipAssertion) -> str:
        return dict(row.qualifiers).get("raw_predicate", row.predicate)

    def match_entities(self, question: str, profile: GraphQueryProfile) -> list[dict[str, Any]]:
        """Score resolved entities by the fraction of their name terms in the query."""
        wanted = query_terms(question)
        scored = []
        for entity in self.entities.values():
            best: tuple[float, int] = (0.0, 0)
            for name in (entity.display_name, *entity.aliases):
                terms = query_terms(name)
                if not terms:
                    continue
                matched = len(terms & wanted)
                best = max(best, (matched / len(terms), matched))
            if best[1] and best[0] >= profile.min_entity_match:
                scored.append((best, entity))
        scored.sort(key=lambda item: (-item[0][0], -item[0][1], item[1].entity_id))
        seeds: list[dict[str, Any]] = []
        names: set[tuple[str, str]] = set()
        for score, entity in scored:
            # Count distinct names so same-name duplicates cannot crowd out seeds.
            name = (entity.display_name.casefold(), entity.entity_type)
            if name not in names and len(names) >= profile.max_seed_entities:
                continue
            names.add(name)
            seeds.append(
                {
                    "entity_id": entity.entity_id,
                    "name": entity.display_name,
                    "entity_type": entity.entity_type,
                    "match": round(score[0], 6),
                }
            )
        return seeds

    def local_search(
        self, question: str, profile: GraphQueryProfile | None = None
    ) -> dict[str, Any]:
        """Match entities, expand a bounded neighborhood and cite its source quotes."""
        profile = profile or GraphQueryProfile()
        _question(question)
        wanted = query_terms(question)
        seeds = self.match_entities(question, profile)
        hops: dict[str, int] = {}
        bridged: dict[str, str] = {}

        def visit(entity_id: str, depth: int, found: list[str]) -> None:
            # Resolution never merges same-name entities without identity
            # evidence; the optional bridge links them at query time only.
            group = self._same_name[entity_id] if profile.bridge_same_name else (entity_id,)
            for member in group:
                if member in hops or len(hops) >= profile.max_neighborhood_entities:
                    continue
                hops[member] = depth
                found.append(member)
                if member != entity_id:
                    bridged[member] = entity_id

        frontier: list[str] = []
        for seed in seeds:
            visit(seed["entity_id"], 0, frontier)
        frontier.sort()
        for depth in range(1, profile.max_hops + 1):
            following: list[str] = []
            for entity_id in frontier:
                for row in self._adjacency.get(entity_id, ()):
                    for other in (row.source_entity_id, row.target_entity_id):
                        visit(other, depth, following)
            frontier = sorted(following)
        selected = [
            row
            for row in self.relationships.values()
            if row.source_entity_id in hops and row.target_entity_id in hops
        ]
        own = {
            row.assertion_id: len(
                query_terms(
                    " ".join(
                        (
                            self._names(row.source_entity_id),
                            self._meaning(row),
                            self._names(row.target_entity_id),
                        )
                    )
                )
                & wanted
            )
            for row in selected
        }
        seed_ids = {seed["entity_id"] for seed in seeds}
        relations = []
        for row in selected:
            # Path relevance: add the best overlap of a relationship that continues
            # this one through a non-seed endpoint, so chains toward matched facts
            # outrank seed-adjacent dead ends.
            chained = 0
            for endpoint in (row.source_entity_id, row.target_entity_id):
                if endpoint in seed_ids:
                    continue
                group = self._same_name[endpoint] if profile.bridge_same_name else (endpoint,)
                for member in group:
                    for neighbor in self._adjacency.get(member, ()):
                        if (
                            neighbor.assertion_id != row.assertion_id
                            and neighbor.assertion_id in own
                        ):
                            chained = max(chained, own[neighbor.assertion_id])
            distance = min(hops[row.source_entity_id], hops[row.target_entity_id])
            score = own[row.assertion_id] + chained
            relations.append((-score, distance, -own[row.assertion_id], row.assertion_id, row))
        relations.sort(key=lambda item: item[:4])
        relations = relations[: profile.max_relationships]
        candidates: list[tuple[int, int, GraphCitation]] = []
        relationship_rows = []
        for order, (negative, distance, _, assertion_id, row) in enumerate(relations):
            cited = self.citations_for(assertion_id)
            relationship_rows.append(
                {
                    "assertion_id": assertion_id,
                    "source": self._names(row.source_entity_id),
                    "meaning": self._meaning(row),
                    "target": self._names(row.target_entity_id),
                    "hops": distance,
                    "path_score": -negative,
                    "epistemic_status": row.epistemic_status,
                    "citations": [citation.citation_id for citation in cited],
                }
            )
            for position, citation in enumerate(cited):
                candidates.append((order, position, citation))
        evidence = _unique(item[2] for item in sorted(candidates, key=lambda item: item[:2]))[
            : profile.max_evidence
        ]
        communities = sorted(
            {
                community.community_id
                for community in self.communities.values()
                if any(seed["entity_id"] in community.member_ids for seed in seeds)
            }
        )
        reports = [
            {
                "report_id": report.report_id,
                "title": report.title,
                "community_id": report.community_id,
            }
            for report in sorted(self.reports.values(), key=lambda item: item.report_id)
            if report.community_id in communities
        ][: profile.max_reports]
        return self._result(
            "local",
            question,
            profile,
            {
                "status": "succeeded" if seeds else "no_entity_match",
                "seeds": seeds,
                "entities": [
                    {
                        "entity_id": entity_id,
                        "name": self._names(entity_id),
                        "entity_type": self.entities[entity_id].entity_type,
                        "hops": depth,
                        "bridged_from": bridged.get(entity_id),
                    }
                    for entity_id, depth in sorted(
                        hops.items(), key=lambda item: (item[1], item[0])
                    )
                ],
                "relationships": relationship_rows,
                "reports": reports,
                "evidence": [citation.to_mapping() for citation in evidence],
            },
        )

    def global_search(
        self, question: str, profile: GraphQueryProfile | None = None
    ) -> dict[str, Any]:
        """Rank community reports and return them with their cited source quotes."""
        profile = profile or GraphQueryProfile()
        _question(question)
        wanted = query_terms(question)
        scored = []
        for report in self.reports.values():
            terms = query_terms(" ".join((report.title, *(s.text for s in _statements(report)))))
            overlap = len(terms & wanted)
            if overlap:
                scored.append(
                    (-overlap, report.report_kind != "community", report.report_id, report)
                )
        scored.sort(key=lambda item: item[:3])
        reports = []
        candidates: list[tuple[int, int, int, int, GraphCitation]] = []
        for report_rank, (negative, _, _, report) in enumerate(scored[: profile.max_reports]):
            statements = []
            for position, (section, statement) in enumerate(_sections(report)):
                cited = _unique(self._statement_citations(statement, profile.max_child_depth))
                overlap = len(query_terms(statement.text) & wanted)
                candidates.extend(
                    (-overlap, report_rank, position, order, citation)
                    for order, citation in enumerate(cited)
                )
                statements.append(
                    {
                        "statement_id": statement.statement_id,
                        "section": section,
                        "kind": statement.kind,
                        "text": statement.text,
                        "citations": [citation.citation_id for citation in cited],
                    }
                )
            reports.append(
                {
                    "report_id": report.report_id,
                    "report_kind": report.report_kind,
                    "community_id": report.community_id,
                    "title": report.title,
                    "term_overlap": -negative,
                    "statements": statements,
                }
            )
        return self._result(
            "global",
            question,
            profile,
            {
                "status": "succeeded" if reports else "no_report_match",
                "reports": reports,
                # Statements most related to the question lead; report rank breaks ties.
                "evidence": [
                    citation.to_mapping()
                    for citation in _unique(
                        item[4] for item in sorted(candidates, key=lambda item: item[:4])
                    )[: profile.max_evidence]
                ],
            },
        )

    def _statement_citations(self, statement: Any, depth: int) -> list[GraphCitation]:
        cited: list[GraphCitation] = []
        for assertion_id in statement.assertion_refs:
            if assertion_id in self.relationships:
                cited.extend(self.citations_for(assertion_id))
        if depth > 0:
            for child in statement.child_statement_refs:
                report = self.reports.get(child.report_id)
                if report is None:
                    continue
                for _, candidate in _sections(report):
                    if candidate.statement_id == child.statement_id:
                        cited.extend(self._statement_citations(candidate, depth - 1))
        return cited

    def _result(
        self, mode: str, question: str, profile: GraphQueryProfile, body: dict[str, Any]
    ) -> dict[str, Any]:
        result = {
            "schema": GRAPH_QUERY_RESULT_SCHEMA,
            "experimental": True,
            "mode": mode,
            "question": question,
            "pin": {**self.pin, "query_profile": profile.identity()},
            **body,
        }
        result["result_sha256"] = hashlib.sha256(canonical_bytes(result)).hexdigest()
        return result


def answer_graph_result(result: Mapping[str, Any], answerer: Answerer | None) -> dict[str, Any]:
    """Generate only through an explicit answerer and validate every exact quote.

    The answerer receives the question, the citations as evidence rows
    (``evidence_id`` and ``quote``) and derived graph context. It must return
    ``{"claims": [{"text", "citations": [{"evidence_id", "quote"}]}]}``.
    """
    evidence = [
        {
            "evidence_id": row["citation_id"],
            "quote": row["quote"],
            "source": row["document_ref"],
        }
        for row in result["evidence"]
    ]
    context = [
        {"kind": "relationship", "text": f"{row['source']} {row['meaning']} {row['target']}"}
        for row in result.get("relationships", [])
    ] + [
        {"kind": "report", "text": statement["text"]}
        for report in result.get("reports", [])
        for statement in report.get("statements", [])
    ]
    base = {"pin": result["pin"], "mode": result["mode"], "result_sha256": result["result_sha256"]}
    if not evidence:
        return {**base, "status": "no_evidence", "claims": []}
    if answerer is None:
        return {**base, "status": "evidence_only", "claims": []}
    claims = validate_cited_answer(answerer(result["question"], evidence, context), evidence)
    return {**base, "status": "answered" if claims else "insufficient_evidence", "claims": claims}


def validate_cited_answer(payload: Any, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reject fabricated, empty, extraneous or unsupported citations."""
    allowed = {row["evidence_id"]: row["quote"] for row in evidence}
    if not isinstance(payload, dict) or set(payload) != {"claims"}:
        raise GraphQueryError("invalid_answer")
    claims = payload["claims"]
    if not isinstance(claims, list) or len(claims) > 20:
        raise GraphQueryError("invalid_answer")
    for claim in claims:
        if (
            not isinstance(claim, dict)
            or set(claim) != {"text", "citations"}
            or not isinstance(claim["text"], str)
            or not claim["text"].strip()
            or len(claim["text"]) > 4000
            or not isinstance(claim["citations"], list)
            or not 1 <= len(claim["citations"]) <= 8
        ):
            raise GraphQueryError("invalid_answer")
        for citation in claim["citations"]:
            if (
                not isinstance(citation, dict)
                or set(citation) != {"evidence_id", "quote"}
                or citation["evidence_id"] not in allowed
                or not isinstance(citation["quote"], str)
                or not citation["quote"].strip()
                or citation["quote"] not in allowed[citation["evidence_id"]]
            ):
                raise GraphQueryError("invalid_answer")
    return claims


def _question(question: object) -> None:
    if not isinstance(question, str) or not question.strip() or len(question) > 4000:
        raise GraphQueryError("invalid_question")


def _sections(report: CommunityReport) -> list[tuple[str, Any]]:
    return [
        (item.name, statement)
        for item in fields(report.sections)
        for statement in getattr(report.sections, item.name)
    ]


def _statements(report: CommunityReport) -> list[Any]:
    return [statement for _, statement in _sections(report)]


def _unique(citations: Iterable[GraphCitation]) -> tuple[GraphCitation, ...]:
    seen: set[str] = set()
    result = []
    for citation in citations:
        if citation.citation_id not in seen:
            seen.add(citation.citation_id)
            result.append(citation)
    return tuple(result)


__all__ = [
    "GRAPH_QUERY_PROFILE_SCHEMA",
    "GRAPH_QUERY_RESULT_SCHEMA",
    "GraphCitation",
    "GraphQueryError",
    "GraphQueryIndex",
    "GraphQueryProfile",
    "answer_graph_result",
    "load_graph_query_profile",
    "query_terms",
    "validate_cited_answer",
]
