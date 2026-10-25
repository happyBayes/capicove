"""Unified native knowledge lifecycle, retrieval, and evidence-bound answers.

This is a single-owner host composition, not a new model or database contract.
Optional runtimes are loaded only for explicitly selected processing modes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from capicove.runtime.knowledge_config import KnowledgeConfiguration
from capicove.runtime.knowledge_store import KnowledgeStore
from capicove.runtime.media_handoff import MediaChanged, MediaDenied, MediaIntake, MediaInvalid


Answerer = Callable[[str, list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]]
_MAX_SOURCES = 1000
_MAX_CHUNKS = 10000


def digest(value: Any) -> str:
    """Bind immutable JSON data with a deterministic content identity."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def terms(text: str) -> set[str]:
    """Tokenize Latin words and CJK characters without an optional runtime."""
    return set(re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]", text.lower()))


class KnowledgeBase:
    """Coordinate private source versions, atomic activation, and checked answers."""

    def __init__(
        self,
        config: KnowledgeConfiguration,
        *,
        answerer: Answerer | None = None,
        read_only: bool = False,
    ) -> None:
        self.config = config
        self.store = KnowledgeStore(
            config.storage_root,
            digest(
                {
                    "workspace": config.workspace_id,
                    "roots": {alias: str(path) for alias, path in config.source_roots.items()},
                }
            ),
            read_only=read_only,
        )
        self.intake = MediaIntake(
            read_roots=tuple(config.source_roots.values()),
            staging_root=config.storage_root / "staging",
            max_input_bytes=128 * 1024 * 1024,
            accepts=lambda _path, _head: True,
            reject_hardlinks=True,
        )
        self.answerer = answerer
        self.deadline: datetime | None = None
        self.expected_provider: str | None = None

    def __enter__(self) -> KnowledgeBase:
        return self

    def __exit__(self, *args: object) -> None:
        self.store.close()

    def path(self, alias: str, relative: str) -> Path:
        """Resolve a caller's relative name against host-owned source authority."""
        if alias not in self.config.source_roots or not isinstance(relative, str):
            raise ValueError("invalid_source")
        path = Path(relative)
        if (
            not relative
            or path.is_absolute()
            or ".." in path.parts
            or str(path) != relative
            or "\x00" in relative
            or "\\" in relative
        ):
            raise ValueError("invalid_source")
        return self.config.source_roots[alias] / path

    def _check_sources(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            try:
                actual = self.intake.digest(self.path(record["root"], record["path"])).sha256
            except (OSError, MediaChanged, MediaDenied, MediaInvalid):
                raise ValueError("source_changed_or_unavailable") from None
            if actual != record["source_sha256"]:
                raise ValueError("source_changed_or_unavailable")

    def _view(self) -> tuple[int, list[dict[str, Any]], Any]:
        with self.store.transaction():
            graph = self.store.get("graph") if self.config.graph else None
            if graph is not None and self.store.get("graph_digest") != digest(graph):
                raise ValueError("corrupt_knowledge_graph")
            return self.store.get("revision"), self.store.records(), graph

    def _fence(self, revision: int, records: list[dict[str, Any]]) -> None:
        if self.deadline is not None and datetime.now(UTC) >= self.deadline:
            raise TimeoutError("knowledge_deadline_exceeded")
        self._check_sources(records)
        if revision != self.store.get("revision"):
            raise ValueError("knowledge_changed")

    def status(self) -> dict[str, Any]:
        """Report active state separately from the outcome of the latest attempt."""
        revision, records, graph = self._view()
        return {
            "schema_version": 1,
            "revision": revision,
            "generation": self.store.get("generation"),
            "sources": len(records),
            "chunks": sum(len(row["chunks"]) for row in records),
            "graph_status": "active" if graph else ("pending" if self.config.graph else "disabled"),
            "graph": graph.get("receipt") if graph else None,
            "semantic": self.config.models is not None,
            "generation_enabled": self.config.generation is not None,
            "last_run": self.store.last_run(),
        }

    def update(self, alias: str, relative: str) -> dict[str, Any]:
        """Prepare one source and atomically replace the complete corpus generation."""
        self.path(alias, relative)
        return self._update(((alias, relative),))

    def rebuild(self) -> dict[str, Any]:
        """Reprepare every active source with the currently selected model profiles."""
        records = self.store.records()
        return self._update(tuple((row["root"], row["path"]) for row in records))

    def _update(self, selections: tuple[tuple[str, str], ...]) -> dict[str, Any]:
        revision, records, _ = self._view()
        try:
            selected = {f"{alias}:{relative}" for alias, relative in selections}
            kept = [row for row in records if row["source"] not in selected]
            self._check_sources(kept)
            prepared = [self._prepare(alias, relative) for alias, relative in selections]
            candidate = sorted([*kept, *prepared], key=lambda row: row["source"])
            if (
                len(candidate) > _MAX_SOURCES
                or sum(len(row["chunks"]) for row in candidate) > _MAX_CHUNKS
            ):
                raise ValueError("knowledge_capacity_exceeded")
            self._embed(candidate)
            graph = None
            if self.config.graph and candidate:
                from capicove.runtime.knowledge_graph import build_graph

                graph = build_graph(self, candidate, revision)
                if graph["receipt"]["status"] != "succeeded":
                    result = {
                        "status": "partial",
                        "activated": False,
                        "graph_status": graph["receipt"]["status"],
                        "graph": graph["receipt"],
                    }
                    self.store.run(result)
                    return result
            self._fence(revision, candidate)
            with self.store.transaction():
                if self.store.get("revision") != revision:
                    raise ValueError("knowledge_changed")
                self.store.db.execute("DELETE FROM sources")
                for row in candidate:
                    row["version"] = digest(
                        {key: value for key, value in row.items() if key != "version"}
                    )
                    encoded = json.dumps(row, ensure_ascii=False, allow_nan=False)
                    self.store.db.execute(
                        "INSERT OR IGNORE INTO versions VALUES (?,?)", (row["version"], encoded)
                    )
                    self.store.db.execute(
                        "INSERT INTO sources VALUES (?,?)", (row["source"], row["version"])
                    )
                self.store.set("revision", revision + 1)
                self.store.set("generation", self.store.get("generation") + 1)
                self.store.set("graph", graph)
                self.store.set("graph_digest", digest(graph))
            result = {
                "status": "succeeded",
                "activated": True,
                "sources": len(candidate),
                "graph_status": "active" if graph else "disabled",
                "generation": self.store.get("generation"),
            }
            self.store.run(result)
            return result
        except Exception:
            self.store.run(
                {"status": "failed", "activated": False, "code": "knowledge_update_failed"}
            )
            raise

    def _prepare(self, alias: str, relative: str) -> dict[str, Any]:
        from capicove.runtime.knowledge_prepare import prepare_source

        path = self.path(alias, relative)
        try:
            with self.intake.stage((path,)) as staged:
                item = staged[0]
                segments, lineage = prepare_source(self.config, item, self.store.root)
                if not segments or not any(row["text"].strip() for row in segments):
                    raise ValueError("source_has_no_searchable_text")
                if sum(len(row["text"]) for row in segments) > 2_000_000:
                    raise ValueError("prepared_source_too_large")
                source = f"{alias}:{relative}"
                chunks = []
                for segment in segments:
                    text = segment["text"]
                    for start in range(0, len(text), 700):
                        quote = text[start : start + 900]
                        if not quote.strip():
                            continue
                        locator = {key: value for key, value in segment.items() if key != "text"}
                        locator.update({"start_char": start, "end_char": start + len(quote)})
                        evidence_id = "kb:" + digest([source, item.sha256, lineage, locator, quote])
                        chunks.append(
                            {"evidence_id": evidence_id, "quote": quote, "locator": locator}
                        )
                self.store.preserve(item.staged, item.sha256)
                return {
                    "source": source,
                    "root": alias,
                    "path": relative,
                    "source_sha256": item.sha256,
                    "segments": segments,
                    "lineage": lineage,
                    "chunks": chunks,
                    "space": None,
                }
        except (MediaDenied, MediaInvalid, MediaChanged):
            raise ValueError("source_denied_or_invalid") from None

    def _embed(self, records: list[dict[str, Any]]) -> None:
        if self.config.models is None:
            for row in records:
                row["space"] = None
                for chunk in row["chunks"]:
                    chunk.pop("vector", None)
            return
        from capicove.runtime.knowledge_models import encode_texts

        chunks = [chunk for row in records for chunk in row["chunks"]]
        if not chunks:
            return
        space, vectors = encode_texts(
            self.config.models, [chunk["quote"] for chunk in chunks], "document"
        )
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk["vector"] = vector
        for row in records:
            row["space"] = space

    def forget(self, alias: str, relative: str) -> dict[str, Any]:
        """Revoke future access immediately, retaining immutable local history."""
        self.path(alias, relative)
        with self.store.transaction():
            self.store.db.execute("DELETE FROM sources WHERE source=?", (f"{alias}:{relative}",))
            self.store.set("revision", self.store.get("revision") + 1)
            self.store.set("graph", None)
        return {"status": "succeeded", "graph_status": "invalidated"}

    def search(self, question: str, *, limit: int = 8) -> dict[str, Any]:
        """Fuse lexical and configured dense retrieval over currently authorized text."""
        if not isinstance(question, str) or not question.strip() or len(question) > 4000:
            raise ValueError("invalid_question")
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("invalid_search_limit")
        revision, records, graph = self._view()
        self._fence(revision, records)
        query_terms = terms(question)
        candidates: list[dict[str, Any]] = []
        for record in records:
            for chunk in record["chunks"]:
                candidates.append(
                    {
                        **chunk,
                        "source": record["source"],
                        "root": record["root"],
                        "path": record["path"],
                        "source_sha256": record["source_sha256"],
                        "lineage": record["lineage"],
                        "space": record["space"],
                    }
                )
        lexical = sorted(
            (
                (len(query_terms & terms(row["quote"])), index)
                for index, row in enumerate(candidates)
            ),
            reverse=True,
        )
        scores: dict[int, float] = {}
        for rank, (score, index) in enumerate(lexical):
            if score:
                scores[index] = 1 / (60 + rank)
        if self.config.models is not None and candidates:
            from capicove.runtime.knowledge_models import encode_texts

            space, vectors = encode_texts(self.config.models, [question], "query")
            dense = []
            for index, row in enumerate(candidates):
                if row["space"] != space or "vector" not in row:
                    raise ValueError("embedding_profile_changed_rebuild_required")
                vector = row["vector"]
                if len(vector) != len(vectors[0]):
                    raise ValueError("corrupt_embedding")
                score = sum(a * b for a, b in zip(vector, vectors[0], strict=True))
                if not math.isfinite(score):
                    raise ValueError("corrupt_embedding")
                dense.append((score, index))
            for rank, (_, index) in enumerate(sorted(dense, reverse=True)):
                scores[index] = scores.get(index, 0) + 1 / (60 + rank)
        chosen = sorted(
            scores, key=lambda index: (-scores[index], candidates[index]["evidence_id"])
        )[:limit]
        evidence = [
            {
                key: value
                for key, value in candidates[index].items()
                if key not in {"vector", "space"}
            }
            for index in chosen
        ]
        context = []
        if graph:
            context = sorted(
                graph["context"], key=lambda row: -len(query_terms & terms(row["text"]))
            )[:8]
        self._fence(revision, records)
        return {
            "revision": revision,
            "evidence": evidence,
            "graph_context": context,
            "retrieval": "hybrid" if self.config.models else "lexical",
        }

    def evidence(self, evidence_id: str) -> dict[str, Any]:
        """Resolve a live citation to verified original bytes and page/time coordinates."""
        revision, records, _ = self._view()
        for row in records:
            for chunk in row["chunks"]:
                if chunk["evidence_id"] == evidence_id:
                    self._fence(revision, [row])
                    return {
                        **{key: value for key, value in chunk.items() if key != "vector"},
                        "source": row["source"],
                        "source_sha256": row["source_sha256"],
                        "original_path": str(self.path(row["root"], row["path"])),
                        "lineage": row["lineage"],
                    }
        raise ValueError("evidence_unavailable")

    def ask(self, question: str) -> dict[str, Any]:
        """Generate only with explicit configuration, then validate every exact citation."""
        result = self.search(question)
        if not result["evidence"]:
            return {**result, "status": "no_evidence", "claims": []}
        answerer = self.answerer
        if answerer is None and self.config.generation:
            from capicove.runtime.knowledge_answer import generate_answer

            generation_path = self.config.generation

            def answerer(
                question: str, evidence: list[dict[str, Any]], graph: list[dict[str, Any]]
            ) -> dict[str, Any]:
                return generate_answer(
                    generation_path,
                    question,
                    evidence,
                    graph,
                    deadline=self.deadline,
                    expected_provider=self.expected_provider,
                )

        if answerer is None:
            return {**result, "status": "evidence_only", "claims": []}
        revision, records, _ = self._view()
        if revision != result["revision"]:
            raise ValueError("knowledge_changed")
        self._fence(revision, records)
        payload = answerer(question, result["evidence"], result["graph_context"])
        self._fence(revision, records)
        claims = self.validate_answer(payload, result["evidence"])
        return {
            **result,
            "status": "answered" if claims else "insufficient_evidence",
            "claims": claims,
        }

    @staticmethod
    def validate_answer(payload: Any, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Reject fabricated, empty, extraneous, or unsupported citation structures."""
        allowed = {row["evidence_id"]: row for row in evidence}
        if not isinstance(payload, dict) or set(payload) != {"claims"}:
            raise ValueError("invalid_answer")
        claims = payload["claims"]
        if not isinstance(claims, list) or len(claims) > 20:
            raise ValueError("invalid_answer")
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
                raise ValueError("invalid_answer")
            for citation in claim["citations"]:
                if not isinstance(citation, dict) or set(citation) != {"evidence_id", "quote"}:
                    raise ValueError("invalid_answer")
                identity, quote = citation["evidence_id"], citation["quote"]
                if (
                    not isinstance(identity, str)
                    or identity not in allowed
                    or not isinstance(quote, str)
                    or not quote.strip()
                    or quote not in allowed[identity]["quote"]
                ):
                    raise ValueError("invalid_answer")
        return claims


__all__ = ["KnowledgeBase", "KnowledgeConfiguration"]
