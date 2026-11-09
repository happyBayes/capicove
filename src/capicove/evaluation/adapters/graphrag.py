"""GraphRAG adapters: local and global graph queries over a scripted native graph.

The authored corpus is built into a real, validated and activated snapshot by
:func:`capicove.graph_build.native.build_native_graph`; the generator,
tokenizer, embedder and partitioner are deterministic fixtures. The generator
replays the dataset's entity and relation annotations. Ranked items are the exact-quote source citations;
the pool is the public citation list (``max_evidence``).
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from capicove.evaluation.adapters._common import corpus_documents, text_labels, union_labels
from capicove.evaluation.harness import (
    EvalDataset,
    EvalQuery,
    EvaluationError,
    QueryOutcome,
    RankedItem,
    RunOptions,
)


class _OpenGate:
    """Single-owner evaluation gate: one fixed scope, no revocation."""

    revision = 0

    def __init__(self, scope: Any) -> None:
        self.scope = scope

    def require_scope(self, scope: Any) -> None:
        if scope != self.scope:
            raise PermissionError("evaluation scope denied")

    def require_access(self, evidence_refs: tuple[str, ...], scope: Any) -> None:
        self.require_scope(scope)

    @contextlib.contextmanager
    def guard(self, expected_revision: int) -> Iterator[None]:
        if expected_revision != self.revision:
            raise PermissionError("evaluation revision changed")
        yield


class _Sources:
    def __init__(self, values: Mapping[str, Any], gate: _OpenGate) -> None:
        self.values = dict(values)
        self.gate = gate

    def read(self, document_ref: str, scope: Any) -> Any:
        self.gate.require_scope(scope)
        return self.values[document_ref]

    def require_access(self, evidence_refs: tuple[str, ...], scope: Any) -> None:
        self.gate.require_scope(scope)


def graph_annotations(dataset: EvalDataset) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Collect unique authored entities and relations from corpus documents."""
    entities: dict[str, dict[str, str]] = {}
    relations: list[dict[str, str]] = []
    for row in dataset.corpus["documents"]:
        graph = row.get("graph", {"entities": [], "relations": []})
        if not isinstance(graph, dict) or set(graph) != {"entities", "relations"}:
            raise EvaluationError("document graph annotations must list entities and relations")
        for entity in graph["entities"]:
            if set(entity) != {"name", "type"}:
                raise EvaluationError("entity annotations must contain name and type")
            entities.setdefault(entity["name"], dict(entity))
        for relation in graph["relations"]:
            if set(relation) != {"source", "target", "description", "quote"}:
                raise EvaluationError("relation annotations have a closed schema")
            relations.append(dict(relation))
    return list(entities.values()), relations


def build_index(dataset: EvalDataset, workdir: Path) -> tuple[Any, Any, dict[str, Any]]:
    """Build and activate the scripted snapshot; return its verified query index."""
    from capicove.contracts.graph_build import GraphScope
    from capicove.evaluation.graph_fakes import (
        HashedEmbedder,
        ScriptedCommunityPartitioner,
        ScriptedGenerationProfile,
        ScriptedGraphGenerator,
        ScriptedTokenizer,
        hashed_text_space,
    )
    from capicove.graph_build.graphrag import GraphQueryIndex
    from capicove.graph_build.native import build_native_graph, open_native_graph
    from capicove.graph_build.representations import text_representation

    scope = GraphScope("evaluation", "graphrag")
    documents = corpus_documents(dataset)
    representations = {
        row["id"]: text_representation(row["id"], row["text"], scope) for row in documents
    }
    entities, relations = graph_annotations(dataset)
    gate = _OpenGate(scope)
    generator = ScriptedGraphGenerator(entities, relations)
    root = workdir / "graph-evaluation"
    built = build_native_graph(
        root,
        scope=scope,
        representations=tuple(representations.values()),
        gate=gate,
        generation=ScriptedGenerationProfile(),  # type: ignore[arg-type]
        embedder=HashedEmbedder(),
        space=hashed_text_space(),
        generator=generator,
        tokenizer=ScriptedTokenizer(),
        partitioner=ScriptedCommunityPartitioner(),
    )
    if built["receipt"]["status"] != "succeeded":
        raise EvaluationError("the scripted graph build did not succeed")
    catalog, workspace = open_native_graph(root, gate=gate)
    ref, _ = catalog.active(scope)
    if ref is None:
        raise EvaluationError("the scripted graph was not activated")
    index = GraphQueryIndex.open(
        catalog, workspace, gate, _Sources(representations, gate), ref, scope
    )
    return index, generator, built["receipt"]


@dataclass(slots=True)
class _PreparedGraph:
    index: Any
    mode: str
    profile: Any
    identity: Mapping[str, Any]

    def search(self, query: EvalQuery, *, k: int) -> QueryOutcome:
        if query.text is None:
            raise EvaluationError("graph queries require text")
        started = time.perf_counter_ns()
        if self.mode == "local":
            result = self.index.local_search(query.text, self.profile)
        else:
            result = self.index.global_search(query.text, self.profile)
        elapsed = time.perf_counter_ns() - started
        items = []
        for citation in result["evidence"]:
            document = citation["document_ref"]
            key = f"{document}@{citation['start_char']}-{citation['end_char']}"
            items.append(
                RankedItem(
                    key,
                    text_labels(query, document, citation["quote"]),
                    len(citation["quote"]),
                )
            )
        return QueryOutcome(
            tuple(items[:k]),
            len(items),
            union_labels(item.labels for item in items),
            diagnostics={"status": result["status"]},
            search_ns=elapsed,
        )

    def close(self) -> None:
        return None


class GraphRagAdapter:
    """Evaluate one GraphRAG mode over a scripted snapshot of the corpus."""

    adapter_version = "2"

    def __init__(self, stack: str) -> None:
        self.stack = stack
        self.mode = stack.removeprefix("graphrag-")

    def prepare(self, dataset: EvalDataset, workdir: Path, options: RunOptions) -> _PreparedGraph:
        from capicove.evaluation.graph_fakes import (
            FAKE_GENERATOR_MODEL,
            ScriptedCommunityPartitioner,
            hashed_text_space,
        )
        from capicove.graph_build.graphrag import GraphQueryProfile, load_graph_query_profile

        if options.reranker != "none" or options.fake_dense:
            raise EvaluationError("graph adapters accept only a graph query profile")
        profile = (
            GraphQueryProfile()
            if options.profile_path is None
            else load_graph_query_profile(options.profile_path)
        )
        index, generator, receipt = build_index(dataset, workdir)
        return _PreparedGraph(
            index,
            self.mode,
            profile,
            {
                "profile": profile.identity(),
                "index": {
                    "kind": "graph-snapshot",
                    "snapshot_id": index.pin["snapshot_id"],
                    "manifest_sha256": index.pin["manifest_sha256"],
                    "graph_profile_fingerprint": index.pin["graph_profile_fingerprint"],
                    "completeness": receipt["completeness"],
                    "entities": len(index.entities),
                    "relationships": len(index.relationships),
                    "reports": len(index.reports),
                },
                "model": {
                    "kind": "deterministic-fake",
                    "components": [
                        f"generator:{FAKE_GENERATOR_MODEL}",
                        f"embedder:{hashed_text_space().space_id}",
                        f"partitioner:{ScriptedCommunityPartitioner.runtime_identity.runtime_id}",
                    ],
                    "generator_calls": len(generator.calls),
                },
            },
        )


def adapter(stack: str) -> GraphRagAdapter:
    """Return the adapter for ``graphrag-local`` or ``graphrag-global``."""
    return GraphRagAdapter(stack)
