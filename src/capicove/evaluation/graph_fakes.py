"""Scripted, fixture-driven stand-ins for the graph build's generator and embedder.

``ScriptedGraphGenerator`` answers the native extraction and report requests of
:func:`capicove.graph_build.native.build_native_graph` from authored annotations:
an entity is emitted when its name occurs exactly once in the core source, and a
relation when its authored quote occurs exactly once and contains both endpoint
names. Reports restate each supplied source assertion as ``source meaning
target``. It never invents text that is not in the annotations or the request,
and it is not a language model. Every result built with it is a deterministic
engineering fixture.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from capicove.evaluation.fakes import hashed_bag_of_words
from capicove.graph_build.ports import CommunityRuntimeIdentity


if TYPE_CHECKING:
    from capicove.graph_build.communities import WeightedGraph
    from capicove.graph_build.profiles import PartitionProfile


FAKE_GENERATOR_PROVIDER = "capicove-deterministic-fake"
FAKE_GENERATOR_MODEL = "capicove/scripted-graph-generator"


class ScriptedGraphGenerator:
    """Replay authored entities and relations; summarize reports mechanically."""

    provider_id = FAKE_GENERATOR_PROVIDER
    model_id = FAKE_GENERATOR_MODEL
    model_revision = "fake-1"

    def __init__(
        self,
        entities: Sequence[Mapping[str, str]],
        relations: Sequence[Mapping[str, str]],
    ) -> None:
        self.entities = tuple(dict(row) for row in entities)
        self.relations = tuple(dict(row) for row in relations)
        self.calls: list[str] = []

    def generate(self, request: Any, *, control: Any) -> Any:
        """Return one native JSON payload for an extraction or report request."""
        from capicove.graph_build.inference import RawGeneration

        self.calls.append(request.role)
        if request.role == "report":
            value = self._report(json.loads(request.input_segments[0][2]))
        elif request.role == "diagnostic":
            value = {"findings": []}
        else:
            core = next(text for _, role, text in request.input_segments if role == "core")
            value = self._extract(core)
        return RawGeneration(json.dumps(value, ensure_ascii=False).encode(), "stop", 10, 10)

    def _extract(self, core: str) -> dict[str, Any]:
        def unique(text: str) -> bool:
            start = core.find(text)
            return start >= 0 and core.find(text, start + 1) < 0

        entities = []
        for row in self.entities:
            quote = unique_quote(core, row["name"])
            if quote is not None:
                entities.append(
                    {
                        "name": row["name"],
                        "entity_type": row["type"],
                        "description": row.get("description", row["name"]),
                        "quote": quote,
                    }
                )
        names = {row["name"] for row in entities}
        relations = [
            {
                "source": row["source"],
                "target": row["target"],
                "description": row["description"],
                "quote": row["quote"],
                "polarity": "positive",
            }
            for row in self.relations
            if unique(row["quote"])
            and row["source"] in names
            and row["target"] in names
            and row["source"] in row["quote"]
            and row["target"] in row["quote"]
        ]
        return {"entities": entities, "relationships": relations}

    @staticmethod
    def _report(task: Mapping[str, Any]) -> dict[str, Any]:
        statements = [
            {
                "text": f"{row['source']} {row['meaning']} {row['target']}.",
                "references": [row["ref"]],
            }
            for row in task.get("sources", [])[:12]
        ]
        statements.extend(
            {"text": row["text"], "references": [row["ref"]]}
            for row in task.get("children", [])[: 16 - len(statements)]
        )
        names = sorted(
            {row["source"] for row in task.get("sources", [])}
            | {row["target"] for row in task.get("sources", [])}
        )
        title = ", ".join(names)[:180] if names else "Overview of community summaries"
        return {"title": title or "Community", "statements": statements}


def unique_quote(core: str, name: str, *, limit: int = 200) -> str | None:
    """Return the shortest right extension of ``name`` that occurs once in ``core``."""
    start = core.find(name)
    while start >= 0:
        for end in range(start + len(name), min(len(core), start + limit) + 1):
            quote = core[start:end]
            first = core.find(quote)
            if quote.strip() == quote and core.find(quote, first + 1) < 0:
                return quote
        start = core.find(name, start + 1)
    return None


class ScriptedCommunityPartitioner:
    """Use connected components for portable graph fixtures."""

    algorithm_id = "louvain"  # Binding to the native graph's closed profile tag.
    algorithm_version = "capicove-fixture-components-v1"
    runtime_identity = CommunityRuntimeIdentity("capicove/fixture-components", "1", ())

    def partition(
        self, graph: WeightedGraph, profile: PartitionProfile
    ) -> tuple[frozenset[str], ...]:
        """Group each node by its reachable fixture neighbors."""
        neighbors: dict[str, set[str]] = {node: set() for node in graph.node_ids}
        for edge in graph.edges:
            neighbors[edge.source_id].add(edge.target_id)
            neighbors[edge.target_id].add(edge.source_id)
        unseen = set(neighbors)
        groups = []
        for node in graph.node_ids:
            if node not in unseen:
                continue
            unseen.remove(node)
            group = {node}
            frontier = [node]
            while frontier:
                discovered = neighbors[frontier.pop()] & unseen
                unseen.difference_update(discovered)
                group.update(discovered)
                frontier.extend(discovered)
            groups.append(frozenset(group))
        return tuple(groups)


class ScriptedTokenizer:
    """Count request bytes as tokens; a deterministic stand-in, not a tokenizer."""

    tokenizer_id = "capicove/byte-count-tokenizer"
    tokenizer_revision = "fake-1"
    fingerprint = hashlib.sha256(b"capicove/byte-count-tokenizer:fake-1").hexdigest()

    def count(self, text: str) -> int:
        """Count whitespace-delimited words for the graph chunker."""
        return max(1, len(text.split()))

    def count_request(self, request: Any) -> int:
        """Return a bounded deterministic count for one structured request."""
        return max(1, len(request.canonical_payload) // 4)


class ScriptedGenerationProfile:
    """Inference identity for the scripted generator (duck-typed profile)."""

    provider_id = FAKE_GENERATOR_PROVIDER
    model = FAKE_GENERATOR_MODEL
    model_revision = "fake-1"
    context_limit = 32768


class HashedEmbedder:
    """Graph identity/retrieval embedder: hashed bags of words in ``space``."""

    def embed(self, texts: tuple[str, ...], *, space: Any, role: str) -> tuple[Any, ...]:
        """Return one normalized hashed vector per text."""
        from capicove.retrieval import TextEmbedding

        return tuple(
            TextEmbedding(space, hashed_bag_of_words(text, space.dimension)) for text in texts
        )


def hashed_text_space() -> Any:
    """Return the 64-dimension text space declared by :class:`HashedEmbedder`."""
    from capicove.contracts.retrieval import (
        DistanceMetric,
        EmbeddingSpaceRef,
        QueryModality,
        TargetType,
        VectorNormalization,
    )

    return EmbeddingSpaceRef(
        space_id="capicove-deterministic-fake:hashed-text-64",
        provider_id=FAKE_GENERATOR_PROVIDER,
        model_revision="fake-1",
        preprocessing_version="hashed-bow-64-v1",
        dimension=64,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.COSINE,
        query_modalities=frozenset({QueryModality.TEXT}),
        target_modalities=frozenset({TargetType.TEXT_CHUNK}),
    )


__all__ = [
    "FAKE_GENERATOR_MODEL",
    "FAKE_GENERATOR_PROVIDER",
    "HashedEmbedder",
    "ScriptedCommunityPartitioner",
    "ScriptedGenerationProfile",
    "ScriptedGraphGenerator",
    "ScriptedTokenizer",
    "hashed_text_space",
]
