"""Bind graph publication and provider egress to the live knowledge source set."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from capicove.contracts.graph_build import GraphScope
from capicove.graph_build.representations import BlockSpan, TextRepresentation


if TYPE_CHECKING:
    from capicove.runtime.native_knowledge import KnowledgeBase


class KnowledgeGate:
    """Fence source revocation through the existing graph access protocol."""

    def __init__(
        self,
        kb: KnowledgeBase,
        records: list[dict[str, Any]],
        revision: int,
        scope: GraphScope,
        refs: tuple[str, ...],
    ) -> None:
        self.kb, self.records, self.expected, self.scope, self.refs = (
            kb,
            records,
            revision,
            scope,
            refs,
        )

    @property
    def revision(self) -> int:
        return int(self.kb.store.get("revision"))

    def require_scope(self, scope: GraphScope) -> None:
        if scope != self.scope:
            raise PermissionError("knowledge_scope_denied")
        self.kb._fence(self.expected, self.records)

    def require_access(self, evidence_refs: tuple[str, ...], scope: GraphScope) -> None:
        self.require_scope(scope)
        if not set(evidence_refs) <= set(self.refs):
            raise PermissionError("knowledge_evidence_denied")

    @contextmanager
    def guard(self, expected_revision: int) -> Iterator[None]:
        with self.kb.store.transaction():
            if expected_revision != self.expected:
                raise PermissionError("knowledge_revision_changed")
            self.require_scope(self.scope)
            yield
            self.require_scope(self.scope)


GRAPH_SCOPE_REF = "knowledge-private"


def representations(kb: KnowledgeBase, records: list[dict[str, Any]]) -> list[TextRepresentation]:
    """Rebuild the exact source representations a native graph was built from."""
    scope = GraphScope(kb.config.workspace_id, GRAPH_SCOPE_REF)
    result = []
    for row in records:
        text = ""
        blocks = []
        for index, segment in enumerate(row["segments"]):
            if index:
                blocks.append(
                    BlockSpan(f"separator:{index}", len(text), len(text) + 2, None, "separator", ())
                )
                text += "\n\n"
            start = len(text)
            text += segment["text"]
            blocks.append(
                BlockSpan(
                    f"segment:{index}", start, len(text), segment.get("page_index"), "paragraph", ()
                )
            )
        result.append(
            TextRepresentation(row["source"], scope, row["source_sha256"], text, tuple(blocks), ())
        )
    return result


def build_graph(kb: KnowledgeBase, records: list[dict[str, Any]], revision: int) -> dict[str, Any]:
    """Use existing models and the full graph pipeline, never a substitute graph."""
    from capicove.contracts import EmbeddingSpaceRef
    from capicove.graph_build.remote_generation import load_remote_generation_profile
    from capicove.model import ModelFactory, load_configuration
    from capicove.retrieval import TextEmbedding, TextEmbeddingService

    if kb.config.models is None or kb.config.generation is None:
        raise ValueError("graph_models_not_configured")
    with ModelFactory(load_configuration(kb.config.models)) as factory:
        with factory.text_encoder("knowledge.embedding") as encoder:
            service = TextEmbeddingService(encoder, space=encoder.space)

            class Embedder:
                def embed(
                    self, texts: tuple[str, ...], *, space: EmbeddingSpaceRef, role: str
                ) -> tuple[TextEmbedding, ...]:
                    return service.encode(texts, space=space, role=role)

            return build_graph_with(
                kb,
                records,
                revision,
                generation=load_remote_generation_profile(kb.config.generation),
                embedder=Embedder(),
                space=encoder.space,
            )


def build_graph_with(
    kb: KnowledgeBase,
    records: list[dict[str, Any]],
    revision: int,
    *,
    generation: Any,
    embedder: Any,
    space: Any,
    generator: Any = None,
    tokenizer: Any = None,
) -> dict[str, Any]:
    """Run the native pipeline with explicit generation, embedding and transport."""
    from capicove.graph_build.native import build_native_graph

    built = representations(kb, records)
    scope = GraphScope(kb.config.workspace_id, GRAPH_SCOPE_REF)
    gate = KnowledgeGate(
        kb, records, revision, scope, tuple(row.representation_ref for row in built)
    )
    # Each attempt has a private namespace. Incomplete graphs can never be
    # discovered by query code, which reads only the atomic corpus pointer.
    root = kb.config.storage_root / ("graph-" + uuid.uuid4().hex)
    return build_native_graph(
        root,
        scope=scope,
        representations=tuple(built),
        gate=gate,
        generation=generation,
        embedder=embedder,
        space=space,
        generator=generator,
        tokenizer=tokenizer,
    )


class _Sources:
    """Serve the rebuilt representations under the knowledge gate's authority."""

    def __init__(self, built: list[TextRepresentation], gate: KnowledgeGate) -> None:
        self.values = {row.document_ref: row for row in built}
        self.gate = gate

    def read(self, document_ref: str, scope: GraphScope) -> TextRepresentation:
        try:
            result = self.values[document_ref]
        except KeyError:
            raise PermissionError("knowledge_evidence_denied") from None
        self.gate.require_access((result.representation_ref,), scope)
        return result

    def require_access(self, evidence_refs: tuple[str, ...], scope: GraphScope) -> None:
        self.gate.require_access(evidence_refs, scope)


def query_graph(
    kb: KnowledgeBase,
    graph: dict[str, Any],
    records: list[dict[str, Any]],
    revision: int,
    question: str,
    *,
    mode: str,
    profile: Any = None,
) -> dict[str, Any]:
    """Open the active native snapshot under the live source gate and query it."""
    from capicove.graph_build.graphrag import GraphQueryIndex
    from capicove.graph_build.native import open_native_graph

    if mode not in {"local", "global"}:
        raise ValueError("invalid_graph_query_mode")
    root_name = graph.get("graph_root")
    if not isinstance(root_name, str) or not root_name.startswith("graph-") or "/" in root_name:
        raise ValueError("graph_root_unrecorded_rebuild_required")
    built = representations(kb, records)
    scope = GraphScope(kb.config.workspace_id, GRAPH_SCOPE_REF)
    gate = KnowledgeGate(
        kb, records, revision, scope, tuple(row.representation_ref for row in built)
    )
    catalog, workspace = open_native_graph(kb.config.storage_root / root_name, gate=gate)
    ref, _ = catalog.active(scope)
    if ref is None or ref.snapshot_id != graph["receipt"]["snapshot_id"]:
        raise ValueError("graph_snapshot_changed")
    index = GraphQueryIndex.open(catalog, workspace, gate, _Sources(built, gate), ref, scope)
    if mode == "local":
        return index.local_search(question, profile)
    return index.global_search(question, profile)
