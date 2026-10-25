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


def build_graph(kb: KnowledgeBase, records: list[dict[str, Any]], revision: int) -> dict[str, Any]:
    """Use existing models and the full graph pipeline, never a substitute graph."""
    from capicove.contracts import EmbeddingSpaceRef
    from capicove.graph_build.native import build_native_graph
    from capicove.graph_build.remote_generation import load_remote_generation_profile
    from capicove.model import ModelFactory, load_configuration
    from capicove.retrieval import TextEmbedding, TextEmbeddingService

    if kb.config.models is None or kb.config.generation is None:
        raise ValueError("graph_models_not_configured")
    scope = GraphScope(kb.config.workspace_id, "knowledge-private")
    representations = []
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
        representations.append(
            TextRepresentation(row["source"], scope, row["source_sha256"], text, tuple(blocks), ())
        )
    gate = KnowledgeGate(
        kb, records, revision, scope, tuple(row.representation_ref for row in representations)
    )
    # Each attempt has a private namespace. Incomplete graphs can never be
    # discovered by query code, which reads only the atomic corpus pointer.
    root = kb.config.storage_root / ("graph-" + uuid.uuid4().hex)
    with ModelFactory(load_configuration(kb.config.models)) as factory:
        with factory.text_encoder("knowledge.embedding") as encoder:
            service = TextEmbeddingService(encoder, space=encoder.space)

            class Embedder:
                def embed(
                    self, texts: tuple[str, ...], *, space: EmbeddingSpaceRef, role: str
                ) -> tuple[TextEmbedding, ...]:
                    return service.encode(texts, space=space, role=role)

            return build_native_graph(
                root,
                scope=scope,
                representations=tuple(representations),
                gate=gate,
                generation=load_remote_generation_profile(kb.config.generation),
                embedder=Embedder(),
                space=encoder.space,
            )
