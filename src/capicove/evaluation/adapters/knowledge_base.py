"""Knowledge-base adapter: native ``KnowledgeBase`` search over authored Markdown.

The default run uses the lexical lane only (no models configured), exactly as
``capicove kb search`` does without a model profile. ``fake_dense`` injects the
deterministic hashed bag-of-words encoder to exercise dense fusion mechanics.
The candidate pool is the deepest public result list (``max_limit``).
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from capicove.evaluation.adapters._common import (
    corpus_documents,
    text_labels,
    union_labels,
    write_documents,
)
from capicove.evaluation.adapters._profiles import resolve_profile
from capicove.evaluation.harness import (
    EvalDataset,
    EvalQuery,
    EvaluationError,
    QueryOutcome,
    RankedItem,
    RunOptions,
)


POOL_DEPTH = 20


@dataclass(slots=True)
class _PreparedKnowledge:
    kb: Any
    documents: Mapping[str, str]
    identity: Mapping[str, Any]
    depth: int

    def search(self, query: EvalQuery, *, k: int) -> QueryOutcome:
        if query.text is None:
            raise EvaluationError("knowledge-base queries require text")
        started = time.perf_counter_ns()
        try:
            result = self.kb.search(query.text, limit=self.depth)
        except (ValueError, RuntimeError) as error:
            return QueryOutcome((), 0, (), status="failed", diagnostics={"error": str(error)})
        elapsed = time.perf_counter_ns() - started
        items = []
        for row in result["evidence"]:
            document = self.documents[row["source"]]
            locator = row["locator"]
            key = f"{document}@{locator['start_char']}-{locator['end_char']}"
            items.append(
                RankedItem(key, text_labels(query, document, row["quote"]), len(row["quote"]))
            )
        diagnostics: dict[str, Any] = {"retrieval": result["retrieval"]}
        if "rerank" in result:
            diagnostics["rerank"] = result["rerank"]
        return QueryOutcome(
            tuple(items[:k]),
            len(items),
            union_labels(item.labels for item in items),
            diagnostics=diagnostics,
            search_ns=elapsed,
        )

    def close(self) -> None:
        self.kb.__exit__(None, None, None)


class KnowledgeBaseAdapter:
    """Build and query a private native knowledge base."""

    stack = "kb"
    adapter_version = "1"

    def prepare(
        self, dataset: EvalDataset, workdir: Path, options: RunOptions
    ) -> _PreparedKnowledge:
        from capicove.runtime.native_knowledge import KnowledgeBase, KnowledgeConfiguration

        documents = corpus_documents(dataset)
        sources = workdir / "sources"
        write_documents(sources, documents)
        config = KnowledgeConfiguration.from_mapping(
            {
                "schema_version": 1,
                "workspace_id": "evaluation",
                "storage_root": str(workdir / "store"),
                "source_roots": {"docs": str(sources)},
                "models": None,
                "generation": None,
                "audio": None,
                "graph": False,
                "network_enabled": False,
            }
        )
        profile, reranker, model = resolve_profile("kb", options)
        keywords: dict[str, Any] = {}
        if profile is not None:
            keywords.update(profile=profile, reranker=reranker)
        if options.fake_dense:
            from capicove.evaluation.fakes import fake_text_encoder

            keywords["text_encoder"] = fake_text_encoder
            model = {
                "kind": "deterministic-fake",
                "components": [*model["components"], "dense:hashed-bag-of-words-64"],
            }
        kb = KnowledgeBase(config, **keywords)
        kb.__enter__()
        try:
            for document in documents:
                outcome = kb.update("docs", document["path"])
                if outcome["status"] != "succeeded":
                    raise EvaluationError("knowledge-base update did not succeed")
            status = kb.status()
        except BaseException:
            kb.__exit__(None, None, None)
            raise
        if profile is None:
            from capicove.retrieval.pipeline_profiles import default_knowledge_profile

            profile_identity = default_knowledge_profile().identity()
        else:
            profile_identity = profile.identity()
        return _PreparedKnowledge(
            kb,
            {f"docs:{document['path']}": document["id"] for document in documents},
            {
                "profile": profile_identity,
                "index": {
                    "kind": "native-knowledge-generation",
                    "generation": status["generation"],
                    "revision": status["revision"],
                    "chunks": status["chunks"],
                },
                "model": model,
            },
            POOL_DEPTH,
        )


def adapter(stack: str) -> KnowledgeBaseAdapter:
    """Return the knowledge-base stack adapter."""
    return KnowledgeBaseAdapter()
