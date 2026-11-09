"""Offline text adapter: the FTS5 workspace profile through canonical Search.

Authored Markdown documents are imported and built exactly as an operator would
with ``capicove source import`` and ``capicove index build``. Queries go through
the canonical Search Tool handler and hits are resolved through the Evidence
handler, so rankings exercise the shipped path. No model is involved.

Each query runs once at the deepest public cutoff (``POOL_DEPTH``); the ranking
is that list's prefix, which equals a direct top-``k`` request because fusion
truncates after ordering.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from capicove.evaluation.adapters._common import (
    corpus_documents,
    host_config,
    invocation_context,
    text_labels,
    union_labels,
    write_documents,
)
from capicove.evaluation.harness import (
    EvalDataset,
    EvalQuery,
    EvaluationError,
    QueryOutcome,
    RankedItem,
    RunOptions,
)


POOL_DEPTH = 100


@dataclass(slots=True)
class _PreparedText:
    runtime: Any
    documents: Mapping[str, str]
    identity: Mapping[str, Any]
    reranking: bool = False
    _evidence: dict[str, Mapping[str, Any]] = field(default_factory=dict)

    def search(self, query: EvalQuery, *, k: int) -> QueryOutcome:
        if query.text is None:
            raise EvaluationError("offline text queries require text")
        context = invocation_context()
        started = time.perf_counter_ns()
        result = self.runtime.search_handler(context, {"query": query.text, "top_k": POOL_DEPTH})
        elapsed = time.perf_counter_ns() - started
        if result.status.value != "succeeded" or result.result is None:
            code = None if result.error is None else result.error.code.value
            return QueryOutcome((), 0, (), status="failed", diagnostics={"error": code})
        items = []
        for hit in result.result["hits"]:
            metadata = self._metadata(context, hit["evidence_id"])
            document = self.documents[metadata["source_ref"]]
            key = f"{document}@{metadata['start_char']}-{metadata['end_char']}"
            items.append(
                RankedItem(
                    key,
                    text_labels(query, document, metadata["text"]),
                    metadata["end_char"] - metadata["start_char"],
                )
            )
        diagnostics: dict[str, Any] = {}
        if result.result.get("degraded"):
            diagnostics["warning_codes"] = sorted(result.result.get("warning_codes", []))
        return QueryOutcome(
            tuple(items[:k]),
            len(items),
            union_labels(item.labels for item in items),
            diagnostics=diagnostics,
            search_ns=elapsed,
        )

    def _metadata(self, context: Any, evidence_id: str) -> Mapping[str, Any]:
        cached = self._evidence.get(evidence_id)
        if cached is None:
            read = self.runtime.evidence_handler(
                context, {"evidence_id": evidence_id, "sections": ["metadata"]}
            )
            if read.status.value != "succeeded" or read.result is None:
                raise EvaluationError("evidence for a returned hit is unavailable")
            cached = read.result["metadata"]
            self._evidence[evidence_id] = cached
        return cached

    def close(self) -> None:
        return None


class OfflineTextAdapter:
    """Build and query the offline FTS5 workspace profile."""

    stack = "offline-text"
    adapter_version = "1"

    def prepare(self, dataset: EvalDataset, workdir: Path, options: RunOptions) -> _PreparedText:
        from capicove.runtime import local_retrieval

        documents = corpus_documents(dataset)
        workspace = workdir / "workspace"
        corpus_root = workdir / "corpus"
        workspace.mkdir(mode=0o700)
        write_documents(corpus_root, documents)
        config = host_config(
            workspace,
            {
                "retrieval": {
                    "schema_version": 1,
                    "source_roots": {"corpus": str(corpus_root)},
                    "index_root": str(workspace / "index"),
                    "corpus_ref": f"corpus:{dataset.dataset_id}",
                    "retrieval_profile_ref": "local-default",
                    "max_source_bytes": 1024 * 1024,
                }
            },
            ["capicove.get_evidence", "capicove.search"],
        )
        for document in documents:
            local_retrieval.import_source(config, alias="corpus", relative_path=document["path"])
        built = local_retrieval.build_index(config)
        profile, reranker, model = _search_policy(options)
        if profile is None:
            runtime = local_retrieval.open_local_retrieval(config)
            profile_identity = {
                "id": "offline-text-default-v1",
                "version": 1,
                "fingerprint": _default_fingerprint(),
            }
        else:
            runtime = local_retrieval.open_local_retrieval(
                config, pipeline_profile=profile, reranker=reranker
            )
            profile_identity = profile.identity()
        return _PreparedText(
            runtime,
            {f"corpus/{document['path']}": document["id"] for document in documents},
            {
                "profile": profile_identity,
                "index": {
                    "kind": "local-fts5-generation",
                    "generation_id": built["generation_id"],
                    "manifest_sha256": built.get("manifest_sha256"),
                },
                "model": model,
            },
        )


def _default_fingerprint() -> str:
    from capicove.retrieval.pipeline_profiles import default_offline_text_profile

    return default_offline_text_profile().fingerprint


def _search_policy(options: RunOptions) -> tuple[Any, Any, dict[str, Any]]:
    """Resolve an optional profile file and the deterministic fake reranker."""
    from capicove.evaluation.adapters._profiles import resolve_profile

    return resolve_profile("offline-text", options)


def adapter(stack: str) -> OfflineTextAdapter:
    """Return the offline text stack adapter."""
    return OfflineTextAdapter()
