"""Unified adapter: ``unified-local-v1`` shared-space search with a fake encoder.

Authored Markdown documents and synthetic PNG images are imported, built,
activated and searched through the production unified runtime. The only model is
:class:`~capicove.evaluation.fakes.FakeSharedSpaceEncoder` (hashed bags of words;
images by mean palette color), so the numbers measure fusion and lifecycle
mechanics with a fake dense lane, never embedding quality. The pool is the
deepest public result list (``POOL_DEPTH``).
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from capicove.evaluation.adapters._common import (
    corpus_documents,
    corpus_images,
    host_config,
    invocation_context,
    render_image,
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


POOL_DEPTH = 100
PROFILE = "unified-local-v1"
FAKE_PROVIDER = "capicove-deterministic-fake"


def fake_space() -> Any:
    """Return the shared embedding space declared by the fake encoder."""
    from capicove.contracts.retrieval import (
        DistanceMetric,
        EmbeddingSpaceRef,
        QueryModality,
        TargetType,
        VectorNormalization,
    )

    return EmbeddingSpaceRef(
        space_id="capicove-deterministic-fake:hashed-bow-64",
        provider_id=FAKE_PROVIDER,
        model_revision="fake-1",
        preprocessing_version="hashed-bow-64-v1",
        dimension=64,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.COSINE,
        query_modalities=frozenset(
            {QueryModality.TEXT, QueryModality.IMAGE, QueryModality.AUDIO_CLIP}
        ),
        target_modalities=frozenset(
            {TargetType.TEXT_CHUNK, TargetType.IMAGE, TargetType.AUDIO_SEGMENT, TargetType.KEYFRAME}
        ),
    )


def fake_model_factory(root: Path) -> Any:
    """Compose a model factory whose only provider is the deterministic fake."""
    from capicove.evaluation.fakes import FakeSharedSpaceEncoder
    from capicove.model.configuration import ModelCapability, ModelConfiguration, ModelProfile
    from capicove.model.factory import ModelFactory
    from capicove.model.registry import ProviderDefinition, ProviderRegistry

    space = fake_space()
    registry = ProviderRegistry()
    registry.register(
        ProviderDefinition(
            FAKE_PROVIDER,
            "1",
            ModelCapability.MULTIMODAL_EMBEDDING,
            lambda _: FakeSharedSpaceEncoder(space),
        )
    )
    profile = ModelProfile(
        "shared", FAKE_PROVIDER, ModelCapability.MULTIMODAL_EMBEDDING, root, "0" * 64
    )
    return ModelFactory(
        ModelConfiguration({"shared": profile}, {"unified": "shared"}), registry=registry
    )


@dataclass(slots=True)
class _PreparedUnified:
    runtime: Any
    models: Any
    sources: Mapping[str, str]
    identity: Mapping[str, Any]
    _evidence: dict[str, Mapping[str, Any]] = field(default_factory=dict)

    def search(self, query: EvalQuery, *, k: int) -> QueryOutcome:
        from capicove.contracts.retrieval import SearchRequest, TargetType, TextQuery

        if query.text is None:
            raise EvaluationError("unified queries require text")
        targets = frozenset(
            TargetType(value) for value in query.payload.get("targets", ("text_chunk", "image"))
        )
        started = time.perf_counter_ns()
        result = self.runtime.search(
            SearchRequest(
                TextQuery(query.text), targets, top_k=POOL_DEPTH, retrieval_profile_ref=PROFILE
            )
        )
        elapsed = time.perf_counter_ns() - started
        if result.status.value != "succeeded":
            code = None if result.failure_code is None else result.failure_code.value
            return QueryOutcome((), 0, (), status="failed", diagnostics={"error": code})
        items = []
        for hit in result.hits:
            metadata = self._metadata(hit.evidence_id)
            document = self.sources[metadata["source_ref"]]
            if hit.target_type is TargetType.TEXT_CHUNK:
                key = f"{document}@{metadata['start_char']}-{metadata['end_char']}"
                labels = text_labels(query, document, metadata["text"])
                chars: int | None = metadata["end_char"] - metadata["start_char"]
            else:
                key = f"{document}#{hit.target_type.value}"
                labels = text_labels(query, document, None)
                chars = None
            items.append(RankedItem(key, labels, chars))
        diagnostics: dict[str, Any] = {}
        warnings = sorted(code.value for code in result.warning_codes)
        if warnings != ["retrieval_preview"]:
            diagnostics["warning_codes"] = warnings
        return QueryOutcome(
            tuple(items[:k]),
            len(items),
            union_labels(item.labels for item in items),
            diagnostics=diagnostics,
            search_ns=elapsed,
        )

    def _metadata(self, evidence_id: str) -> Mapping[str, Any]:
        cached = self._evidence.get(evidence_id)
        if cached is None:
            cached = dict(
                self.runtime.read_evidence(invocation_context(), evidence_id, ("metadata",))[
                    "metadata"
                ]
            )
            self._evidence[evidence_id] = cached
        return cached

    def close(self) -> None:
        self.runtime.close()
        self.models.close()


class UnifiedAdapter:
    """Build and query the unified shared-space generation."""

    stack = "unified"
    adapter_version = "1"

    def prepare(self, dataset: EvalDataset, workdir: Path, options: RunOptions) -> _PreparedUnified:
        from capicove.runtime import unified_retrieval

        if options.fake_dense:
            raise EvaluationError("the unified adapter always uses its fake shared-space encoder")
        documents = corpus_documents(dataset)
        images = corpus_images(dataset)
        workspace = workdir / "workspace"
        media = workspace / "media"
        workspace.mkdir(mode=0o700)
        write_documents(media, documents)
        for image in images:
            path = media / image["path"]
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            path.write_bytes(render_image(image["spec"]))
        ffmpeg = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
        ffprobe = shutil.which("ffprobe") or "/usr/bin/ffprobe"
        config = host_config(
            workspace,
            {
                "multimodal": {
                    "schema_version": 1,
                    "profile": PROFILE,
                    "source_roots": {"media": str(media)},
                    "index_root": str(workspace / "index"),
                    "staging_root": str(workspace / "staging"),
                    "corpus_ref": f"corpus:{dataset.dataset_id}",
                    "model_config": None,
                    "model_binding": "unified",
                    "ffmpeg": ffmpeg,
                    "ffprobe": ffprobe,
                    "max_source_bytes": 64 * 1024 * 1024,
                }
            },
            ["capicove.search.multimodal"],
        )
        models = fake_model_factory(workdir)
        try:
            for row in (*documents, *images):
                unified_retrieval.import_media(config, "media", row["path"])
            built = unified_retrieval.build_generation(config, models=models)
            unified_retrieval.activate_generation(config, built["generation_id"])
            profile, reranker, model = resolve_profile("unified", options)
            if profile is None:
                from capicove.retrieval.pipeline_profiles import default_unified_profile

                runtime = unified_retrieval.open_unified_retrieval(config, models=models)
                profile_identity = default_unified_profile().identity()
            else:
                runtime = unified_retrieval.open_unified_retrieval(
                    config, models=models, pipeline_profile=profile, reranker=reranker
                )
                profile_identity = profile.identity()
        except BaseException:
            models.close()
            raise
        return _PreparedUnified(
            runtime,
            models,
            {f"media:{row['path']}": row["id"] for row in (*documents, *images)},
            {
                "profile": profile_identity,
                "index": {
                    "kind": "unified-generation",
                    "generation_id": built["generation_id"],
                    "units": built["units"],
                },
                "model": {
                    "kind": "deterministic-fake",
                    "components": [f"shared-space:{fake_space().space_id}", *model["components"]],
                },
            },
        )


def adapter(stack: str) -> UnifiedAdapter:
    """Return the unified stack adapter."""
    return UnifiedAdapter()
