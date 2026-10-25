"""Authored public-contract fixtures for the FAISS plugin tests."""

from __future__ import annotations

from collections.abc import Iterable

from capicove.contracts import (
    DistanceMetric,
    EmbeddingSpaceRef,
    ProcessingFingerprint,
    QueryModality,
    SearchableUnit,
    TargetType,
    TemporalRange,
    VectorNormalization,
)
from capicove.retrieval.video_models import (
    VisualIndexEntry,
    VisualIndexGenerationManifest,
    visual_entries_content_sha256,
)


def visual_space(dimension: int = 2) -> EmbeddingSpaceRef:
    """Return one compatible visual embedding space."""
    return EmbeddingSpaceRef(
        space_id=f"clip-v1-{dimension}",
        provider_id="capicove.model.clip",
        model_revision="revision-1",
        preprocessing_version="clip-v1",
        dimension=dimension,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.INNER_PRODUCT,
        query_modalities=frozenset({QueryModality.TEXT, QueryModality.IMAGE}),
        target_modalities=frozenset({TargetType.KEYFRAME, TargetType.VIDEO_SEGMENT}),
    )


def entry(
    unit_id: str,
    vector: tuple[float, ...],
    *,
    target_type: TargetType = TargetType.KEYFRAME,
    artifact_ref: str = "artifact-1",
    timestamp_ms: int = 1_000,
) -> VisualIndexEntry:
    """Return one canonical visual entry."""
    space = visual_space(len(vector))
    locator = (
        {
            "timestamp_ms": timestamp_ms,
            "scene_range": TemporalRange(max(0, timestamp_ms - 1_000), timestamp_ms + 1_000),
        }
        if target_type is TargetType.KEYFRAME
        else {"time_range": TemporalRange(timestamp_ms, timestamp_ms + 1_000)}
    )
    return VisualIndexEntry(
        unit=SearchableUnit(
            unit_id=unit_id,
            unit_type=target_type,
            source_hash="a" * 64,
            artifact_ref=artifact_ref,
            evidence_ref=f"evidence:{unit_id}",
            representation_refs=(f"representation:{unit_id}",),
            processing_run_ref="run-1",
            processing_version="video-v1",
            processing_fingerprint=ProcessingFingerprint("b" * 64),
            embedding_spaces=(space,),
            **locator,
        ),
        vector=vector,
    )


def manifest(
    entries: Iterable[VisualIndexEntry],
    *,
    generation_id: str = "keyframes-1",
    target_type: TargetType = TargetType.KEYFRAME,
) -> VisualIndexGenerationManifest:
    """Return one immutable FAISS generation manifest."""
    materialized = tuple(entries)
    dimension = len(materialized[0].vector) if materialized else 2
    return VisualIndexGenerationManifest(
        generation_id=generation_id,
        workspace_ref="workspace-1",
        corpus_refs=("corpus-1",),
        target_type=target_type,
        searchable_unit_manifest_sha256="c" * 64,
        content_sha256=visual_entries_content_sha256(materialized),
        embedding_space=visual_space(dimension),
        retriever_id="capicove.retriever.faiss",
        index_format_version="faiss-hnsw-flat-ip-v1",
        build_processing_fingerprint=ProcessingFingerprint("d" * 64),
        expected_unit_count=len(materialized),
    )
