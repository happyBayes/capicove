"""Public-contract fixtures for the opt-in native FAISS integration test."""

from __future__ import annotations

from dataclasses import replace

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


def _space() -> EmbeddingSpaceRef:
    return EmbeddingSpaceRef(
        "clip-v1-2",
        "capicove.model.clip",
        "revision-1",
        "clip-v1",
        2,
        VectorNormalization.L2,
        DistanceMetric.INNER_PRODUCT,
        frozenset({QueryModality.TEXT, QueryModality.IMAGE}),
        frozenset({TargetType.KEYFRAME, TargetType.VIDEO_SEGMENT}),
    )


def entry(unit_id: str, vector: tuple[float, ...]) -> VisualIndexEntry:
    """Return one native-integration keyframe vector."""
    return VisualIndexEntry(
        SearchableUnit(
            unit_id=unit_id,
            unit_type=TargetType.KEYFRAME,
            source_hash="a" * 64,
            artifact_ref="artifact-1",
            evidence_ref=f"evidence:{unit_id}",
            representation_refs=(f"representation:{unit_id}",),
            processing_run_ref="run-1",
            processing_version="video-v1",
            processing_fingerprint=ProcessingFingerprint("b" * 64),
            embedding_spaces=(
                replace(_space(), dimension=len(vector), space_id=f"clip-v1-{len(vector)}"),
            ),
            timestamp_ms=1_000,
            scene_range=TemporalRange(0, 2_000),
        ),
        vector,
    )


def manifest(
    entries: tuple[VisualIndexEntry, ...],
) -> VisualIndexGenerationManifest:
    """Return the matching native-integration generation manifest."""
    return VisualIndexGenerationManifest(
        generation_id="keyframes-1",
        workspace_ref="workspace-1",
        corpus_refs=("corpus-1",),
        target_type=TargetType.KEYFRAME,
        searchable_unit_manifest_sha256="c" * 64,
        content_sha256=visual_entries_content_sha256(entries),
        embedding_space=entries[0].unit.embedding_spaces[0],
        retriever_id="capicove.retriever.faiss",
        index_format_version="faiss-hnsw-flat-ip-v1",
        build_processing_fingerprint=ProcessingFingerprint("d" * 64),
        expected_unit_count=len(entries),
    )
