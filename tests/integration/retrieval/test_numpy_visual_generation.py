"""Corruption checks for the persisted NumPy exact visual generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

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
from capicove.retrieval._adapters.numpy_visual import (
    NumpyExactVisualConfiguration,
    NumpyExactVisualRetriever,
)
from capicove.retrieval.video_models import (
    VisualIndexEntry,
    VisualIndexGenerationManifest,
    visual_entries_content_sha256,
)


def test_validation_detects_vector_payload_corruption(tmp_path: Path) -> None:
    space = EmbeddingSpaceRef(
        "clip-v1",
        "provider-1",
        "revision-1",
        "preprocess-1",
        2,
        VectorNormalization.L2,
        DistanceMetric.INNER_PRODUCT,
        frozenset({QueryModality.TEXT}),
        frozenset({TargetType.KEYFRAME}),
    )
    unit = SearchableUnit(
        "keyframe-1",
        TargetType.KEYFRAME,
        "a" * 64,
        "artifact-1",
        "evidence-1",
        ("representation-1",),
        "run-1",
        "video-v1",
        ProcessingFingerprint("b" * 64),
        (space,),
        timestamp_ms=1,
        scene_range=TemporalRange(0, 2),
    )
    entries = (VisualIndexEntry(unit, (1.0, 0.0)),)
    manifest = VisualIndexGenerationManifest(
        "generation-1",
        "workspace-1",
        ("corpus-1",),
        TargetType.KEYFRAME,
        "c" * 64,
        visual_entries_content_sha256(entries),
        space,
        "capicove.retriever.numpy",
        "numpy-exact-v1",
        ProcessingFingerprint("d" * 64),
        1,
    )
    backend = NumpyExactVisualRetriever(NumpyExactVisualConfiguration(tmp_path.resolve()))
    path = Path(backend.generation_lifecycle.create(manifest))
    backend.index_writer.add(index_generation_id=manifest.generation_id, entries=entries)
    assert backend.generation_lifecycle.validate(manifest).valid

    payload = np.load(path / "vectors.npy", mmap_mode="r+")
    payload[0, 0] = 0.0
    payload.flush()
    del payload

    report = backend.generation_lifecycle.validate(manifest)
    assert report.valid is False
    assert "checksum_mismatch" in report.issue_codes


def test_validation_reports_missing_payload_instead_of_raising(tmp_path: Path) -> None:
    space = EmbeddingSpaceRef(
        "clip-v1",
        "provider-1",
        "revision-1",
        "preprocess-1",
        2,
        VectorNormalization.L2,
        DistanceMetric.INNER_PRODUCT,
        frozenset({QueryModality.TEXT}),
        frozenset({TargetType.KEYFRAME}),
    )
    manifest = VisualIndexGenerationManifest(
        "generation-1",
        "workspace-1",
        ("corpus-1",),
        TargetType.KEYFRAME,
        "c" * 64,
        "d" * 64,
        space,
        "capicove.retriever.numpy",
        "numpy-exact-v1",
        ProcessingFingerprint("e" * 64),
        1,
    )
    backend = NumpyExactVisualRetriever(NumpyExactVisualConfiguration(tmp_path.resolve()))
    backend.generation_lifecycle.create(manifest)

    report = backend.generation_lifecycle.validate(manifest)

    assert report.valid is False
    assert "payload_corrupt" in report.issue_codes
