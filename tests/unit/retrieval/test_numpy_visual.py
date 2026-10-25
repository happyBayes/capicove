"""Tests for the exact immutable NumPy visual retriever."""

from __future__ import annotations

from pathlib import Path

import pytest

from capicove.contracts import (
    DistanceMetric,
    EmbeddingSpaceRef,
    ProcessingFingerprint,
    QueryModality,
    SearchableUnit,
    SearchFilters,
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


def _space() -> EmbeddingSpaceRef:
    return EmbeddingSpaceRef(
        "clip-v1",
        "capicove.model.clip",
        "revision-1",
        "clip-v1",
        2,
        VectorNormalization.L2,
        DistanceMetric.INNER_PRODUCT,
        frozenset({QueryModality.TEXT, QueryModality.IMAGE}),
        frozenset({TargetType.KEYFRAME, TargetType.VIDEO_SEGMENT}),
    )


def _entry(
    unit_id: str,
    vector: tuple[float, float],
    *,
    artifact_ref: str = "artifact-1",
) -> VisualIndexEntry:
    return VisualIndexEntry(
        SearchableUnit(
            unit_id=unit_id,
            unit_type=TargetType.KEYFRAME,
            source_hash="a" * 64,
            artifact_ref=artifact_ref,
            evidence_ref=f"evidence:{unit_id}",
            representation_refs=(f"representation:{unit_id}",),
            processing_run_ref="run-1",
            processing_version="video-v1",
            processing_fingerprint=ProcessingFingerprint("b" * 64),
            embedding_spaces=(_space(),),
            timestamp_ms=1_000,
            scene_range=TemporalRange(0, 2_000),
        ),
        vector,
    )


def _manifest(entries: tuple[VisualIndexEntry, ...]) -> VisualIndexGenerationManifest:
    return VisualIndexGenerationManifest(
        generation_id="keyframes-1",
        workspace_ref="workspace-1",
        corpus_refs=("corpus-1",),
        target_type=TargetType.KEYFRAME,
        searchable_unit_manifest_sha256="c" * 64,
        content_sha256=visual_entries_content_sha256(entries),
        embedding_space=_space(),
        retriever_id="capicove.retriever.numpy",
        index_format_version="numpy-exact-v1",
        build_processing_fingerprint=ProcessingFingerprint("d" * 64),
        expected_unit_count=len(entries),
    )


def test_exact_retriever_returns_stable_inner_product_order_and_reopens(
    tmp_path: Path,
) -> None:
    entries = (
        _entry("unit-b", (1.0, 0.0)),
        _entry("unit-a", (1.0, 0.0)),
        _entry("unit-c", (0.0, 1.0)),
    )
    manifest = _manifest(entries)
    configuration = NumpyExactVisualConfiguration(tmp_path.resolve())
    backend = NumpyExactVisualRetriever(configuration)
    physical = backend.generation_lifecycle.create(manifest)
    backend.index_writer.add(
        index_generation_id=manifest.generation_id,
        entries=entries,
    )
    report = backend.generation_lifecycle.validate(manifest)
    pin = backend.generation_lifecycle.activate(manifest, report)
    backend.close()

    reopened = NumpyExactVisualRetriever(configuration)
    candidates = reopened.candidate_retriever.retrieve(
        query_vector=(1.0, 0.0),
        embedding_space=_space(),
        filters=SearchFilters(),
        generation=pin,
        top_k=2,
    )

    assert Path(physical).is_absolute()
    assert report.valid
    assert [item.unit_id for item in candidates] == ["unit-a", "unit-b"]
    assert [item.rank for item in candidates] == [1, 2]
    first = reopened.candidate_retriever.retrieve(
        query_vector=(1.0, 0.0),
        embedding_space=_space(),
        filters=SearchFilters(),
        generation=pin,
        top_k=1,
    )
    assert [item.unit_id for item in first] == ["unit-a"]


def test_exact_retriever_applies_closed_filters(tmp_path: Path) -> None:
    entries = (
        _entry("unit-a", (1.0, 0.0), artifact_ref="artifact-a"),
        _entry("unit-b", (0.8, 0.6), artifact_ref="artifact-b"),
    )
    manifest = _manifest(entries)
    backend = NumpyExactVisualRetriever(NumpyExactVisualConfiguration(tmp_path.resolve()))
    backend.generation_lifecycle.create(manifest)
    backend.index_writer.add(index_generation_id=manifest.generation_id, entries=entries)
    report = backend.generation_lifecycle.validate(manifest)
    pin = backend.generation_lifecycle.activate(manifest, report)

    candidates = backend.candidate_retriever.retrieve(
        query_vector=(1.0, 0.0),
        embedding_space=_space(),
        filters=SearchFilters(artifact_refs=frozenset({"artifact-b"})),
        generation=pin,
        top_k=2,
    )
    assert [item.unit_id for item in candidates] == ["unit-b"]


def test_exact_retriever_rejects_capacity_dimension_and_nonfinite_values(
    tmp_path: Path,
) -> None:
    entries = (_entry("unit-a", (1.0, 0.0)), _entry("unit-b", (0.0, 1.0)))
    with pytest.raises(ValueError, match="capacity"):
        NumpyExactVisualRetriever(
            NumpyExactVisualConfiguration(tmp_path.resolve(), maximum_units_per_target=1)
        ).generation_lifecycle.create(_manifest(entries))

    backend = NumpyExactVisualRetriever(NumpyExactVisualConfiguration(tmp_path.resolve()))
    manifest = _manifest(entries)
    backend.generation_lifecycle.create(manifest)
    with pytest.raises(ValueError, match="dimension"):
        backend.index_writer.add(
            index_generation_id=manifest.generation_id,
            entries=(VisualIndexEntry(entries[0].unit, (1.0, 0.0, 0.0)), entries[1]),
        )
    backend.index_writer.add(index_generation_id=manifest.generation_id, entries=entries)
    report = backend.generation_lifecycle.validate(manifest)
    pin = backend.generation_lifecycle.activate(manifest, report)
    with pytest.raises(ValueError, match="finite"):
        backend.candidate_retriever.retrieve(
            query_vector=(float("nan"), 0.0),
            embedding_space=_space(),
            filters=SearchFilters(),
            generation=pin,
            top_k=1,
        )
