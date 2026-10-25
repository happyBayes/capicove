"""Tests for immutable video visual-index and generation-set values."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace

import pytest

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
from capicove.retrieval.generations import (
    PinnedTextIndexGeneration,
    TextIndexGenerationManifest,
)
from capicove.retrieval.video_models import (
    PinnedVideoGenerationSet,
    PinnedVisualIndexGeneration,
    VideoGenerationSetManifest,
    VisualCandidate,
    VisualIndexEntry,
    VisualIndexGenerationManifest,
    VisualIndexValidationReport,
    searchable_units_manifest_sha256,
    visual_entries_content_sha256,
)


def visual_space() -> EmbeddingSpaceRef:
    return EmbeddingSpaceRef(
        space_id="clip-v1",
        provider_id="capicove.model.clip",
        model_revision="revision-1",
        preprocessing_version="clip-v1",
        dimension=2,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.INNER_PRODUCT,
        query_modalities=frozenset({QueryModality.TEXT, QueryModality.IMAGE}),
        target_modalities=frozenset({TargetType.KEYFRAME, TargetType.VIDEO_SEGMENT}),
    )


def unit(unit_id: str = "keyframe-1", target: TargetType = TargetType.KEYFRAME) -> SearchableUnit:
    values: dict[str, object] = {
        "unit_id": unit_id,
        "unit_type": target,
        "source_hash": "a" * 64,
        "artifact_ref": "artifact-1",
        "evidence_ref": f"evidence:{unit_id}",
        "representation_refs": (f"representation:{unit_id}",),
        "processing_run_ref": "run-1",
        "processing_version": "video-retrieval-v1",
        "processing_fingerprint": ProcessingFingerprint("b" * 64),
        "embedding_spaces": (visual_space(),),
    }
    if target is TargetType.KEYFRAME:
        values["timestamp_ms"] = 1_000
        values["scene_range"] = TemporalRange(0, 2_000)
    else:
        values["time_range"] = TemporalRange(0, 2_000)
    return SearchableUnit(**values)  # type: ignore[arg-type]


def visual_manifest(
    target: TargetType = TargetType.KEYFRAME,
    *,
    generation_id: str | None = None,
    unit_hash: str = "c" * 64,
) -> VisualIndexGenerationManifest:
    entries = (VisualIndexEntry(unit(f"{target.value}-1", target), (1.0, 0.0)),)
    return VisualIndexGenerationManifest(
        generation_id=generation_id or f"{target.value}-generation-1",
        workspace_ref="workspace-1",
        corpus_refs=("corpus-b", "corpus-a"),
        target_type=target,
        searchable_unit_manifest_sha256=unit_hash,
        content_sha256=visual_entries_content_sha256(entries),
        embedding_space=visual_space(),
        retriever_id="capicove.retriever.numpy",
        index_format_version="numpy-exact-v1",
        build_processing_fingerprint=ProcessingFingerprint("d" * 64),
        expected_unit_count=1,
    )


def text_manifest(origin: str) -> TextIndexGenerationManifest:
    return TextIndexGenerationManifest(
        generation_id=f"{origin}-generation-1",
        workspace_ref="workspace-1",
        corpus_refs=("corpus-a", "corpus-b"),
        searchable_unit_manifest_sha256="c" * 64,
        content_sha256="e" * 64,
        mapping_sha256="f" * 64,
        embedding_spaces=(),
        retriever_id=f"capicove.retriever.{origin}",
        index_format_version="text-v1",
        build_processing_fingerprint=ProcessingFingerprint("1" * 64),
        expected_unit_count=0,
    )


def test_visual_entry_and_candidate_enforce_finite_immutable_values() -> None:
    entry = VisualIndexEntry(unit(), [1.0, 0.0])  # type: ignore[arg-type]
    candidate = VisualCandidate(
        unit_id="keyframe-1",
        target_type=TargetType.KEYFRAME,
        lane_id="keyframe_visual",
        raw_score=0.75,
        score_semantics="inner_product_similarity",
        rank=1,
        provider_id="capicove.retriever.numpy",
        generation_id="visual-generation-1",
    )

    assert entry.vector == (1.0, 0.0)
    assert candidate.rank == 1
    with pytest.raises(FrozenInstanceError):
        candidate.rank = 2  # type: ignore[misc]
    with pytest.raises(ValueError, match="finite"):
        replace(candidate, raw_score=math.nan)
    with pytest.raises(ValueError, match="positive"):
        replace(candidate, rank=0)


def test_visual_entry_rejects_nonvisual_units_and_bad_vectors() -> None:
    with pytest.raises(ValueError, match="visual target"):
        VisualIndexEntry(
            replace(
                unit(),
                unit_type=TargetType.IMAGE,
                timestamp_ms=None,
                scene_range=None,
            ),
            (1.0,),
        )
    with pytest.raises(ValueError, match="finite"):
        VisualIndexEntry(unit(), (1.0, math.inf))
    with pytest.raises(ValueError, match="not be empty"):
        VisualIndexEntry(unit(), ())


def test_visual_manifest_is_canonical_and_rejects_incompatible_target_space() -> None:
    manifest = visual_manifest()
    assert manifest.corpus_refs == ("corpus-a", "corpus-b")
    assert len(manifest.manifest_sha256) == 64

    incompatible = replace(visual_space(), target_modalities=frozenset({TargetType.VIDEO_SEGMENT}))
    with pytest.raises(ValueError, match="target"):
        replace(manifest, embedding_space=incompatible)
    with pytest.raises(ValueError, match="visual target"):
        replace(manifest, target_type=TargetType.TEXT_CHUNK)


def test_manifest_and_report_validate_count_digest_and_activation_identity() -> None:
    manifest = visual_manifest()
    report = VisualIndexValidationReport(
        generation_id=manifest.generation_id,
        manifest_sha256=manifest.manifest_sha256,
        physical_index_ref="/tmp/index-1",
        content_sha256=manifest.content_sha256,
        expected_unit_count=1,
        observed_unit_count=1,
        vector_dimension=2,
        issue_codes=(),
    )
    assert report.valid
    with pytest.raises(ValueError, match="unit_count_mismatch"):
        replace(report, observed_unit_count=0)
    assert replace(report, issue_codes=("checksum_mismatch",)).valid is False


def test_manifest_helpers_are_ordered_and_content_sensitive() -> None:
    first = unit("keyframe-1")
    second = unit("keyframe-2")
    assert searchable_units_manifest_sha256((first, second)) != searchable_units_manifest_sha256(
        (second, first)
    )
    assert visual_entries_content_sha256(
        (VisualIndexEntry(first, (1.0, 0.0)),)
    ) != visual_entries_content_sha256((VisualIndexEntry(first, (0.0, 1.0)),))
    with pytest.raises(ValueError, match="duplicate"):
        searchable_units_manifest_sha256((first, first))


def test_video_generation_set_rejects_mixed_members_and_builds_pin() -> None:
    keyframe = visual_manifest(TargetType.KEYFRAME)
    segment = visual_manifest(TargetType.VIDEO_SEGMENT)
    asr = text_manifest("asr")
    manifest = VideoGenerationSetManifest(
        set_id="video-set-1",
        workspace_ref="workspace-1",
        profile_ref="video-local-exact",
        corpus_refs=("corpus-b", "corpus-a"),
        unit_catalog_ref="/tmp/units.sqlite3",
        unit_manifest_sha256="c" * 64,
        keyframe_visual=keyframe,
        segment_visual=segment,
        asr_text=asr,
        ocr_text=None,
        fusion_profile_ref="video-weighted-rrf-v1",
    )
    keyframe_pin = PinnedVisualIndexGeneration(
        keyframe.generation_id,
        keyframe.manifest_sha256,
        "/tmp/keyframes",
        TargetType.KEYFRAME,
        visual_space(),
    )
    segment_pin = PinnedVisualIndexGeneration(
        segment.generation_id,
        segment.manifest_sha256,
        "/tmp/segments",
        TargetType.VIDEO_SEGMENT,
        visual_space(),
    )
    pinned = PinnedVideoGenerationSet(
        set_id=manifest.set_id,
        manifest_sha256=manifest.manifest_sha256,
        workspace_ref=manifest.workspace_ref,
        profile_ref=manifest.profile_ref,
        unit_catalog_ref=manifest.unit_catalog_ref,
        unit_manifest_sha256=manifest.unit_manifest_sha256,
        keyframe_visual=keyframe_pin,
        segment_visual=segment_pin,
        asr_text=PinnedTextIndexGeneration(asr.generation_id, asr.manifest_sha256, "/tmp/asr"),
        ocr_text=None,
        fusion_profile_ref=manifest.fusion_profile_ref,
    )
    assert pinned.set_id == "video-set-1"
    with pytest.raises(ValueError, match="unit manifest"):
        replace(
            manifest,
            segment_visual=visual_manifest(TargetType.VIDEO_SEGMENT, unit_hash="9" * 64),
        )
