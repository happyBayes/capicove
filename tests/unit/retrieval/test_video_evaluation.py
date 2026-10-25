"""Tests for reproducible video-retrieval qualification contracts."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from capicove.contracts import (
    ProcessingFingerprint,
    QueryModality,
    SearchableUnit,
    TargetType,
    TemporalRange,
)
from capicove.retrieval.temporal import TemporalCandidateAssembler
from capicove.retrieval.video_evaluation import (
    VideoEvaluationEvidence,
    VideoEvaluationEvidenceKind,
    VideoEvaluationMetrics,
    VideoEvaluationProfile,
    VideoEvaluationReport,
    VideoEvaluationThresholds,
    VideoRankedResult,
    VideoRetrievalJudgment,
    evaluate_rankings,
    render_qualification_summary,
    temporal_iou,
)
from capicove.retrieval.video_fusion import (
    CandidateLane,
    VideoCandidate,
    weighted_video_rrf,
)


class _UnitCatalog:
    """Serve the minimum immutable relations used by temporal assembly."""

    def __init__(self, units: tuple[SearchableUnit, ...]) -> None:
        self._units = {unit.unit_id: unit for unit in units}

    @property
    def unit_manifest_sha256(self) -> str:
        return "a" * 64

    def get(self, unit_id: str) -> SearchableUnit | None:
        return self._units.get(unit_id)

    def get_many(self, unit_ids: tuple[str, ...]) -> tuple[SearchableUnit, ...]:
        return tuple(self._units[unit_id] for unit_id in unit_ids if unit_id in self._units)

    def segments_for_keyframe(self, unit_id: str) -> tuple[str, ...]:
        return ("segment-1",) if unit_id == "keyframe-1" else ()

    def segments_for_keyframes(self, unit_ids: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
        return {unit_id: self.segments_for_keyframe(unit_id) for unit_id in unit_ids}

    def segments_for_text(self, unit_id: str) -> tuple[tuple[str, int], ...]:
        del unit_id
        return ()

    def segments_for_texts(
        self, unit_ids: tuple[str, ...]
    ) -> dict[str, tuple[tuple[str, int], ...]]:
        return {unit_id: () for unit_id in unit_ids}


def _profile(**changes: object) -> VideoEvaluationProfile:
    values: dict[str, object] = {
        "corpus_sha256": "1" * 64,
        "query_set_sha256": "2" * 64,
        "model_bundle_sha256": "3" * 64,
        "preprocessing_version": "clip-vit-b32-v1",
        "index_manifest_sha256": "4" * 64,
        "generation_set_manifest_sha256": "5" * 64,
        "fusion_profile_ref": "video-weighted-rrf-v1",
        "temporal_policy_ref": "scene-transcript-segments-v1",
        "host_fingerprint_sha256": "6" * 64,
        "host_class": "arm64-16g-local-ssd",
        "operating_system": "macos",
        "storage_class": "local-ssd",
        "keyframe_count": 1_000_000,
        "embedding_dimension": 512,
        "network_rtt_ms": 1.0,
        "concurrency": 4,
        "warmup_requests": 100,
        "sample_requests": 1_000,
        "evidence": VideoEvaluationEvidence(
            semantic_quality=VideoEvaluationEvidenceKind.REAL_SEMANTIC_CORPUS,
            ann_recall=VideoEvaluationEvidenceKind.REAL_ANN_INDEX,
            capacity=VideoEvaluationEvidenceKind.REAL_ANN_INDEX,
            memory=VideoEvaluationEvidenceKind.REAL_PROCESS,
            client_latency=VideoEvaluationEvidenceKind.REAL_ENDPOINT,
        ),
    }
    values.update(changes)
    return VideoEvaluationProfile(**values)  # type: ignore[arg-type]


def _passing_metrics(**changes: object) -> VideoEvaluationMetrics:
    values: dict[str, object] = {
        "ann_recall_at_20": 0.96,
        "text_keyframe_recall_at_20": 0.91,
        "segment_recall_at_20": 0.86,
        "segment_temporal_iou_at_1": 0.60,
        "mean_ndcg_at_20": 0.90,
        "mean_reciprocal_rank": 0.90,
        "client_p50_ms": 100.0,
        "client_p95_ms": 499.0,
        "client_p99_ms": 550.0,
        "maximum_compact_20_bytes": 16_000,
        "opened_visual_index_bytes": 7 * 1024**3,
        "process_rss_bytes": 15 * 1024**3,
        "steady_state_rss_growth_fraction": 0.04,
        "measured_requests": 1_000,
        "failed_requests": 0,
        "degraded_requests": 0,
    }
    values.update(changes)
    return VideoEvaluationMetrics(**values)  # type: ignore[arg-type]


def _unit(
    unit_id: str,
    target_type: TargetType,
    *,
    timestamp_ms: int | None = None,
    time_range: TemporalRange | None = None,
) -> SearchableUnit:
    return SearchableUnit(
        unit_id=unit_id,
        unit_type=target_type,
        source_hash="7" * 64,
        artifact_ref="artifact-1",
        evidence_ref=f"evidence:{unit_id}",
        representation_refs=(f"representation:{unit_id}",),
        processing_run_ref="processing-run-1",
        processing_version="video-retrieval-v1",
        processing_fingerprint=ProcessingFingerprint("8" * 64),
        timestamp_ms=timestamp_ms,
        time_range=time_range,
    )


def test_evaluation_profile_hash_changes_for_every_behavior_identity() -> None:
    baseline = _profile()
    changed_profiles = (
        ("corpus_sha256", replace(baseline, corpus_sha256="a" * 64)),
        ("query_set_sha256", replace(baseline, query_set_sha256="b" * 64)),
        ("model_bundle_sha256", replace(baseline, model_bundle_sha256="c" * 64)),
        (
            "preprocessing_version",
            replace(baseline, preprocessing_version="clip-vit-b32-v2"),
        ),
        ("index_manifest_sha256", replace(baseline, index_manifest_sha256="d" * 64)),
        (
            "generation_set_manifest_sha256",
            replace(baseline, generation_set_manifest_sha256="e" * 64),
        ),
        (
            "fusion_profile_ref",
            replace(baseline, fusion_profile_ref="video-weighted-rrf-v2"),
        ),
        (
            "temporal_policy_ref",
            replace(baseline, temporal_policy_ref="scene-transcript-segments-v2"),
        ),
        (
            "host_fingerprint_sha256",
            replace(baseline, host_fingerprint_sha256="f" * 64),
        ),
        ("network_rtt_ms", replace(baseline, network_rtt_ms=2.0)),
        ("concurrency", replace(baseline, concurrency=1)),
        ("warmup_requests", replace(baseline, warmup_requests=101)),
        ("sample_requests", replace(baseline, sample_requests=1_001)),
    )

    for field_name, changed in changed_profiles:
        assert changed.profile_sha256 != baseline.profile_sha256, field_name


def test_profile_digest_is_canonical_and_stable() -> None:
    first = _profile()
    second = _profile()

    assert first.profile_sha256 == second.profile_sha256
    assert len(first.profile_sha256) == 64
    assert first.to_dict()["profile_sha256"] == first.profile_sha256


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("corpus_sha256", "A" * 64, "lowercase SHA-256"),
        ("network_rtt_ms", math.nan, "finite"),
        ("concurrency", True, "positive integer"),
        ("warmup_requests", 99, "at least 100"),
        ("sample_requests", 999, "at least 1000"),
        ("keyframe_count", 1_000_001, "at most 1000000"),
    ],
)
def test_profile_rejects_non_reproducible_identity(
    field_name: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _profile(**{field_name: value})


def test_judgment_and_ranking_enforce_target_and_temporal_identity() -> None:
    judgment = VideoRetrievalJudgment(
        query_id="query-1",
        query_modality=QueryModality.TEXT,
        target_type=TargetType.VIDEO_SEGMENT,
        relevant_unit_ids=("segment-1",),
        relevant_ranges=(TemporalRange(100, 200),),
    )
    result = VideoRankedResult(
        unit_id="segment-1",
        target_type=TargetType.VIDEO_SEGMENT,
        time_range=TemporalRange(110, 190),
    )

    assert judgment.relevant_unit_ids == ("segment-1",)
    assert result.time_range == TemporalRange(110, 190)

    with pytest.raises(ValueError, match="relevant_ranges"):
        replace(judgment, relevant_ranges=())
    with pytest.raises(ValueError, match="forbid a time_range"):
        VideoRankedResult("keyframe-1", TargetType.KEYFRAME, TemporalRange(0, 1))


def test_temporal_iou_uses_half_open_ranges() -> None:
    assert temporal_iou(TemporalRange(100, 200), TemporalRange(150, 250)) == pytest.approx(
        1.0 / 3.0
    )
    assert temporal_iou(TemporalRange(100, 200), TemporalRange(200, 250)) == 0.0


def test_evaluate_rankings_returns_stable_per_query_and_aggregate_metrics() -> None:
    judgments = (
        VideoRetrievalJudgment(
            query_id="keyframe-query",
            query_modality=QueryModality.TEXT,
            target_type=TargetType.KEYFRAME,
            relevant_unit_ids=("keyframe-a", "keyframe-b"),
        ),
        VideoRetrievalJudgment(
            query_id="segment-query",
            query_modality=QueryModality.IMAGE,
            target_type=TargetType.VIDEO_SEGMENT,
            relevant_unit_ids=("segment-a",),
            relevant_ranges=(TemporalRange(100, 200),),
        ),
    )
    rankings = {
        "segment-query": (
            VideoRankedResult(
                "segment-a",
                TargetType.VIDEO_SEGMENT,
                TemporalRange(120, 200),
            ),
        ),
        "keyframe-query": (
            VideoRankedResult("irrelevant", TargetType.KEYFRAME),
            VideoRankedResult("keyframe-a", TargetType.KEYFRAME),
        ),
    }

    report = evaluate_rankings(
        profile=_profile(
            evidence=VideoEvaluationEvidence(
                semantic_quality=VideoEvaluationEvidenceKind.AUTHORED_METADATA
            )
        ),
        judgments=judgments,
        rankings=rankings,
        recorded_at=datetime(2026, 9, 14, tzinfo=UTC),
    )

    assert tuple(item.query_id for item in report.metrics.per_query) == (
        "keyframe-query",
        "segment-query",
    )
    assert report.metrics.text_keyframe_recall_at_20 == 0.5
    assert report.metrics.segment_recall_at_20 == 1.0
    assert report.metrics.segment_temporal_iou_at_1 == 0.8
    assert report.thresholds_met is False
    assert report.passed is False


def test_evaluation_consumes_real_rrf_and_temporal_assembly_outputs() -> None:
    keyframe = _unit("keyframe-1", TargetType.KEYFRAME, timestamp_ms=5_000)
    segment = _unit(
        "segment-1",
        TargetType.VIDEO_SEGMENT,
        time_range=TemporalRange(0, 10_000),
    )
    lanes = (
        CandidateLane(
            lane_id="keyframe_visual",
            weight=1.0,
            candidates=(
                VideoCandidate(
                    unit_id=keyframe.unit_id,
                    target_type=TargetType.KEYFRAME,
                    lane_id="keyframe_visual",
                    raw_score=-1_000.0,
                    rank=1,
                    provider_id="visual-provider",
                    generation_id="visual-generation",
                    score_semantics="inner_product_similarity",
                ),
            ),
        ),
        CandidateLane(
            lane_id="ocr_lexical",
            weight=0.5,
            candidates=(
                VideoCandidate(
                    unit_id=keyframe.unit_id,
                    target_type=TargetType.KEYFRAME,
                    lane_id="ocr_lexical",
                    raw_score=1_000_000.0,
                    rank=1,
                    provider_id="text-provider",
                    generation_id="text-generation",
                    score_semantics="bm25",
                ),
            ),
        ),
    )

    fused = weighted_video_rrf(lanes)
    refined = TemporalCandidateAssembler(_UnitCatalog((keyframe, segment))).assemble(fused=fused)
    report = evaluate_rankings(
        profile=_profile(
            evidence=VideoEvaluationEvidence(
                semantic_quality=VideoEvaluationEvidenceKind.AUTHORED_METADATA
            )
        ),
        judgments=(
            VideoRetrievalJudgment(
                query_id="query-1",
                query_modality=QueryModality.TEXT,
                target_type=TargetType.VIDEO_SEGMENT,
                relevant_unit_ids=(segment.unit_id,),
                relevant_ranges=(TemporalRange(0, 10_000),),
            ),
        ),
        rankings={
            "query-1": tuple(
                VideoRankedResult(item.unit.unit_id, item.unit.unit_type, item.unit.time_range)
                for item in refined
            )
        },
        recorded_at=datetime(2026, 9, 15, tzinfo=UTC),
    )

    assert tuple(name for name, _ in fused[0].lane_scores) == (
        "keyframe_visual",
        "ocr_lexical",
    )
    assert fused[0].lane_scores[0][1] == pytest.approx(1.0 / 61.0)
    assert fused[0].lane_scores[1][1] == pytest.approx(0.5 / 61.0)
    assert refined[0].score == pytest.approx(1.5 / 61.0)
    assert report.metrics.segment_recall_at_20 == 1.0
    assert report.metrics.segment_temporal_iou_at_1 == 1.0
    assert report.passed is False


@pytest.mark.parametrize(
    ("rankings", "message"),
    [
        (
            {
                "query": (
                    VideoRankedResult("same", TargetType.KEYFRAME),
                    VideoRankedResult("same", TargetType.KEYFRAME),
                )
            },
            "duplicate",
        ),
        (
            {"unknown": (VideoRankedResult("unit", TargetType.KEYFRAME),)},
            "same query IDs",
        ),
        (
            {
                "query": (
                    VideoRankedResult(
                        "unit",
                        TargetType.VIDEO_SEGMENT,
                        TemporalRange(0, 1),
                    ),
                )
            },
            "target type",
        ),
    ],
)
def test_evaluate_rankings_rejects_invalid_results(
    rankings: dict[str, tuple[VideoRankedResult, ...]],
    message: str,
) -> None:
    judgments = (
        VideoRetrievalJudgment(
            query_id="query",
            query_modality=QueryModality.TEXT,
            target_type=TargetType.KEYFRAME,
            relevant_unit_ids=("unit",),
        ),
    )

    with pytest.raises(ValueError, match=message):
        evaluate_rankings(
            profile=_profile(),
            judgments=judgments,
            rankings=rankings,
            recorded_at=datetime(2026, 9, 14, tzinfo=UTC),
        )


def test_report_pass_is_derived_and_strict() -> None:
    report = VideoEvaluationReport(
        profile=_profile(),
        thresholds=VideoEvaluationThresholds(),
        metrics=_passing_metrics(),
        recorded_at=datetime(2026, 9, 14, tzinfo=UTC),
    )

    assert report.thresholds_met is True
    assert report.claim_evidence_complete is True
    assert report.passed is True
    assert (
        replace(
            report,
            metrics=_passing_metrics(client_p95_ms=500.0),
        ).passed
        is False
    )
    assert (
        replace(
            report,
            metrics=_passing_metrics(ann_recall_at_20=0.90),
        ).passed
        is False
    )


def test_report_cannot_weaken_published_thresholds_to_claim_qualification() -> None:
    report = VideoEvaluationReport(
        profile=_profile(),
        thresholds=replace(VideoEvaluationThresholds(), client_p95_ms=1_000.0),
        metrics=_passing_metrics(client_p95_ms=750.0, client_p99_ms=800.0),
        recorded_at=datetime(2026, 9, 15, tzinfo=UTC),
    )

    assert report.thresholds_met is True
    assert report.uses_standard_thresholds is False
    assert report.passed is False


def test_fixed_metric_names_reject_a_non_twenty_cutoff() -> None:
    judgment = VideoRetrievalJudgment(
        query_id="query",
        target_type=TargetType.KEYFRAME,
        relevant_unit_ids=("keyframe",),
    )

    with pytest.raises(ValueError, match="k=20"):
        evaluate_rankings(
            profile=_profile(),
            judgments=(judgment,),
            rankings={"query": (VideoRankedResult("keyframe", TargetType.KEYFRAME),)},
            recorded_at=datetime(2026, 9, 15, tzinfo=UTC),
            k=10,
        )


def test_report_cannot_turn_synthetic_metrics_into_public_qualification() -> None:
    report = VideoEvaluationReport(
        profile=_profile(
            evidence=VideoEvaluationEvidence(
                semantic_quality=VideoEvaluationEvidenceKind.AUTHORED_METADATA,
                ann_recall=VideoEvaluationEvidenceKind.SYNTHETIC_VECTORS,
                capacity=VideoEvaluationEvidenceKind.SYNTHETIC_VECTORS,
                memory=VideoEvaluationEvidenceKind.REAL_PROCESS,
                client_latency=VideoEvaluationEvidenceKind.REAL_ENDPOINT,
            )
        ),
        thresholds=VideoEvaluationThresholds(),
        metrics=_passing_metrics(),
        recorded_at=datetime(2026, 9, 14, tzinfo=UTC),
    )

    assert report.thresholds_met is True
    assert report.claim_evidence_complete is False
    assert report.passed is False
    assert "Status: Unqualified" in render_qualification_summary(report)


def test_report_rejects_naive_timestamp_and_inconsistent_latency() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        VideoEvaluationReport(
            profile=_profile(),
            thresholds=VideoEvaluationThresholds(),
            metrics=_passing_metrics(),
            recorded_at=datetime(2026, 9, 14),
        )
    with pytest.raises(ValueError, match=r"p50.*p95.*p99"):
        _passing_metrics(client_p50_ms=200.0, client_p95_ms=100.0)


def test_qualification_summary_contains_identities_without_private_host_data() -> None:
    report = VideoEvaluationReport(
        profile=_profile(),
        thresholds=VideoEvaluationThresholds(),
        metrics=_passing_metrics(),
        recorded_at=datetime(2026, 9, 14, tzinfo=UTC),
    )

    summary = render_qualification_summary(report)

    assert "Status: Qualified" in summary
    assert "generation_set_manifest_sha256" in summary
    assert "client_observed_p95_ms" in summary
    assert report.report_sha256 in summary
    assert "hostname" not in summary.lower()
    assert "Human review: pending" in summary
    assert "Thresholds" in summary
    assert report.profile.query_set_sha256 in summary
