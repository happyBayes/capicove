"""Deterministic contracts and metrics for video-retrieval qualification."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from capicove.contracts import QueryModality, TargetType, TemporalRange


_SHA256_LENGTH = 64
_REPORT_SCHEMA_VERSION = 1
_MAX_QUALIFIED_KEYFRAMES = 1_000_000


class VideoEvaluationEvidenceKind(StrEnum):
    """Identify what produced one group of qualification measurements."""

    NOT_MEASURED = "not_measured"
    AUTHORED_METADATA = "authored_metadata"
    SYNTHETIC_VECTORS = "synthetic_vectors"
    REAL_SEMANTIC_CORPUS = "real_semantic_corpus"
    REAL_ANN_INDEX = "real_ann_index"
    REAL_PROCESS = "real_process"
    REAL_ENDPOINT = "real_endpoint"


@dataclass(frozen=True, slots=True)
class VideoEvaluationEvidence:
    """Bind every metric family to its measurement provenance."""

    semantic_quality: VideoEvaluationEvidenceKind = VideoEvaluationEvidenceKind.NOT_MEASURED
    ann_recall: VideoEvaluationEvidenceKind = VideoEvaluationEvidenceKind.NOT_MEASURED
    capacity: VideoEvaluationEvidenceKind = VideoEvaluationEvidenceKind.NOT_MEASURED
    memory: VideoEvaluationEvidenceKind = VideoEvaluationEvidenceKind.NOT_MEASURED
    client_latency: VideoEvaluationEvidenceKind = VideoEvaluationEvidenceKind.NOT_MEASURED

    def __post_init__(self) -> None:
        for name in (
            "semantic_quality",
            "ann_recall",
            "capacity",
            "memory",
            "client_latency",
        ):
            if not isinstance(getattr(self, name), VideoEvaluationEvidenceKind):
                raise TypeError(f"{name} must be a VideoEvaluationEvidenceKind")

    def to_dict(self) -> dict[str, str]:
        """Return the stable serialized evidence identities."""
        return {
            "ann_recall": self.ann_recall.value,
            "capacity": self.capacity.value,
            "client_latency": self.client_latency.value,
            "memory": self.memory.value,
            "semantic_quality": self.semantic_quality.value,
        }


@dataclass(frozen=True, slots=True)
class VideoEvaluationThresholds:
    """Hold the fixed gates for the one-million-keyframe profile."""

    ann_recall_at_20: float = 0.95
    text_keyframe_recall_at_20: float = 0.90
    segment_recall_at_20: float = 0.85
    segment_temporal_iou_at_1: float = 0.50
    client_p95_ms: float = 500.0
    compact_20_max_bytes: int = 16_384
    opened_visual_index_max_bytes: int = 8 * 1024**3
    process_rss_max_bytes: int = 16 * 1024**3
    steady_state_rss_growth_max_fraction: float = 0.05
    network_rtt_max_ms: float = 20.0
    minimum_warmup_requests: int = 100
    minimum_sample_requests: int = 1_000
    required_concurrency: int = 4
    qualified_keyframe_count: int = _MAX_QUALIFIED_KEYFRAMES

    def __post_init__(self) -> None:
        for name in (
            "ann_recall_at_20",
            "text_keyframe_recall_at_20",
            "segment_recall_at_20",
            "segment_temporal_iou_at_1",
            "steady_state_rss_growth_max_fraction",
        ):
            _require_unit_interval(name, getattr(self, name))
        _require_positive_finite("client_p95_ms", self.client_p95_ms)
        _require_non_negative_finite("network_rtt_max_ms", self.network_rtt_max_ms)
        for name in (
            "compact_20_max_bytes",
            "opened_visual_index_max_bytes",
            "process_rss_max_bytes",
            "minimum_warmup_requests",
            "minimum_sample_requests",
            "required_concurrency",
            "qualified_keyframe_count",
        ):
            _require_positive_integer(name, getattr(self, name))

    def to_dict(self) -> dict[str, int | float]:
        """Return a canonical JSON-compatible threshold mapping."""
        return {
            "ann_recall_at_20": self.ann_recall_at_20,
            "client_p95_ms": self.client_p95_ms,
            "compact_20_max_bytes": self.compact_20_max_bytes,
            "minimum_sample_requests": self.minimum_sample_requests,
            "minimum_warmup_requests": self.minimum_warmup_requests,
            "network_rtt_max_ms": self.network_rtt_max_ms,
            "opened_visual_index_max_bytes": self.opened_visual_index_max_bytes,
            "process_rss_max_bytes": self.process_rss_max_bytes,
            "qualified_keyframe_count": self.qualified_keyframe_count,
            "required_concurrency": self.required_concurrency,
            "segment_recall_at_20": self.segment_recall_at_20,
            "segment_temporal_iou_at_1": self.segment_temporal_iou_at_1,
            "steady_state_rss_growth_max_fraction": (self.steady_state_rss_growth_max_fraction),
            "text_keyframe_recall_at_20": self.text_keyframe_recall_at_20,
        }


@dataclass(frozen=True, slots=True)
class VideoRetrievalJudgment:
    """Describe relevant immutable units for one frozen evaluation query."""

    query_id: str
    target_type: TargetType
    relevant_unit_ids: tuple[str, ...]
    relevant_ranges: tuple[TemporalRange, ...] = ()
    query_modality: QueryModality = QueryModality.TEXT

    def __post_init__(self) -> None:
        _require_non_blank("query_id", self.query_id)
        if self.query_modality not in (QueryModality.TEXT, QueryModality.IMAGE):
            raise ValueError("query_modality must be text or image")
        if self.target_type not in (TargetType.KEYFRAME, TargetType.VIDEO_SEGMENT):
            raise ValueError("target_type must be keyframe or video_segment")
        unit_ids = _freeze_unique_strings("relevant_unit_ids", self.relevant_unit_ids)
        if not unit_ids:
            raise ValueError("relevant_unit_ids must not be empty")
        object.__setattr__(self, "relevant_unit_ids", unit_ids)
        ranges = tuple(self.relevant_ranges)
        if any(not isinstance(item, TemporalRange) for item in ranges):
            raise TypeError("relevant_ranges must contain TemporalRange values")
        if self.target_type is TargetType.KEYFRAME and ranges:
            raise ValueError("keyframe judgments forbid relevant_ranges")
        if self.target_type is TargetType.VIDEO_SEGMENT and len(ranges) != len(unit_ids):
            raise ValueError("segment relevant_ranges must align one-to-one with relevant_unit_ids")
        object.__setattr__(self, "relevant_ranges", ranges)


@dataclass(frozen=True, slots=True)
class VideoRankedResult:
    """Carry only the immutable result identity needed for evaluation."""

    unit_id: str
    target_type: TargetType
    time_range: TemporalRange | None = None

    def __post_init__(self) -> None:
        _require_non_blank("unit_id", self.unit_id)
        if self.target_type not in (TargetType.KEYFRAME, TargetType.VIDEO_SEGMENT):
            raise ValueError("target_type must be keyframe or video_segment")
        if self.target_type is TargetType.KEYFRAME and self.time_range is not None:
            raise ValueError("keyframe results forbid a time_range")
        if self.target_type is TargetType.VIDEO_SEGMENT and not isinstance(
            self.time_range, TemporalRange
        ):
            raise ValueError("video_segment results require a time_range")


@dataclass(frozen=True, slots=True)
class VideoQueryMetrics:
    """Record deterministic metrics for one query."""

    query_id: str
    target_type: TargetType
    recall_at_20: float
    ndcg_at_20: float
    reciprocal_rank: float
    temporal_iou_at_1: float | None = None

    def __post_init__(self) -> None:
        _require_non_blank("query_id", self.query_id)
        for name in ("recall_at_20", "ndcg_at_20", "reciprocal_rank"):
            _require_unit_interval(name, getattr(self, name))
        if self.temporal_iou_at_1 is not None:
            _require_unit_interval("temporal_iou_at_1", self.temporal_iou_at_1)

    def to_dict(self) -> dict[str, object]:
        """Return a canonical JSON-compatible per-query mapping."""
        return {
            "ndcg_at_20": self.ndcg_at_20,
            "query_id": self.query_id,
            "recall_at_20": self.recall_at_20,
            "reciprocal_rank": self.reciprocal_rank,
            "target_type": self.target_type.value,
            "temporal_iou_at_1": self.temporal_iou_at_1,
        }


@dataclass(frozen=True, slots=True)
class VideoEvaluationMetrics:
    """Combine quality, index, memory, and client-observed measurements."""

    ann_recall_at_20: float | None = None
    text_keyframe_recall_at_20: float | None = None
    segment_recall_at_20: float | None = None
    segment_temporal_iou_at_1: float | None = None
    mean_ndcg_at_20: float | None = None
    mean_reciprocal_rank: float | None = None
    client_p50_ms: float | None = None
    client_p95_ms: float | None = None
    client_p99_ms: float | None = None
    maximum_compact_20_bytes: int | None = None
    opened_visual_index_bytes: int | None = None
    process_rss_bytes: int | None = None
    steady_state_rss_growth_fraction: float | None = None
    measured_requests: int = 0
    failed_requests: int = 0
    degraded_requests: int = 0
    per_query: tuple[VideoQueryMetrics, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "ann_recall_at_20",
            "text_keyframe_recall_at_20",
            "segment_recall_at_20",
            "segment_temporal_iou_at_1",
            "mean_ndcg_at_20",
            "mean_reciprocal_rank",
            "steady_state_rss_growth_fraction",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_unit_interval(name, value)
        for name in ("client_p50_ms", "client_p95_ms", "client_p99_ms"):
            value = getattr(self, name)
            if value is not None:
                _require_non_negative_finite(name, value)
        latency_values = (self.client_p50_ms, self.client_p95_ms, self.client_p99_ms)
        if any(value is not None for value in latency_values):
            if any(value is None for value in latency_values):
                raise ValueError("client p50, p95, and p99 must be recorded together")
            p50, p95, p99 = latency_values
            if not p50 <= p95 <= p99:  # type: ignore[operator]
                raise ValueError("client p50, p95, and p99 must be non-decreasing")
        for name in (
            "maximum_compact_20_bytes",
            "opened_visual_index_bytes",
            "process_rss_bytes",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_non_negative_integer(name, value)
        for name in ("measured_requests", "failed_requests", "degraded_requests"):
            _require_non_negative_integer(name, getattr(self, name))
        if self.failed_requests > self.measured_requests:
            raise ValueError("failed_requests cannot exceed measured_requests")
        if self.degraded_requests > self.measured_requests:
            raise ValueError("degraded_requests cannot exceed measured_requests")
        per_query = tuple(self.per_query)
        if any(not isinstance(item, VideoQueryMetrics) for item in per_query):
            raise TypeError("per_query must contain VideoQueryMetrics values")
        if tuple(sorted(item.query_id for item in per_query)) != tuple(
            item.query_id for item in per_query
        ):
            raise ValueError("per_query metrics must use stable query-ID order")
        if len({item.query_id for item in per_query}) != len(per_query):
            raise ValueError("per_query metrics must not repeat query IDs")
        object.__setattr__(self, "per_query", per_query)

    def to_dict(self) -> dict[str, object]:
        """Return a canonical JSON-compatible metrics mapping."""
        return {
            "ann_recall_at_20": self.ann_recall_at_20,
            "client_p50_ms": self.client_p50_ms,
            "client_p95_ms": self.client_p95_ms,
            "client_p99_ms": self.client_p99_ms,
            "degraded_requests": self.degraded_requests,
            "failed_requests": self.failed_requests,
            "maximum_compact_20_bytes": self.maximum_compact_20_bytes,
            "mean_ndcg_at_20": self.mean_ndcg_at_20,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "measured_requests": self.measured_requests,
            "opened_visual_index_bytes": self.opened_visual_index_bytes,
            "per_query": [item.to_dict() for item in self.per_query],
            "process_rss_bytes": self.process_rss_bytes,
            "segment_recall_at_20": self.segment_recall_at_20,
            "segment_temporal_iou_at_1": self.segment_temporal_iou_at_1,
            "steady_state_rss_growth_fraction": self.steady_state_rss_growth_fraction,
            "text_keyframe_recall_at_20": self.text_keyframe_recall_at_20,
        }


@dataclass(frozen=True, slots=True)
class VideoEvaluationProfile:
    """Bind a qualification run to every behavior-changing identity."""

    corpus_sha256: str
    query_set_sha256: str
    model_bundle_sha256: str
    preprocessing_version: str
    index_manifest_sha256: str
    generation_set_manifest_sha256: str
    fusion_profile_ref: str
    temporal_policy_ref: str
    host_fingerprint_sha256: str
    host_class: str
    operating_system: str
    storage_class: str
    keyframe_count: int
    embedding_dimension: int
    network_rtt_ms: float
    concurrency: int
    warmup_requests: int
    sample_requests: int
    evidence: VideoEvaluationEvidence = field(default_factory=VideoEvaluationEvidence)

    def __post_init__(self) -> None:
        for name in (
            "corpus_sha256",
            "query_set_sha256",
            "model_bundle_sha256",
            "index_manifest_sha256",
            "generation_set_manifest_sha256",
            "host_fingerprint_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        for name in (
            "preprocessing_version",
            "fusion_profile_ref",
            "temporal_policy_ref",
            "host_class",
            "operating_system",
            "storage_class",
        ):
            _require_non_blank(name, getattr(self, name))
        _require_positive_integer("keyframe_count", self.keyframe_count)
        if self.keyframe_count > _MAX_QUALIFIED_KEYFRAMES:
            raise ValueError("keyframe_count must be at most 1000000")
        _require_positive_integer("embedding_dimension", self.embedding_dimension)
        _require_non_negative_finite("network_rtt_ms", self.network_rtt_ms)
        _require_positive_integer("concurrency", self.concurrency)
        if self.concurrency not in (1, 4):
            raise ValueError("concurrency must be 1 or 4")
        _require_positive_integer("warmup_requests", self.warmup_requests)
        if self.warmup_requests < 100:
            raise ValueError("warmup_requests must be at least 100")
        _require_positive_integer("sample_requests", self.sample_requests)
        if self.sample_requests < 1_000:
            raise ValueError("sample_requests must be at least 1000")
        if not isinstance(self.evidence, VideoEvaluationEvidence):
            raise TypeError("evidence must be a VideoEvaluationEvidence")

    @property
    def profile_sha256(self) -> str:
        """Return the canonical digest of every profile identity."""
        return _sha256_json(self._identity_dict())

    def _identity_dict(self) -> dict[str, object]:
        return {
            "concurrency": self.concurrency,
            "corpus_sha256": self.corpus_sha256,
            "embedding_dimension": self.embedding_dimension,
            "evidence": self.evidence.to_dict(),
            "fusion_profile_ref": self.fusion_profile_ref,
            "generation_set_manifest_sha256": self.generation_set_manifest_sha256,
            "host_class": self.host_class,
            "host_fingerprint_sha256": self.host_fingerprint_sha256,
            "index_manifest_sha256": self.index_manifest_sha256,
            "keyframe_count": self.keyframe_count,
            "model_bundle_sha256": self.model_bundle_sha256,
            "network_rtt_ms": self.network_rtt_ms,
            "operating_system": self.operating_system,
            "preprocessing_version": self.preprocessing_version,
            "query_set_sha256": self.query_set_sha256,
            "sample_requests": self.sample_requests,
            "storage_class": self.storage_class,
            "temporal_policy_ref": self.temporal_policy_ref,
            "warmup_requests": self.warmup_requests,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the profile and its canonical digest."""
        payload = self._identity_dict()
        payload["profile_sha256"] = self.profile_sha256
        return payload


@dataclass(frozen=True, slots=True)
class VideoEvaluationReport:
    """Record measurements while deriving, never trusting, qualification state."""

    profile: VideoEvaluationProfile
    thresholds: VideoEvaluationThresholds
    metrics: VideoEvaluationMetrics
    recorded_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.profile, VideoEvaluationProfile):
            raise TypeError("profile must be a VideoEvaluationProfile")
        if not isinstance(self.thresholds, VideoEvaluationThresholds):
            raise TypeError("thresholds must be VideoEvaluationThresholds")
        if not isinstance(self.metrics, VideoEvaluationMetrics):
            raise TypeError("metrics must be VideoEvaluationMetrics")
        if self.recorded_at.tzinfo is None or self.recorded_at.utcoffset() is None:
            raise ValueError("recorded_at must be timezone-aware")

    @property
    def thresholds_met(self) -> bool:
        """Return whether every required measured value passes its strict gate."""
        metrics = self.metrics
        thresholds = self.thresholds
        ann_recall = metrics.ann_recall_at_20
        text_keyframe_recall = metrics.text_keyframe_recall_at_20
        segment_recall = metrics.segment_recall_at_20
        temporal_iou_at_1 = metrics.segment_temporal_iou_at_1
        client_p50_ms = metrics.client_p50_ms
        client_p95_ms = metrics.client_p95_ms
        client_p99_ms = metrics.client_p99_ms
        maximum_response_bytes = metrics.maximum_compact_20_bytes
        opened_index_bytes = metrics.opened_visual_index_bytes
        process_rss_bytes = metrics.process_rss_bytes
        rss_growth = metrics.steady_state_rss_growth_fraction
        if (
            ann_recall is None
            or text_keyframe_recall is None
            or segment_recall is None
            or temporal_iou_at_1 is None
            or client_p50_ms is None
            or client_p95_ms is None
            or client_p99_ms is None
            or maximum_response_bytes is None
            or opened_index_bytes is None
            or process_rss_bytes is None
            or rss_growth is None
        ):
            return False
        return bool(
            ann_recall >= thresholds.ann_recall_at_20
            and text_keyframe_recall >= thresholds.text_keyframe_recall_at_20
            and segment_recall >= thresholds.segment_recall_at_20
            and temporal_iou_at_1 >= thresholds.segment_temporal_iou_at_1
            and client_p95_ms < thresholds.client_p95_ms
            and maximum_response_bytes <= thresholds.compact_20_max_bytes
            and opened_index_bytes <= thresholds.opened_visual_index_max_bytes
            and process_rss_bytes <= thresholds.process_rss_max_bytes
            and rss_growth <= thresholds.steady_state_rss_growth_max_fraction
            and self.profile.network_rtt_ms <= thresholds.network_rtt_max_ms
            and self.profile.concurrency == thresholds.required_concurrency
            and self.profile.warmup_requests >= thresholds.minimum_warmup_requests
            and self.profile.sample_requests >= thresholds.minimum_sample_requests
            and self.profile.keyframe_count == thresholds.qualified_keyframe_count
            and metrics.measured_requests >= self.profile.sample_requests
            and metrics.failed_requests == 0
        )

    @property
    def claim_evidence_complete(self) -> bool:
        """Reject support claims backed only by synthetic or metadata fixtures."""
        evidence = self.profile.evidence
        return bool(
            evidence.semantic_quality is VideoEvaluationEvidenceKind.REAL_SEMANTIC_CORPUS
            and evidence.ann_recall is VideoEvaluationEvidenceKind.REAL_ANN_INDEX
            and evidence.capacity is VideoEvaluationEvidenceKind.REAL_ANN_INDEX
            and evidence.memory is VideoEvaluationEvidenceKind.REAL_PROCESS
            and evidence.client_latency is VideoEvaluationEvidenceKind.REAL_ENDPOINT
        )

    @property
    def uses_standard_thresholds(self) -> bool:
        """Return whether the report uses the fixed published qualification gates."""
        return self.thresholds == VideoEvaluationThresholds()

    @property
    def passed(self) -> bool:
        """Return the derived public-qualification decision."""
        return (
            self.thresholds_met and self.claim_evidence_complete and self.uses_standard_thresholds
        )

    @property
    def report_sha256(self) -> str:
        """Return the canonical report digest excluding the digest itself."""
        return _sha256_json(self._report_dict())

    def _report_dict(self) -> dict[str, object]:
        return {
            "claim_evidence_complete": self.claim_evidence_complete,
            "metrics": self.metrics.to_dict(),
            "passed": self.passed,
            "profile": self.profile.to_dict(),
            "recorded_at": self.recorded_at.isoformat(),
            "schema_version": _REPORT_SCHEMA_VERSION,
            "thresholds": self.thresholds.to_dict(),
            "thresholds_met": self.thresholds_met,
            "uses_standard_thresholds": self.uses_standard_thresholds,
        }

    def to_dict(self) -> dict[str, object]:
        """Return a canonical JSON-compatible report mapping."""
        payload = self._report_dict()
        payload["report_sha256"] = self.report_sha256
        return payload

    def to_json(self) -> str:
        """Serialize the report deterministically with one trailing newline."""
        return _canonical_json(self.to_dict()) + "\n"


def evaluate_rankings(
    *,
    profile: VideoEvaluationProfile,
    judgments: Sequence[VideoRetrievalJudgment],
    rankings: Mapping[str, Sequence[VideoRankedResult]],
    recorded_at: datetime,
    k: int = 20,
    thresholds: VideoEvaluationThresholds | None = None,
) -> VideoEvaluationReport:
    """Evaluate frozen rankings in stable query-ID order."""
    _require_positive_integer("k", k)
    if k != 20:
        raise ValueError("video qualification rankings require k=20")
    frozen_judgments = tuple(judgments)
    if not frozen_judgments:
        raise ValueError("judgments must contain at least one query")
    if any(not isinstance(item, VideoRetrievalJudgment) for item in frozen_judgments):
        raise TypeError("judgments must contain VideoRetrievalJudgment values")
    judgment_by_id = {item.query_id: item for item in frozen_judgments}
    if len(judgment_by_id) != len(frozen_judgments):
        raise ValueError("judgments must not repeat query IDs")
    if set(rankings) != set(judgment_by_id):
        raise ValueError("rankings and judgments must contain the same query IDs")

    per_query: list[VideoQueryMetrics] = []
    for query_id in sorted(judgment_by_id):
        judgment = judgment_by_id[query_id]
        results = tuple(rankings[query_id])
        if any(not isinstance(item, VideoRankedResult) for item in results):
            raise TypeError("rankings must contain VideoRankedResult values")
        if len({item.unit_id for item in results}) != len(results):
            raise ValueError(f"rankings for query {query_id!r} contain a duplicate unit ID")
        if any(item.target_type is not judgment.target_type for item in results):
            raise ValueError(f"rankings for query {query_id!r} contain a mixed target type")
        per_query.append(_evaluate_one(judgment, results, k=k))

    metrics = VideoEvaluationMetrics(
        text_keyframe_recall_at_20=_mean_or_none(
            item.recall_at_20
            for item, judgment in zip(
                per_query, sorted(frozen_judgments, key=lambda x: x.query_id), strict=True
            )
            if judgment.query_modality is QueryModality.TEXT
            and judgment.target_type is TargetType.KEYFRAME
        ),
        segment_recall_at_20=_mean_or_none(
            item.recall_at_20 for item in per_query if item.target_type is TargetType.VIDEO_SEGMENT
        ),
        segment_temporal_iou_at_1=_mean_or_none(
            item.temporal_iou_at_1 for item in per_query if item.temporal_iou_at_1 is not None
        ),
        mean_ndcg_at_20=_mean_or_none(item.ndcg_at_20 for item in per_query),
        mean_reciprocal_rank=_mean_or_none(item.reciprocal_rank for item in per_query),
        per_query=tuple(per_query),
    )
    return VideoEvaluationReport(
        profile=profile,
        thresholds=thresholds or VideoEvaluationThresholds(),
        metrics=metrics,
        recorded_at=recorded_at,
    )


def temporal_iou(left: TemporalRange, right: TemporalRange) -> float:
    """Return intersection over union for two half-open millisecond ranges."""
    if not isinstance(left, TemporalRange) or not isinstance(right, TemporalRange):
        raise TypeError("temporal_iou requires TemporalRange values")
    intersection = max(0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))
    union = max(left.end_ms, right.end_ms) - min(left.start_ms, right.start_ms)
    return intersection / union


def recall_between_rankings(
    exact_unit_ids: Sequence[str],
    approximate_unit_ids: Sequence[str],
    *,
    k: int,
) -> float:
    """Return ANN Recall@K against an exact top-K identity set."""
    _require_positive_integer("k", k)
    exact = tuple(exact_unit_ids[:k])
    approximate = tuple(approximate_unit_ids[:k])
    if len(exact) != len(set(exact)) or len(approximate) != len(set(approximate)):
        raise ValueError("rankings must not contain duplicate unit IDs")
    if not exact:
        raise ValueError("exact ranking must not be empty")
    return len(set(exact).intersection(approximate)) / len(exact)


def render_qualification_summary(report: VideoEvaluationReport) -> str:
    """Render a deterministic, non-sensitive reviewed-summary candidate."""
    if not isinstance(report, VideoEvaluationReport):
        raise TypeError("report must be a VideoEvaluationReport")
    status = "Qualified" if report.passed else "Unqualified"
    metrics = report.metrics
    lines = [
        "# Video Retrieval Qualification Summary",
        "",
        f"- Status: {status}",
        "- Human review: pending (technical status is not publication approval)",
        f"- recorded_at: `{report.recorded_at.isoformat()}`",
        f"- report_sha256: `{report.report_sha256}`",
        f"- profile_sha256: `{report.profile.profile_sha256}`",
        f"- corpus_sha256: `{report.profile.corpus_sha256}`",
        f"- query_set_sha256: `{report.profile.query_set_sha256}`",
        f"- model_bundle_sha256: `{report.profile.model_bundle_sha256}`",
        f"- index_manifest_sha256: `{report.profile.index_manifest_sha256}`",
        f"- preprocessing_version: `{report.profile.preprocessing_version}`",
        f"- fusion_profile_ref: `{report.profile.fusion_profile_ref}`",
        f"- temporal_policy_ref: `{report.profile.temporal_policy_ref}`",
        f"- generation_set_manifest_sha256: `{report.profile.generation_set_manifest_sha256}`",
        f"- host_class: `{report.profile.host_class}`",
        "",
        "| Metric | Measured |",
        "| --- | ---: |",
        f"| ann_recall_at_20 | {_display(metrics.ann_recall_at_20)} |",
        f"| text_keyframe_recall_at_20 | {_display(metrics.text_keyframe_recall_at_20)} |",
        f"| segment_recall_at_20 | {_display(metrics.segment_recall_at_20)} |",
        f"| segment_temporal_iou_at_1 | {_display(metrics.segment_temporal_iou_at_1)} |",
        f"| client_observed_p95_ms | {_display(metrics.client_p95_ms)} |",
        f"| maximum_compact_20_bytes | {_display(metrics.maximum_compact_20_bytes)} |",
        f"| opened_visual_index_bytes | {_display(metrics.opened_visual_index_bytes)} |",
        f"| process_rss_bytes | {_display(metrics.process_rss_bytes)} |",
        "",
        "## Thresholds",
        "",
        "```json",
        json.dumps(report.thresholds.to_dict(), sort_keys=True, indent=2, allow_nan=False),
        "```",
        "",
        "This summary contains no credentials, absolute paths, raw queries, media, "
        "vectors, or private machine names.",
    ]
    return "\n".join(lines) + "\n"


def _evaluate_one(
    judgment: VideoRetrievalJudgment,
    results: tuple[VideoRankedResult, ...],
    *,
    k: int,
) -> VideoQueryMetrics:
    relevant = set(judgment.relevant_unit_ids)
    top_results = results[:k]
    hits = sum(item.unit_id in relevant for item in top_results)
    recall = hits / len(relevant)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(top_results, start=1)
        if item.unit_id in relevant
    )
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    reciprocal_rank = next(
        (1.0 / rank for rank, item in enumerate(results, start=1) if item.unit_id in relevant),
        0.0,
    )
    best_iou: float | None = None
    if judgment.target_type is TargetType.VIDEO_SEGMENT:
        best_iou = 0.0
        if results:
            result_range = results[0].time_range
            if result_range is None:  # Defensive; VideoRankedResult already validates this.
                raise ValueError("video_segment results require a time_range")
            best_iou = max(temporal_iou(result_range, item) for item in judgment.relevant_ranges)
    return VideoQueryMetrics(
        query_id=judgment.query_id,
        target_type=judgment.target_type,
        recall_at_20=recall,
        ndcg_at_20=dcg / ideal_dcg,
        reciprocal_rank=reciprocal_rank,
        temporal_iou_at_1=best_iou,
    )


def _mean_or_none(values: Iterable[float]) -> float | None:
    materialized: tuple[float, ...] = tuple(values)
    if not materialized:
        return None
    return sum(materialized) / len(materialized)


def _display(value: object) -> str:
    return "not measured" if value is None else str(value)


def _require_non_blank(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def _require_sha256(field_name: str, value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hash")


def _require_positive_integer(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


def _require_non_negative_integer(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _require_non_negative_finite(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _require_positive_finite(field_name: str, value: object) -> None:
    _require_non_negative_finite(field_name, value)
    if value <= 0:  # type: ignore[operator]
        raise ValueError(f"{field_name} must be positive")


def _require_unit_interval(field_name: str, value: object) -> None:
    _require_non_negative_finite(field_name, value)
    if value > 1:  # type: ignore[operator]
        raise ValueError(f"{field_name} must be within [0, 1]")


def _freeze_unique_strings(field_name: str, values: Sequence[str]) -> tuple[str, ...]:
    frozen = tuple(values)
    for value in frozen:
        _require_non_blank(field_name, value)
    if len(set(frozen)) != len(frozen):
        raise ValueError(f"{field_name} must not contain duplicates")
    return frozen


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


__all__ = [
    "VideoEvaluationEvidence",
    "VideoEvaluationEvidenceKind",
    "VideoEvaluationMetrics",
    "VideoEvaluationProfile",
    "VideoEvaluationReport",
    "VideoEvaluationThresholds",
    "VideoQueryMetrics",
    "VideoRankedResult",
    "VideoRetrievalJudgment",
    "evaluate_rankings",
    "recall_between_rankings",
    "render_qualification_summary",
    "temporal_iou",
]
