"""Pure personal reverse-image quality and scale-gate logic.

This module deliberately performs no filesystem, runner, activation, or backend
work. It cannot prove that caller-supplied identities or measurements came from
trusted local evidence. A passing value is only a provisional structural precheck,
never formal capacity evidence or activation authority. A future verifier must
re-read the canonical report and re-run this precheck before making any decision.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol, TypeVar, cast, runtime_checkable

from capicove.contracts.retrieval import SearchIntent
from capicove.retrieval.evaluation import ndcg_at_k, recall_at_k
from capicove.retrieval.search_models import RetrievalLane


PERSONAL_PROFILE_REF = "personal-lite"
PERSONAL_GATE_POLICY_ID = "reverse-image-personal-precheck-v1"
PERSONAL_EXACT_UNIT_COUNT = 10_000
PERSONAL_MAXIMUM_P95_MS = 200.0
PERSONAL_WARMUP_QUERIES_PER_CELL = 100
PERSONAL_MEASURED_QUERIES_PER_CELL = 1_000
PERSONAL_DECLARED_CONCURRENCY = 1
# This only prevents a one-query self-check. It is not an approved domain-quality floor.
MINIMUM_PERSONAL_EVALUATION_QUERY_COUNT = 2
# Exact-reader correctness requires identical ordered results.
PERSONAL_REQUIRED_EXACT_PARITY_METRIC = 1.0
PERSONAL_SEMANTIC_INDEX_FORMAT = "numpy-flat-f32-v1"
MAX_EVALUATION_RANKED_IDS = 100
RESOURCE_FRACTION_CEILING = 0.8

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ValueT = TypeVar("_ValueT")
_REQUIRED_INTENTS = (
    SearchIntent.INSTANCE,
    SearchIntent.CONCEPT,
    SearchIntent.BALANCED,
)
_REQUIRED_QUERY_STAGES = (
    "decode",
    "encode",
    "phash",
    "exact_scan",
    "fusion",
    "catalog",
    "projection",
    "serialization",
)
_REQUIRED_BUILD_STAGES = (
    "unit_manifest_preflight",
    "represent",
    "catalog",
    "perceptual",
    "instance",
    "concept",
    "exact_oracle",
    "validate",
    "publish",
)


class ScaleWorkload(StrEnum):
    """Identify one frozen personal latency selector workload."""

    UNFILTERED = "unfiltered"
    SPARSE_FILTER = "sparse-filter"
    DENSE_BITMAP = "dense-bitmap"


class EvaluationQualityKind(StrEnum):
    """Separate authored domain relevance from exact-index parity."""

    DOMAIN_RELEVANCE = "domain_relevance"
    EXACT_PARITY = "exact_parity"


class PersonalGateReason(StrEnum):
    """Closed, deterministic reasons why the personal precheck failed."""

    BUILD_STAGE_SEQUENCE_MISMATCH = "build_stage_sequence_mismatch"
    BYTES_PER_VECTOR_MISMATCH = "bytes_per_vector_mismatch"
    DOMAIN_QUALITY_FAILED = "domain_quality_failed"
    EXACT_TARGET_COUNT_MISMATCH = "exact_target_count_mismatch"
    EXACT_PARITY_FAILED = "exact_parity_failed"
    UNIT_COUNT_MISMATCH = "unit_count_mismatch"
    LATENCY_CELL_SET_MISMATCH = "latency_cell_set_mismatch"
    LATENCY_EVIDENCE_MISMATCH = "latency_evidence_mismatch"
    LATENCY_P95_EXCEEDED = "latency_p95_exceeded"
    MEMORY_BUDGET_EXCEEDED = "memory_budget_exceeded"
    POLICY_IDENTITY_MISMATCH = "policy_identity_mismatch"
    QUALITY_CELL_SET_MISMATCH = "quality_cell_set_mismatch"
    QUERY_STAGE_SEQUENCE_MISMATCH = "query_stage_sequence_mismatch"
    RUNTIME_INTEGRITY_FAILED = "runtime_integrity_failed"
    SUBJECT_IDENTITY_MISMATCH = "subject_identity_mismatch"
    THREAD_BUDGET_EXCEEDED = "thread_budget_exceeded"
    WORK_ROOT_BUDGET_EXCEEDED = "work_root_budget_exceeded"


@dataclass(frozen=True, slots=True)
class PersonalScaleGatePolicy:
    """Contain the one non-overridable provisional personal precheck policy."""

    policy_id: str
    profile_ref: str
    exact_unit_count: int
    maximum_p95_ms: float
    warmup_queries_per_cell: int
    measured_queries_per_cell: int
    exact_target_count: int
    minimum_evaluation_query_count: int
    required_exact_parity_metric: float
    required_intents: tuple[SearchIntent, ...]
    required_workloads: tuple[ScaleWorkload, ...]
    declared_concurrency: int
    required_query_stages: tuple[str, ...]
    required_build_stages: tuple[str, ...]
    resource_fraction_ceiling: float
    policy_sha256: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        _non_blank("policy_id", self.policy_id)
        _non_blank("profile_ref", self.profile_ref)
        for name in (
            "exact_unit_count",
            "warmup_queries_per_cell",
            "measured_queries_per_cell",
            "exact_target_count",
            "minimum_evaluation_query_count",
            "declared_concurrency",
        ):
            _positive_int(name, getattr(self, name))
        _finite_between("maximum_p95_ms", self.maximum_p95_ms, lower=0.0)
        _finite_between(
            "required_exact_parity_metric",
            self.required_exact_parity_metric,
            lower=0.0,
            upper=1.0,
        )
        _finite_between(
            "resource_fraction_ceiling",
            self.resource_fraction_ceiling,
            lower=0.0,
            upper=1.0,
            lower_exclusive=True,
            upper_exclusive=True,
        )
        _typed_tuple("required_intents", self.required_intents, SearchIntent)
        _typed_tuple("required_workloads", self.required_workloads, ScaleWorkload)
        _unique_non_blank_tuple("required_query_stages", self.required_query_stages)
        _unique_non_blank_tuple("required_build_stages", self.required_build_stages)
        _schema_version(self.schema_version)
        _sha256("policy_sha256", self.policy_sha256)
        if self.policy_sha256 != _policy_sha256(self):
            raise ValueError("policy_sha256 must match the canonical policy body")


def _new_personal_policy() -> PersonalScaleGatePolicy:
    values: dict[str, object] = {
        "policy_id": PERSONAL_GATE_POLICY_ID,
        "profile_ref": PERSONAL_PROFILE_REF,
        "exact_unit_count": PERSONAL_EXACT_UNIT_COUNT,
        "maximum_p95_ms": PERSONAL_MAXIMUM_P95_MS,
        "warmup_queries_per_cell": PERSONAL_WARMUP_QUERIES_PER_CELL,
        "measured_queries_per_cell": PERSONAL_MEASURED_QUERIES_PER_CELL,
        "exact_target_count": PERSONAL_EXACT_UNIT_COUNT,
        "minimum_evaluation_query_count": MINIMUM_PERSONAL_EVALUATION_QUERY_COUNT,
        "required_exact_parity_metric": PERSONAL_REQUIRED_EXACT_PARITY_METRIC,
        "required_intents": _REQUIRED_INTENTS,
        "required_workloads": tuple(ScaleWorkload),
        "declared_concurrency": PERSONAL_DECLARED_CONCURRENCY,
        "required_query_stages": _REQUIRED_QUERY_STAGES,
        "required_build_stages": _REQUIRED_BUILD_STAGES,
        "resource_fraction_ceiling": RESOURCE_FRACTION_CEILING,
        "schema_version": 1,
    }
    digest = hashlib.sha256(
        json.dumps(
            values,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return PersonalScaleGatePolicy(**values, policy_sha256=digest)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class LatencyWorkloadBinding:
    """Bind one latency cell to its preregistered queries and filter."""

    intent: SearchIntent
    workload: ScaleWorkload
    workload_manifest_sha256: str
    query_set_sha256: str
    warmup_query_sha256s: tuple[str, ...]
    measured_query_sha256s: tuple[str, ...]
    filter_sha256: str
    matching_unit_count: int

    def __post_init__(self) -> None:
        _enum("intent", self.intent, SearchIntent)
        _enum("workload", self.workload, ScaleWorkload)
        for name in ("workload_manifest_sha256", "query_set_sha256", "filter_sha256"):
            _sha256(name, getattr(self, name))
        warmup = _unique_sha256_tuple("warmup_query_sha256s", self.warmup_query_sha256s)
        measured = _unique_sha256_tuple("measured_query_sha256s", self.measured_query_sha256s)
        if len(warmup) != PERSONAL_SCALE_POLICY.warmup_queries_per_cell:
            raise ValueError("warmup_query_sha256s must contain exactly 100 values")
        if len(measured) != PERSONAL_SCALE_POLICY.measured_queries_per_cell:
            raise ValueError("measured_query_sha256s must contain exactly 1000 values")
        if set(warmup) & set(measured):
            raise ValueError("warmup and measured query hashes must be disjoint")
        _validate_matching_unit_count(self.workload, self.matching_unit_count)


@dataclass(frozen=True, slots=True)
class PersonalSemanticIndexObservation:
    """Bind one semantic lane to its real personal exact-index parameters."""

    lane: RetrievalLane
    index_format: str
    index_parameters_sha256: str
    exact_candidate_depth: int

    def __post_init__(self) -> None:
        if not isinstance(self.lane, RetrievalLane):
            raise TypeError("lane must be a RetrievalLane")
        if self.lane not in (RetrievalLane.INSTANCE, RetrievalLane.CONCEPT):
            raise ValueError("exact search policy requires instance or concept")
        if self.index_format != PERSONAL_SEMANTIC_INDEX_FORMAT:
            raise ValueError(f"personal semantic index requires {PERSONAL_SEMANTIC_INDEX_FORMAT}")
        _sha256("index_parameters_sha256", self.index_parameters_sha256)
        _positive_int("exact_candidate_depth", self.exact_candidate_depth)
        if self.exact_candidate_depth != PERSONAL_EXACT_UNIT_COUNT:
            raise ValueError("personal exact search requires exact_candidate_depth 10,000")


@dataclass(frozen=True, slots=True)
class PersonalGateBindings:
    """Carry precheck subject identities that a future verifier must resolve."""

    activation_set_id: str
    generation_manifest_sha256: str
    profile_snapshot_sha256: str
    dataset_manifest_sha256: str
    bundle_provisioning_manifest_sha256: str
    unit_manifest_sha256: str
    target_representation_manifest_sha256: str
    evaluation_definition_sha256: str
    evaluation_manifest_sha256: str
    evaluation_query_count: int
    exact_target_selection_sha256: str
    tool_schema_sha256: str
    generation_ids: tuple[str, ...]
    bundle_manifest_sha256s: tuple[str, ...]
    semantic_index_policy: tuple[PersonalSemanticIndexObservation, ...]
    fusion_policy_id: str
    fusion_policy_sha256: str
    domain_quality_policy_id: str
    domain_quality_policy_sha256: str
    instance_labels_sha256: str
    concept_labels_sha256: str
    hard_negatives_sha256: str
    hardware_fingerprint_sha256: str
    work_root_fingerprint_sha256: str
    latency_workloads: tuple[LatencyWorkloadBinding, ...]

    def __post_init__(self) -> None:
        for name in (
            "activation_set_id",
            "fusion_policy_id",
            "domain_quality_policy_id",
        ):
            _non_blank(name, getattr(self, name))
        for name in (
            "generation_manifest_sha256",
            "profile_snapshot_sha256",
            "dataset_manifest_sha256",
            "bundle_provisioning_manifest_sha256",
            "unit_manifest_sha256",
            "target_representation_manifest_sha256",
            "evaluation_definition_sha256",
            "evaluation_manifest_sha256",
            "exact_target_selection_sha256",
            "tool_schema_sha256",
            "fusion_policy_sha256",
            "domain_quality_policy_sha256",
            "instance_labels_sha256",
            "concept_labels_sha256",
            "hard_negatives_sha256",
            "hardware_fingerprint_sha256",
            "work_root_fingerprint_sha256",
        ):
            _sha256(name, getattr(self, name))
        _minimum_int(
            "evaluation_query_count",
            self.evaluation_query_count,
            minimum=PERSONAL_SCALE_POLICY.minimum_evaluation_query_count,
        )
        generation_ids = _unique_non_blank_tuple("generation_ids", self.generation_ids)
        if len(generation_ids) != 3:
            raise ValueError("generation_ids must contain the three personal lane generations")
        bundles = _unique_sha256_tuple(
            "bundle_manifest_sha256s",
            self.bundle_manifest_sha256s,
        )
        if len(bundles) != 2:
            raise ValueError("bundle_manifest_sha256s must contain instance and concept bundles")
        semantic_policy = _typed_tuple(
            "semantic_index_policy",
            self.semantic_index_policy,
            PersonalSemanticIndexObservation,
        )
        semantic_lanes = tuple(item.lane for item in semantic_policy)
        if len(semantic_lanes) != len(set(semantic_lanes)) or set(semantic_lanes) != {
            RetrievalLane.INSTANCE,
            RetrievalLane.CONCEPT,
        }:
            raise ValueError("semantic_index_policy must contain exactly instance and concept")
        if (
            len(
                {
                    self.instance_labels_sha256,
                    self.concept_labels_sha256,
                    self.hard_negatives_sha256,
                }
            )
            != 3
        ):
            raise ValueError("trusted domain evidence digests must be distinct")
        workloads = _typed_tuple(
            "latency_workloads",
            self.latency_workloads,
            LatencyWorkloadBinding,
        )
        keys = tuple((item.intent, item.workload) for item in workloads)
        if len(keys) != len(set(keys)) or set(keys) != _required_latency_keys():
            raise ValueError("latency_workloads must contain the exact nine personal cells")
        for item in workloads:
            _validate_matching_unit_count(item.workload, item.matching_unit_count)


@dataclass(frozen=True, slots=True)
class StageLatency:
    """Record one stage's three finite ordered latency percentiles."""

    stage: str
    p50_ms: float
    p95_ms: float
    p99_ms: float

    def __post_init__(self) -> None:
        _non_blank("stage", self.stage)
        _ordered_percentiles(self.p50_ms, self.p95_ms, self.p99_ms)


@dataclass(frozen=True, slots=True)
class IntentLatency:
    """Record one complete measured personal latency cell."""

    intent: SearchIntent
    workload: ScaleWorkload
    workload_manifest_sha256: str
    query_set_sha256: str
    warmup_query_sha256s: tuple[str, ...]
    measured_query_sha256s: tuple[str, ...]
    filter_sha256: str
    matching_unit_count: int
    sample_count: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    cache_mode: str
    declared_concurrency: int
    stages: tuple[StageLatency, ...]

    def __post_init__(self) -> None:
        _enum("intent", self.intent, SearchIntent)
        _enum("workload", self.workload, ScaleWorkload)
        for name in ("workload_manifest_sha256", "query_set_sha256", "filter_sha256"):
            _sha256(name, getattr(self, name))
        warmup = _unique_sha256_tuple(
            "warmup_query_sha256s",
            self.warmup_query_sha256s,
        )
        measured = _unique_sha256_tuple(
            "measured_query_sha256s",
            self.measured_query_sha256s,
        )
        if set(warmup) & set(measured):
            raise ValueError("warmup and measured query hashes must be disjoint")
        _positive_int("sample_count", self.sample_count)
        if self.sample_count != len(measured):
            raise ValueError("sample_count must equal measured_query_sha256s length")
        _validate_matching_unit_count(self.workload, self.matching_unit_count)
        _ordered_percentiles(self.p50_ms, self.p95_ms, self.p99_ms)
        _non_blank("cache_mode", self.cache_mode)
        _positive_int("declared_concurrency", self.declared_concurrency)
        _typed_tuple("stages", self.stages, StageLatency)


@dataclass(frozen=True, slots=True)
class LaneQuality:
    """Contain one lane and quality-kind metric cell."""

    lane: RetrievalLane
    quality_kind: EvaluationQualityKind
    recall_at_1: float
    recall_at_10: float
    recall_at_100: float
    ndcg_at_10: float

    def __post_init__(self) -> None:
        if not isinstance(self.lane, RetrievalLane):
            raise TypeError("lane must be a RetrievalLane")
        if self.lane not in (RetrievalLane.INSTANCE, RetrievalLane.CONCEPT):
            raise ValueError("lane quality requires instance or concept")
        _enum("quality_kind", self.quality_kind, EvaluationQualityKind)
        for name in ("recall_at_1", "recall_at_10", "recall_at_100", "ndcg_at_10"):
            _finite_between(name, getattr(self, name), lower=0.0, upper=1.0)


@dataclass(frozen=True, slots=True)
class DomainQualityVerdict:
    """Bind an external domain verdict to the exact provisional evidence."""

    policy_id: str
    policy_sha256: str
    instance_labels_sha256: str
    concept_labels_sha256: str
    hard_negatives_sha256: str
    evaluation_manifest_sha256: str
    evaluation_query_count: int
    lane_quality_sha256: str
    passed: bool

    def __post_init__(self) -> None:
        _non_blank("policy_id", self.policy_id)
        for name in (
            "policy_sha256",
            "instance_labels_sha256",
            "concept_labels_sha256",
            "hard_negatives_sha256",
            "evaluation_manifest_sha256",
            "lane_quality_sha256",
        ):
            _sha256(name, getattr(self, name))
        _minimum_int(
            "evaluation_query_count",
            self.evaluation_query_count,
            minimum=PERSONAL_SCALE_POLICY.minimum_evaluation_query_count,
        )
        _boolean("passed", self.passed)
        if (
            len(
                {
                    self.instance_labels_sha256,
                    self.concept_labels_sha256,
                    self.hard_negatives_sha256,
                }
            )
            != 3
        ):
            raise ValueError("domain label and hard-negative digests must be distinct")


@dataclass(frozen=True, slots=True)
class BuildStageObservation:
    """Record bounded resource observations for one build stage."""

    stage: str
    elapsed_seconds: float
    peak_rss_bytes: int
    peak_temp_bytes: int

    def __post_init__(self) -> None:
        _non_blank("stage", self.stage)
        _finite_between("elapsed_seconds", self.elapsed_seconds, lower=0.0)
        _non_negative_int("peak_rss_bytes", self.peak_rss_bytes)
        _non_negative_int("peak_temp_bytes", self.peak_temp_bytes)


@dataclass(frozen=True, slots=True)
class HardwareObservation:
    """Identify the measured offline machine and its resource ceilings."""

    fingerprint_sha256: str
    architecture: str
    logical_cpu_count: int
    memory_bytes: int
    local_free_bytes_at_preflight: int

    def __post_init__(self) -> None:
        _sha256("fingerprint_sha256", self.fingerprint_sha256)
        _non_blank("architecture", self.architecture)
        _positive_int("logical_cpu_count", self.logical_cpu_count)
        _positive_int("memory_bytes", self.memory_bytes)
        _positive_int("local_free_bytes_at_preflight", self.local_free_bytes_at_preflight)


@dataclass(frozen=True, slots=True)
class PersonalScaleReport:
    """Contain pure evidence for one provisional personal precheck."""

    report_id: str
    gate_policy_id: str
    gate_policy_sha256: str
    profile_ref: str
    profile_snapshot_sha256: str
    dataset_manifest_sha256: str
    bundle_provisioning_manifest_sha256: str
    activation_set_id: str
    generation_manifest_sha256: str
    unit_manifest_sha256: str
    target_representation_manifest_sha256: str
    unit_count: int
    evaluation_definition_sha256: str
    evaluation_manifest_sha256: str
    evaluation_query_count: int
    exact_target_selection_sha256: str
    exact_target_count: int
    tool_schema_sha256: str
    generation_ids: tuple[str, ...]
    bundle_manifest_sha256s: tuple[str, ...]
    semantic_index_policy: tuple[PersonalSemanticIndexObservation, ...]
    fusion_policy_id: str
    fusion_policy_sha256: str
    hardware: HardwareObservation
    work_root_fingerprint_sha256: str
    lane_schedule_id: str
    cpu_total_threads: int
    max_parallel_lanes: int
    per_lane_compute_threads: int
    onnx_inter_op_threads: int
    intent_latency: tuple[IntentLatency, ...]
    lane_quality: tuple[LaneQuality, ...]
    domain_quality: DomainQualityVerdict
    build_stages: tuple[BuildStageObservation, ...]
    build_peak_rss_bytes: int
    build_peak_temp_bytes: int
    peak_work_root_bytes: int
    query_peak_rss_bytes: int
    final_index_bytes: int
    bytes_per_vector: float
    cold_start_ms: float
    page_faults: int
    io_read_bytes: int
    generation_switch_ms: float
    readonly_mmap: bool
    candidate_built_on_reported_machine: bool
    production_alias_unchanged: bool
    production_alias_before_sha256: str
    production_alias_after_sha256: str
    created_at: datetime
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in (
            "report_id",
            "gate_policy_id",
            "profile_ref",
            "activation_set_id",
            "fusion_policy_id",
            "lane_schedule_id",
        ):
            _non_blank(name, getattr(self, name))
        for name in (
            "gate_policy_sha256",
            "profile_snapshot_sha256",
            "dataset_manifest_sha256",
            "bundle_provisioning_manifest_sha256",
            "generation_manifest_sha256",
            "unit_manifest_sha256",
            "target_representation_manifest_sha256",
            "evaluation_definition_sha256",
            "evaluation_manifest_sha256",
            "exact_target_selection_sha256",
            "tool_schema_sha256",
            "fusion_policy_sha256",
            "work_root_fingerprint_sha256",
            "production_alias_before_sha256",
            "production_alias_after_sha256",
        ):
            _sha256(name, getattr(self, name))
        _positive_int("unit_count", self.unit_count)
        _positive_int("exact_target_count", self.exact_target_count)
        _minimum_int(
            "evaluation_query_count",
            self.evaluation_query_count,
            minimum=PERSONAL_SCALE_POLICY.minimum_evaluation_query_count,
        )
        _unique_non_blank_tuple("generation_ids", self.generation_ids)
        _unique_sha256_tuple("bundle_manifest_sha256s", self.bundle_manifest_sha256s)
        semantic_policy = _typed_tuple(
            "semantic_index_policy",
            self.semantic_index_policy,
            PersonalSemanticIndexObservation,
        )
        semantic_lanes = tuple(item.lane for item in semantic_policy)
        if len(semantic_lanes) != len(set(semantic_lanes)) or set(semantic_lanes) != {
            RetrievalLane.INSTANCE,
            RetrievalLane.CONCEPT,
        }:
            raise ValueError("semantic_index_policy must contain exactly instance and concept")
        if not isinstance(self.hardware, HardwareObservation):
            raise TypeError("hardware must be a HardwareObservation")
        for name in (
            "cpu_total_threads",
            "max_parallel_lanes",
            "per_lane_compute_threads",
            "onnx_inter_op_threads",
        ):
            _positive_int(name, getattr(self, name))
        _typed_tuple("intent_latency", self.intent_latency, IntentLatency)
        _typed_tuple("lane_quality", self.lane_quality, LaneQuality)
        if not isinstance(self.domain_quality, DomainQualityVerdict):
            raise TypeError("domain_quality must be a DomainQualityVerdict")
        _typed_tuple("build_stages", self.build_stages, BuildStageObservation)
        for name in (
            "build_peak_rss_bytes",
            "build_peak_temp_bytes",
            "peak_work_root_bytes",
            "query_peak_rss_bytes",
            "final_index_bytes",
            "page_faults",
            "io_read_bytes",
        ):
            _non_negative_int(name, getattr(self, name))
        if self.final_index_bytes == 0:
            raise ValueError("final_index_bytes must be positive")
        _finite_between("bytes_per_vector", self.bytes_per_vector, lower=0.0)
        if self.bytes_per_vector == 0:
            raise ValueError("bytes_per_vector must be positive")
        for name in ("cold_start_ms", "generation_switch_ms"):
            _finite_between(name, getattr(self, name), lower=0.0)
        for name in (
            "readonly_mmap",
            "candidate_built_on_reported_machine",
            "production_alias_unchanged",
        ):
            _boolean(name, getattr(self, name))
        _aware_datetime("created_at", self.created_at)
        _schema_version(self.schema_version)


@dataclass(frozen=True, slots=True)
class PersonalScalePrecheckVerdict:
    """Return an internal precheck result that a future verifier must recompute."""

    passed: bool
    tier: str
    formal_capacity_supported: bool
    gate_policy_id: str
    gate_policy_sha256: str
    report_sha256: str
    reason_codes: tuple[PersonalGateReason, ...]
    evaluated_at: datetime
    schema_version: int = 1

    def __post_init__(self) -> None:
        _boolean("passed", self.passed)
        _boolean("formal_capacity_supported", self.formal_capacity_supported)
        if self.tier != "provisional":
            raise ValueError("the personal precheck emits only provisional verdicts")
        if self.formal_capacity_supported:
            raise ValueError("a provisional precheck cannot claim formal capacity support")
        _non_blank("gate_policy_id", self.gate_policy_id)
        _sha256("gate_policy_sha256", self.gate_policy_sha256)
        if (
            self.gate_policy_id != PERSONAL_SCALE_POLICY.policy_id
            or self.gate_policy_sha256 != PERSONAL_SCALE_POLICY.policy_sha256
        ):
            raise ValueError("verdict must bind the fixed personal policy")
        _sha256("report_sha256", self.report_sha256)
        reasons = _typed_tuple("reason_codes", self.reason_codes, PersonalGateReason)
        if tuple(sorted(set(reasons), key=lambda item: item.value)) != reasons:
            raise ValueError("reason_codes must be unique and canonically sorted")
        if self.passed == bool(reasons):
            raise ValueError("passed must be true exactly when reason_codes is empty")
        _aware_datetime("evaluated_at", self.evaluated_at)
        _schema_version(self.schema_version)


def report_sha256(report: PersonalScaleReport) -> str:
    """Hash the exact canonical report document bytes."""
    if not isinstance(report, PersonalScaleReport):
        raise TypeError("report must be a PersonalScaleReport")
    # Local import keeps the pure model module independent from its strict wire codec.
    from capicove.retrieval.scale_codec import encode_personal_scale_report

    return hashlib.sha256(encode_personal_scale_report(report)).hexdigest()


def lane_quality_sha256(observations: tuple[LaneQuality, ...]) -> str:
    """Hash quality cells in their canonical lane/kind order."""
    typed = _typed_tuple("observations", observations, LaneQuality)
    ordered = tuple(sorted(typed, key=lambda item: (item.lane.value, item.quality_kind.value)))
    return _document_sha256(ordered)


def evaluate_personal_scale_precheck(
    report: PersonalScaleReport,
    bindings: PersonalGateBindings,
    *,
    evaluated_at: datetime,
) -> PersonalScalePrecheckVerdict:
    """Evaluate provisional structure without issuing activation authority."""
    if not isinstance(report, PersonalScaleReport):
        raise TypeError("report must be a PersonalScaleReport")
    if not isinstance(bindings, PersonalGateBindings):
        raise TypeError("bindings must be PersonalGateBindings")
    _aware_datetime("evaluated_at", evaluated_at)
    if evaluated_at < report.created_at:
        raise ValueError("evaluated_at must not precede report.created_at")
    reasons: set[PersonalGateReason] = set()

    if (
        report.gate_policy_id != PERSONAL_SCALE_POLICY.policy_id
        or report.gate_policy_sha256 != PERSONAL_SCALE_POLICY.policy_sha256
    ):
        reasons.add(PersonalGateReason.POLICY_IDENTITY_MISMATCH)
    if not _subject_matches(report, bindings):
        reasons.add(PersonalGateReason.SUBJECT_IDENTITY_MISMATCH)
    if report.unit_count != PERSONAL_SCALE_POLICY.exact_unit_count:
        reasons.add(PersonalGateReason.UNIT_COUNT_MISMATCH)
    if report.exact_target_count != PERSONAL_SCALE_POLICY.exact_target_count:
        reasons.add(PersonalGateReason.EXACT_TARGET_COUNT_MISMATCH)

    _evaluate_latency(report, bindings, reasons)
    _evaluate_quality(report, bindings, reasons)
    _evaluate_resources(report, reasons)

    digest = report_sha256(report)
    ordered_reasons = tuple(sorted(reasons, key=lambda item: item.value))
    return PersonalScalePrecheckVerdict(
        passed=not ordered_reasons,
        tier="provisional",
        formal_capacity_supported=False,
        gate_policy_id=PERSONAL_SCALE_POLICY.policy_id,
        gate_policy_sha256=PERSONAL_SCALE_POLICY.policy_sha256,
        report_sha256=digest,
        reason_codes=ordered_reasons,
        evaluated_at=evaluated_at,
    )


def _subject_matches(report: PersonalScaleReport, bindings: PersonalGateBindings) -> bool:
    comparisons = (
        (report.profile_ref, PERSONAL_PROFILE_REF),
        (report.activation_set_id, bindings.activation_set_id),
        (report.generation_manifest_sha256, bindings.generation_manifest_sha256),
        (report.profile_snapshot_sha256, bindings.profile_snapshot_sha256),
        (report.dataset_manifest_sha256, bindings.dataset_manifest_sha256),
        (
            report.bundle_provisioning_manifest_sha256,
            bindings.bundle_provisioning_manifest_sha256,
        ),
        (report.unit_manifest_sha256, bindings.unit_manifest_sha256),
        (
            report.target_representation_manifest_sha256,
            bindings.target_representation_manifest_sha256,
        ),
        (
            report.evaluation_definition_sha256,
            bindings.evaluation_definition_sha256,
        ),
        (report.evaluation_manifest_sha256, bindings.evaluation_manifest_sha256),
        (report.evaluation_query_count, bindings.evaluation_query_count),
        (
            report.exact_target_selection_sha256,
            bindings.exact_target_selection_sha256,
        ),
        (report.tool_schema_sha256, bindings.tool_schema_sha256),
        (report.generation_ids, bindings.generation_ids),
        (report.bundle_manifest_sha256s, bindings.bundle_manifest_sha256s),
        (report.semantic_index_policy, bindings.semantic_index_policy),
        (report.fusion_policy_id, bindings.fusion_policy_id),
        (report.fusion_policy_sha256, bindings.fusion_policy_sha256),
        (report.hardware.fingerprint_sha256, bindings.hardware_fingerprint_sha256),
        (report.work_root_fingerprint_sha256, bindings.work_root_fingerprint_sha256),
    )
    return all(observed == expected for observed, expected in comparisons)


def _evaluate_latency(
    report: PersonalScaleReport,
    bindings: PersonalGateBindings,
    reasons: set[PersonalGateReason],
) -> None:
    observed_keys = tuple((item.intent, item.workload) for item in report.intent_latency)
    if len(observed_keys) != len(set(observed_keys)) or set(observed_keys) != (
        _required_latency_keys()
    ):
        reasons.add(PersonalGateReason.LATENCY_CELL_SET_MISMATCH)
    observed_by_key = {(item.intent, item.workload): item for item in report.intent_latency}
    expected_by_key = {(item.intent, item.workload): item for item in bindings.latency_workloads}
    for key, expected in expected_by_key.items():
        observed = observed_by_key.get(key)
        if observed is None:
            continue
        if (
            observed.workload_manifest_sha256 != expected.workload_manifest_sha256
            or observed.query_set_sha256 != expected.query_set_sha256
            or observed.warmup_query_sha256s != expected.warmup_query_sha256s
            or observed.measured_query_sha256s != expected.measured_query_sha256s
            or observed.filter_sha256 != expected.filter_sha256
            or observed.matching_unit_count != expected.matching_unit_count
            or observed.matching_unit_count > report.unit_count
            or observed.sample_count != PERSONAL_SCALE_POLICY.measured_queries_per_cell
            or len(observed.warmup_query_sha256s) != PERSONAL_SCALE_POLICY.warmup_queries_per_cell
            or observed.cache_mode != "miss"
            or observed.declared_concurrency != PERSONAL_SCALE_POLICY.declared_concurrency
        ):
            reasons.add(PersonalGateReason.LATENCY_EVIDENCE_MISMATCH)
        stage_names = tuple(item.stage for item in observed.stages)
        if stage_names != PERSONAL_SCALE_POLICY.required_query_stages:
            reasons.add(PersonalGateReason.QUERY_STAGE_SEQUENCE_MISMATCH)
        if any(
            stage.p50_ms > observed.p50_ms
            or stage.p95_ms > observed.p95_ms
            or stage.p99_ms > observed.p99_ms
            for stage in observed.stages
        ):
            reasons.add(PersonalGateReason.LATENCY_EVIDENCE_MISMATCH)
        if observed.p95_ms > PERSONAL_SCALE_POLICY.maximum_p95_ms:
            reasons.add(PersonalGateReason.LATENCY_P95_EXCEEDED)


def _evaluate_quality(
    report: PersonalScaleReport,
    bindings: PersonalGateBindings,
    reasons: set[PersonalGateReason],
) -> None:
    expected_keys = {
        (lane, kind)
        for lane in (RetrievalLane.INSTANCE, RetrievalLane.CONCEPT)
        for kind in EvaluationQualityKind
    }
    observed_keys = tuple((item.lane, item.quality_kind) for item in report.lane_quality)
    if len(observed_keys) != len(set(observed_keys)) or set(observed_keys) != expected_keys:
        reasons.add(PersonalGateReason.QUALITY_CELL_SET_MISMATCH)
    for observation in report.lane_quality:
        if observation.quality_kind is EvaluationQualityKind.EXACT_PARITY and any(
            metric != PERSONAL_SCALE_POLICY.required_exact_parity_metric
            for metric in (
                observation.recall_at_1,
                observation.recall_at_10,
                observation.recall_at_100,
                observation.ndcg_at_10,
            )
        ):
            reasons.add(PersonalGateReason.EXACT_PARITY_FAILED)
    domain = report.domain_quality
    if (
        not domain.passed
        or domain.policy_id != bindings.domain_quality_policy_id
        or domain.policy_sha256 != bindings.domain_quality_policy_sha256
        or domain.instance_labels_sha256 != bindings.instance_labels_sha256
        or domain.concept_labels_sha256 != bindings.concept_labels_sha256
        or domain.hard_negatives_sha256 != bindings.hard_negatives_sha256
        or domain.evaluation_manifest_sha256 != report.evaluation_manifest_sha256
        or domain.evaluation_manifest_sha256 != bindings.evaluation_manifest_sha256
        or domain.evaluation_query_count != report.evaluation_query_count
        or domain.evaluation_query_count != bindings.evaluation_query_count
        or domain.lane_quality_sha256 != lane_quality_sha256(report.lane_quality)
    ):
        reasons.add(PersonalGateReason.DOMAIN_QUALITY_FAILED)


def _evaluate_resources(
    report: PersonalScaleReport,
    reasons: set[PersonalGateReason],
) -> None:
    stages = tuple(item.stage for item in report.build_stages)
    if stages != PERSONAL_SCALE_POLICY.required_build_stages:
        reasons.add(PersonalGateReason.BUILD_STAGE_SEQUENCE_MISMATCH)
    if (
        report.max_parallel_lanes * report.per_lane_compute_threads > report.cpu_total_threads
        or report.onnx_inter_op_threads > report.cpu_total_threads
        or report.cpu_total_threads > report.hardware.logical_cpu_count
    ):
        reasons.add(PersonalGateReason.THREAD_BUDGET_EXCEEDED)
    if max(report.build_peak_rss_bytes, report.query_peak_rss_bytes) >= (
        PERSONAL_SCALE_POLICY.resource_fraction_ceiling * report.hardware.memory_bytes
    ):
        reasons.add(PersonalGateReason.MEMORY_BUDGET_EXCEEDED)
    if report.peak_work_root_bytes >= (
        PERSONAL_SCALE_POLICY.resource_fraction_ceiling
        * report.hardware.local_free_bytes_at_preflight
    ):
        reasons.add(PersonalGateReason.WORK_ROOT_BUDGET_EXCEEDED)
    expected_bytes_per_vector = report.final_index_bytes / report.unit_count
    if not math.isclose(
        report.bytes_per_vector,
        expected_bytes_per_vector,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        reasons.add(PersonalGateReason.BYTES_PER_VECTOR_MISMATCH)
    if report.build_stages and (
        report.build_peak_rss_bytes < max(item.peak_rss_bytes for item in report.build_stages)
        or report.build_peak_temp_bytes < max(item.peak_temp_bytes for item in report.build_stages)
    ):
        reasons.add(PersonalGateReason.RUNTIME_INTEGRITY_FAILED)
    if (
        report.peak_work_root_bytes < report.final_index_bytes
        or report.peak_work_root_bytes < report.build_peak_temp_bytes
    ):
        reasons.add(PersonalGateReason.RUNTIME_INTEGRITY_FAILED)
    if (
        not report.readonly_mmap
        or not report.candidate_built_on_reported_machine
        or not report.production_alias_unchanged
        or report.production_alias_before_sha256 != report.production_alias_after_sha256
    ):
        reasons.add(PersonalGateReason.RUNTIME_INTEGRITY_FAILED)


@dataclass(frozen=True, slots=True)
class EvaluationQueryEvidence:
    """Contain verified authored relevance and exact top-100 evidence for one query."""

    query_ref: str
    query_sha256: str
    instance_relevance: tuple[tuple[str, int], ...]
    concept_relevance: tuple[tuple[str, int], ...]
    instance_exact_unit_ids: tuple[str, ...]
    concept_exact_unit_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _non_blank("query_ref", self.query_ref)
        _sha256("query_sha256", self.query_sha256)
        _relevance_tuple("instance_relevance", self.instance_relevance)
        _relevance_tuple("concept_relevance", self.concept_relevance)
        for name in ("instance_exact_unit_ids", "concept_exact_unit_ids"):
            values = _unique_non_blank_tuple(name, getattr(self, name))
            if len(values) != MAX_EVALUATION_RANKED_IDS:
                raise ValueError(f"{name} must contain exactly 100 unit IDs")


@dataclass(frozen=True, slots=True)
class PersonalEvaluationManifest:
    """Carry already verified, bounded pure evidence consumed by the evaluator."""

    evaluation_id: str
    definition_sha256: str
    unit_manifest_sha256: str
    target_representation_manifest_sha256: str
    target_selection_sha256: str
    exact_target_count: int
    generation_manifest_sha256: str
    query_count: int
    queries: tuple[EvaluationQueryEvidence, ...]

    def __post_init__(self) -> None:
        _non_blank("evaluation_id", self.evaluation_id)
        for name in (
            "definition_sha256",
            "unit_manifest_sha256",
            "target_representation_manifest_sha256",
            "target_selection_sha256",
            "generation_manifest_sha256",
        ):
            _sha256(name, getattr(self, name))
        _positive_int("exact_target_count", self.exact_target_count)
        if self.exact_target_count != PERSONAL_EXACT_UNIT_COUNT:
            raise ValueError("exact_target_count must contain all 10,000 personal units")
        queries = _typed_tuple("queries", self.queries, EvaluationQueryEvidence)
        _minimum_int(
            "query_count",
            self.query_count,
            minimum=PERSONAL_SCALE_POLICY.minimum_evaluation_query_count,
        )
        if self.query_count != len(queries):
            raise ValueError("query_count must equal the number of evaluation queries")
        refs = tuple(query.query_ref for query in queries)
        hashes = tuple(query.query_sha256 for query in queries)
        if len(refs) != len(set(refs)) or len(hashes) != len(set(hashes)):
            raise ValueError("evaluation query refs and hashes must be unique")


@dataclass(frozen=True, slots=True)
class RankedEvaluationResult:
    """Carry one bounded candidate ranking for one lane and quality kind."""

    query_ref: str
    query_sha256: str
    intent: SearchIntent
    quality_kind: EvaluationQualityKind
    ranked_unit_ids: tuple[str, ...]
    target_selection_sha256: str
    generation_manifest_sha256: str

    def __post_init__(self) -> None:
        _non_blank("query_ref", self.query_ref)
        _sha256("query_sha256", self.query_sha256)
        if not isinstance(self.intent, SearchIntent):
            raise TypeError("intent must be a SearchIntent")
        if self.intent not in (SearchIntent.INSTANCE, SearchIntent.CONCEPT):
            raise ValueError("evaluation intent must be instance or concept")
        _enum("quality_kind", self.quality_kind, EvaluationQualityKind)
        ranked = _unique_non_blank_tuple("ranked_unit_ids", self.ranked_unit_ids)
        if not ranked:
            raise ValueError("ranked_unit_ids must not be empty")
        if len(ranked) > MAX_EVALUATION_RANKED_IDS:
            raise ValueError("ranked_unit_ids must contain at most 100 values")
        _sha256("target_selection_sha256", self.target_selection_sha256)
        _sha256("generation_manifest_sha256", self.generation_manifest_sha256)


@runtime_checkable
class RetrievalEvaluator(Protocol):
    """Compute separate closed instance/concept quality observations."""

    def evaluate(
        self,
        manifest: PersonalEvaluationManifest,
        rankings: Iterable[RankedEvaluationResult],
        *,
        deadline: datetime,
    ) -> tuple[LaneQuality, ...]: ...


class ExactRetrievalEvaluator:
    """Evaluate verified bounded rankings using the shared metric functions."""

    def evaluate(
        self,
        manifest: PersonalEvaluationManifest,
        rankings: Iterable[RankedEvaluationResult],
        *,
        deadline: datetime,
    ) -> tuple[LaneQuality, ...]:
        if not isinstance(manifest, PersonalEvaluationManifest):
            raise TypeError("manifest must be a PersonalEvaluationManifest")
        _check_deadline(deadline)
        materialized = _consume_rankings(
            rankings,
            expected_count=manifest.query_count * 4,
            deadline=deadline,
        )
        queries: dict[str, EvaluationQueryEvidence] = {}
        expected_keys: set[tuple[str, SearchIntent, EvaluationQualityKind]] = set()
        for query in manifest.queries:
            _check_deadline(deadline)
            queries[query.query_ref] = query
            for intent in (SearchIntent.INSTANCE, SearchIntent.CONCEPT):
                for kind in EvaluationQualityKind:
                    expected_keys.add((query.query_ref, intent, kind))
        observed_keys = tuple(
            (item.query_ref, item.intent, item.quality_kind) for item in materialized
        )
        if set(observed_keys) != expected_keys or any(
            count != 1 for count in Counter(observed_keys).values()
        ):
            raise ValueError("rankings must contain exactly one result for every query cell")
        for result in materialized:
            _check_deadline(deadline)
            query = queries[result.query_ref]
            if result.query_sha256 != query.query_sha256:
                raise ValueError("ranking query hash does not match evaluation evidence")
            if result.target_selection_sha256 != manifest.target_selection_sha256:
                raise ValueError("ranking target selection does not match evaluation evidence")
            if result.generation_manifest_sha256 != manifest.generation_manifest_sha256:
                raise ValueError("ranking generation manifest does not match evaluation evidence")

        by_key = {(item.query_ref, item.intent, item.quality_kind): item for item in materialized}
        qualities: list[LaneQuality] = []
        for lane, intent in (
            (RetrievalLane.INSTANCE, SearchIntent.INSTANCE),
            (RetrievalLane.CONCEPT, SearchIntent.CONCEPT),
        ):
            _check_deadline(deadline)
            for kind in EvaluationQualityKind:
                candidate_rankings: dict[str, tuple[str, ...]] = {}
                for query in manifest.queries:
                    _check_deadline(deadline)
                    candidate_rankings[query.query_ref] = by_key[
                        (query.query_ref, intent, kind)
                    ].ranked_unit_ids
                if kind is EvaluationQualityKind.DOMAIN_RELEVANCE:
                    relevance: dict[str, dict[str, int]] = {}
                    for query in manifest.queries:
                        _check_deadline(deadline)
                        relevance[query.query_ref] = dict(
                            query.instance_relevance
                            if lane is RetrievalLane.INSTANCE
                            else query.concept_relevance
                        )
                    recalls = tuple(
                        recall_at_k(candidate_rankings, relevance, k=cutoff)
                        for cutoff in (1, 10, 100)
                    )
                    ndcg = ndcg_at_k(candidate_rankings, relevance, k=10)
                else:
                    exact: dict[str, tuple[str, ...]] = {}
                    for query in manifest.queries:
                        _check_deadline(deadline)
                        exact[query.query_ref] = (
                            query.instance_exact_unit_ids
                            if lane is RetrievalLane.INSTANCE
                            else query.concept_exact_unit_ids
                        )
                    for query_ref, exact_ranking in exact.items():
                        _check_deadline(deadline)
                        if candidate_rankings[query_ref] != exact_ranking:
                            raise ValueError(
                                "exact parity ranking must equal the complete ordered top-100"
                            )
                    recalls = tuple(
                        recall_at_k(
                            candidate_rankings,
                            {
                                query_ref: {unit_id: 1 for unit_id in exact_ids[:cutoff]}
                                for query_ref, exact_ids in exact.items()
                            },
                            k=cutoff,
                        )
                        for cutoff in (1, 10, 100)
                    )
                    ndcg = ndcg_at_k(
                        candidate_rankings,
                        {
                            query_ref: {
                                unit_id: _exact_rank_grade(rank)
                                for rank, unit_id in enumerate(exact_ids[:10], start=1)
                            }
                            for query_ref, exact_ids in exact.items()
                        },
                        k=10,
                    )
                qualities.append(
                    LaneQuality(
                        lane=lane,
                        quality_kind=kind,
                        recall_at_1=recalls[0],
                        recall_at_10=recalls[1],
                        recall_at_100=recalls[2],
                        ndcg_at_10=ndcg,
                    )
                )
        return tuple(qualities)


def _consume_rankings(
    rankings: Iterable[RankedEvaluationResult],
    *,
    expected_count: int,
    deadline: datetime,
) -> tuple[RankedEvaluationResult, ...]:
    """Consume no more than the expected cells plus one overflow probe."""
    _positive_int("expected_count", expected_count)
    iterator = iter(rankings)
    materialized: list[RankedEvaluationResult] = []
    while True:
        _check_deadline(deadline)
        try:
            item = next(iterator)
        except StopIteration:
            break
        _check_deadline(deadline)
        if len(materialized) == expected_count:
            raise ValueError("rankings must contain exactly one result for every query cell")
        if not isinstance(item, RankedEvaluationResult):
            raise TypeError("rankings must contain RankedEvaluationResult values")
        materialized.append(item)
    if len(materialized) != expected_count:
        raise ValueError("rankings must contain exactly one result for every query cell")
    return tuple(materialized)


def _exact_rank_grade(rank: int) -> int:
    if rank == 1:
        return 3
    if rank <= 3:
        return 2
    return 1


def _required_latency_keys() -> set[tuple[SearchIntent, ScaleWorkload]]:
    return {
        (intent, workload)
        for intent in PERSONAL_SCALE_POLICY.required_intents
        for workload in PERSONAL_SCALE_POLICY.required_workloads
    }


def _policy_sha256(policy: PersonalScaleGatePolicy) -> str:
    return _document_sha256(policy, omitted_fields=frozenset({"policy_sha256"}))


def _document_sha256(
    value: object,
    *,
    omitted_fields: frozenset[str] = frozenset(),
) -> str:
    document = _canonical_value(value, omitted_fields=omitted_fields)
    payload = json.dumps(
        document,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_value(value: object, *, omitted_fields: frozenset[str]) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical_value(
                getattr(value, field.name),
                omitted_fields=frozenset(),
            )
            for field in fields(value)
            if field.name not in omitted_fields
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical mappings must use string keys")
        return {
            cast(str, key): _canonical_value(item, omitted_fields=frozenset())
            for key, item in value.items()
            if key not in omitted_fields
        }
    if isinstance(value, tuple):
        return [_canonical_value(item, omitted_fields=frozenset()) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, datetime):
        _aware_datetime("datetime", value)
        return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical floats must be finite")
        return value
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def _enum(name: str, value: object, enum_type: type[StrEnum]) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(f"{name} must be a {enum_type.__name__}")


def _typed_tuple(
    name: str,
    value: object,
    item_type: type[_ValueT],
) -> tuple[_ValueT, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")
    if any(not isinstance(item, item_type) for item in value):
        raise TypeError(f"{name} contains an invalid value")
    return cast(tuple[_ValueT, ...], value)


def _unique_non_blank_tuple(name: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")
    if any(not isinstance(item, str) for item in value):
        raise TypeError(f"{name} must contain strings")
    result = cast(tuple[str, ...], value)
    for item in result:
        _non_blank(name, item)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} values must be unique")
    return result


def _unique_sha256_tuple(name: str, value: object) -> tuple[str, ...]:
    result = _unique_non_blank_tuple(name, value)
    for item in result:
        _sha256(name, item)
    return result


def _relevance_tuple(name: str, value: object) -> None:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{name} must be a non-empty tuple")
    identifiers: list[str] = []
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(f"{name} must contain (unit_id, grade) tuples")
        unit_id, grade = item
        _non_blank("unit_id", unit_id)
        if isinstance(grade, bool) or not isinstance(grade, int) or not 0 <= grade <= 3:
            raise ValueError("relevance grades must be integers from zero through three")
        identifiers.append(cast(str, unit_id))
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{name} must not repeat a unit ID")
    if not any(cast(int, item[1]) > 0 for item in value):
        raise ValueError(f"{name} must contain positive relevance")


def _ordered_percentiles(p50_ms: object, p95_ms: object, p99_ms: object) -> None:
    for name, value in (("p50_ms", p50_ms), ("p95_ms", p95_ms), ("p99_ms", p99_ms)):
        _finite_between(name, value, lower=0.0)
    if not cast(float, p50_ms) <= cast(float, p95_ms) <= cast(float, p99_ms):
        raise ValueError("p50_ms, p95_ms, and p99_ms must be ordered")


def _finite_between(
    name: str,
    value: object,
    *,
    lower: float,
    upper: float | None = None,
    lower_exclusive: bool = False,
    upper_exclusive: bool = False,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if normalized < lower or (lower_exclusive and normalized == lower):
        raise ValueError(f"{name} is below its valid range")
    if upper is not None and (normalized > upper or (upper_exclusive and normalized == upper)):
        raise ValueError(f"{name} is above its valid range")


def _non_blank(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a non-blank string")


def _sha256(name: str, value: object) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _minimum_int(name: str, value: object, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _non_negative_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_matching_unit_count(workload: ScaleWorkload, value: object) -> None:
    _enum("workload", workload, ScaleWorkload)
    _non_negative_int("matching_unit_count", value)
    count = cast(int, value)
    if workload is ScaleWorkload.UNFILTERED:
        if count != PERSONAL_EXACT_UNIT_COUNT:
            raise ValueError("unfiltered workload must match all 10,000 personal units")
        return
    if not 0 < count < PERSONAL_EXACT_UNIT_COUNT:
        raise ValueError("filtered workload must match from 1 through 9,999 personal units")


def _boolean(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")


def _aware_datetime(name: str, value: object) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")


def _schema_version(value: object) -> None:
    if isinstance(value, bool) or value != 1:
        raise ValueError("schema_version must be 1")


def _check_deadline(deadline: datetime) -> None:
    _aware_datetime("deadline", deadline)
    if datetime.now(UTC) >= deadline.astimezone(UTC):
        raise TimeoutError("evaluation deadline exceeded")


PERSONAL_SCALE_POLICY = _new_personal_policy()


__all__ = [
    "MAX_EVALUATION_RANKED_IDS",
    "MINIMUM_PERSONAL_EVALUATION_QUERY_COUNT",
    "PERSONAL_SCALE_POLICY",
    "BuildStageObservation",
    "DomainQualityVerdict",
    "EvaluationQualityKind",
    "EvaluationQueryEvidence",
    "ExactRetrievalEvaluator",
    "HardwareObservation",
    "IntentLatency",
    "LaneQuality",
    "LatencyWorkloadBinding",
    "PersonalEvaluationManifest",
    "PersonalGateBindings",
    "PersonalGateReason",
    "PersonalScaleGatePolicy",
    "PersonalScalePrecheckVerdict",
    "PersonalScaleReport",
    "PersonalSemanticIndexObservation",
    "RankedEvaluationResult",
    "RetrievalEvaluator",
    "ScaleWorkload",
    "StageLatency",
    "evaluate_personal_scale_precheck",
    "lane_quality_sha256",
    "report_sha256",
]
