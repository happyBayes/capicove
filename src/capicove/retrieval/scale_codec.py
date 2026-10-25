"""Strict canonical JSON codec for personal scale precheck evidence.

This module only maps immutable in-memory evidence to and from bytes.  It performs
no filesystem, store, activation, runner, or backend work. Decoding a verdict does
not authenticate it: a future verifier must re-read its report and re-run the
personal precheck before making any decision.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import TypeVar, cast

from capicove.contracts.retrieval import SearchIntent
from capicove.retrieval.scale import (
    BuildStageObservation,
    DomainQualityVerdict,
    EvaluationQualityKind,
    HardwareObservation,
    IntentLatency,
    LaneQuality,
    PersonalGateReason,
    PersonalScalePrecheckVerdict,
    PersonalScaleReport,
    PersonalSemanticIndexObservation,
    ScaleWorkload,
    StageLatency,
)
from capicove.retrieval.search_models import RetrievalLane


MAX_PERSONAL_SCALE_REPORT_BYTES = 16 * 1024 * 1024
MAX_PERSONAL_SCALE_VERDICT_BYTES = 1024 * 1024

_TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")
_EnumT = TypeVar("_EnumT", bound=StrEnum)


def encode_personal_scale_report(report: PersonalScaleReport, /) -> bytes:
    """Encode one report as bounded canonical UTF-8 JSON plus one line feed."""
    if not isinstance(report, PersonalScaleReport):
        raise TypeError("report must be a PersonalScaleReport")
    return _encode_document(
        _encode_report(report),
        maximum_bytes=MAX_PERSONAL_SCALE_REPORT_BYTES,
        context="personal scale report",
    )


def decode_personal_scale_report(payload: bytes, /) -> PersonalScaleReport:
    """Strictly decode one bounded canonical personal scale report."""
    document = _parse_document(
        payload,
        maximum_bytes=MAX_PERSONAL_SCALE_REPORT_BYTES,
        context="personal scale report",
    )
    report = _decode_report(document)
    if encode_personal_scale_report(report) != payload:
        raise ValueError("personal scale report must use canonical JSON encoding")
    return report


def encode_personal_scale_precheck_verdict(
    verdict: PersonalScalePrecheckVerdict,
    /,
) -> bytes:
    """Encode one provisional verdict as bounded canonical JSON plus one line feed."""
    if not isinstance(verdict, PersonalScalePrecheckVerdict):
        raise TypeError("verdict must be a PersonalScalePrecheckVerdict")
    return _encode_document(
        _encode_verdict(verdict),
        maximum_bytes=MAX_PERSONAL_SCALE_VERDICT_BYTES,
        context="personal scale precheck verdict",
    )


def decode_personal_scale_precheck_verdict(
    payload: bytes,
    /,
) -> PersonalScalePrecheckVerdict:
    """Strictly decode one bounded canonical provisional verdict."""
    document = _parse_document(
        payload,
        maximum_bytes=MAX_PERSONAL_SCALE_VERDICT_BYTES,
        context="personal scale precheck verdict",
    )
    verdict = _decode_verdict(document)
    if encode_personal_scale_precheck_verdict(verdict) != payload:
        raise ValueError("personal scale precheck verdict must use canonical JSON encoding")
    return verdict


def verdict_sha256(verdict: PersonalScalePrecheckVerdict, /) -> str:
    """Hash the exact canonical verdict document bytes."""
    return hashlib.sha256(encode_personal_scale_precheck_verdict(verdict)).hexdigest()


def _encode_report(report: PersonalScaleReport) -> dict[str, object]:
    return {
        "activation_set_id": report.activation_set_id,
        "build_peak_rss_bytes": report.build_peak_rss_bytes,
        "build_peak_temp_bytes": report.build_peak_temp_bytes,
        "build_stages": [_encode_build_stage(item) for item in report.build_stages],
        "bundle_manifest_sha256s": list(report.bundle_manifest_sha256s),
        "bundle_provisioning_manifest_sha256": (report.bundle_provisioning_manifest_sha256),
        "bytes_per_vector": float(report.bytes_per_vector),
        "candidate_built_on_reported_machine": (report.candidate_built_on_reported_machine),
        "cold_start_ms": float(report.cold_start_ms),
        "cpu_total_threads": report.cpu_total_threads,
        "created_at": _encode_timestamp(report.created_at),
        "dataset_manifest_sha256": report.dataset_manifest_sha256,
        "domain_quality": _encode_domain_quality(report.domain_quality),
        "evaluation_definition_sha256": report.evaluation_definition_sha256,
        "evaluation_manifest_sha256": report.evaluation_manifest_sha256,
        "evaluation_query_count": report.evaluation_query_count,
        "exact_target_count": report.exact_target_count,
        "exact_target_selection_sha256": report.exact_target_selection_sha256,
        "final_index_bytes": report.final_index_bytes,
        "fusion_policy_id": report.fusion_policy_id,
        "fusion_policy_sha256": report.fusion_policy_sha256,
        "gate_policy_id": report.gate_policy_id,
        "gate_policy_sha256": report.gate_policy_sha256,
        "generation_ids": list(report.generation_ids),
        "generation_manifest_sha256": report.generation_manifest_sha256,
        "generation_switch_ms": float(report.generation_switch_ms),
        "hardware": _encode_hardware(report.hardware),
        "intent_latency": [_encode_intent_latency(item) for item in report.intent_latency],
        "io_read_bytes": report.io_read_bytes,
        "lane_quality": [_encode_lane_quality(item) for item in report.lane_quality],
        "lane_schedule_id": report.lane_schedule_id,
        "max_parallel_lanes": report.max_parallel_lanes,
        "onnx_inter_op_threads": report.onnx_inter_op_threads,
        "page_faults": report.page_faults,
        "peak_work_root_bytes": report.peak_work_root_bytes,
        "per_lane_compute_threads": report.per_lane_compute_threads,
        "production_alias_after_sha256": report.production_alias_after_sha256,
        "production_alias_before_sha256": report.production_alias_before_sha256,
        "production_alias_unchanged": report.production_alias_unchanged,
        "profile_ref": report.profile_ref,
        "profile_snapshot_sha256": report.profile_snapshot_sha256,
        "query_peak_rss_bytes": report.query_peak_rss_bytes,
        "readonly_mmap": report.readonly_mmap,
        "report_id": report.report_id,
        "schema_version": report.schema_version,
        "semantic_index_policy": [
            _encode_semantic_index(item) for item in report.semantic_index_policy
        ],
        "target_representation_manifest_sha256": (report.target_representation_manifest_sha256),
        "tool_schema_sha256": report.tool_schema_sha256,
        "unit_count": report.unit_count,
        "unit_manifest_sha256": report.unit_manifest_sha256,
        "work_root_fingerprint_sha256": report.work_root_fingerprint_sha256,
    }


def _encode_semantic_index(
    observation: PersonalSemanticIndexObservation,
) -> dict[str, object]:
    return {
        "exact_candidate_depth": observation.exact_candidate_depth,
        "index_format": observation.index_format,
        "index_parameters_sha256": observation.index_parameters_sha256,
        "lane": observation.lane.value,
    }


def _encode_hardware(observation: HardwareObservation) -> dict[str, object]:
    return {
        "architecture": observation.architecture,
        "fingerprint_sha256": observation.fingerprint_sha256,
        "local_free_bytes_at_preflight": observation.local_free_bytes_at_preflight,
        "logical_cpu_count": observation.logical_cpu_count,
        "memory_bytes": observation.memory_bytes,
    }


def _encode_intent_latency(observation: IntentLatency) -> dict[str, object]:
    return {
        "cache_mode": observation.cache_mode,
        "declared_concurrency": observation.declared_concurrency,
        "filter_sha256": observation.filter_sha256,
        "intent": observation.intent.value,
        "matching_unit_count": observation.matching_unit_count,
        "measured_query_sha256s": list(observation.measured_query_sha256s),
        "p50_ms": float(observation.p50_ms),
        "p95_ms": float(observation.p95_ms),
        "p99_ms": float(observation.p99_ms),
        "query_set_sha256": observation.query_set_sha256,
        "sample_count": observation.sample_count,
        "stages": [_encode_stage_latency(item) for item in observation.stages],
        "warmup_query_sha256s": list(observation.warmup_query_sha256s),
        "workload": observation.workload.value,
        "workload_manifest_sha256": observation.workload_manifest_sha256,
    }


def _encode_stage_latency(observation: StageLatency) -> dict[str, object]:
    return {
        "p50_ms": float(observation.p50_ms),
        "p95_ms": float(observation.p95_ms),
        "p99_ms": float(observation.p99_ms),
        "stage": observation.stage,
    }


def _encode_lane_quality(observation: LaneQuality) -> dict[str, object]:
    return {
        "lane": observation.lane.value,
        "ndcg_at_10": float(observation.ndcg_at_10),
        "quality_kind": observation.quality_kind.value,
        "recall_at_1": float(observation.recall_at_1),
        "recall_at_10": float(observation.recall_at_10),
        "recall_at_100": float(observation.recall_at_100),
    }


def _encode_domain_quality(verdict: DomainQualityVerdict) -> dict[str, object]:
    return {
        "concept_labels_sha256": verdict.concept_labels_sha256,
        "evaluation_manifest_sha256": verdict.evaluation_manifest_sha256,
        "evaluation_query_count": verdict.evaluation_query_count,
        "hard_negatives_sha256": verdict.hard_negatives_sha256,
        "instance_labels_sha256": verdict.instance_labels_sha256,
        "lane_quality_sha256": verdict.lane_quality_sha256,
        "passed": verdict.passed,
        "policy_id": verdict.policy_id,
        "policy_sha256": verdict.policy_sha256,
    }


def _encode_build_stage(observation: BuildStageObservation) -> dict[str, object]:
    return {
        "elapsed_seconds": float(observation.elapsed_seconds),
        "peak_rss_bytes": observation.peak_rss_bytes,
        "peak_temp_bytes": observation.peak_temp_bytes,
        "stage": observation.stage,
    }


def _encode_verdict(verdict: PersonalScalePrecheckVerdict) -> dict[str, object]:
    return {
        "evaluated_at": _encode_timestamp(verdict.evaluated_at),
        "formal_capacity_supported": verdict.formal_capacity_supported,
        "gate_policy_id": verdict.gate_policy_id,
        "gate_policy_sha256": verdict.gate_policy_sha256,
        "passed": verdict.passed,
        "reason_codes": [reason.value for reason in verdict.reason_codes],
        "report_sha256": verdict.report_sha256,
        "schema_version": verdict.schema_version,
        "tier": verdict.tier,
    }


def _decode_report(raw: object) -> PersonalScaleReport:
    values = _closed_object(
        raw,
        frozenset(
            {
                "activation_set_id",
                "build_peak_rss_bytes",
                "build_peak_temp_bytes",
                "build_stages",
                "bundle_manifest_sha256s",
                "bundle_provisioning_manifest_sha256",
                "bytes_per_vector",
                "candidate_built_on_reported_machine",
                "cold_start_ms",
                "cpu_total_threads",
                "created_at",
                "dataset_manifest_sha256",
                "domain_quality",
                "evaluation_definition_sha256",
                "evaluation_manifest_sha256",
                "evaluation_query_count",
                "exact_target_count",
                "exact_target_selection_sha256",
                "final_index_bytes",
                "fusion_policy_id",
                "fusion_policy_sha256",
                "gate_policy_id",
                "gate_policy_sha256",
                "generation_ids",
                "generation_manifest_sha256",
                "generation_switch_ms",
                "hardware",
                "intent_latency",
                "io_read_bytes",
                "lane_quality",
                "lane_schedule_id",
                "max_parallel_lanes",
                "onnx_inter_op_threads",
                "page_faults",
                "peak_work_root_bytes",
                "per_lane_compute_threads",
                "production_alias_after_sha256",
                "production_alias_before_sha256",
                "production_alias_unchanged",
                "profile_ref",
                "profile_snapshot_sha256",
                "query_peak_rss_bytes",
                "readonly_mmap",
                "report_id",
                "schema_version",
                "semantic_index_policy",
                "target_representation_manifest_sha256",
                "tool_schema_sha256",
                "unit_count",
                "unit_manifest_sha256",
                "work_root_fingerprint_sha256",
            }
        ),
        "personal scale report",
    )
    return PersonalScaleReport(
        report_id=_string(values["report_id"], "report_id"),
        gate_policy_id=_string(values["gate_policy_id"], "gate_policy_id"),
        gate_policy_sha256=_string(
            values["gate_policy_sha256"],
            "gate_policy_sha256",
        ),
        profile_ref=_string(values["profile_ref"], "profile_ref"),
        profile_snapshot_sha256=_string(
            values["profile_snapshot_sha256"],
            "profile_snapshot_sha256",
        ),
        dataset_manifest_sha256=_string(
            values["dataset_manifest_sha256"],
            "dataset_manifest_sha256",
        ),
        bundle_provisioning_manifest_sha256=_string(
            values["bundle_provisioning_manifest_sha256"],
            "bundle_provisioning_manifest_sha256",
        ),
        activation_set_id=_string(values["activation_set_id"], "activation_set_id"),
        generation_manifest_sha256=_string(
            values["generation_manifest_sha256"],
            "generation_manifest_sha256",
        ),
        unit_manifest_sha256=_string(
            values["unit_manifest_sha256"],
            "unit_manifest_sha256",
        ),
        target_representation_manifest_sha256=_string(
            values["target_representation_manifest_sha256"],
            "target_representation_manifest_sha256",
        ),
        unit_count=_integer(values["unit_count"], "unit_count"),
        evaluation_definition_sha256=_string(
            values["evaluation_definition_sha256"],
            "evaluation_definition_sha256",
        ),
        evaluation_manifest_sha256=_string(
            values["evaluation_manifest_sha256"],
            "evaluation_manifest_sha256",
        ),
        evaluation_query_count=_integer(
            values["evaluation_query_count"],
            "evaluation_query_count",
        ),
        exact_target_selection_sha256=_string(
            values["exact_target_selection_sha256"],
            "exact_target_selection_sha256",
        ),
        exact_target_count=_integer(values["exact_target_count"], "exact_target_count"),
        tool_schema_sha256=_string(values["tool_schema_sha256"], "tool_schema_sha256"),
        generation_ids=_string_tuple(values["generation_ids"], "generation_ids"),
        bundle_manifest_sha256s=_string_tuple(
            values["bundle_manifest_sha256s"],
            "bundle_manifest_sha256s",
        ),
        semantic_index_policy=tuple(
            _decode_semantic_index(item)
            for item in _array(values["semantic_index_policy"], "semantic_index_policy")
        ),
        fusion_policy_id=_string(values["fusion_policy_id"], "fusion_policy_id"),
        fusion_policy_sha256=_string(
            values["fusion_policy_sha256"],
            "fusion_policy_sha256",
        ),
        hardware=_decode_hardware(values["hardware"]),
        work_root_fingerprint_sha256=_string(
            values["work_root_fingerprint_sha256"],
            "work_root_fingerprint_sha256",
        ),
        lane_schedule_id=_string(values["lane_schedule_id"], "lane_schedule_id"),
        cpu_total_threads=_integer(values["cpu_total_threads"], "cpu_total_threads"),
        max_parallel_lanes=_integer(
            values["max_parallel_lanes"],
            "max_parallel_lanes",
        ),
        per_lane_compute_threads=_integer(
            values["per_lane_compute_threads"],
            "per_lane_compute_threads",
        ),
        onnx_inter_op_threads=_integer(
            values["onnx_inter_op_threads"],
            "onnx_inter_op_threads",
        ),
        intent_latency=tuple(
            _decode_intent_latency(item)
            for item in _array(values["intent_latency"], "intent_latency")
        ),
        lane_quality=tuple(
            _decode_lane_quality(item) for item in _array(values["lane_quality"], "lane_quality")
        ),
        domain_quality=_decode_domain_quality(values["domain_quality"]),
        build_stages=tuple(
            _decode_build_stage(item) for item in _array(values["build_stages"], "build_stages")
        ),
        build_peak_rss_bytes=_integer(
            values["build_peak_rss_bytes"],
            "build_peak_rss_bytes",
        ),
        build_peak_temp_bytes=_integer(
            values["build_peak_temp_bytes"],
            "build_peak_temp_bytes",
        ),
        peak_work_root_bytes=_integer(
            values["peak_work_root_bytes"],
            "peak_work_root_bytes",
        ),
        query_peak_rss_bytes=_integer(
            values["query_peak_rss_bytes"],
            "query_peak_rss_bytes",
        ),
        final_index_bytes=_integer(values["final_index_bytes"], "final_index_bytes"),
        bytes_per_vector=_float(values["bytes_per_vector"], "bytes_per_vector"),
        cold_start_ms=_float(values["cold_start_ms"], "cold_start_ms"),
        page_faults=_integer(values["page_faults"], "page_faults"),
        io_read_bytes=_integer(values["io_read_bytes"], "io_read_bytes"),
        generation_switch_ms=_float(
            values["generation_switch_ms"],
            "generation_switch_ms",
        ),
        readonly_mmap=_boolean(values["readonly_mmap"], "readonly_mmap"),
        candidate_built_on_reported_machine=_boolean(
            values["candidate_built_on_reported_machine"],
            "candidate_built_on_reported_machine",
        ),
        production_alias_unchanged=_boolean(
            values["production_alias_unchanged"],
            "production_alias_unchanged",
        ),
        production_alias_before_sha256=_string(
            values["production_alias_before_sha256"],
            "production_alias_before_sha256",
        ),
        production_alias_after_sha256=_string(
            values["production_alias_after_sha256"],
            "production_alias_after_sha256",
        ),
        created_at=_decode_timestamp(values["created_at"], "created_at"),
        schema_version=_integer(values["schema_version"], "schema_version"),
    )


def _decode_semantic_index(raw: object) -> PersonalSemanticIndexObservation:
    values = _closed_object(
        raw,
        frozenset(
            {
                "exact_candidate_depth",
                "index_format",
                "index_parameters_sha256",
                "lane",
            }
        ),
        "semantic index observation",
    )
    return PersonalSemanticIndexObservation(
        lane=_enum(RetrievalLane, values["lane"], "lane"),
        index_format=_string(values["index_format"], "index_format"),
        index_parameters_sha256=_string(
            values["index_parameters_sha256"],
            "index_parameters_sha256",
        ),
        exact_candidate_depth=_integer(
            values["exact_candidate_depth"],
            "exact_candidate_depth",
        ),
    )


def _decode_hardware(raw: object) -> HardwareObservation:
    values = _closed_object(
        raw,
        frozenset(
            {
                "architecture",
                "fingerprint_sha256",
                "local_free_bytes_at_preflight",
                "logical_cpu_count",
                "memory_bytes",
            }
        ),
        "hardware observation",
    )
    return HardwareObservation(
        fingerprint_sha256=_string(
            values["fingerprint_sha256"],
            "fingerprint_sha256",
        ),
        architecture=_string(values["architecture"], "architecture"),
        logical_cpu_count=_integer(
            values["logical_cpu_count"],
            "logical_cpu_count",
        ),
        memory_bytes=_integer(values["memory_bytes"], "memory_bytes"),
        local_free_bytes_at_preflight=_integer(
            values["local_free_bytes_at_preflight"],
            "local_free_bytes_at_preflight",
        ),
    )


def _decode_intent_latency(raw: object) -> IntentLatency:
    values = _closed_object(
        raw,
        frozenset(
            {
                "cache_mode",
                "declared_concurrency",
                "filter_sha256",
                "intent",
                "matching_unit_count",
                "measured_query_sha256s",
                "p50_ms",
                "p95_ms",
                "p99_ms",
                "query_set_sha256",
                "sample_count",
                "stages",
                "warmup_query_sha256s",
                "workload",
                "workload_manifest_sha256",
            }
        ),
        "intent latency observation",
    )
    return IntentLatency(
        intent=_enum(SearchIntent, values["intent"], "intent"),
        workload=_enum(ScaleWorkload, values["workload"], "workload"),
        workload_manifest_sha256=_string(
            values["workload_manifest_sha256"],
            "workload_manifest_sha256",
        ),
        query_set_sha256=_string(values["query_set_sha256"], "query_set_sha256"),
        warmup_query_sha256s=_string_tuple(
            values["warmup_query_sha256s"],
            "warmup_query_sha256s",
        ),
        measured_query_sha256s=_string_tuple(
            values["measured_query_sha256s"],
            "measured_query_sha256s",
        ),
        filter_sha256=_string(values["filter_sha256"], "filter_sha256"),
        matching_unit_count=_integer(
            values["matching_unit_count"],
            "matching_unit_count",
        ),
        sample_count=_integer(values["sample_count"], "sample_count"),
        p50_ms=_float(values["p50_ms"], "p50_ms"),
        p95_ms=_float(values["p95_ms"], "p95_ms"),
        p99_ms=_float(values["p99_ms"], "p99_ms"),
        cache_mode=_string(values["cache_mode"], "cache_mode"),
        declared_concurrency=_integer(
            values["declared_concurrency"],
            "declared_concurrency",
        ),
        stages=tuple(_decode_stage_latency(item) for item in _array(values["stages"], "stages")),
    )


def _decode_stage_latency(raw: object) -> StageLatency:
    values = _closed_object(
        raw,
        frozenset({"p50_ms", "p95_ms", "p99_ms", "stage"}),
        "stage latency observation",
    )
    return StageLatency(
        stage=_string(values["stage"], "stage"),
        p50_ms=_float(values["p50_ms"], "p50_ms"),
        p95_ms=_float(values["p95_ms"], "p95_ms"),
        p99_ms=_float(values["p99_ms"], "p99_ms"),
    )


def _decode_lane_quality(raw: object) -> LaneQuality:
    values = _closed_object(
        raw,
        frozenset(
            {
                "lane",
                "ndcg_at_10",
                "quality_kind",
                "recall_at_1",
                "recall_at_10",
                "recall_at_100",
            }
        ),
        "lane quality observation",
    )
    return LaneQuality(
        lane=_enum(RetrievalLane, values["lane"], "lane"),
        quality_kind=_enum(
            EvaluationQualityKind,
            values["quality_kind"],
            "quality_kind",
        ),
        recall_at_1=_float(values["recall_at_1"], "recall_at_1"),
        recall_at_10=_float(values["recall_at_10"], "recall_at_10"),
        recall_at_100=_float(values["recall_at_100"], "recall_at_100"),
        ndcg_at_10=_float(values["ndcg_at_10"], "ndcg_at_10"),
    )


def _decode_domain_quality(raw: object) -> DomainQualityVerdict:
    values = _closed_object(
        raw,
        frozenset(
            {
                "concept_labels_sha256",
                "evaluation_manifest_sha256",
                "evaluation_query_count",
                "hard_negatives_sha256",
                "instance_labels_sha256",
                "lane_quality_sha256",
                "passed",
                "policy_id",
                "policy_sha256",
            }
        ),
        "domain quality verdict",
    )
    return DomainQualityVerdict(
        policy_id=_string(values["policy_id"], "policy_id"),
        policy_sha256=_string(values["policy_sha256"], "policy_sha256"),
        instance_labels_sha256=_string(
            values["instance_labels_sha256"],
            "instance_labels_sha256",
        ),
        concept_labels_sha256=_string(
            values["concept_labels_sha256"],
            "concept_labels_sha256",
        ),
        hard_negatives_sha256=_string(
            values["hard_negatives_sha256"],
            "hard_negatives_sha256",
        ),
        evaluation_manifest_sha256=_string(
            values["evaluation_manifest_sha256"],
            "evaluation_manifest_sha256",
        ),
        evaluation_query_count=_integer(
            values["evaluation_query_count"],
            "evaluation_query_count",
        ),
        lane_quality_sha256=_string(
            values["lane_quality_sha256"],
            "lane_quality_sha256",
        ),
        passed=_boolean(values["passed"], "passed"),
    )


def _decode_build_stage(raw: object) -> BuildStageObservation:
    values = _closed_object(
        raw,
        frozenset({"elapsed_seconds", "peak_rss_bytes", "peak_temp_bytes", "stage"}),
        "build stage observation",
    )
    return BuildStageObservation(
        stage=_string(values["stage"], "stage"),
        elapsed_seconds=_float(values["elapsed_seconds"], "elapsed_seconds"),
        peak_rss_bytes=_integer(values["peak_rss_bytes"], "peak_rss_bytes"),
        peak_temp_bytes=_integer(values["peak_temp_bytes"], "peak_temp_bytes"),
    )


def _decode_verdict(raw: object) -> PersonalScalePrecheckVerdict:
    values = _closed_object(
        raw,
        frozenset(
            {
                "evaluated_at",
                "formal_capacity_supported",
                "gate_policy_id",
                "gate_policy_sha256",
                "passed",
                "reason_codes",
                "report_sha256",
                "schema_version",
                "tier",
            }
        ),
        "personal scale precheck verdict",
    )
    return PersonalScalePrecheckVerdict(
        passed=_boolean(values["passed"], "passed"),
        tier=_string(values["tier"], "tier"),
        formal_capacity_supported=_boolean(
            values["formal_capacity_supported"],
            "formal_capacity_supported",
        ),
        gate_policy_id=_string(values["gate_policy_id"], "gate_policy_id"),
        gate_policy_sha256=_string(
            values["gate_policy_sha256"],
            "gate_policy_sha256",
        ),
        report_sha256=_string(values["report_sha256"], "report_sha256"),
        reason_codes=tuple(
            _enum(PersonalGateReason, item, "reason_codes item")
            for item in _array(values["reason_codes"], "reason_codes")
        ),
        evaluated_at=_decode_timestamp(values["evaluated_at"], "evaluated_at"),
        schema_version=_integer(values["schema_version"], "schema_version"),
    )


def _parse_document(
    payload: bytes,
    *,
    maximum_bytes: int,
    context: str,
) -> dict[str, object]:
    if not isinstance(payload, bytes):
        raise TypeError(f"{context} payload must be bytes")
    if not payload:
        raise ValueError(f"{context} payload must not be empty")
    if len(payload) > maximum_bytes:
        raise ValueError(f"{context} exceeds {maximum_bytes} bytes")
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_non_finite_json_number,
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"{context} must be valid UTF-8 JSON") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{context} must be valid JSON") from error
    if not isinstance(document, dict):
        raise ValueError(f"{context} must be a JSON object")
    return cast(dict[str, object], document)


def _encode_document(
    document: dict[str, object],
    *,
    maximum_bytes: int,
    context: str,
) -> bytes:
    try:
        payload = (
            json.dumps(
                document,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must contain finite JSON values") from error
    if len(payload) > maximum_bytes:
        raise ValueError(f"{context} exceeds {maximum_bytes} bytes")
    return payload


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    values: dict[str, object] = {}
    for name, value in pairs:
        if name in values:
            raise ValueError(f"duplicate JSON field: {name}")
        values[name] = value
    return values


def _reject_non_finite_json_number(value: str) -> object:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _closed_object(
    raw: object,
    expected_fields: frozenset[str],
    context: str,
) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be a JSON object")
    values = cast(dict[str, object], raw)
    unknown = sorted(set(values) - expected_fields)
    if unknown:
        raise ValueError(f"unknown {context} fields: {', '.join(unknown)}")
    missing = sorted(expected_fields - set(values))
    if missing:
        raise ValueError(f"missing {context} fields: {', '.join(missing)}")
    return values


def _array(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a JSON array")
    return cast(list[object], value)


def _string_tuple(value: object, context: str) -> tuple[str, ...]:
    return tuple(_string(item, f"{context} item") for item in _array(value, context))


def _string(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string")
    return value


def _integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    return value


def _float(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite number")
    result = float(value)
    if result == float("inf") or result == float("-inf") or result != result:
        raise ValueError(f"{context} must be a finite number")
    return result


def _boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be a boolean")
    return value


def _enum(enum_type: type[_EnumT], value: object, context: str) -> _EnumT:
    encoded = _string(value, context)
    try:
        return enum_type(encoded)
    except ValueError as error:
        raise ValueError(f"{context} must be a valid {enum_type.__name__}") from error


def _encode_timestamp(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _decode_timestamp(value: object, context: str) -> datetime:
    encoded = _string(value, context)
    if _TIMESTAMP_PATTERN.fullmatch(encoded) is None:
        raise ValueError(f"{context} must be a canonical UTC timestamp")
    try:
        return datetime.fromisoformat(encoded[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{context} must be a valid UTC timestamp") from error


__all__ = [
    "MAX_PERSONAL_SCALE_REPORT_BYTES",
    "MAX_PERSONAL_SCALE_VERDICT_BYTES",
    "decode_personal_scale_precheck_verdict",
    "decode_personal_scale_report",
    "encode_personal_scale_precheck_verdict",
    "encode_personal_scale_report",
    "verdict_sha256",
]
