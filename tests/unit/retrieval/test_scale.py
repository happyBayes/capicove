"""Pure personal reverse-image scale-gate and evaluator tests."""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from capicove.contracts.retrieval import SearchIntent
from capicove.retrieval.scale import (
    MINIMUM_PERSONAL_EVALUATION_QUERY_COUNT,
    PERSONAL_SCALE_POLICY,
    BuildStageObservation,
    DomainQualityVerdict,
    EvaluationQualityKind,
    EvaluationQueryEvidence,
    ExactRetrievalEvaluator,
    HardwareObservation,
    IntentLatency,
    LaneQuality,
    LatencyWorkloadBinding,
    PersonalEvaluationManifest,
    PersonalGateBindings,
    PersonalGateReason,
    PersonalScaleReport,
    PersonalSemanticIndexObservation,
    RankedEvaluationResult,
    RetrievalEvaluator,
    ScaleWorkload,
    StageLatency,
    evaluate_personal_scale_precheck,
    lane_quality_sha256,
    report_sha256,
)
from capicove.retrieval.search_models import RetrievalLane


NOW = datetime(2026, 8, 28, 12, tzinfo=UTC)
FUTURE = datetime.now(UTC) + timedelta(hours=1)
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _query_hashes(cell: str, phase: str, count: int) -> tuple[str, ...]:
    return tuple(_sha(f"{cell}:{phase}:{index}") for index in range(count))


def _workload_bindings() -> tuple[LatencyWorkloadBinding, ...]:
    result: list[LatencyWorkloadBinding] = []
    for intent in SearchIntent:
        for workload in ScaleWorkload:
            cell = f"{intent.value}:{workload.value}"
            result.append(
                LatencyWorkloadBinding(
                    intent=intent,
                    workload=workload,
                    workload_manifest_sha256=_sha(f"{cell}:manifest"),
                    query_set_sha256=_sha(f"{cell}:query-set"),
                    warmup_query_sha256s=_query_hashes(cell, "warmup", 100),
                    measured_query_sha256s=_query_hashes(cell, "measured", 1_000),
                    filter_sha256=_sha(f"{cell}:filter"),
                    matching_unit_count=(10_000 if workload is ScaleWorkload.UNFILTERED else 300),
                )
            )
    return tuple(result)


def _bindings() -> PersonalGateBindings:
    return PersonalGateBindings(
        activation_set_id="set-1",
        generation_manifest_sha256=SHA_A,
        profile_snapshot_sha256=SHA_B,
        dataset_manifest_sha256=SHA_C,
        bundle_provisioning_manifest_sha256=SHA_D,
        unit_manifest_sha256=_sha("unit-manifest"),
        target_representation_manifest_sha256=_sha("representations"),
        evaluation_definition_sha256=_sha("definition"),
        evaluation_manifest_sha256=_sha("evaluation"),
        evaluation_query_count=MINIMUM_PERSONAL_EVALUATION_QUERY_COUNT,
        exact_target_selection_sha256=_sha("selection"),
        tool_schema_sha256=_sha("tool-schema"),
        generation_ids=("generation-perceptual", "generation-instance", "generation-concept"),
        bundle_manifest_sha256s=(_sha("instance-bundle"), _sha("concept-bundle")),
        semantic_index_policy=(
            PersonalSemanticIndexObservation(
                lane=RetrievalLane.INSTANCE,
                index_format="numpy-flat-f32-v1",
                index_parameters_sha256=_sha("instance-index-parameters"),
                exact_candidate_depth=10_000,
            ),
            PersonalSemanticIndexObservation(
                lane=RetrievalLane.CONCEPT,
                index_format="numpy-flat-f32-v1",
                index_parameters_sha256=_sha("concept-index-parameters"),
                exact_candidate_depth=10_000,
            ),
        ),
        fusion_policy_id="weighted-rrf-v1",
        fusion_policy_sha256=_sha("fusion-policy"),
        domain_quality_policy_id="education-film-domain-v1",
        domain_quality_policy_sha256=_sha("domain-policy"),
        instance_labels_sha256=_sha("instance-labels"),
        concept_labels_sha256=_sha("concept-labels"),
        hard_negatives_sha256=_sha("hard-negatives"),
        hardware_fingerprint_sha256=_sha("hardware"),
        work_root_fingerprint_sha256=_sha("work-root"),
        latency_workloads=_workload_bindings(),
    )


def _stage_latencies(p95_ms: float) -> tuple[StageLatency, ...]:
    return tuple(
        StageLatency(stage, p50_ms=1.0, p95_ms=min(p95_ms, 2.0), p99_ms=3.0)
        for stage in PERSONAL_SCALE_POLICY.required_query_stages
    )


def _intent_latency(
    binding: LatencyWorkloadBinding,
    *,
    p95_ms: float = 200.0,
) -> IntentLatency:
    return IntentLatency(
        intent=binding.intent,
        workload=binding.workload,
        workload_manifest_sha256=binding.workload_manifest_sha256,
        query_set_sha256=binding.query_set_sha256,
        warmup_query_sha256s=binding.warmup_query_sha256s,
        measured_query_sha256s=binding.measured_query_sha256s,
        filter_sha256=binding.filter_sha256,
        matching_unit_count=binding.matching_unit_count,
        sample_count=len(binding.measured_query_sha256s),
        p50_ms=100.0,
        p95_ms=p95_ms,
        p99_ms=max(p95_ms, 210.0),
        cache_mode="miss",
        declared_concurrency=1,
        stages=_stage_latencies(p95_ms),
    )


def _lane_quality() -> tuple[LaneQuality, ...]:
    return tuple(
        LaneQuality(
            lane=lane,
            quality_kind=quality_kind,
            recall_at_1=1.0,
            recall_at_10=1.0,
            recall_at_100=1.0,
            ndcg_at_10=1.0,
        )
        for lane in (RetrievalLane.INSTANCE, RetrievalLane.CONCEPT)
        for quality_kind in EvaluationQualityKind
    )


def _report(
    bindings: PersonalGateBindings,
    *,
    p95_ms: float = 200.0,
) -> PersonalScaleReport:
    final_index_bytes = 12_000_000
    quality = _lane_quality()
    return PersonalScaleReport(
        report_id="personal-provisional-report-1",
        gate_policy_id=PERSONAL_SCALE_POLICY.policy_id,
        gate_policy_sha256=PERSONAL_SCALE_POLICY.policy_sha256,
        profile_ref="personal-lite",
        profile_snapshot_sha256=bindings.profile_snapshot_sha256,
        dataset_manifest_sha256=bindings.dataset_manifest_sha256,
        bundle_provisioning_manifest_sha256=(bindings.bundle_provisioning_manifest_sha256),
        activation_set_id=bindings.activation_set_id,
        generation_manifest_sha256=bindings.generation_manifest_sha256,
        unit_manifest_sha256=bindings.unit_manifest_sha256,
        target_representation_manifest_sha256=(bindings.target_representation_manifest_sha256),
        unit_count=10_000,
        evaluation_definition_sha256=bindings.evaluation_definition_sha256,
        evaluation_manifest_sha256=bindings.evaluation_manifest_sha256,
        evaluation_query_count=bindings.evaluation_query_count,
        exact_target_selection_sha256=bindings.exact_target_selection_sha256,
        exact_target_count=10_000,
        tool_schema_sha256=bindings.tool_schema_sha256,
        generation_ids=bindings.generation_ids,
        bundle_manifest_sha256s=bindings.bundle_manifest_sha256s,
        semantic_index_policy=bindings.semantic_index_policy,
        fusion_policy_id=bindings.fusion_policy_id,
        fusion_policy_sha256=bindings.fusion_policy_sha256,
        hardware=HardwareObservation(
            fingerprint_sha256=bindings.hardware_fingerprint_sha256,
            architecture="arm64",
            logical_cpu_count=12,
            memory_bytes=10_000_000,
            local_free_bytes_at_preflight=20_000_000,
        ),
        work_root_fingerprint_sha256=bindings.work_root_fingerprint_sha256,
        lane_schedule_id="personal-cpu-v1",
        cpu_total_threads=8,
        max_parallel_lanes=2,
        per_lane_compute_threads=4,
        onnx_inter_op_threads=1,
        intent_latency=tuple(
            _intent_latency(binding, p95_ms=p95_ms) for binding in bindings.latency_workloads
        ),
        lane_quality=quality,
        domain_quality=DomainQualityVerdict(
            policy_id=bindings.domain_quality_policy_id,
            policy_sha256=bindings.domain_quality_policy_sha256,
            instance_labels_sha256=bindings.instance_labels_sha256,
            concept_labels_sha256=bindings.concept_labels_sha256,
            hard_negatives_sha256=bindings.hard_negatives_sha256,
            evaluation_manifest_sha256=bindings.evaluation_manifest_sha256,
            evaluation_query_count=bindings.evaluation_query_count,
            lane_quality_sha256=lane_quality_sha256(quality),
            passed=True,
        ),
        build_stages=tuple(
            BuildStageObservation(
                stage=stage,
                elapsed_seconds=1.0,
                peak_rss_bytes=1_000_000,
                peak_temp_bytes=2_000_000,
            )
            for stage in PERSONAL_SCALE_POLICY.required_build_stages
        ),
        build_peak_rss_bytes=7_999_999,
        build_peak_temp_bytes=2_000_000,
        peak_work_root_bytes=15_999_999,
        query_peak_rss_bytes=7_000_000,
        final_index_bytes=final_index_bytes,
        bytes_per_vector=final_index_bytes / 10_000,
        cold_start_ms=300.0,
        page_faults=50,
        io_read_bytes=1_000_000,
        generation_switch_ms=5.0,
        readonly_mmap=True,
        candidate_built_on_reported_machine=True,
        production_alias_unchanged=True,
        production_alias_before_sha256=_sha("alias"),
        production_alias_after_sha256=_sha("alias"),
        created_at=NOW,
    )


def test_personal_policy_is_fixed_and_precheck_never_claims_formal_support() -> None:
    bindings = _bindings()
    report = _report(bindings, p95_ms=200.0)

    verdict = evaluate_personal_scale_precheck(report, bindings, evaluated_at=NOW)

    assert set(inspect.signature(evaluate_personal_scale_precheck).parameters) == {
        "report",
        "bindings",
        "evaluated_at",
    }
    assert verdict.passed is True
    assert verdict.reason_codes == ()
    assert verdict.tier == "provisional"
    assert verdict.formal_capacity_supported is False
    assert verdict.report_sha256 == report_sha256(report)
    assert not hasattr(verdict, "qualification")


@pytest.mark.parametrize("unit_count", (9_999, 10_001))
def test_precheck_requires_exact_personal_capacity(unit_count: int) -> None:
    bindings = _bindings()

    verdict = evaluate_personal_scale_precheck(
        replace(_report(bindings), unit_count=unit_count),
        bindings,
        evaluated_at=NOW,
    )

    assert verdict.passed is False
    assert PersonalGateReason.UNIT_COUNT_MISMATCH in verdict.reason_codes


def test_precheck_requires_the_bound_full_exact_target_selection() -> None:
    bindings = _bindings()
    count_drift = replace(_report(bindings), exact_target_count=9_999)
    selection_drift = replace(
        _report(bindings),
        exact_target_selection_sha256=_sha("different-selection"),
    )

    count_verdict = evaluate_personal_scale_precheck(count_drift, bindings, evaluated_at=NOW)
    selection_verdict = evaluate_personal_scale_precheck(
        selection_drift,
        bindings,
        evaluated_at=NOW,
    )

    assert PersonalGateReason.EXACT_TARGET_COUNT_MISMATCH in count_verdict.reason_codes
    assert PersonalGateReason.SUBJECT_IDENTITY_MISMATCH in selection_verdict.reason_codes


def test_precheck_rejects_generation_parameter_drift() -> None:
    bindings = _bindings()
    report = _report(bindings)
    changed_policy = replace(
        report.semantic_index_policy[0],
        index_parameters_sha256=_sha("changed-instance-index-parameters"),
    )
    drifted = replace(
        report,
        semantic_index_policy=(changed_policy, *report.semantic_index_policy[1:]),
    )

    verdict = evaluate_personal_scale_precheck(drifted, bindings, evaluated_at=NOW)

    assert PersonalGateReason.SUBJECT_IDENTITY_MISMATCH in verdict.reason_codes


def test_gate_rejects_missing_cell_and_one_cell_over_p95() -> None:
    bindings = _bindings()
    report = _report(bindings)
    missing = replace(report, intent_latency=report.intent_latency[:-1])
    over = replace(
        report,
        intent_latency=(
            replace(report.intent_latency[0], p95_ms=200.0001, p99_ms=210.0),
            *report.intent_latency[1:],
        ),
    )

    missing_verdict = evaluate_personal_scale_precheck(missing, bindings, evaluated_at=NOW)
    over_verdict = evaluate_personal_scale_precheck(over, bindings, evaluated_at=NOW)

    assert PersonalGateReason.LATENCY_CELL_SET_MISMATCH in missing_verdict.reason_codes
    assert PersonalGateReason.LATENCY_P95_EXCEEDED in over_verdict.reason_codes


def test_gate_rejects_stage_latency_that_exceeds_its_total_cell() -> None:
    bindings = _bindings()
    report = _report(bindings)
    cell = report.intent_latency[0]
    impossible_stage = replace(cell.stages[0], p95_ms=201.0, p99_ms=211.0)
    impossible_cell = replace(cell, stages=(impossible_stage, *cell.stages[1:]))
    impossible = replace(report, intent_latency=(impossible_cell, *report.intent_latency[1:]))

    verdict = evaluate_personal_scale_precheck(impossible, bindings, evaluated_at=NOW)

    assert PersonalGateReason.LATENCY_EVIDENCE_MISMATCH in verdict.reason_codes


def test_precheck_rejects_reordered_query_and_build_stages() -> None:
    bindings = _bindings()
    report = _report(bindings)
    reordered = replace(
        report,
        intent_latency=tuple(
            replace(cell, stages=tuple(reversed(cell.stages))) for cell in report.intent_latency
        ),
        build_stages=tuple(reversed(report.build_stages)),
    )

    verdict = evaluate_personal_scale_precheck(reordered, bindings, evaluated_at=NOW)

    assert PersonalGateReason.QUERY_STAGE_SEQUENCE_MISMATCH in verdict.reason_codes
    assert PersonalGateReason.BUILD_STAGE_SEQUENCE_MISMATCH in verdict.reason_codes


@pytest.mark.parametrize("matching_unit_count", (0, 10_000))
def test_filtered_workloads_require_a_nonempty_strict_subset(
    matching_unit_count: int,
) -> None:
    filtered = next(
        item
        for item in _bindings().latency_workloads
        if item.workload is ScaleWorkload.SPARSE_FILTER
    )

    with pytest.raises(ValueError, match="filtered workload"):
        replace(filtered, matching_unit_count=matching_unit_count)


def test_unfiltered_workload_requires_all_personal_units() -> None:
    unfiltered = next(
        item for item in _bindings().latency_workloads if item.workload is ScaleWorkload.UNFILTERED
    )

    with pytest.raises(ValueError, match="all 10,000"):
        replace(unfiltered, matching_unit_count=9_999)


def test_gate_rejects_query_evidence_drift_and_policy_drift() -> None:
    bindings = _bindings()
    report = _report(bindings)
    first = report.intent_latency[0]
    shortened = replace(
        first,
        measured_query_sha256s=first.measured_query_sha256s[:-1],
        sample_count=999,
    )
    evidence_drift = replace(report, intent_latency=(shortened, *report.intent_latency[1:]))
    policy_drift = replace(report, gate_policy_sha256=SHA_A)

    evidence_verdict = evaluate_personal_scale_precheck(
        evidence_drift,
        bindings,
        evaluated_at=NOW,
    )
    policy_verdict = evaluate_personal_scale_precheck(policy_drift, bindings, evaluated_at=NOW)

    assert PersonalGateReason.LATENCY_EVIDENCE_MISMATCH in evidence_verdict.reason_codes
    assert PersonalGateReason.POLICY_IDENTITY_MISMATCH in policy_verdict.reason_codes


def test_latency_observation_rejects_repeated_queries_and_non_finite_percentiles() -> None:
    binding = _bindings().latency_workloads[0]
    observation = _intent_latency(binding)

    with pytest.raises(ValueError, match="unique"):
        replace(
            observation,
            measured_query_sha256s=(
                observation.measured_query_sha256s[0],
                *observation.measured_query_sha256s[:-1],
            ),
        )
    with pytest.raises(ValueError, match="disjoint"):
        replace(
            observation,
            warmup_query_sha256s=(
                observation.measured_query_sha256s[0],
                *observation.warmup_query_sha256s[1:],
            ),
        )
    with pytest.raises(ValueError, match="finite"):
        replace(observation, p95_ms=float("nan"))
    with pytest.raises(ValueError, match=r"p50_ms.*p95_ms.*p99_ms"):
        replace(observation, p50_ms=201.0, p95_ms=200.0)


@pytest.mark.parametrize(
    ("metric", "value"),
    (
        ("recall_at_1", 0.0),
        ("recall_at_10", 0.99),
        ("recall_at_100", 0.99),
        ("ndcg_at_10", 0.99),
    ),
)
def test_gate_requires_four_quality_cells_and_full_exact_personal_parity(
    metric: str,
    value: float,
) -> None:
    bindings = _bindings()
    report = _report(bindings)
    missing = replace(report, lane_quality=report.lane_quality[:-1])
    parity_index = next(
        index
        for index, item in enumerate(report.lane_quality)
        if item.lane is RetrievalLane.INSTANCE
        and item.quality_kind is EvaluationQualityKind.EXACT_PARITY
    )
    degraded_items = list(report.lane_quality)
    degraded_items[parity_index] = _replace_quality_metric(
        degraded_items[parity_index],
        metric,
        value,
    )
    degraded = replace(report, lane_quality=tuple(degraded_items))

    missing_verdict = evaluate_personal_scale_precheck(missing, bindings, evaluated_at=NOW)
    degraded_verdict = evaluate_personal_scale_precheck(degraded, bindings, evaluated_at=NOW)

    assert PersonalGateReason.QUALITY_CELL_SET_MISMATCH in missing_verdict.reason_codes
    assert PersonalGateReason.EXACT_PARITY_FAILED in degraded_verdict.reason_codes


def _replace_quality_metric(item: LaneQuality, metric: str, value: float) -> LaneQuality:
    if metric == "recall_at_1":
        return replace(item, recall_at_1=value)
    if metric == "recall_at_10":
        return replace(item, recall_at_10=value)
    if metric == "recall_at_100":
        return replace(item, recall_at_100=value)
    if metric == "ndcg_at_10":
        return replace(item, ndcg_at_10=value)
    raise AssertionError(f"unknown test metric: {metric}")


def test_personal_semantic_index_policy_is_the_real_exact_numpy_index() -> None:
    observation = _bindings().semantic_index_policy[0]

    with pytest.raises(ValueError, match="numpy-flat-f32-v1"):
        replace(observation, index_format="hnsw-v1")
    with pytest.raises(ValueError, match="positive integer"):
        replace(observation, exact_candidate_depth=True)
    with pytest.raises(ValueError, match="exact_candidate_depth 10,000"):
        replace(observation, exact_candidate_depth=9_999)


def test_quality_and_semantic_index_cells_reject_string_enum_impersonation() -> None:
    with pytest.raises(TypeError, match="RetrievalLane"):
        replace(_lane_quality()[0], lane="instance")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="SearchIntent"):
        replace(_rankings(_evaluation_manifest())[0], intent="instance")  # type: ignore[arg-type]


def test_gate_requires_passed_bound_domain_policy() -> None:
    bindings = _bindings()
    report = _report(bindings)

    failed = replace(report, domain_quality=replace(report.domain_quality, passed=False))
    drifted = replace(
        report,
        domain_quality=replace(report.domain_quality, policy_sha256=SHA_A),
    )
    relabelled = replace(
        report,
        domain_quality=replace(
            report.domain_quality,
            instance_labels_sha256=_sha("different-instance-labels"),
        ),
    )
    changed_quality = list(report.lane_quality)
    domain_index = next(
        index
        for index, item in enumerate(changed_quality)
        if item.lane is RetrievalLane.INSTANCE
        and item.quality_kind is EvaluationQualityKind.DOMAIN_RELEVANCE
    )
    changed_quality[domain_index] = replace(
        changed_quality[domain_index],
        recall_at_10=0.0,
    )
    stale_quality_digest = replace(report, lane_quality=tuple(changed_quality))
    wrong_evaluation = replace(
        report,
        domain_quality=replace(
            report.domain_quality,
            evaluation_manifest_sha256=_sha("different-evaluation-manifest"),
            evaluation_query_count=report.evaluation_query_count + 1,
        ),
    )

    assert (
        PersonalGateReason.DOMAIN_QUALITY_FAILED
        in evaluate_personal_scale_precheck(
            failed,
            bindings,
            evaluated_at=NOW,
        ).reason_codes
    )
    assert (
        PersonalGateReason.DOMAIN_QUALITY_FAILED
        in evaluate_personal_scale_precheck(
            stale_quality_digest,
            bindings,
            evaluated_at=NOW,
        ).reason_codes
    )
    assert (
        PersonalGateReason.DOMAIN_QUALITY_FAILED
        in evaluate_personal_scale_precheck(
            wrong_evaluation,
            bindings,
            evaluated_at=NOW,
        ).reason_codes
    )
    assert (
        PersonalGateReason.DOMAIN_QUALITY_FAILED
        in evaluate_personal_scale_precheck(
            relabelled,
            bindings,
            evaluated_at=NOW,
        ).reason_codes
    )
    assert (
        PersonalGateReason.DOMAIN_QUALITY_FAILED
        in evaluate_personal_scale_precheck(
            drifted,
            bindings,
            evaluated_at=NOW,
        ).reason_codes
    )


def test_resource_ratios_are_strict_and_alias_must_remain_unchanged() -> None:
    bindings = _bindings()
    report = _report(bindings)
    memory_boundary = replace(report, build_peak_rss_bytes=8_000_000)
    work_boundary = replace(report, peak_work_root_bytes=16_000_000)
    alias_drift = replace(
        report,
        production_alias_after_sha256=_sha("changed-alias"),
        production_alias_unchanged=False,
    )
    inconsistent_peak = replace(report, build_peak_rss_bytes=999_999)

    assert (
        PersonalGateReason.MEMORY_BUDGET_EXCEEDED
        in evaluate_personal_scale_precheck(
            memory_boundary,
            bindings,
            evaluated_at=NOW,
        ).reason_codes
    )
    assert (
        PersonalGateReason.WORK_ROOT_BUDGET_EXCEEDED
        in evaluate_personal_scale_precheck(
            work_boundary,
            bindings,
            evaluated_at=NOW,
        ).reason_codes
    )
    assert (
        PersonalGateReason.RUNTIME_INTEGRITY_FAILED
        in evaluate_personal_scale_precheck(
            alias_drift,
            bindings,
            evaluated_at=NOW,
        ).reason_codes
    )
    assert (
        PersonalGateReason.RUNTIME_INTEGRITY_FAILED
        in evaluate_personal_scale_precheck(
            inconsistent_peak,
            bindings,
            evaluated_at=NOW,
        ).reason_codes
    )


def test_precheck_report_identity_changes_without_creating_qualification() -> None:
    bindings = _bindings()
    first = evaluate_personal_scale_precheck(_report(bindings), bindings, evaluated_at=NOW)
    second = evaluate_personal_scale_precheck(
        _report(bindings, p95_ms=199.0),
        bindings,
        evaluated_at=NOW,
    )

    assert first.report_sha256 != second.report_sha256
    assert not hasattr(first, "qualification")
    with pytest.raises(ValueError, match="formal capacity"):
        replace(first, formal_capacity_supported=True)
    with pytest.raises(ValueError, match="provisional"):
        replace(first, tier="formal")


def test_gate_rejects_evaluation_time_before_report_creation() -> None:
    bindings = _bindings()

    with pytest.raises(ValueError, match="must not precede"):
        evaluate_personal_scale_precheck(
            _report(bindings),
            bindings,
            evaluated_at=NOW - timedelta(microseconds=1),
        )


def _evaluation_manifest() -> PersonalEvaluationManifest:
    queries: list[EvaluationQueryEvidence] = []
    for query_index in range(2):
        prefix = f"q{query_index}"
        instance_exact = tuple(f"{prefix}-instance-{index}" for index in range(100))
        concept_exact = tuple(f"{prefix}-concept-{index}" for index in range(100))
        queries.append(
            EvaluationQueryEvidence(
                query_ref=prefix,
                query_sha256=_sha(prefix),
                instance_relevance=((instance_exact[0], 3), (instance_exact[3], 1)),
                concept_relevance=((concept_exact[1], 3), (concept_exact[4], 1)),
                instance_exact_unit_ids=instance_exact,
                concept_exact_unit_ids=concept_exact,
            )
        )
    return PersonalEvaluationManifest(
        evaluation_id="evaluation-1",
        definition_sha256=_sha("definition"),
        unit_manifest_sha256=_sha("units"),
        target_representation_manifest_sha256=_sha("representations"),
        target_selection_sha256=_sha("selection"),
        exact_target_count=10_000,
        generation_manifest_sha256=_sha("generation"),
        query_count=len(queries),
        queries=tuple(queries),
    )


def _rankings(manifest: PersonalEvaluationManifest) -> tuple[RankedEvaluationResult, ...]:
    results: list[RankedEvaluationResult] = []
    for query in manifest.queries:
        for intent, exact_ids in (
            (SearchIntent.INSTANCE, query.instance_exact_unit_ids),
            (SearchIntent.CONCEPT, query.concept_exact_unit_ids),
        ):
            results.append(
                RankedEvaluationResult(
                    query_ref=query.query_ref,
                    query_sha256=query.query_sha256,
                    intent=intent,
                    quality_kind=EvaluationQualityKind.DOMAIN_RELEVANCE,
                    ranked_unit_ids=exact_ids,
                    target_selection_sha256=manifest.target_selection_sha256,
                    generation_manifest_sha256=manifest.generation_manifest_sha256,
                )
            )
            results.append(
                RankedEvaluationResult(
                    query_ref=query.query_ref,
                    query_sha256=query.query_sha256,
                    intent=intent,
                    quality_kind=EvaluationQualityKind.EXACT_PARITY,
                    ranked_unit_ids=exact_ids,
                    target_selection_sha256=manifest.target_selection_sha256,
                    generation_manifest_sha256=manifest.generation_manifest_sha256,
                )
            )
    return tuple(results)


def test_evaluation_manifest_requires_two_bound_queries() -> None:
    manifest = _evaluation_manifest()

    with pytest.raises(ValueError, match="at least 2"):
        replace(manifest, query_count=1, queries=manifest.queries[:1])
    with pytest.raises(ValueError, match="must equal"):
        replace(manifest, queries=manifest.queries[:1])


def test_evaluator_rejects_non_identical_exact_ranking_before_metrics() -> None:
    manifest = _evaluation_manifest()
    rankings = list(_rankings(manifest))
    index = next(
        index
        for index, result in enumerate(rankings)
        if result.query_ref == "q0"
        and result.intent is SearchIntent.INSTANCE
        and result.quality_kind is EvaluationQualityKind.EXACT_PARITY
    )
    original = rankings[index]
    rankings[index] = replace(
        original,
        ranked_unit_ids=(
            original.ranked_unit_ids[0],
            "foreign",
            *original.ranked_unit_ids[1:9],
            *original.ranked_unit_ids[10:],
        ),
    )

    with pytest.raises(ValueError, match="complete ordered top-100"):
        ExactRetrievalEvaluator().evaluate(manifest, rankings, deadline=FUTURE)


def test_evaluator_exact_parity_detects_order_drift_below_top_ten() -> None:
    manifest = _evaluation_manifest()
    rankings = list(_rankings(manifest))
    index = next(
        index
        for index, result in enumerate(rankings)
        if result.query_ref == "q0"
        and result.intent is SearchIntent.INSTANCE
        and result.quality_kind is EvaluationQualityKind.EXACT_PARITY
    )
    original = rankings[index]
    reordered = list(original.ranked_unit_ids)
    reordered[20], reordered[21] = reordered[21], reordered[20]
    rankings[index] = replace(original, ranked_unit_ids=tuple(reordered))

    with pytest.raises(ValueError, match="complete ordered top-100"):
        ExactRetrievalEvaluator().evaluate(manifest, rankings, deadline=FUTURE)


def test_evaluator_returns_separate_four_cell_quality_and_conforms_to_port() -> None:
    manifest = _evaluation_manifest()
    evaluator: RetrievalEvaluator = ExactRetrievalEvaluator()

    qualities = evaluator.evaluate(manifest, _rankings(manifest), deadline=FUTURE)

    assert isinstance(evaluator, RetrievalEvaluator)
    assert {(quality.lane, quality.quality_kind) for quality in qualities} == {
        (lane, kind)
        for lane in (RetrievalLane.INSTANCE, RetrievalLane.CONCEPT)
        for kind in EvaluationQualityKind
    }
    assert all(quality.recall_at_10 == 1.0 for quality in qualities)


def test_evaluator_rejects_balanced_missing_duplicate_and_oversized_rankings() -> None:
    manifest = _evaluation_manifest()
    rankings = _rankings(manifest)

    with pytest.raises(ValueError, match="instance or concept"):
        replace(rankings[0], intent=SearchIntent.BALANCED)
    with pytest.raises(ValueError, match="exactly one"):
        ExactRetrievalEvaluator().evaluate(manifest, rankings[:-1], deadline=FUTURE)
    with pytest.raises(ValueError, match="exactly one"):
        ExactRetrievalEvaluator().evaluate(
            manifest,
            (*rankings, rankings[0]),
            deadline=FUTURE,
        )
    with pytest.raises(ValueError, match="at most 100"):
        replace(rankings[0], ranked_unit_ids=tuple(f"unit-{index}" for index in range(101)))


def test_evaluator_consumes_only_expected_rankings_plus_one_overflow_probe() -> None:
    manifest = _evaluation_manifest()
    rankings = _rankings(manifest)
    consumed: list[RankedEvaluationResult] = []

    def oversized_rankings() -> Iterator[RankedEvaluationResult]:
        for item in (*rankings, rankings[0], rankings[0]):
            consumed.append(item)
            yield item

    with pytest.raises(ValueError, match="exactly one"):
        ExactRetrievalEvaluator().evaluate(
            manifest,
            oversized_rankings(),
            deadline=FUTURE,
        )

    assert len(consumed) == len(rankings) + 1


def test_evaluator_rejects_query_and_generation_binding_drift() -> None:
    manifest = _evaluation_manifest()
    rankings = list(_rankings(manifest))
    rankings[0] = replace(rankings[0], generation_manifest_sha256=SHA_A)

    with pytest.raises(ValueError, match="generation manifest"):
        ExactRetrievalEvaluator().evaluate(manifest, rankings, deadline=FUTURE)
