"""Qualification metrics require independent held-out labels and fixed baselines."""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

import pytest

import capicove.graph_build as graph_build
from capicove.graph_build.evaluation import (
    BaselinePrediction,
    DecisionLabel,
    DiagnosticLabel,
    EvaluationProfile,
    ExtractionLabel,
    LabeledCorpus,
    MetricGate,
    OccurrenceLabel,
    PredictedGraphRun,
    ReportLabel,
    RuntimeIdentity,
    evaluate_graph_run,
    precision_recall,
)


def test_evaluation_contracts_are_lazily_public() -> None:
    assert graph_build.EvaluationProfile is EvaluationProfile
    assert graph_build.QualificationReport.__name__ == "QualificationReport"
    assert graph_build.evaluate_graph_run is evaluate_graph_run


def test_metrics_do_not_turn_abstention_into_perfect_precision() -> None:
    assert precision_recall(frozenset(), frozenset({"a"})) == (None, 0.0)
    assert precision_recall(frozenset({"a", "b"}), frozenset({"a", "c"})) == (
        0.5,
        0.5,
    )
    assert precision_recall(frozenset(), frozenset()) == (None, None)


def _corpus(*, independent: bool = True) -> LabeledCorpus:
    return LabeledCorpus(
        corpus_id="corpus:held-out-v1",
        split_id="held-out:v1",
        annotation_ref="review:labels-17",
        annotator_identity_sha256="1" * 64,
        independent_annotation=independent,
        extraction_labels=(
            ExtractionLabel("e-base-bad", "base", "chinese", False),
            ExtractionLabel("e-base-good", "base", "chinese", True),
            ExtractionLabel("e-diag", "diagnostic", "feedback", True),
            ExtractionLabel("e-event", "event", "reflection", True),
        ),
        merge_labels=(
            DecisionLabel("merge-false", "distinct"),
            DecisionLabel("merge-good", "matched"),
        ),
        alignment_labels=(
            DecisionLabel("align-abstain", "matched"),
            DecisionLabel("align-distinct", "distinct"),
            DecisionLabel("align-good", "matched"),
        ),
        occurrence_labels=(
            OccurrenceLabel("occ-other", False),
            OccurrenceLabel("occ-same", True),
        ),
        diagnostic_labels=(
            DiagnosticLabel("diag-bad", False),
            DiagnosticLabel("diag-good", True),
        ),
        report_labels=(
            ReportLabel("statement-bad", False, False),
            ReportLabel("statement-good", True, True),
        ),
        confidence_item_ids=frozenset({"decision:1", "decision:2"}),
    )


def _baseline(baseline_id: Literal["fixed-token", "name-type-only"]) -> BaselinePrediction:
    return BaselinePrediction(
        baseline_id=baseline_id,
        extraction_ids=frozenset({"e-base-good"}),
        merged_pair_ids=frozenset({"merge-good"}),
    )


def _runtime_identities() -> tuple[RuntimeIdentity, ...]:
    return (
        RuntimeIdentity("community-partitioner", "networkx-3.6.1", "a" * 64),
        RuntimeIdentity("device", "cpu", "b" * 64),
        RuntimeIdentity("embedding", "bge-v1", "3" * 64),
        RuntimeIdentity("embedding-space", "space:v1", "c" * 64),
        RuntimeIdentity("generation-profile", "profile:v1", "d" * 64),
        RuntimeIdentity("generator", "local-v1", "4" * 64),
        RuntimeIdentity("retrieval-generation", "generation:17", "e" * 64),
        RuntimeIdentity("tokenizer", "tokenizer:v1", "f" * 64),
    )


def _run() -> PredictedGraphRun:
    return PredictedGraphRun(
        run_id="run:17",
        run_sha256="2" * 64,
        runtime_identities=_runtime_identities(),
        extraction_ids=frozenset({"e-base-good", "e-event", "e-diag"}),
        merged_pair_ids=frozenset({"merge-good", "merge-false"}),
        alignment_decisions=(
            DecisionLabel("align-abstain", "ambiguous"),
            DecisionLabel("align-distinct", "distinct"),
            DecisionLabel("align-good", "matched"),
        ),
        duplicate_occurrence_ids=frozenset({"occ-same"}),
        diagnostic_finding_ids=frozenset({"diag-good", "diag-bad"}),
        supported_diagnostic_ids=frozenset({"diag-good"}),
        report_statement_ids=frozenset({"statement-good", "statement-bad"}),
        entailed_statement_ids=frozenset({"statement-good"}),
        counter_evidence_statement_ids=frozenset({"statement-good"}),
        confidence_assessed_ids=frozenset({"decision:1", "decision:2"}),
        confidence_accepted_ids=frozenset({"decision:1"}),
        baselines=(_baseline("fixed-token"), _baseline("name-type-only")),
        elapsed_ms=1500,
        input_tokens=100,
        output_tokens=40,
        cost_microunits=0,
        generation_calls=2,
        cache_hits=3,
        structural_checks_passed=True,
    )


def _profile() -> EvaluationProfile:
    return EvaluationProfile(
        profile_id="qualification:v1",
        held_out_split_id="held-out:v1",
        reviewer_approval_ref="approval:quality-17",
        reviewer_identity_sha256="5" * 64,
        gates=(
            MetricGate("alignment_abstention_rate", "maximum", 0.5),
            MetricGate("diagnostic_support_rate", "minimum", 0.5),
            MetricGate("extraction.base.precision", "minimum", 0.9),
            MetricGate("false_merge_rate", "maximum", 0.5),
        ),
    )


def test_evaluate_graph_run_measures_all_required_families() -> None:
    report = evaluate_graph_run(_run(), _corpus(), _profile())

    values = {metric.name: metric.value for metric in report.metrics}
    assert values == {
        "alignment_abstention_rate": 1 / 3,
        "alignment_recall": 0.5,
        "citation_entailment_rate": 0.5,
        "counter_evidence_coverage": 1.0,
        "diagnostic_support_rate": 0.5,
        "extraction.base.precision": 1.0,
        "extraction.base.recall": 1.0,
        "extraction.diagnostic.precision": 1.0,
        "extraction.diagnostic.recall": 1.0,
        "extraction.event.precision": 1.0,
        "extraction.event.recall": 1.0,
        "extraction.stratum.chinese.precision": 1.0,
        "extraction.stratum.chinese.recall": 1.0,
        "extraction.stratum.feedback.precision": 1.0,
        "extraction.stratum.feedback.recall": 1.0,
        "extraction.stratum.mixed-language.precision": None,
        "extraction.stratum.mixed-language.recall": None,
        "extraction.stratum.reference-book.precision": None,
        "extraction.stratum.reference-book.recall": None,
        "extraction.stratum.reflection.precision": 1.0,
        "extraction.stratum.reflection.recall": 1.0,
        "extraction.stratum.table.precision": None,
        "extraction.stratum.table.recall": None,
        "false_merge_rate": 0.5,
        "occurrence_dedup_accuracy": 1.0,
    }
    assert report.counts == (
        ("alignment_assessed", 3),
        ("diagnostic_findings", 2),
        ("extraction_candidates", 4),
        ("report_statements", 2),
    )
    assert report.corpus_sha256 != report.profile_sha256
    assert report.model_sha256 not in {"3" * 64, "4" * 64}
    changed_identity = dataclasses.replace(
        _run(),
        runtime_identities=tuple(
            RuntimeIdentity("embedding", "bge-v2", "6" * 64)
            if item.component_id == "embedding"
            else item
            for item in _runtime_identities()
        ),
    )
    assert (
        evaluate_graph_run(changed_identity, _corpus(), _profile()).model_sha256
        != report.model_sha256
    )
    assert report.qualified is True
    assert report.failure_reasons == ()
    assert (report.elapsed_ms, report.cost_microunits) == (1500, 0)
    assert (
        report.confidence_accepted,
        report.confidence_denominator,
        report.abstention_count,
        report.abstention_denominator,
    ) == (1, 2, 1, 3)
    assert tuple(item.baseline_id for item in report.baselines) == (
        "fixed-token",
        "name-type-only",
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("independent_annotation", False),
        ("split_id", "training:v1"),
        ("annotation_ref", "generator:self-label"),
    ],
)
def test_corpus_rejects_non_independent_or_non_held_out_labels(
    field: str, replacement: Any
) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(_corpus(), **{field: replacement})


def test_profile_is_reviewed_closed_and_declared_before_results() -> None:
    with pytest.raises(ValueError, match="review"):
        dataclasses.replace(_profile(), reviewer_approval_ref="")
    with pytest.raises(ValueError, match="metric"):
        dataclasses.replace(_profile(), gates=(MetricGate("invented_metric", "minimum", 0.9),))
    with pytest.raises(ValueError, match="duplicate"):
        dataclasses.replace(_profile(), gates=(_profile().gates[0],) * 2)
    with pytest.raises(ValueError, match="at least one"):
        dataclasses.replace(_profile(), gates=())


def test_evaluation_rejects_foreign_predictions_and_missing_fixed_baselines() -> None:
    with pytest.raises(ValueError, match="outside labeled corpus"):
        evaluate_graph_run(
            dataclasses.replace(_run(), extraction_ids=frozenset({"foreign"})),
            _corpus(),
            _profile(),
        )
    with pytest.raises(ValueError, match="baseline"):
        evaluate_graph_run(
            dataclasses.replace(_run(), baselines=(_baseline("fixed-token"),)),
            _corpus(),
            _profile(),
        )


def test_missing_denominator_fails_a_gate_instead_of_fabricating_a_score() -> None:
    corpus = dataclasses.replace(
        _corpus(),
        extraction_labels=tuple(
            item for item in _corpus().extraction_labels if item.ontology_layer != "diagnostic"
        ),
    )
    profile = dataclasses.replace(
        _profile(),
        gates=(MetricGate("extraction.diagnostic.recall", "minimum", 0.5),),
    )
    run = dataclasses.replace(_run(), extraction_ids=frozenset({"e-base-good", "e-event"}))

    report = evaluate_graph_run(run, corpus, profile)

    assert (
        dict((item.name, item.value) for item in report.metrics)["extraction.diagnostic.recall"]
        is None
    )
    assert report.qualified is False
    assert report.failure_reasons == ("metric_without_denominator:extraction.diagnostic.recall",)


def test_structural_failure_blocks_qualification_and_output_is_deterministic() -> None:
    run = dataclasses.replace(_run(), structural_checks_passed=False)
    first = evaluate_graph_run(run, _corpus(), _profile())
    second = evaluate_graph_run(run, _corpus(), _profile())

    assert first == second
    assert first.qualified is False
    assert first.failure_reasons == ("structural_checks_failed",)


def test_missing_runtime_identity_cannot_qualify_provider() -> None:
    run = dataclasses.replace(
        _run(),
        runtime_identities=tuple(
            item for item in _runtime_identities() if item.component_id != "device"
        ),
    )

    report = evaluate_graph_run(run, _corpus(), _profile())

    assert report.qualified is False
    assert report.failure_reasons == ("missing_runtime_identity:device",)


def test_runtime_components_are_unique_and_confidence_is_corpus_bound() -> None:
    with pytest.raises(ValueError, match="unique"):
        dataclasses.replace(_run(), runtime_identities=(_runtime_identities()[0],) * 2)
    with pytest.raises(ValueError, match="confidence"):
        evaluate_graph_run(
            dataclasses.replace(_run(), confidence_assessed_ids=frozenset({"decision:1"})),
            _corpus(),
            _profile(),
        )


def test_public_result_types_reject_fabricated_or_inconsistent_values() -> None:
    report = evaluate_graph_run(_run(), _corpus(), _profile())
    with pytest.raises(ValueError, match="metric"):
        dataclasses.replace(report.metrics[0], value=2.0)
    with pytest.raises(ValueError, match="baseline"):
        dataclasses.replace(report.baselines[0], false_merge_rate=2.0)
    with pytest.raises(ValueError, match="runtime"):
        dataclasses.replace(report, model_sha256="0" * 64)
    with pytest.raises(ValueError, match="qualification"):
        dataclasses.replace(report, qualified=True, failure_reasons=("structural_checks_failed",))


@pytest.mark.parametrize(
    "bad",
    [
        lambda: ExtractionLabel("x", "unknown", "chinese", True),  # type: ignore[arg-type]
        lambda: DecisionLabel("x", "unknown"),  # type: ignore[arg-type]
        lambda: RuntimeIdentity("generator", "v1", "BAD"),
        lambda: MetricGate("false_merge_rate", "minimum", 0.5),
        lambda: MetricGate("false_merge_rate", "maximum", float("nan")),
    ],
)
def test_closed_values_and_numeric_bounds_fail_fast(bad: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        bad()
