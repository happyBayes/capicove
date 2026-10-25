"""Closed, deterministic qualification metrics for independently labeled runs.

This module evaluates already-produced graph outputs.  It does not invoke models,
select thresholds, or certify prose by itself.  A qualification requires an
explicit held-out corpus, independent annotation, a pre-reviewed gate profile,
both declared comparison baselines, and successful structural checks.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Literal


OntologyLayer = Literal["base", "event", "diagnostic"]
MaterialStratum = Literal[
    "chinese", "mixed-language", "table", "reflection", "feedback", "reference-book"
]
Decision = Literal["matched", "distinct", "ambiguous"]
GateDirection = Literal["minimum", "maximum"]

_LAYERS: tuple[OntologyLayer, ...] = ("base", "event", "diagnostic")
_STRATA: tuple[MaterialStratum, ...] = (
    "chinese",
    "mixed-language",
    "table",
    "reflection",
    "feedback",
    "reference-book",
)
_GATE_DIRECTIONS = {"false_merge_rate": "maximum", "alignment_abstention_rate": "maximum"}
_REQUIRED_RUNTIME_COMPONENTS = frozenset(
    {
        "community-partitioner",
        "device",
        "embedding",
        "embedding-space",
        "generation-profile",
        "generator",
        "retrieval-generation",
        "tokenizer",
    }
)
_METRIC_NAMES = {
    "alignment_abstention_rate",
    "alignment_recall",
    "citation_entailment_rate",
    "counter_evidence_coverage",
    "diagnostic_support_rate",
    "false_merge_rate",
    "occurrence_dedup_accuracy",
    *(f"extraction.{layer}.{measure}" for layer in _LAYERS for measure in ("precision", "recall")),
    *(
        f"extraction.stratum.{stratum}.{measure}"
        for stratum in _STRATA
        for measure in ("precision", "recall")
    ),
}
_REF = re.compile(r"\S{1,256}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _ref(value: object, name: str) -> str:
    if not isinstance(value, str) or _REF.fullmatch(value) is None:
        raise ValueError(f"invalid {name}")
    return value


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"invalid {name}")
    return value


def _closed_tuple(value: object, kind: type[object], name: str) -> tuple[object, ...]:
    if not isinstance(value, tuple) or any(not isinstance(item, kind) for item in value):
        raise TypeError(f"{name} must be a tuple of {kind.__name__}")
    identities: list[str] = []
    for item in value:
        identity = next(
            (
                candidate
                for attribute in ("item_id", "component_id", "baseline_id")
                if isinstance(candidate := getattr(item, attribute, None), str)
            ),
            None,
        )
        if identity is None:
            raise TypeError(f"{name} items require a stable identity")
        identities.append(identity)
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        raise ValueError(f"{name} must have unique, canonical order")
    return value


def _hash(value: LabeledCorpus | EvaluationProfile) -> str:
    def default(item: object) -> list[str]:
        if isinstance(item, frozenset) and all(isinstance(value, str) for value in item):
            return sorted(item)
        raise TypeError(f"unsupported canonical evaluation value: {type(item).__name__}")

    payload = json.dumps(
        asdict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=default,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _runtime_hash(values: tuple[RuntimeIdentity, ...]) -> str:
    payload = json.dumps(
        [asdict(item) for item in values],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ExtractionLabel:
    """One exhaustively adjudicated extraction candidate."""

    item_id: str
    ontology_layer: OntologyLayer
    stratum: MaterialStratum
    positive: bool

    def __post_init__(self) -> None:
        _ref(self.item_id, "extraction item ID")
        if self.ontology_layer not in _LAYERS:
            raise ValueError("unknown ontology layer")
        if self.stratum not in _STRATA:
            raise ValueError("unknown material stratum")
        if type(self.positive) is not bool:
            raise TypeError("positive must be bool")


@dataclass(frozen=True, slots=True)
class DecisionLabel:
    """One identity pair and its closed decision."""

    item_id: str
    decision: Decision

    def __post_init__(self) -> None:
        _ref(self.item_id, "decision item ID")
        if self.decision not in {"matched", "distinct", "ambiguous"}:
            raise ValueError("unknown decision")


@dataclass(frozen=True, slots=True)
class OccurrenceLabel:
    """Gold duplicate/distinct judgment for an event occurrence pair."""

    item_id: str
    duplicate: bool

    def __post_init__(self) -> None:
        _ref(self.item_id, "occurrence item ID")
        if type(self.duplicate) is not bool:
            raise TypeError("duplicate must be bool")


@dataclass(frozen=True, slots=True)
class DiagnosticLabel:
    """Independent judgment that a proposed diagnostic is evidence-supported."""

    item_id: str
    supported: bool

    def __post_init__(self) -> None:
        _ref(self.item_id, "diagnostic item ID")
        if type(self.supported) is not bool:
            raise TypeError("supported must be bool")


@dataclass(frozen=True, slots=True)
class ReportLabel:
    """Independent citation-entailment and counter-evidence requirement label."""

    item_id: str
    entailed: bool
    counter_evidence_required: bool

    def __post_init__(self) -> None:
        _ref(self.item_id, "statement item ID")
        if type(self.entailed) is not bool or type(self.counter_evidence_required) is not bool:
            raise TypeError("report judgments must be bool")


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    """Exact identity of a model, tokenizer, embedding, or algorithm component."""

    component_id: str
    version: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        _ref(self.component_id, "runtime component ID")
        _ref(self.version, "runtime version")
        _digest(self.artifact_sha256, "runtime artifact digest")


@dataclass(frozen=True, slots=True)
class BaselinePrediction:
    """Predictions from one fixed comparison method on the identical corpus."""

    baseline_id: Literal["fixed-token", "name-type-only"]
    extraction_ids: frozenset[str]
    merged_pair_ids: frozenset[str]

    def __post_init__(self) -> None:
        if self.baseline_id not in {"fixed-token", "name-type-only"}:
            raise ValueError("unknown baseline")
        _refs(self.extraction_ids, "baseline extraction IDs")
        _refs(self.merged_pair_ids, "baseline merge IDs")


@dataclass(frozen=True, slots=True)
class LabeledCorpus:
    """Exhaustively labeled, independently reviewed held-out candidate inventory."""

    corpus_id: str
    split_id: str
    annotation_ref: str
    annotator_identity_sha256: str
    independent_annotation: bool
    extraction_labels: tuple[ExtractionLabel, ...]
    merge_labels: tuple[DecisionLabel, ...]
    alignment_labels: tuple[DecisionLabel, ...]
    occurrence_labels: tuple[OccurrenceLabel, ...]
    diagnostic_labels: tuple[DiagnosticLabel, ...]
    report_labels: tuple[ReportLabel, ...]
    confidence_item_ids: frozenset[str]
    schema_version: int = 1

    def __post_init__(self) -> None:
        _ref(self.corpus_id, "corpus ID")
        _ref(self.split_id, "split ID")
        _ref(self.annotation_ref, "annotation reference")
        _digest(self.annotator_identity_sha256, "annotator identity digest")
        if type(self.independent_annotation) is not bool or not self.independent_annotation:
            raise ValueError("independent annotation is required")
        if "held-out" not in self.split_id.lower():
            raise ValueError("labels must identify a held-out split")
        if any(token in self.annotation_ref.lower() for token in ("self-label", "llm-judge")):
            raise ValueError("annotation must be independent of the generator")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported evaluation corpus schema")
        _closed_tuple(self.extraction_labels, ExtractionLabel, "extraction labels")
        _closed_tuple(self.merge_labels, DecisionLabel, "merge labels")
        _closed_tuple(self.alignment_labels, DecisionLabel, "alignment labels")
        _closed_tuple(self.occurrence_labels, OccurrenceLabel, "occurrence labels")
        _closed_tuple(self.diagnostic_labels, DiagnosticLabel, "diagnostic labels")
        _closed_tuple(self.report_labels, ReportLabel, "report labels")
        _refs(self.confidence_item_ids, "confidence item IDs")
        if any(item.decision == "ambiguous" for item in self.merge_labels):
            raise ValueError("merge gold labels must be matched or distinct")


@dataclass(frozen=True, slots=True)
class MetricGate:
    """A numeric threshold declared before observing the qualification run."""

    metric_name: str
    direction: GateDirection
    threshold: float

    def __post_init__(self) -> None:
        if self.metric_name not in _METRIC_NAMES:
            raise ValueError("unknown metric")
        expected = _GATE_DIRECTIONS.get(self.metric_name, "minimum")
        if self.direction != expected:
            raise ValueError("metric gate direction is not allowed")
        if type(self.threshold) not in (int, float) or not math.isfinite(self.threshold):
            raise ValueError("metric threshold must be finite")
        if not 0.0 <= float(self.threshold) <= 1.0:
            raise ValueError("metric threshold must be between zero and one")


@dataclass(frozen=True, slots=True)
class EvaluationProfile:
    """Reviewed, immutable qualification gates for one held-out split."""

    profile_id: str
    held_out_split_id: str
    reviewer_approval_ref: str
    reviewer_identity_sha256: str
    gates: tuple[MetricGate, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        _ref(self.profile_id, "evaluation profile ID")
        _ref(self.held_out_split_id, "held-out split ID")
        _ref(self.reviewer_approval_ref, "reviewer approval reference")
        _digest(self.reviewer_identity_sha256, "reviewer identity digest")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported evaluation profile schema")
        if not isinstance(self.gates, tuple) or any(
            not isinstance(item, MetricGate) for item in self.gates
        ):
            raise TypeError("gates must be a tuple of MetricGate")
        if not self.gates:
            raise ValueError("at least one reviewed metric gate is required")
        names = [item.metric_name for item in self.gates]
        if len(names) != len(set(names)):
            raise ValueError("duplicate metric gate")
        if names != sorted(names):
            raise ValueError("metric gates must use canonical order")


@dataclass(frozen=True, slots=True)
class PredictedGraphRun:
    """Bounded predictions and measured execution facts from one exact run."""

    run_id: str
    run_sha256: str
    runtime_identities: tuple[RuntimeIdentity, ...]
    extraction_ids: frozenset[str]
    merged_pair_ids: frozenset[str]
    alignment_decisions: tuple[DecisionLabel, ...]
    duplicate_occurrence_ids: frozenset[str]
    diagnostic_finding_ids: frozenset[str]
    supported_diagnostic_ids: frozenset[str]
    report_statement_ids: frozenset[str]
    entailed_statement_ids: frozenset[str]
    counter_evidence_statement_ids: frozenset[str]
    confidence_assessed_ids: frozenset[str]
    confidence_accepted_ids: frozenset[str]
    baselines: tuple[BaselinePrediction, ...]
    elapsed_ms: int
    input_tokens: int
    output_tokens: int
    cost_microunits: int
    generation_calls: int
    cache_hits: int
    structural_checks_passed: bool
    schema_version: int = 1

    def __post_init__(self) -> None:
        _ref(self.run_id, "run ID")
        _digest(self.run_sha256, "run digest")
        _closed_tuple(self.runtime_identities, RuntimeIdentity, "runtime identities")
        _closed_tuple(self.alignment_decisions, DecisionLabel, "alignment decisions")
        _closed_tuple(self.baselines, BaselinePrediction, "baselines")
        for name in (
            "extraction_ids",
            "merged_pair_ids",
            "duplicate_occurrence_ids",
            "diagnostic_finding_ids",
            "supported_diagnostic_ids",
            "report_statement_ids",
            "entailed_statement_ids",
            "counter_evidence_statement_ids",
            "confidence_assessed_ids",
            "confidence_accepted_ids",
        ):
            _refs(getattr(self, name), name)
        if not self.supported_diagnostic_ids <= self.diagnostic_finding_ids:
            raise ValueError("supported diagnostics must be predicted findings")
        if not self.entailed_statement_ids <= self.report_statement_ids:
            raise ValueError("entailed statements must be predicted statements")
        if not self.counter_evidence_statement_ids <= self.report_statement_ids:
            raise ValueError("counter evidence must belong to predicted statements")
        if not self.confidence_accepted_ids <= self.confidence_assessed_ids:
            raise ValueError("accepted confidence decisions must be assessed")
        for name in (
            "elapsed_ms",
            "input_tokens",
            "output_tokens",
            "cost_microunits",
            "generation_calls",
            "cache_hits",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if type(self.structural_checks_passed) is not bool:
            raise TypeError("structural_checks_passed must be bool")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported predicted-run schema")


@dataclass(frozen=True, slots=True)
class MetricResult:
    """One metric with explicit numerator and optional denominator."""

    name: str
    value: float | None
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.name not in _METRIC_NAMES:
            raise ValueError("unknown metric result")
        if (
            type(self.numerator) is not int
            or type(self.denominator) is not int
            or not 0 <= self.numerator <= self.denominator
        ):
            raise ValueError("invalid metric numerator or denominator")
        expected = self.numerator / self.denominator if self.denominator else None
        if expected is None:
            if self.value is not None:
                raise ValueError("metric without denominator must be None")
        else:
            if self.value is None:
                raise ValueError("metric value does not match its counts")
            value = float(self.value)
            if (
                type(self.value) not in (int, float)
                or not math.isfinite(value)
                or not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-15)
            ):
                raise ValueError("metric value does not match its counts")


@dataclass(frozen=True, slots=True)
class BaselineResult:
    """Comparable extraction and false-merge metrics for one fixed baseline."""

    baseline_id: str
    extraction_precision: float | None
    extraction_recall: float | None
    false_merge_rate: float | None

    def __post_init__(self) -> None:
        if self.baseline_id not in {"fixed-token", "name-type-only"}:
            raise ValueError("unknown baseline result")
        for value in (
            self.extraction_precision,
            self.extraction_recall,
            self.false_merge_rate,
        ):
            if value is not None and (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError("invalid baseline metric")


@dataclass(frozen=True, slots=True)
class QualificationReport:
    """Immutable results; ``qualified`` is false if any gate is undefined."""

    corpus_sha256: str
    profile_sha256: str
    model_sha256: str
    run_sha256: str
    runtime_identities: tuple[RuntimeIdentity, ...]
    counts: tuple[tuple[str, int], ...]
    metrics: tuple[MetricResult, ...]
    baselines: tuple[BaselineResult, ...]
    elapsed_ms: int
    input_tokens: int
    output_tokens: int
    cost_microunits: int
    generation_calls: int
    cache_hits: int
    confidence_accepted: int
    confidence_denominator: int
    abstention_count: int
    abstention_denominator: int
    qualified: bool
    failure_reasons: tuple[str, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in ("corpus_sha256", "profile_sha256", "model_sha256", "run_sha256"):
            _digest(getattr(self, name), name)
        _closed_tuple(self.runtime_identities, RuntimeIdentity, "runtime identities")
        if self.model_sha256 != _runtime_hash(self.runtime_identities):
            raise ValueError("runtime identities do not match model digest")
        if (
            not isinstance(self.counts, tuple)
            or any(
                not isinstance(item, tuple)
                or len(item) != 2
                or not isinstance(item[0], str)
                or type(item[1]) is not int
                or item[1] < 0
                for item in self.counts
            )
            or list(self.counts) != sorted(self.counts)
            or len({item[0] for item in self.counts}) != len(self.counts)
        ):
            raise ValueError("invalid qualification counts")
        if not isinstance(self.metrics, tuple) or any(
            not isinstance(item, MetricResult) for item in self.metrics
        ):
            raise TypeError("metrics must be a tuple of MetricResult")
        metric_names = [item.name for item in self.metrics]
        if metric_names != sorted(metric_names) or len(metric_names) != len(set(metric_names)):
            raise ValueError("metric results must have unique canonical order")
        _closed_tuple(self.baselines, BaselineResult, "baseline results")
        if tuple(item.baseline_id for item in self.baselines) != (
            "fixed-token",
            "name-type-only",
        ):
            raise ValueError("invalid baseline result inventory")
        for name in (
            "elapsed_ms",
            "input_tokens",
            "output_tokens",
            "cost_microunits",
            "generation_calls",
            "cache_hits",
            "confidence_accepted",
            "confidence_denominator",
            "abstention_count",
            "abstention_denominator",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"invalid qualification {name}")
        if self.confidence_accepted > self.confidence_denominator:
            raise ValueError("invalid qualification confidence denominator")
        if self.abstention_count > self.abstention_denominator:
            raise ValueError("invalid qualification abstention denominator")
        if (
            not isinstance(self.failure_reasons, tuple)
            or any(
                not isinstance(item, str) or _REF.fullmatch(item) is None
                for item in self.failure_reasons
            )
            or tuple(sorted(set(self.failure_reasons))) != self.failure_reasons
        ):
            raise ValueError("invalid qualification failure reasons")
        if type(self.qualified) is not bool or self.qualified != (not self.failure_reasons):
            raise ValueError("qualification status and failure reasons disagree")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported qualification report schema")


def _refs(values: object, name: str) -> frozenset[str]:
    if not isinstance(values, frozenset):
        raise TypeError(f"{name} must be frozenset")
    for value in values:
        _ref(value, name)
    return values


def precision_recall(
    predicted: frozenset[str], gold: frozenset[str]
) -> tuple[float | None, float | None]:
    """Return precision and recall without treating absent denominators as one."""
    _refs(predicted, "predicted IDs")
    _refs(gold, "gold IDs")
    correct = len(predicted & gold)
    precision = correct / len(predicted) if predicted else None
    recall = correct / len(gold) if gold else None
    return precision, recall


def _ratio(name: str, numerator: int, denominator: int) -> MetricResult:
    return MetricResult(
        name, numerator / denominator if denominator else None, numerator, denominator
    )


def _inventory(corpus: LabeledCorpus) -> dict[str, frozenset[str]]:
    return {
        "extraction": frozenset(item.item_id for item in corpus.extraction_labels),
        "merge": frozenset(item.item_id for item in corpus.merge_labels),
        "alignment": frozenset(item.item_id for item in corpus.alignment_labels),
        "occurrence": frozenset(item.item_id for item in corpus.occurrence_labels),
        "diagnostic": frozenset(item.item_id for item in corpus.diagnostic_labels),
        "report": frozenset(item.item_id for item in corpus.report_labels),
    }


def _subset(values: frozenset[str], allowed: frozenset[str], name: str) -> None:
    if not values <= allowed:
        raise ValueError(f"{name} contains predictions outside labeled corpus")


def _extraction_metrics(run: PredictedGraphRun, corpus: LabeledCorpus) -> list[MetricResult]:
    result: list[MetricResult] = []
    for prefix, values in (
        ("", _LAYERS),
        ("stratum.", _STRATA),
    ):
        for value in values:
            labels = tuple(
                item
                for item in corpus.extraction_labels
                if getattr(item, "stratum" if prefix else "ontology_layer") == value
            )
            universe = frozenset(item.item_id for item in labels)
            gold = frozenset(item.item_id for item in labels if item.positive)
            predicted = run.extraction_ids & universe
            precision, recall = precision_recall(predicted, gold)
            correct = len(predicted & gold)
            result.extend(
                (
                    MetricResult(
                        f"extraction.{prefix}{value}.precision", precision, correct, len(predicted)
                    ),
                    MetricResult(f"extraction.{prefix}{value}.recall", recall, correct, len(gold)),
                )
            )
    return result


def _baseline_results(
    baselines: tuple[BaselinePrediction, ...], corpus: LabeledCorpus
) -> tuple[BaselineResult, ...]:
    inventory = _inventory(corpus)
    gold_extraction = frozenset(item.item_id for item in corpus.extraction_labels if item.positive)
    distinct = frozenset(
        item.item_id for item in corpus.merge_labels if item.decision == "distinct"
    )
    output = []
    for baseline in baselines:
        _subset(baseline.extraction_ids, inventory["extraction"], "baseline extraction")
        _subset(baseline.merged_pair_ids, inventory["merge"], "baseline merge")
        precision, recall = precision_recall(baseline.extraction_ids, gold_extraction)
        false = len(baseline.merged_pair_ids & distinct)
        output.append(
            BaselineResult(
                baseline.baseline_id,
                precision,
                recall,
                false / len(baseline.merged_pair_ids) if baseline.merged_pair_ids else None,
            )
        )
    return tuple(output)


def evaluate_graph_run(
    predicted: PredictedGraphRun,
    labeled_corpus: LabeledCorpus,
    evaluation_profile: EvaluationProfile,
) -> QualificationReport:
    """Evaluate one exact run against one independently reviewed held-out corpus."""
    if not isinstance(predicted, PredictedGraphRun):
        raise TypeError("predicted must be PredictedGraphRun")
    if not isinstance(labeled_corpus, LabeledCorpus):
        raise TypeError("labeled_corpus must be LabeledCorpus")
    if not isinstance(evaluation_profile, EvaluationProfile):
        raise TypeError("evaluation_profile must be EvaluationProfile")
    if evaluation_profile.held_out_split_id != labeled_corpus.split_id:
        raise ValueError("evaluation profile and corpus split mismatch")
    inventory = _inventory(labeled_corpus)
    _subset(predicted.extraction_ids, inventory["extraction"], "extraction")
    _subset(predicted.merged_pair_ids, inventory["merge"], "merge")
    decisions = {item.item_id: item.decision for item in predicted.alignment_decisions}
    if frozenset(decisions) != inventory["alignment"]:
        raise ValueError("alignment predictions must cover the labeled corpus exactly")
    if predicted.confidence_assessed_ids != labeled_corpus.confidence_item_ids:
        raise ValueError("confidence assessments must exactly cover the labeled corpus")
    _subset(predicted.duplicate_occurrence_ids, inventory["occurrence"], "occurrence")
    _subset(predicted.diagnostic_finding_ids, inventory["diagnostic"], "diagnostic")
    _subset(predicted.report_statement_ids, inventory["report"], "report")
    baseline_ids = tuple(item.baseline_id for item in predicted.baselines)
    if baseline_ids != ("fixed-token", "name-type-only"):
        raise ValueError("both fixed baselines are required in canonical order")
    generators = [item for item in predicted.runtime_identities if item.component_id == "generator"]
    if len(generators) != 1:
        raise ValueError("one exact generator identity is required")

    metrics = _extraction_metrics(predicted, labeled_corpus)
    distinct_merges = frozenset(
        item.item_id for item in labeled_corpus.merge_labels if item.decision == "distinct"
    )
    metrics.append(
        _ratio(
            "false_merge_rate",
            len(predicted.merged_pair_ids & distinct_merges),
            len(predicted.merged_pair_ids),
        )
    )
    gold_matches = frozenset(
        item.item_id for item in labeled_corpus.alignment_labels if item.decision == "matched"
    )
    predicted_matches = frozenset(key for key, value in decisions.items() if value == "matched")
    metrics.append(
        _ratio("alignment_recall", len(gold_matches & predicted_matches), len(gold_matches))
    )
    metrics.append(
        _ratio(
            "alignment_abstention_rate",
            sum(value == "ambiguous" for value in decisions.values()),
            len(decisions),
        )
    )
    expected_duplicates = frozenset(
        item.item_id for item in labeled_corpus.occurrence_labels if item.duplicate
    )
    occurrence_correct = len(inventory["occurrence"]) - len(
        predicted.duplicate_occurrence_ids ^ expected_duplicates
    )
    metrics.append(
        _ratio("occurrence_dedup_accuracy", occurrence_correct, len(inventory["occurrence"]))
    )
    gold_supported = frozenset(
        item.item_id for item in labeled_corpus.diagnostic_labels if item.supported
    )
    metrics.append(
        _ratio(
            "diagnostic_support_rate",
            len(predicted.supported_diagnostic_ids & gold_supported),
            len(predicted.diagnostic_finding_ids),
        )
    )
    gold_entailed = frozenset(
        item.item_id for item in labeled_corpus.report_labels if item.entailed
    )
    metrics.append(
        _ratio(
            "citation_entailment_rate",
            len(predicted.entailed_statement_ids & gold_entailed),
            len(predicted.report_statement_ids),
        )
    )
    counter_required = frozenset(
        item.item_id for item in labeled_corpus.report_labels if item.counter_evidence_required
    )
    metrics.append(
        _ratio(
            "counter_evidence_coverage",
            len(predicted.counter_evidence_statement_ids & counter_required),
            len(counter_required),
        )
    )
    metrics_by_name = {item.name: item for item in metrics}
    failures: list[str] = []
    runtime_components = {item.component_id for item in predicted.runtime_identities}
    failures.extend(
        f"missing_runtime_identity:{component}"
        for component in sorted(_REQUIRED_RUNTIME_COMPONENTS - runtime_components)
    )
    if not predicted.structural_checks_passed:
        failures.append("structural_checks_failed")
    for gate in evaluation_profile.gates:
        result = metrics_by_name[gate.metric_name]
        if result.value is None:
            failures.append(f"metric_without_denominator:{gate.metric_name}")
        elif gate.direction == "minimum" and result.value < gate.threshold:
            failures.append(f"gate_failed:{gate.metric_name}")
        elif gate.direction == "maximum" and result.value > gate.threshold:
            failures.append(f"gate_failed:{gate.metric_name}")
    counts = (
        ("alignment_assessed", len(labeled_corpus.alignment_labels)),
        ("diagnostic_findings", len(predicted.diagnostic_finding_ids)),
        ("extraction_candidates", len(labeled_corpus.extraction_labels)),
        ("report_statements", len(predicted.report_statement_ids)),
    )
    return QualificationReport(
        corpus_sha256=_hash(labeled_corpus),
        profile_sha256=_hash(evaluation_profile),
        model_sha256=_runtime_hash(predicted.runtime_identities),
        run_sha256=predicted.run_sha256,
        runtime_identities=predicted.runtime_identities,
        counts=counts,
        metrics=tuple(sorted(metrics, key=lambda item: item.name)),
        baselines=_baseline_results(predicted.baselines, labeled_corpus),
        elapsed_ms=predicted.elapsed_ms,
        input_tokens=predicted.input_tokens,
        output_tokens=predicted.output_tokens,
        cost_microunits=predicted.cost_microunits,
        generation_calls=predicted.generation_calls,
        cache_hits=predicted.cache_hits,
        confidence_accepted=len(predicted.confidence_accepted_ids),
        confidence_denominator=len(predicted.confidence_assessed_ids),
        abstention_count=sum(value == "ambiguous" for value in decisions.values()),
        abstention_denominator=len(decisions),
        qualified=not failures,
        failure_reasons=tuple(sorted(failures)),
    )


__all__ = [
    "BaselinePrediction",
    "BaselineResult",
    "DecisionLabel",
    "DiagnosticLabel",
    "EvaluationProfile",
    "ExtractionLabel",
    "LabeledCorpus",
    "MetricGate",
    "MetricResult",
    "OccurrenceLabel",
    "PredictedGraphRun",
    "QualificationReport",
    "ReportLabel",
    "RuntimeIdentity",
    "evaluate_graph_run",
    "precision_recall",
]
