"""Deterministic, bounded personal-growth investigation and synthesis validation.

The caller supplies authorized current source assertions and an explicitly bound
subject. Names, scope ownership and narrator labels never bind a person. Source
lineage must already be verified upstream: opaque evidence IDs cannot prove their
own provenance. This module rejects declared derivations and generated sources.

Bundles preserve complete assertion paths and counter closures. Their readable
kind and content digest bind eligibility, not truth or authenticity. They must
come from ``diagnostic_candidates`` in the trusted host, never from a provider.
The provider receives the bundled assertions separately via Task 8; neither pure
function invokes generation. Textual explanations still require entailment
evaluation; structural validation cannot prove arbitrary prose true.
"""

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

from capicove.contracts.graph_build import (
    DiagnosticFinding,
    Entity,
    EvidenceBundle,
    RelationshipAssertion,
)
from capicove.contracts.graph_build.vocabulary import DIAGNOSTICS
from capicove.graph_build.aggregation import aggregate_assertions
from capicove.graph_build.codec import fingerprint
from capicove.graph_build.ontology import relation_allowed
from capicove.graph_build.validation import StructuralOutputError


_KINDS = {
    "error_pattern": "ERROR_PATTERN",
    "difficulty": "DIFFICULTY",
    "knowledge_gap": "KNOWLEDGE_GAP",
    "feedback": "DIFFICULTY",
    "cause_hypothesis": "CAUSE_HYPOTHESIS",
    "false_assumption": "FALSE_ASSUMPTION",
    "intervention_recommendation": "INTERVENTION",
    "intervention_outcome": "INTERVENTION",
}
_REPEATED = frozenset({"error_pattern", "feedback"})
_GENERATED_PREFIXES = (
    "report:",
    "finding:",
    "diagnostic-finding:",
    "generated:",
    "generated_report:",
    "generated-report:",
    "community-report:",
)
_UNCERTAIN_SUPPORT = ("unknown-support:", "composite-support:")
_MAX_RESPONSE_BYTES = 65536


def _ref(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 256
        or any(char.isspace() or ord(char) < 32 for char in value)
    ):
        raise ValueError("invalid_reference")
    return value


@dataclass(frozen=True, slots=True)
class DiagnosticProfile:
    """Explicit subject and local evidence budgets for personal-growth-v1.

    Assertion and counter IDs share a budget. Units with incomplete path/counter
    closure are omitted whole; omitted_count counts distinct unselected relevant
    assertions. max_bundles caps returned candidates, not assertion omissions.
    """

    subject_ref: str
    max_assertions_per_bundle: int
    max_bundles: int
    version: int = 1
    source_kind: str = "source"

    def __post_init__(self) -> None:
        _ref(self.subject_ref)
        if type(self.version) is not int or self.version != 1:
            raise ValueError("unsupported_diagnostic_profile")
        if (
            type(self.max_assertions_per_bundle) is not int
            or not 1 <= self.max_assertions_per_bundle <= 128
            or type(self.max_bundles) is not int
            or not 0 <= self.max_bundles <= 128
        ):
            raise ValueError("invalid_diagnostic_budget")
        _ref(self.source_kind)


def repeat_pattern_eligible(
    event_ids: tuple[str, ...], personal_attribution: bool, source_kind: str
) -> bool:
    """Apply the conservative recurrence kernel to established personal events."""
    return personal_attribution and source_kind == "source" and len(set(event_ids)) >= 2


def _source(row: RelationshipAssertion) -> bool:
    return row.derivation_ref is None and not any(
        ref.startswith(_GENERATED_PREFIXES) for ref in (row.assertion_id, *row.evidence_refs)
    )


def _positive(row: RelationshipAssertion) -> bool:
    return _source(row) and row.epistemic_status == "explicit" and row.polarity == "positive"


def _graph(
    assertions: tuple[RelationshipAssertion, ...],
    entities: tuple[Entity, ...],
    profile: DiagnosticProfile,
) -> tuple[dict[str, RelationshipAssertion], dict[str, Entity]]:
    if not isinstance(profile, DiagnosticProfile):
        raise TypeError("profile must be DiagnosticProfile")
    if not isinstance(assertions, tuple) or any(
        not isinstance(row, RelationshipAssertion) for row in assertions
    ):
        raise TypeError("assertions must be a tuple of RelationshipAssertion")
    if not isinstance(entities, tuple) or any(not isinstance(row, Entity) for row in entities):
        raise TypeError("entities must be a tuple of Entity")
    rows = {row.assertion_id: row for row in assertions}
    nodes = {node.entity_id: node for node in entities}
    if len(rows) != len(assertions) or len(nodes) != len(entities):
        raise ValueError("duplicate_graph_identity")
    subject = nodes.get(profile.subject_ref)
    if subject is None or subject.entity_type != "PERSON":
        raise ValueError("explicit_person_subject_required")
    for node in nodes.values():
        _ref(node.entity_id)
        if node.scope != subject.scope:
            raise ValueError("mixed_graph_scopes")
    for row in rows.values():
        for ref in (row.assertion_id, *row.evidence_refs, row.support_group_id):
            _ref(ref)
        source, target = nodes.get(row.source_entity_id), nodes.get(row.target_entity_id)
        if (
            source is None
            or target is None
            or not relation_allowed(source.entity_type, row.predicate, target.entity_type)
        ):
            raise ValueError("invalid_assertion_endpoints")
        if row.attribution_ref is not None:
            speaker = nodes.get(row.attribution_ref)
            if speaker is None or speaker.entity_type != "PERSON":
                raise ValueError("invalid_assertion_attribution")
    aggregate_assertions(assertions)  # Validate relevant explicit counters, including cycles.
    return rows, nodes


@dataclass(frozen=True, slots=True)
class _Path:
    event: str
    group: str | None
    refs: frozenset[str]


def _extend(path: _Path, row: RelationshipAssertion) -> _Path:
    return _Path(path.event, path.group, path.refs | {row.assertion_id})


def _paths(
    rows: dict[str, RelationshipAssertion],
    nodes: dict[str, Entity],
    subject: str,
) -> dict[tuple[str, str], list[_Path]]:
    participation: dict[str, list[RelationshipAssertion]] = {}
    for row in rows.values():
        if (
            _positive(row)
            and row.predicate == "PARTICIPATED_IN"
            and row.source_entity_id == subject
        ):
            participation.setdefault(row.target_entity_id, []).append(row)
    candidates: dict[tuple[str, str], list[_Path]] = {}
    personal_targets: dict[str, list[_Path]] = {}
    for row in rows.values():
        if not _source(row) or row.epistemic_status != "explicit":
            continue
        event_type = nodes[row.source_entity_id].entity_type
        kind = None
        if row.predicate == "EVALUATES" and event_type == "FEEDBACK_EVENT":
            kind = "feedback"
        elif row.polarity == "positive":
            if row.predicate == "EXHIBITS":
                kind = "error_pattern"
            elif row.predicate == "REVEALS":
                kind = nodes[row.target_entity_id].entity_type.lower()
            elif row.predicate == "EVALUATES_EFFECT_OF":
                kind = "intervention_outcome"
        if row.polarity != "positive" and kind != "feedback":
            continue
        for attribution in participation.get(row.source_entity_id, ()):
            known = not any(
                item.support_group_id.startswith(_UNCERTAIN_SUPPORT) for item in (row, attribution)
            )
            path = _Path(
                row.source_entity_id,
                row.support_group_id if known else None,
                frozenset({row.assertion_id, attribution.assertion_id}),
            )
            if kind is not None:
                candidates.setdefault((kind, row.target_entity_id), []).append(path)
            if kind in {
                "error_pattern",
                "difficulty",
                "knowledge_gap",
                "feedback",
            } or row.predicate in {"PRACTICES_SKILL", "APPLIES_CONCEPT"}:
                personal_targets.setdefault(row.target_entity_id, []).append(path)
    # Causal/misconception candidates require a supplied source path. The rule
    # neither invents causes nor declares the premise mistaken by itself.
    for row in rows.values():
        if _positive(row) and row.predicate in {"MAY_BE_CAUSED_BY", "BASED_ON"}:
            kind = "cause_hypothesis" if row.predicate == "MAY_BE_CAUSED_BY" else "false_assumption"
            extensions = [
                _extend(path, row) for path in personal_targets.get(row.source_entity_id, ())
            ]
            if extensions:
                candidates.setdefault((kind, row.target_entity_id), []).extend(extensions)
    for (kind, target), paths in tuple(candidates.items()):
        if kind == "false_assumption":
            personal_targets.setdefault(target, []).extend(paths)
    for row in rows.values():
        if _positive(row) and row.predicate in {"ADDRESSES", "TARGETS"}:
            extensions = [
                _extend(path, row) for path in personal_targets.get(row.target_entity_id, ())
            ]
            if extensions:
                candidates.setdefault(
                    ("intervention_recommendation", row.source_entity_id), []
                ).extend(extensions)
    return candidates


def _counter_links(rows: dict[str, RelationshipAssertion]) -> dict[str, set[str]]:
    links = {ref: set(row.counter_assertion_ids) for ref, row in rows.items()}
    for row in rows.values():
        for other in row.counter_assertion_ids:
            links[other].add(row.assertion_id)
    return links


def _closure(refs: frozenset[str], links: dict[str, set[str]]) -> frozenset[str]:
    result, pending = set(refs), list(refs)
    while pending:
        for ref in links[pending.pop()] - result:
            result.add(ref)
            pending.append(ref)
    return frozenset(result)


def _independent(paths: Iterable[_Path]) -> tuple[str, ...]:
    """Count only distinct events supported by distinct established groups.

    Matching permits duplicate observations without assigning one support group
    to two events. Reserved unknown/composite markers never enter the matching.
    """
    groups: dict[str, set[str]] = {}
    for path in paths:
        if path.group is not None:
            groups.setdefault(path.event, set()).add(path.group)
    matched: dict[str, str] = {}

    def assign(event: str, seen: set[str]) -> bool:
        for group in sorted(groups[event] - seen):
            seen.add(group)
            if group not in matched or assign(matched[group], seen):
                matched[group] = event
                return True
        return False

    for event in sorted(groups):
        assign(event, set())
    return tuple(sorted(matched.values()))


def _bundle_id(
    kind: str,
    subject: str,
    target: str,
    supports: tuple[str, ...],
    counters: tuple[str, ...],
    events: tuple[str, ...],
    omitted: int,
) -> str:
    return f"diagnostic-bundle:v1:{kind}:" + fingerprint(
        {
            "version": 1,
            "kind": kind,
            "subject": subject,
            "target": target,
            "assertions": supports,
            "counters": counters,
            "events": events,
            "omitted": omitted,
        }
    )


@dataclass(frozen=True, slots=True)
class _ClosedPath:
    path: _Path
    refs: frozenset[str]

    @property
    def order(self) -> tuple[int, str, str, tuple[str, ...], tuple[str, ...]]:
        return (
            len(self.refs),
            self.path.event,
            self.path.group or "",
            tuple(sorted(self.path.refs)),
            tuple(sorted(self.refs)),
        )


def _select_paths(choices: list[_ClosedPath], budget: int, repeated: bool) -> list[_ClosedPath]:
    """Establish eligibility before allowing redundant evidence to spend budget.

    Reduce alternatives to one minimum closure per event/group. Recurrence needs
    exactly two independent paths, so compare only pairs of these representatives
    (quadratic time, linear storage), never subsets or an unbounded backtracking
    search. Selected closures and subsequent matching are bounded by 128 refs.
    Add paths increasing event/group matching before a final redundant-evidence
    pass. This is deterministic eligibility preservation, not a claim of global
    maximum evidence coverage under a set-union budget.
    """
    ordered = sorted(choices, key=lambda choice: choice.order)
    representatives: dict[tuple[str, str | None], _ClosedPath] = {}
    for choice in ordered:
        representatives.setdefault((choice.path.event, choice.path.group), choice)
    reduced = list(representatives.values())
    selected: list[_ClosedPath] = []
    if repeated:
        best: tuple[int, int, int] | None = None
        for i, left in enumerate(reduced):
            for j in range(i + 1, len(reduced)):
                right = reduced[j]
                if left.path.event == right.path.event or left.path.group == right.path.group:
                    continue
                size = len(left.refs | right.refs)
                rank = (size, i, j)
                if size <= budget and (best is None or rank < best):
                    best = rank
        if best is None:
            return []
        selected = [reduced[best[1]], reduced[best[2]]]
    elif reduced:
        selected = [reduced[0]]
    refs = set(ref for choice in selected for ref in choice.refs)
    for independent_only in (True, False):
        for choice in ordered:
            if choice in selected or len(refs | choice.refs) > budget:
                continue
            if independent_only and len(
                _independent(item.path for item in (*selected, choice))
            ) <= len(_independent(item.path for item in selected)):
                continue
            selected.append(choice)
            refs.update(choice.refs)
    return selected


def _eligible_closure(
    refs: frozenset[str],
    rows: dict[str, RelationshipAssertion],
    nodes: dict[str, Entity],
    subject: str,
) -> bool:
    for ref in refs:
        row = rows[ref]
        if not _source(row) or row.support_group_id.startswith(_UNCERTAIN_SUPPORT):
            return False
        for node_id in (row.source_entity_id, row.target_entity_id):
            node = nodes[node_id]
            if node.entity_type in DIAGNOSTICS and node.subject_ref not in (None, subject):
                return False
    return True


def diagnostic_candidates(
    assertions: tuple[RelationshipAssertion, ...],
    entities: tuple[Entity, ...],
    profile: DiagnosticProfile,
) -> tuple[EvidenceBundle, ...]:
    """Select complete source paths; eligibility only triggers investigation.

    Stable ordering selects whole paths with bidirectional counter closure. If
    closure touches generated/uncertain evidence, conflicts with a diagnostic
    entity's explicit subject, or cannot fit the budget, omit that path.
    Recurrence must still hold after selection. Counter references share the
    evidence budget and never masquerade as positive supporting assertions.
    """
    rows, nodes = _graph(assertions, entities, profile)
    if profile.source_kind != "source" or profile.max_bundles == 0:
        return ()
    links = _counter_links(rows)
    result = []
    for (kind, target), paths in sorted(_paths(rows, nodes, profile.subject_ref).items()):
        all_refs: set[str] = set()
        choices = []
        for path in paths:
            closure = _closure(path.refs, links)
            all_refs.update(closure)
            if (
                path.group is None
                or len(closure) > profile.max_assertions_per_bundle
                or not _eligible_closure(closure, rows, nodes, profile.subject_ref)
            ):
                continue
            choices.append(_ClosedPath(path, closure))
        chosen = _select_paths(choices, profile.max_assertions_per_bundle, kind in _REPEATED)
        selected = {ref for choice in chosen for ref in choice.refs}
        selected_paths = [choice.path for choice in chosen]
        support_seeds = {ref for path in selected_paths for ref in path.refs}
        events = _independent(selected_paths)
        if not selected_paths or (
            kind in _REPEATED and not repeat_pattern_eligible(events, True, profile.source_kind)
        ):
            continue
        counters = tuple(
            sorted(
                ref
                for ref in selected
                if ref not in support_seeds or rows[ref].counter_assertion_ids
            )
        )
        supports = tuple(sorted(selected - set(counters)))
        if not supports:
            continue
        omitted = len(all_refs - selected)
        result.append(
            EvidenceBundle(
                _bundle_id(kind, profile.subject_ref, target, supports, counters, events, omitted),
                profile.subject_ref,
                target,
                supports,
                counters,
                events,
                omitted,
            )
        )
        if len(result) == profile.max_bundles:
            break
    return tuple(result)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _schema(value: Any, schema: dict[str, Any], definitions: dict[str, Any]) -> None:
    """Validate only the closed schema's small, dependency-free JSON subset."""
    if "$ref" in schema:
        _schema(value, definitions[schema["$ref"].rsplit("/", 1)[1]], definitions)
        return
    expected = schema["type"]
    if isinstance(expected, list):
        if value is None and "null" in expected:
            return
        expected = "string"
    valid = {
        "object": type(value) is dict,
        "array": type(value) is list,
        "string": type(value) is str,
        "integer": type(value) is int,
    }
    if (
        not valid[expected]
        or ("const" in schema and value != schema["const"])
        or ("enum" in schema and value not in schema["enum"])
    ):
        raise ValueError("invalid_diagnostic_schema")
    if expected == "object":
        if set(value) != set(schema["required"]):
            raise ValueError("invalid_diagnostic_fields")
        for key, item in value.items():
            _schema(item, schema["properties"][key], definitions)
    elif expected == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema["maxItems"]:
            raise ValueError("invalid_diagnostic_array_bound")
        for item in value:
            _schema(item, schema["items"], definitions)
        if schema.get("uniqueItems") and len(set(value)) != len(value):
            raise ValueError("duplicate_diagnostic_reference")
    elif expected == "string":
        if not value.strip() or not schema.get("minLength", 0) <= len(value) <= schema["maxLength"]:
            raise ValueError("invalid_diagnostic_text_bound")


def _validate_bundle(bundle: EvidenceBundle, receipt: str) -> str:
    if not isinstance(bundle, EvidenceBundle):
        raise TypeError("bundle must be EvidenceBundle")
    _ref(receipt)
    match = re.fullmatch(r"diagnostic-bundle:v1:([a-z_]+):[0-9a-f]{64}", bundle.bundle_id)
    if match is None or match[1] not in _KINDS:
        raise ValueError("unknown_diagnostic_bundle_kind")
    kind = match[1]
    sources = (*bundle.assertion_ids, *bundle.counter_assertion_ids)
    refs = (bundle.subject_ref, bundle.target_ref, *sources, *bundle.independent_event_ids)
    for ref in refs:
        _ref(ref)
    if (
        not bundle.assertion_ids
        or len(sources) > 128
        or len(sources) != len(set(sources))
        or set(sources) & {bundle.subject_ref, bundle.target_ref, *bundle.independent_event_ids}
        or receipt in {*refs, bundle.bundle_id}
        or any(ref.startswith(_GENERATED_PREFIXES) for ref in sources)
        or (kind in _REPEATED and len(bundle.independent_event_ids) < 2)
        or bundle.bundle_id
        != _bundle_id(
            kind,
            bundle.subject_ref,
            bundle.target_ref,
            bundle.assertion_ids,
            bundle.counter_assertion_ids,
            bundle.independent_event_ids,
            bundle.omitted_count,
        )
    ):
        raise ValueError("invalid_diagnostic_bundle")
    return kind


def validate_findings(
    payload: bytes, bundle: EvidenceBundle, receipt_ref: str
) -> tuple[DiagnosticFinding, ...]:
    """Reject unsupported structured conclusions and bind accepted output lineage.

    Every finding discloses eligibility uncertainty, all selected counter refs,
    and exact evidence omissions. Recommendations require a proposed check and
    an explicit not-executed limitation. Outcome findings describe documented
    associations; neither missing outcomes nor recommendations establish effect.
    IDs bind the receipt, bundle and normalized complete content. Model-local IDs
    only detect duplicate response records and cannot supply persistent identity.
    """
    kind = _validate_bundle(bundle, receipt_ref)
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if len(payload) > _MAX_RESPONSE_BYTES:
        raise ValueError("diagnostic_response_too_large")
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_pairs)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise StructuralOutputError("invalid_diagnostic_json") from error
    schema = json.loads(
        files("capicove.graph_build").joinpath("schemas/diagnostics-v1.json").read_bytes()
    )
    try:
        _schema(value, schema, schema["$defs"])
    except ValueError as error:
        raise StructuralOutputError(str(error)) from error
    local_ids: set[str] = set()
    findings: dict[str, DiagnosticFinding] = {}
    for row in value["findings"]:
        _ref(row["local_id"])
        if row["local_id"] in {
            bundle.bundle_id,
            bundle.subject_ref,
            bundle.target_ref,
            receipt_ref,
            *bundle.assertion_ids,
            *bundle.counter_assertion_ids,
            *bundle.independent_event_ids,
        }:
            raise ValueError("self_referencing_finding_id")
        if row["local_id"] in local_ids:
            raise ValueError("duplicate_finding_id")
        local_ids.add(row["local_id"])
        if (
            row["candidate_kind"] != kind
            or row["diagnostic_type"] != _KINDS[kind]
            or row["subject_ref"] != bundle.subject_ref
            or row["target_ref"] != bundle.target_ref
        ):
            raise ValueError("finding_outside_candidate")
        supports, counters = set(row["support_refs"]), set(row["counter_refs"])
        if not supports <= set(bundle.assertion_ids) or counters != set(
            bundle.counter_assertion_ids
        ):
            raise ValueError("finding_evidence_mismatch")
        required = {"eligibility_not_diagnosis"}
        if not bundle.independent_event_ids:
            required.add("independence_unestablished")
        if bundle.omitted_count:
            required.add(f"omitted_assertions:{bundle.omitted_count}")
        if kind == "intervention_recommendation":
            required.add("not_executed_or_verified")
            if row["epistemic_status"] != "hypothesized" or row["verification_step"] is None:
                raise ValueError("recommendation_requires_hypothesis_and_verification")
        elif kind == "intervention_outcome":
            required.add("association_not_causation")
        if not required <= set(row["limitations"]):
            raise ValueError("missing_finding_limitations")
        content = {key: item for key, item in row.items() if key != "local_id"}
        for key in ("support_refs", "counter_refs", "limitations"):
            content[key] = sorted(content[key])
        finding_id = "diagnostic-finding:" + fingerprint(
            {"bundle": bundle.bundle_id, "receipt": receipt_ref, "content": content}
        )
        if finding_id in findings:
            raise ValueError("duplicate_finding_content")
        findings[finding_id] = DiagnosticFinding(
            finding_id,
            row["diagnostic_type"],
            row["subject_ref"],
            row["target_ref"],
            row["epistemic_status"],
            tuple(sorted(supports)),
            tuple(sorted(counters)),
            row["explanation"],
            tuple(sorted(row["limitations"])),
            row["verification_step"],
            receipt_ref,
        )
    return tuple(findings[ref] for ref in sorted(findings))
