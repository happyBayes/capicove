"""Bounded report waves and source-closed validation over trusted host inputs.

Inputs must be current, authorized snapshots. Opaque source IDs cannot establish
authenticity or authorization: Task 8 rechecks ``task.source_refs`` in its scope,
and Task 22 supplies the authorized input set. A bare EvidenceBundle has no
source content or scope and is deliberately insufficient here. Generated text
is usable only through validated child citations, never as independent evidence.

This module validates structure, not prose entailment or personality judgments.
The closed schema has no score, execution, or effectiveness fields. Task 8 owns
full-request token preflight and output-token enforcement using the pinned model
tokenizer; byte/codepoint limits here are additional bounds, not token estimates.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields, replace
from importlib.resources import files
from typing import Any

from capicove.contracts.graph_build import (
    ChildStatementRef,
    Community,
    CommunityReport,
    DiagnosticFinding,
    EvidenceBundle,
    GraphScope,
    RelationshipAssertion,
    ReportSections,
    ReportStatement,
    StageCoverage,
)
from capicove.graph_build.codec import canonical_bytes, fingerprint
from capicove.graph_build.validation import StructuralOutputError


_SECTIONS = tuple(item.name for item in fields(ReportSections))
_GENERATED = (
    "report:",
    "community-report:",
    "finding:",
    "diagnostic-finding:",
    "generated:",
    "generated_report:",
    "generated-report:",
)


def _ref(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 256
        or any(char.isspace() or ord(char) < 32 for char in value)
    ):
        raise ValueError("invalid_report_reference")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("invalid_report_unicode") from error
    return value


def _digest(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("invalid_report_digest")


def report_cache_key(
    community_id: str | None,
    bundle_digest: str,
    child_digests: tuple[str, ...],
    profile_digest: str,
) -> str:
    """Hash actual evidence, child content, and profile; scope enters bundle digest."""
    if community_id is not None:
        _ref(community_id)
    for digest in (bundle_digest, profile_digest, *child_digests):
        _digest(digest)
    if not isinstance(child_digests, tuple) or len(set(child_digests)) != len(child_digests):
        raise ValueError("invalid_child_digests")
    return fingerprint(
        {
            "community": community_id,
            "bundle": bundle_digest,
            "children": sorted(child_digests),
            "profile": profile_digest,
        }
    )


@dataclass(frozen=True, slots=True)
class ReportProfile:
    """Closed local selection/output limits, additionally bound by Task 8 policy."""

    max_assertions: int = 128
    max_findings: int = 32
    max_child_statements: int = 256
    max_response_bytes: int = 65536
    max_codepoints: int = 32768
    max_statements: int = 100
    max_statement_codepoints: int = 2000
    version: int = 1
    prompt_version: str = "report-v1"
    ontology_version: str = "personal-growth-v1"

    def __post_init__(self) -> None:
        bounds = {
            "max_assertions": 128,
            "max_findings": 32,
            "max_child_statements": 256,
            "max_response_bytes": 65536,
            "max_codepoints": 32768,
            "max_statements": 100,
            "max_statement_codepoints": 2000,
        }
        for name, maximum in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError("invalid_report_budget")
        if type(self.version) is not int or self.version != 1:
            raise ValueError("unsupported_report_profile")
        _ref(self.prompt_version)
        _ref(self.ontology_version)


@dataclass(frozen=True, slots=True)
class ReportEvidence:
    """Bind one immutable evidence bundle to its exact source assertion snapshot."""

    scope: GraphScope
    bundle: EvidenceBundle
    assertions: tuple[RelationshipAssertion, ...]
    source_kind: str = "source"

    def __post_init__(self) -> None:
        if not isinstance(self.scope, GraphScope) or not isinstance(self.bundle, EvidenceBundle):
            raise TypeError("report_evidence_requires_scope_and_bundle")
        rows = tuple(self.assertions)
        if any(not isinstance(row, RelationshipAssertion) for row in rows):
            raise TypeError("report_evidence_requires_assertions")
        refs = (*self.bundle.assertion_ids, *self.bundle.counter_assertion_ids)
        if (
            self.source_kind != "source"
            or len(refs) != len(set(refs))
            or len(rows) != len(refs)
            or {row.assertion_id for row in rows} != set(refs)
        ):
            raise ValueError("invalid_report_evidence_inventory")
        for ref in (
            self.bundle.bundle_id,
            self.bundle.subject_ref,
            self.bundle.target_ref,
            *self.bundle.independent_event_ids,
        ):
            _ref(ref)
        for row in rows:
            for ref in (row.assertion_id, row.support_group_id, *row.evidence_refs):
                _ref(ref)
            if (
                row.derivation_ref is not None
                or not row.evidence_refs
                or any(ref.startswith(_GENERATED) for ref in (row.assertion_id, *row.evidence_refs))
                or not set(row.counter_assertion_ids) <= set(refs)
            ):
                raise ValueError("generated_or_unclosed_report_evidence")
        object.__setattr__(
            self, "assertions", tuple(sorted(rows, key=lambda row: row.assertion_id))
        )
        object.__setattr__(
            self,
            "bundle",
            EvidenceBundle(
                self.bundle.bundle_id,
                self.bundle.subject_ref,
                self.bundle.target_ref,
                tuple(sorted(self.bundle.assertion_ids)),
                tuple(sorted(self.bundle.counter_assertion_ids)),
                tuple(sorted(self.bundle.independent_event_ids)),
                self.bundle.omitted_count,
            ),
        )


_DEFAULT_PROFILE = ReportProfile()


@dataclass(frozen=True, slots=True)
class FindingSource:
    """Source and uncertainty metadata for a cited finding, without its prose."""

    finding_id: str
    support_refs: tuple[str, ...]
    counter_refs: tuple[str, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        _ref(self.finding_id)
        for name in ("support_refs", "counter_refs", "limitations"):
            values = tuple(getattr(self, name))
            if len(set(values)) != len(values):
                raise ValueError("duplicate_finding_source_metadata")
            object.__setattr__(self, name, tuple(sorted(values)))
        if not self.support_refs:
            raise ValueError("finding_requires_original_sources")


@dataclass(frozen=True, slots=True)
class StatementPath:
    """Text-free citation edges retained through successive report levels."""

    ref: ChildStatementRef
    kind: str
    assertion_refs: tuple[str, ...]
    finding_refs: tuple[str, ...]
    child_statement_refs: tuple[ChildStatementRef, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.ref, ChildStatementRef) or self.kind not in {
            "fact",
            "hypothesis",
            "recommendation",
            "limitation",
        }:
            raise ValueError("invalid_statement_path")
        for name in ("assertion_refs", "finding_refs", "child_statement_refs"):
            values = tuple(getattr(self, name))
            if len(set(values)) != len(values):
                raise ValueError("duplicate_statement_path_reference")
            object.__setattr__(self, name, values)


def _path(report_id: str, statement: ReportStatement) -> StatementPath:
    return StatementPath(
        ChildStatementRef(report_id, statement.statement_id),
        statement.kind,
        statement.assertion_refs,
        statement.finding_refs,
        statement.child_statement_refs,
    )


@dataclass(frozen=True, slots=True)
class ChildCitation:
    """Only selected prose, with a self-validating transitive source proof.

    Ancestor path records contain references and kinds, never ancestor prose.
    Closure fields are derived and cannot be supplied or independently replaced.
    Source authenticity is a host precondition; report_tasks revalidates every
    completed task against current evidence before constructing these records.
    """

    ref: ChildStatementRef
    statement: ReportStatement
    section: str
    scope: GraphScope
    paths: tuple[StatementPath, ...]
    assertions: tuple[RelationshipAssertion, ...]
    finding_sources: tuple[FindingSource, ...]
    counter_assertion_ids: tuple[str, ...]
    structural_limitations: tuple[str, ...]
    kind: str = field(init=False)
    assertion_ids: tuple[str, ...] = field(init=False)
    finding_ids: tuple[str, ...] = field(init=False)
    source_refs: tuple[str, ...] = field(init=False)
    required_limitations: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.ref, ChildStatementRef) or not isinstance(self.scope, GraphScope):
            raise TypeError("child_citation_requires_namespaced_ref")
        if self.section not in _SECTIONS or self.ref.statement_id != self.statement.statement_id:
            raise ValueError("invalid_selected_child_statement")
        if self.statement.statement_id != "statement:" + fingerprint(
            {key: value for key, value in asdict(self.statement).items() if key != "statement_id"}
        ):
            raise ValueError("child_statement_content_mismatch")
        for name in (
            "paths",
            "assertions",
            "finding_sources",
            "counter_assertion_ids",
            "structural_limitations",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        paths = {item.ref: item for item in self.paths}
        findings = {item.finding_id: item for item in self.finding_sources}
        assertions = {item.assertion_id: item for item in self.assertions}
        if (
            len(paths) != len(self.paths)
            or len(findings) != len(self.finding_sources)
            or len(assertions) != len(self.assertions)
            or paths.get(self.ref) != _path(self.ref.report_id, self.statement)
        ):
            raise ValueError("invalid_child_statement_proof")
        # Iterative topological traversal rejects cycles and foreign/dangling refs.
        pending = [(self.ref, False)]
        visiting: set[ChildStatementRef] = set()
        visited: set[ChildStatementRef] = set()
        source_ids: set[str] = set()
        finding_ids: set[str] = set()
        while pending:
            ref, exiting = pending.pop()
            if exiting:
                visiting.remove(ref)
                visited.add(ref)
                continue
            if ref in visiting or ref not in paths:
                raise ValueError("cyclic_or_foreign_child_path")
            if ref in visited:
                continue
            node = paths[ref]
            visiting.add(ref)
            source_ids.update(node.assertion_refs)
            finding_ids.update(node.finding_refs)
            pending.append((ref, True))
            pending.extend((child, False) for child in node.child_statement_refs)
        if visited != paths.keys() or finding_ids != findings.keys():
            raise ValueError("extraneous_child_path")
        for item in findings.values():
            source_ids.update(item.support_refs + item.counter_refs)
        if source_ids != assertions.keys() or not source_ids:
            raise ValueError("child_requires_exact_original_source_closure")
        for node in paths.values():
            if node.kind == "fact" and (
                node.finding_refs
                or any(paths[ref].kind != "fact" for ref in node.child_statement_refs)
                or any(
                    assertions[ref].epistemic_status != "explicit" for ref in node.assertion_refs
                )
            ):
                raise ValueError("child_fact_has_inferred_sources")
        for row in assertions.values():
            if (
                row.derivation_ref is not None
                or not row.evidence_refs
                or any(ref.startswith(_GENERATED) for ref in (row.assertion_id, *row.evidence_refs))
            ):
                raise ValueError("generated_child_source")
        if not set(self.counter_assertion_ids) <= source_ids:
            raise ValueError("foreign_child_counter")
        required = set(self.structural_limitations)
        required.update(limit for item in findings.values() for limit in item.limitations)
        if self.statement.kind == "limitation":
            required.add(self.statement.text)
        if any(item.kind == "recommendation" for item in paths.values()):
            required.add("not_executed_or_verified")
        object.__setattr__(self, "kind", self.statement.kind)
        object.__setattr__(self, "assertion_ids", tuple(sorted(source_ids)))
        object.__setattr__(self, "finding_ids", tuple(sorted(finding_ids)))
        object.__setattr__(
            self,
            "source_refs",
            tuple(sorted({ref for row in self.assertions for ref in row.evidence_refs})),
        )
        object.__setattr__(self, "required_limitations", tuple(sorted(required)))


@dataclass(frozen=True, slots=True)
class ChildStatementBinding:
    """Commit a host-validated selection proof without retaining its text."""

    statement_id: str
    proof_digest: str

    def __post_init__(self) -> None:
        _ref(self.statement_id)
        _digest(self.proof_digest)


@dataclass(frozen=True, slots=True)
class ChildReportIdentity:
    """Complete terminal identity and coverage; contains no report prose."""

    community_id: str
    report_id: str
    task_digest: str
    content_digest: str
    status: str
    coverage: StageCoverage
    statement_count: int
    statement_bindings: tuple[ChildStatementBinding, ...]

    def __post_init__(self) -> None:
        _ref(self.community_id)
        _ref(self.report_id)
        _digest(self.task_digest)
        _digest(self.content_digest)
        if (
            self.status not in {"succeeded", "partial"}
            or type(self.statement_count) is not int
            or self.statement_count < 0
        ):
            raise ValueError("invalid_terminal_report_identity")
        bindings = tuple(self.statement_bindings)
        if (
            len({item.statement_id for item in bindings}) != len(bindings)
            or len(bindings) > self.statement_count
        ):
            raise ValueError("invalid_child_statement_bindings")
        object.__setattr__(
            self, "statement_bindings", tuple(sorted(bindings, key=lambda item: item.statement_id))
        )


@dataclass(frozen=True, slots=True)
class ReportTask:
    """Frozen host selection; construct via report_tasks, never from model JSON.

    Coverage counts direct child communities. Assertion/finding/statement
    omissions have separate counts so unlike populations are never added.
    Source-free tasks explicitly disable generation and are materialized by
    materialize_inventory_report; they never enter the inference runner.
    """

    community_id: str | None
    scope: GraphScope | None
    member_ids: tuple[str, ...]
    bundle: EvidenceBundle
    evidence: tuple[ReportEvidence, ...]
    findings: tuple[DiagnosticFinding, ...]
    child_report_identities: tuple[ChildReportIdentity, ...]
    child_citations: tuple[ChildCitation, ...]
    source_assertions: tuple[RelationshipAssertion, ...]
    coverage: StageCoverage
    failed_community_ids: tuple[str, ...]
    omitted_count: int
    omitted_findings: int
    omitted_child_statements: int
    required_limitations: tuple[str, ...]
    profile: ReportProfile
    input_digest: str
    report_kind: str = field(init=False)
    profile_fingerprint: str = field(init=False)
    cache_key: str = field(init=False)
    allowed_assertion_ids: tuple[str, ...] = field(init=False)
    allowed_finding_ids: tuple[str, ...] = field(init=False)
    allowed_child_statement_refs: tuple[ChildStatementRef, ...] = field(init=False)
    source_refs: tuple[str, ...] = field(init=False)
    requires_generation: bool = field(init=False)

    def __post_init__(self) -> None:
        _digest(self.input_digest)
        if not isinstance(self.profile, ReportProfile):
            raise TypeError("invalid_report_profile")
        if self.community_id is not None:
            _ref(self.community_id)
        for name in (
            "member_ids",
            "evidence",
            "findings",
            "child_report_identities",
            "child_citations",
            "source_assertions",
            "failed_community_ids",
            "required_limitations",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if any(item.scope != self.scope for item in self.evidence):
            raise ValueError("mixed_report_scopes")
        for count in (self.omitted_count, self.omitted_findings, self.omitted_child_statements):
            if type(count) is not int or count < 0:
                raise ValueError("invalid_report_omissions")
        assertion_ids = tuple(
            sorted(set(self.bundle.assertion_ids + self.bundle.counter_assertion_ids))
        )
        selected_ids = {row.assertion_id for item in self.evidence for row in item.assertions}
        selected_counters = {
            ref for item in self.evidence for ref in item.bundle.counter_assertion_ids
        }
        if (
            set(self.bundle.assertion_ids) != selected_ids - selected_counters
            or set(self.bundle.counter_assertion_ids) != selected_counters
        ):
            raise ValueError("report_task_selection_mismatch")
        expected_sources: dict[str, RelationshipAssertion] = {}
        for row in tuple(row for item in self.evidence for row in item.assertions) + tuple(
            row for item in self.child_citations for row in item.assertions
        ):
            if row.assertion_id in expected_sources and expected_sources[row.assertion_id] != row:
                raise ValueError("conflicting_report_source_snapshot")
            expected_sources[row.assertion_id] = row
        source_ids = {row.assertion_id for row in self.source_assertions}
        if (
            {row.assertion_id: row for row in self.source_assertions} != expected_sources
            or len(source_ids) != len(self.source_assertions)
            or self.bundle.omitted_count != self.omitted_count
        ):
            raise ValueError("invalid_report_task_sources")
        child_ids = {item.report_id for item in self.child_report_identities}
        if (
            len(child_ids) != len(self.child_report_identities)
            or len(self.child_citations) > self.profile.max_child_statements
            or len({item.ref for item in self.child_citations}) != len(self.child_citations)
        ):
            raise ValueError("invalid_child_selection_bound")
        for citation in self.child_citations:
            if citation.ref.report_id not in child_ids or citation.scope != self.scope:
                raise ValueError("invalid_child_citation_closure")
            terminal = next(
                item
                for item in self.child_report_identities
                if item.report_id == citation.ref.report_id
            )
            bindings = {
                item.statement_id: item.proof_digest for item in terminal.statement_bindings
            }
            if bindings.get(citation.ref.statement_id) != fingerprint(asdict(citation)):
                raise ValueError("child_selection_changed_since_validation")
        expected_limits = _structural_limits(self)
        expected_limits.update(limit for item in self.findings for limit in item.limitations)
        expected_limits.update(
            limit for item in self.child_citations for limit in item.required_limitations
        )
        if set(self.required_limitations) != expected_limits:
            raise ValueError("invalid_report_task_limitations")
        object.__setattr__(
            self, "report_kind", "overview" if self.community_id is None else "community"
        )
        object.__setattr__(self, "allowed_assertion_ids", assertion_ids)
        object.__setattr__(
            self, "allowed_finding_ids", tuple(item.finding_id for item in self.findings)
        )
        object.__setattr__(
            self, "allowed_child_statement_refs", tuple(item.ref for item in self.child_citations)
        )
        object.__setattr__(
            self,
            "source_refs",
            tuple(sorted({ref for row in self.source_assertions for ref in row.evidence_refs})),
        )
        object.__setattr__(self, "requires_generation", bool(self.source_refs))
        if not self.source_refs and (self.findings or self.child_citations):
            raise ValueError("provider_text_without_sources")
        object.__setattr__(self, "profile_fingerprint", fingerprint(asdict(self.profile)))
        content = {item.name: getattr(self, item.name) for item in fields(self) if item.init}
        # asdict on this closed value recursively freezes exact evidence, findings,
        # coverage, all selected/omitted inputs and the actual child content.
        content = {
            key: asdict(value) if hasattr(value, "__dataclass_fields__") else value
            for key, value in content.items()
        }
        bundle_digest = fingerprint(_json_value(content))
        object.__setattr__(
            self,
            "cache_key",
            report_cache_key(
                self.community_id,
                bundle_digest,
                tuple(fingerprint(asdict(item)) for item in self.child_report_identities),
                self.profile_fingerprint,
            ),
        )


def _structural_limits(task: ReportTask) -> set[str]:
    limits: set[str] = set()
    if (
        not task.bundle.assertion_ids
        and not task.bundle.counter_assertion_ids
        and not task.child_citations
    ):
        limits.add("insufficient_evidence")
    if task.failed_community_ids or any(
        item.status == "partial" for item in task.child_report_identities
    ):
        limits.add("partial_child_coverage")
    for prefix, count in (
        ("omitted_assertions", task.omitted_count),
        ("omitted_findings", task.omitted_findings),
        ("omitted_child_statements", task.omitted_child_statements),
    ):
        if count:
            limits.add(f"{prefix}:{count}")
    return limits


def _json_value(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class CompletedReport:
    """A validated report bound to its original immutable task, including sources."""

    report: CommunityReport
    task: ReportTask

    def __post_init__(self) -> None:
        _check_completed(self.report, self.task)


def _hierarchy(communities: tuple[Community, ...]) -> dict[str | None, tuple[str, ...]]:
    rows = {row.community_id: row for row in communities}
    if len(rows) != len(communities):
        raise ValueError("duplicate_community")
    children: dict[str | None, list[str]] = {None: []}
    for row in communities:
        _ref(row.community_id)
        children.setdefault(row.community_id, [])
        if row.parent_id is not None:
            parent = rows.get(row.parent_id)
            if (
                parent is None
                or row.depth != parent.depth + 1
                or not set(row.member_ids) < set(parent.member_ids)
            ):
                raise ValueError("invalid_community_hierarchy")
        children.setdefault(row.parent_id, []).append(row.community_id)
    for parent_id, child_ids in children.items():
        members: set[str] = set()
        for ref in child_ids:
            child_members = set(rows[ref].member_ids)
            if child_members & members:
                raise ValueError("overlapping_communities")
            members.update(child_members)
        if parent_id is not None and child_ids and members != set(rows[parent_id].member_ids):
            raise ValueError("incomplete_community_partition")
    return {key: tuple(sorted(value)) for key, value in children.items()}


def _statements(report: CommunityReport) -> tuple[ReportStatement, ...]:
    return tuple(item for name in _SECTIONS for item in getattr(report.sections, name))


def _inputs(
    bundles: tuple[ReportEvidence, ...],
    findings: tuple[DiagnosticFinding, ...],
) -> tuple[GraphScope | None, dict[str, RelationshipAssertion]]:
    if any(not isinstance(item, ReportEvidence) for item in bundles):
        raise TypeError("bundles_require_report_evidence")
    if any(not isinstance(item, DiagnosticFinding) for item in findings):
        raise TypeError("findings_require_diagnostic_findings")
    scopes = {item.scope for item in bundles}
    if len(scopes) > 1:
        raise ValueError("mixed_report_scopes")
    if len({item.bundle.bundle_id for item in bundles}) != len(bundles) or len(
        {item.finding_id for item in findings}
    ) != len(findings):
        raise ValueError("duplicate_report_input")
    assertions: dict[str, RelationshipAssertion] = {}
    for item in bundles:
        for row in item.assertions:
            if row.assertion_id in assertions and assertions[row.assertion_id] != row:
                raise ValueError("conflicting_assertion_snapshot")
            assertions[row.assertion_id] = row
    for finding in findings:
        _ref(finding.finding_id)
        if finding.finding_id in assertions or not any(
            item.bundle.target_ref == finding.target_ref
            and item.bundle.subject_ref == finding.subject_ref
            and set(finding.support_refs) <= set(item.bundle.assertion_ids)
            and set(finding.counter_refs) == set(item.bundle.counter_assertion_ids)
            for item in bundles
        ):
            raise ValueError("finding_outside_report_evidence")
    return next(iter(scopes), None), assertions


def _task(
    community: Community | None,
    scope: GraphScope | None,
    candidates: tuple[ReportEvidence, ...],
    findings: tuple[DiagnosticFinding, ...],
    child_ids: tuple[str, ...],
    completed: dict[str, CompletedReport],
    failed: frozenset[str],
    profile: ReportProfile,
    assertions: dict[str, RelationshipAssertion],
) -> ReportTask:
    members = set(community.member_ids) if community is not None else set()
    # Target association wins over a ubiquitous subject/owner member.
    relevant = tuple(
        item for item in candidates if community is None or item.bundle.target_ref in members
    )
    relevant_findings = tuple(
        item for item in findings if community is None or item.target_ref in members
    )
    selected: list[ReportEvidence] = []
    selected_ids: set[str] = set()
    seen_groups: set[str] = set()
    remaining = list(relevant)
    while remaining:
        remaining.sort(
            key=lambda item: (
                not bool(item.bundle.counter_assertion_ids),
                not bool({row.support_group_id for row in item.assertions} - seen_groups),
                -len(item.bundle.independent_event_ids),
                item.bundle.target_ref,
                item.bundle.bundle_id,
            )
        )
        item = remaining.pop(0)
        refs = {row.assertion_id for row in item.assertions}
        if len(selected_ids | refs) <= profile.max_assertions:
            selected.append(item)
            selected_ids.update(refs)
            seen_groups.update(row.support_group_id for row in item.assertions)
    selected.sort(key=lambda item: item.bundle.bundle_id)
    selected_findings = tuple(
        item
        for item in relevant_findings
        if set(item.support_refs + item.counter_refs) <= selected_ids
    )[: profile.max_findings]
    omitted_findings = len(relevant_findings) - len(selected_findings)
    counter_ids = {ref for item in selected for ref in item.bundle.counter_assertion_ids}
    omitted = len({row.assertion_id for item in relevant for row in item.assertions} - selected_ids)
    omitted += sum(item.bundle.omitted_count for item in relevant)
    done = tuple(completed[ref] for ref in child_ids if ref in completed)
    citations: list[ChildCitation] = []
    for completed_child in done:
        for section in _SECTIONS:
            for row in getattr(completed_child.report.sections, section):
                # Uncited generated limitations are coverage metadata only. They
                # are never supplied as fresh child prose or independent evidence.
                if _closure(row, completed_child.task)[0]:
                    citations.append(_child_citation(completed_child, row, section))
    child_identities = tuple(
        ChildReportIdentity(
            item.report.community_id or "overview",
            item.report.report_id,
            item.task.cache_key,
            fingerprint(asdict(item.report)),
            "partial"
            if item.task.failed_community_ids
            or "partial_child_coverage" in item.task.required_limitations
            else "succeeded",
            item.task.coverage,
            len(_statements(item.report)),
            tuple(
                ChildStatementBinding(citation.ref.statement_id, fingerprint(asdict(citation)))
                for citation in citations
                if citation.ref.report_id == item.report.report_id
            ),
        )
        for item in done
    )
    citations.sort(key=lambda item: (item.ref.report_id, item.ref.statement_id))
    # Counter-bearing child statements get reserved space before other prose.
    child_counters = {ref for item in done for ref in item.task.bundle.counter_assertion_ids}
    citations.sort(
        key=lambda item: (
            not bool(set(item.assertion_ids) & child_counters),
            item.ref.report_id,
            item.ref.statement_id,
        )
    )
    citations = citations[: profile.max_child_statements]
    omitted_children = sum(item.statement_count for item in child_identities) - len(citations)
    citations.sort(key=lambda item: (item.ref.report_id, item.ref.statement_id))
    failures = tuple(ref for ref in child_ids if ref in failed)
    limits: set[str] = set()
    limits.update(limit for finding in selected_findings for limit in finding.limitations)
    limits.update(limit for citation in citations for limit in citation.required_limitations)
    if not selected_ids and not any(item.assertion_ids for item in citations):
        limits.add("insufficient_evidence")
    if failures or any("partial_child_coverage" in item.task.required_limitations for item in done):
        limits.add("partial_child_coverage")
    if omitted:
        limits.add(f"omitted_assertions:{omitted}")
    if omitted_findings:
        limits.add(f"omitted_findings:{omitted_findings}")
    if omitted_children:
        limits.add(f"omitted_child_statements:{omitted_children}")
    closure_ids = selected_ids | {ref for item in citations for ref in item.assertion_ids}
    input_digest = fingerprint(
        {
            "community": asdict(community) if community else None,
            "evidence": [asdict(item) for item in relevant],
            "findings": [asdict(item) for item in relevant_findings],
            "failed": failures,
            "children": [(item.report.report_id, item.task.cache_key) for item in done],
        }
    )
    bundle = EvidenceBundle(
        "report-bundle:" + input_digest,
        "report-selection",
        community.community_id if community else "overview",
        tuple(sorted(selected_ids - counter_ids)),
        tuple(sorted(counter_ids)),
        tuple(sorted({ref for item in selected for ref in item.bundle.independent_event_ids})),
        omitted,
    )
    return ReportTask(
        community.community_id if community else None,
        scope,
        tuple(sorted(members)),
        bundle,
        tuple(selected),
        selected_findings,
        child_identities,
        tuple(citations),
        tuple(assertions[ref] for ref in sorted(closure_ids)),
        StageCoverage(
            "child_communities", len(child_ids), len(done), len(done), 0, len(failures), 0
        ),
        failures,
        omitted,
        omitted_findings,
        omitted_children,
        tuple(sorted(limits)),
        profile,
        input_digest,
    )


def _child_citation(
    completed: CompletedReport, row: ReportStatement, section: str
) -> ChildCitation:
    task = completed.task
    if task.scope is None:
        raise ValueError("sourced_child_requires_scope")
    refs, finding_ids, _ = _closure(row, task)
    root = _path(completed.report.report_id, row)
    paths = {root.ref: root}
    findings = {
        item.finding_id: FindingSource(
            item.finding_id, item.support_refs, item.counter_refs, item.limitations
        )
        for item in task.findings
        if item.finding_id in row.finding_refs
    }
    structural = _structural_limits(task)
    counters = set(task.bundle.counter_assertion_ids) & refs
    for child in task.child_citations:
        if child.ref in row.child_statement_refs:
            for path in child.paths:
                if path.ref in paths and paths[path.ref] != path:
                    raise ValueError("conflicting_transitive_child_path")
                paths[path.ref] = path
            for finding in child.finding_sources:
                if finding.finding_id in findings and findings[finding.finding_id] != finding:
                    raise ValueError("conflicting_transitive_finding_source")
                findings[finding.finding_id] = finding
            structural.update(child.required_limitations)
            counters.update(child.counter_assertion_ids)
    if section == "counter_evidence":
        counters.update(refs)
    if set(findings) != finding_ids:
        raise ValueError("missing_transitive_finding_source")
    return ChildCitation(
        root.ref,
        row,
        section,
        task.scope,
        tuple(
            paths[ref]
            for ref in sorted(paths, key=lambda item: (item.report_id, item.statement_id))
        ),
        tuple(item for item in task.source_assertions if item.assertion_id in refs),
        tuple(findings[ref] for ref in sorted(findings)),
        tuple(sorted(counters)),
        tuple(sorted(structural)),
    )


def report_tasks(
    communities: tuple[Community, ...],
    bundles: tuple[ReportEvidence, ...],
    findings: tuple[DiagnosticFinding, ...],
    *,
    completed_reports: tuple[CompletedReport, ...] = (),
    failed_community_ids: tuple[str, ...] = (),
    profile: ReportProfile = _DEFAULT_PROFILE,
) -> tuple[ReportTask, ...]:
    """Return the next deterministic leaves-before-parents wave, then overview.

    Missing children are pending. Explicit failures permit a partial parent;
    remove their failure marker to retry. Supply all completed descendants on
    subsequent calls: each original task is rebuilt against current inputs and
    compared before its report is reused. Stale reports fail rather than silently
    surviving changed evidence. Tasks with ``requires_generation`` need their own
    Task 8 receipt; source-free tasks use :func:`materialize_inventory_report`.
    """
    communities, bundles, findings = tuple(communities), tuple(bundles), tuple(findings)
    if any(not isinstance(row, Community) for row in communities):
        raise TypeError("communities_required")
    communities = tuple(
        replace(
            row,
            member_ids=tuple(sorted(row.member_ids)),
            issue_codes=tuple(sorted(row.issue_codes)),
        )
        for row in communities
    )
    children = _hierarchy(communities)
    scope, assertions = _inputs(bundles, findings)
    bundles = tuple(sorted(bundles, key=lambda item: item.bundle.bundle_id))
    findings = tuple(
        replace(
            item,
            support_refs=tuple(sorted(item.support_refs)),
            counter_refs=tuple(sorted(item.counter_refs)),
            limitations=tuple(sorted(item.limitations)),
        )
        for item in sorted(findings, key=lambda item: item.finding_id)
    )
    by_id = {row.community_id: row for row in communities}
    failed = frozenset(failed_community_ids)
    if len(failed) != len(failed_community_ids) or not failed <= by_id.keys():
        raise ValueError("invalid_failed_communities")
    completed: dict[str, CompletedReport] = {}
    overview: CompletedReport | None = None
    for item in completed_reports:
        if not isinstance(item, CompletedReport):
            raise TypeError("completed_report_requires_task_binding")
        ref = item.report.community_id
        if ref is None:
            if overview is not None:
                raise ValueError("duplicate_overview")
            overview = item
        elif ref not in by_id or ref in completed or ref in failed:
            raise ValueError("foreign_or_conflicting_completed_report")
        else:
            completed[ref] = item
    ready = []
    terminal = set(completed) | failed
    ordered = sorted(communities, key=lambda row: (-row.depth, row.community_id))
    for community in (*ordered, None):
        ref = community.community_id if community else None
        existing = completed.get(ref) if ref is not None else overview
        if not set(children[ref]) <= terminal:
            if existing is not None:
                raise ValueError("completed_parent_has_pending_children")
            continue
        if ref in failed:
            continue
        task = _task(
            community,
            scope,
            bundles,
            findings,
            children[ref],
            completed,
            failed,
            profile,
            assertions,
        )
        if existing is not None:
            if existing.task != task:
                raise ValueError("stale_completed_report_task")
            _check_completed(existing.report, task)
        else:
            ready.append(task)
    return tuple(ready)


def _closure(row: ReportStatement, task: ReportTask) -> tuple[set[str], set[str], set[str]]:
    refs, finding_ids = set(row.assertion_refs), set(row.finding_refs)
    findings = {item.finding_id: item for item in task.findings}
    children = {item.ref: item for item in task.child_citations}
    for ref in row.finding_refs:
        finding = findings[ref]
        refs.update(finding.support_refs + finding.counter_refs)
    for child_ref in row.child_statement_refs:
        child = children[child_ref]
        refs.update(child.assertion_ids)
        finding_ids.update(child.finding_ids)
    assertions = {item.assertion_id: item for item in task.source_assertions}
    return refs, finding_ids, {source for ref in refs for source in assertions[ref].evidence_refs}


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_report_json_key")
        result[key] = value
    return result


def _schema(value: Any, schema: dict[str, Any], definitions: dict[str, Any]) -> None:
    if "$ref" in schema:
        _schema(value, definitions[schema["$ref"].split("/")[-1]], definitions)
        return
    expected = schema["type"]
    if isinstance(expected, list):
        if value is None:
            return
        expected = "string"
    valid = {
        "object": type(value) is dict,
        "array": type(value) is list,
        "string": type(value) is str,
        "integer": type(value) is int,
    }
    if not valid[expected]:
        raise ValueError("invalid_report_schema")
    if ("const" in schema and value != schema["const"]) or (
        "enum" in schema and value not in schema["enum"]
    ):
        raise ValueError("invalid_report_schema")
    if expected == "object":
        if set(value) != set(schema["required"]):
            raise ValueError("invalid_report_fields")
        for key, item in value.items():
            _schema(item, schema["properties"][key], definitions)
    elif expected == "array":
        if len(value) > schema["maxItems"]:
            raise ValueError("report_array_limit")
        for item in value:
            _schema(item, schema["items"], definitions)
        if schema.get("uniqueItems") and len({canonical_bytes(item) for item in value}) != len(
            value
        ):
            raise ValueError("duplicate_report_reference")
    elif expected == "string":
        if not value.strip() or not 1 <= len(value) <= schema["maxLength"]:
            raise ValueError("report_text_limit")
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("invalid_report_unicode") from error


def validate_report(payload: bytes, task: ReportTask, receipt_ref: str) -> CommunityReport:
    """Validate exact allowlists and kinds and bind content to task and receipt."""
    if not isinstance(task, ReportTask):
        raise TypeError("report_requires_task")
    if not task.requires_generation:
        raise ValueError("source_free_task_requires_inventory_materialization")
    return _validate_report(payload, task, receipt_ref, check_wire_bounds=True)


def materialize_inventory_report(task: ReportTask) -> CommunityReport:
    """Materialize a source-free insufficiency result without Task 8 or a model.

    The deterministic receipt identifies this local rule, not a provider call or
    authorization decision. Output identity binds that receipt, the exact task,
    report kind and generated limitations; the result remains derived content.
    """
    if not isinstance(task, ReportTask) or task.requires_generation:
        raise ValueError("inventory_materialization_requires_source_free_task")
    statements = [
        {
            "local_id": f"notice:{i}",
            "text": limit,
            "kind": "limitation",
            "assertion_refs": [],
            "finding_refs": [],
            "child_statement_refs": [],
        }
        for i, limit in enumerate(task.required_limitations)
    ]
    payload = canonical_bytes(
        {
            "version": 1,
            "task_key": task.cache_key,
            "report_kind": task.report_kind,
            "community_id": task.community_id,
            "title": "Insufficient source evidence",
            "sections": {name: statements if name == "limitations" else [] for name in _SECTIONS},
        }
    )
    return _validate_report(
        payload, task, "report-inventory-receipt:" + task.cache_key, check_wire_bounds=False
    )


def _validate_report(
    payload: bytes, task: ReportTask, receipt_ref: str, *, check_wire_bounds: bool
) -> CommunityReport:
    if not isinstance(task, ReportTask) or not isinstance(payload, bytes):
        raise TypeError("report_requires_bytes_and_task")
    _ref(receipt_ref)
    if receipt_ref in {
        task.cache_key,
        task.bundle.bundle_id,
        task.community_id,
        *task.allowed_assertion_ids,
        *task.allowed_finding_ids,
        *task.source_refs,
        *(item.report_id for item in task.child_report_identities),
    }:
        raise ValueError("invalid_report_receipt_binding")
    if check_wire_bounds and len(payload) > task.profile.max_response_bytes:
        raise ValueError("report_byte_limit")
    try:
        text = payload.decode("utf-8")
    except UnicodeError as error:
        raise StructuralOutputError("invalid_report_json") from error
    if check_wire_bounds and len(text) > task.profile.max_codepoints:
        raise ValueError("report_codepoint_limit")
    try:
        value = json.loads(text, object_pairs_hook=_pairs)
    except (ValueError, RecursionError) as error:
        raise StructuralOutputError("invalid_report_json") from error
    schema = json.loads(
        files("capicove.graph_build").joinpath("schemas/report-v1.json").read_bytes()
    )
    try:
        _schema(value, schema, schema["$defs"])
    except ValueError as error:
        raise StructuralOutputError(str(error)) from error
    if (
        value["task_key"] != task.cache_key
        or value["community_id"] != task.community_id
        or value["report_kind"] != task.report_kind
    ):
        raise ValueError("report_task_mismatch")
    sections: dict[str, tuple[ReportStatement, ...]] = {}
    local_ids: set[str] = set()
    content_ids: set[str] = set()
    child_by_ref = {item.ref: item for item in task.child_citations}
    assertions = {item.assertion_id: item for item in task.source_assertions}
    for name in _SECTIONS:
        statements = []
        for row in value["sections"][name]:
            _ref(row["local_id"])
            if row["local_id"] in local_ids or row["local_id"] in {
                receipt_ref,
                task.bundle.bundle_id,
                task.cache_key,
                *task.allowed_assertion_ids,
                *task.allowed_finding_ids,
                *(item.report_id for item in task.child_report_identities),
            }:
                raise ValueError("duplicate_or_foreign_statement_id")
            local_ids.add(row["local_id"])
            if (
                len(local_ids) > task.profile.max_statements
                or len(row["text"]) > task.profile.max_statement_codepoints
            ):
                raise ValueError("report_statement_limit")
            refs = tuple(ChildStatementRef(**item) for item in row["child_statement_refs"])
            if (
                not set(row["assertion_refs"]) <= set(task.allowed_assertion_ids)
                or not set(row["finding_refs"]) <= set(task.allowed_finding_ids)
                or not set(refs) <= child_by_ref.keys()
            ):
                raise ValueError("report_citation_outside_task")
            kind = row["kind"]
            if kind == "fact" and (
                row["finding_refs"]
                or any(child_by_ref[ref].kind != "fact" for ref in refs)
                or any(
                    assertions[ref].epistemic_status != "explicit" for ref in row["assertion_refs"]
                )
            ):
                raise ValueError("inference_cannot_become_fact")
            if (
                (name == "suggested_next_steps" and kind != "recommendation")
                or (kind == "recommendation" and name != "suggested_next_steps")
                or (name == "possible_causes" and kind not in {"hypothesis", "limitation"})
                or (name == "limitations" and kind != "limitation")
                or (name in {"existing_interventions", "observed_outcomes"} and kind != "fact")
            ):
                raise ValueError("report_section_kind_mismatch")
            content = {key: item for key, item in row.items() if key != "local_id"}
            content["assertion_refs"] = sorted(row["assertion_refs"])
            content["finding_refs"] = sorted(row["finding_refs"])
            content["child_statement_refs"] = [
                asdict(ref)
                for ref in sorted(refs, key=lambda item: (item.report_id, item.statement_id))
            ]
            statement_id = "statement:" + fingerprint(content)
            if statement_id in content_ids:
                raise ValueError("duplicate_report_statement_content")
            content_ids.add(statement_id)
            statement = ReportStatement(
                statement_id,
                row["text"],
                kind,
                tuple(content["assertion_refs"]),
                tuple(content["finding_refs"]),
                tuple(ChildStatementRef(**item) for item in content["child_statement_refs"]),
            )
            if kind == "recommendation" and not any(_closure(statement, task)):
                raise ValueError("unsupported_recommendation")
            statements.append(statement)
        sections[name] = tuple(statements)
    limitations = {item.text for item in sections["limitations"]}
    required = set(task.required_limitations)
    if sections["suggested_next_steps"]:
        required.add("not_executed_or_verified")
    if not required <= limitations:
        raise ValueError("missing_report_limitations")
    counters = {ref for row in sections["counter_evidence"] for ref in _closure(row, task)[0]}
    required_counters = set(task.bundle.counter_assertion_ids)
    for child in task.child_citations:
        required_counters.update(child.counter_assertion_ids)
    if not required_counters <= counters:
        raise ValueError("missing_report_counter_evidence")
    report_sections = ReportSections(**sections)
    report_id = "community-report:" + fingerprint(
        {
            "task": task.cache_key,
            "receipt": receipt_ref,
            "title": value["title"],
            "sections": asdict(report_sections),
        }
    )
    return CommunityReport(
        report_id,
        task.report_kind,
        task.community_id,
        value["title"],
        report_sections,
        task.bundle.bundle_id,
        task.omitted_count,
        receipt_ref,
    )


def _check_completed(report: CommunityReport, task: ReportTask) -> None:
    if not isinstance(report, CommunityReport) or not isinstance(task, ReportTask):
        raise TypeError("completed_report_requires_report_and_task")
    if not task.requires_generation:
        if report != materialize_inventory_report(task):
            raise ValueError("invalid_source_free_report_lineage")
        return
    sections = {
        name: [
            {
                "local_id": row.statement_id,
                **{key: value for key, value in asdict(row).items() if key != "statement_id"},
            }
            for row in getattr(report.sections, name)
        ]
        for name in _SECTIONS
    }
    payload = canonical_bytes(
        {
            "version": 1,
            "task_key": task.cache_key,
            "report_kind": report.report_kind,
            "community_id": report.community_id,
            "title": report.title,
            "sections": sections,
        }
    )
    # Host content IDs are longer than provider-local IDs. Recheck all content,
    # count, citation and hash invariants without treating our reconstructed
    # envelope as the original bounded provider response.
    if _validate_report(payload, task, report.receipt_ref, check_wire_bounds=False) != report:
        raise ValueError("invalid_completed_report_binding")
