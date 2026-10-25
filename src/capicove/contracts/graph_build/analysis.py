"""Immutable analysis records with explicit synthesis and citation boundaries."""

from dataclasses import dataclass, fields

from capicove.contracts.graph_build._checks import choice, refs, text, values
from capicove.contracts.graph_build.evidence import _integer
from capicove.contracts.graph_build.vocabulary import DIAGNOSTICS


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """Bound selected source assertions without treating repeats as support."""

    bundle_id: str
    subject_ref: str
    target_ref: str
    assertion_ids: tuple[str, ...]
    counter_assertion_ids: tuple[str, ...]
    independent_event_ids: tuple[str, ...]
    omitted_count: int

    def __post_init__(self) -> None:
        for value in (self.bundle_id, self.subject_ref, self.target_ref):
            text(value)
        for name in ("assertion_ids", "counter_assertion_ids", "independent_event_ids"):
            object.__setattr__(self, name, refs(getattr(self, name)))
        _integer("omitted_count", self.omitted_count)


@dataclass(frozen=True, slots=True)
class DiagnosticFinding:
    """A generated inference, never an explicit source fact or medical diagnosis."""

    finding_id: str
    diagnostic_type: str
    subject_ref: str
    target_ref: str
    epistemic_status: str
    support_refs: tuple[str, ...]
    counter_refs: tuple[str, ...]
    explanation: str
    limitations: tuple[str, ...]
    verification_step: str | None
    derivation_ref: str

    def __post_init__(self) -> None:
        for value in (
            self.finding_id,
            self.subject_ref,
            self.target_ref,
            self.explanation,
            self.derivation_ref,
        ):
            text(value)
        choice(self.diagnostic_type, DIAGNOSTICS)
        choice(self.epistemic_status, frozenset({"inferred", "hypothesized"}))
        object.__setattr__(self, "support_refs", refs(self.support_refs, required=True))
        object.__setattr__(self, "counter_refs", refs(self.counter_refs))
        object.__setattr__(self, "limitations", refs(self.limitations))
        text(self.verification_step, optional=True)


@dataclass(frozen=True, slots=True)
class Community:
    """Snapshot-local partition; cross-record hierarchy checks belong to publication."""

    community_id: str
    parent_id: str | None
    depth: int
    member_ids: tuple[str, ...]
    issue_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        text(self.community_id)
        text(self.parent_id, optional=True)
        _integer("depth", self.depth)
        if (self.parent_id is None) != (self.depth == 0):
            raise ValueError("Only root communities have depth zero and no parent")
        if self.community_id == self.parent_id:
            raise ValueError("Community cannot parent itself")
        object.__setattr__(self, "member_ids", refs(self.member_ids, required=True))
        object.__setattr__(self, "issue_codes", refs(self.issue_codes))


@dataclass(frozen=True, slots=True)
class CommunityMembership:
    """Associate a community with a resolved entity."""

    community_id: str
    entity_id: str

    def __post_init__(self) -> None:
        text(self.community_id)
        text(self.entity_id)


@dataclass(frozen=True, slots=True)
class ChildStatementRef:
    """Namespace a child-local statement ID without a circular content hash."""

    report_id: str
    statement_id: str

    def __post_init__(self) -> None:
        text(self.report_id)
        text(self.statement_id)


@dataclass(frozen=True, slots=True)
class ReportStatement:
    """Cited content; reference existence and entailment need external validation."""

    statement_id: str
    text: str
    kind: str
    assertion_refs: tuple[str, ...]
    finding_refs: tuple[str, ...]
    child_statement_refs: tuple[ChildStatementRef, ...]

    def __post_init__(self) -> None:
        text(self.statement_id)
        text(self.text)
        choice(self.kind, frozenset({"fact", "hypothesis", "recommendation", "limitation"}))
        object.__setattr__(self, "assertion_refs", refs(self.assertion_refs))
        object.__setattr__(self, "finding_refs", refs(self.finding_refs))
        children = values(self.child_statement_refs, ChildStatementRef)
        if len(set(children)) != len(children):
            raise ValueError("Duplicate child statement references")
        object.__setattr__(self, "child_statement_refs", children)
        if self.kind == "fact" and not (self.assertion_refs or children):
            raise ValueError("Facts require assertion or child statement citations")
        if self.kind == "hypothesis" and not (self.assertion_refs or self.finding_refs or children):
            raise ValueError("Hypotheses require supporting citations")


@dataclass(frozen=True, slots=True)
class ReportSections:
    """Ten closed content sections; empty tuples support sparse evidence inventories."""

    summary: tuple[ReportStatement, ...] = ()
    observed_patterns: tuple[ReportStatement, ...] = ()
    strengths: tuple[ReportStatement, ...] = ()
    difficulties: tuple[ReportStatement, ...] = ()
    possible_causes: tuple[ReportStatement, ...] = ()
    existing_interventions: tuple[ReportStatement, ...] = ()
    observed_outcomes: tuple[ReportStatement, ...] = ()
    suggested_next_steps: tuple[ReportStatement, ...] = ()
    counter_evidence: tuple[ReportStatement, ...] = ()
    limitations: tuple[ReportStatement, ...] = ()

    def __post_init__(self) -> None:
        for field in fields(self):
            object.__setattr__(self, field.name, values(getattr(self, field.name), ReportStatement))


@dataclass(frozen=True, slots=True)
class CommunityReport:
    """A derived report; cross-report cycle and citation checks precede publication."""

    report_id: str
    report_kind: str
    community_id: str | None
    title: str
    sections: ReportSections
    selected_bundle_id: str
    omitted_count: int
    receipt_ref: str

    def __post_init__(self) -> None:
        for value in (self.report_id, self.title, self.selected_bundle_id, self.receipt_ref):
            text(value)
        choice(self.report_kind, frozenset({"community", "overview"}))
        text(self.community_id, optional=True)
        if (self.community_id is None) != (self.report_kind == "overview"):
            raise ValueError("Only overview reports have no community")
        _integer("omitted_count", self.omitted_count)
        if not isinstance(self.sections, ReportSections):
            raise TypeError("sections must be ReportSections")
        statements = [
            item for field in fields(self.sections) for item in getattr(self.sections, field.name)
        ]
        refs([item.statement_id for item in statements])
        if any(
            ref.report_id == self.report_id
            for item in statements
            for ref in item.child_statement_refs
        ):
            raise ValueError("Report cannot cite itself as a child")
