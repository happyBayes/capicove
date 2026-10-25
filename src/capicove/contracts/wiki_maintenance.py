"""Immutable, dependency-light values for reviewed Wiki maintenance.

These values enforce structural bounds only. Source authorization, evidence
resolution, semantic review and publication belong to application operations.
"""

from __future__ import annotations

import json
import re
import types
from dataclasses import asdict, dataclass, field, fields
from types import MappingProxyType
from typing import Any, Literal, get_args, get_origin, get_type_hints


MAX_RECORD_BYTES = 1_048_576
_HASH = r"[0-9a-f]{64}"
_PAGE = r"page:[A-Z0-9]{10,64}"


def bounded(maximum: int, *, minimum: int = 0, pattern: str | None = None) -> Any:
    """Declare a serialized field's byte, tuple-count or integer bounds."""
    return field(metadata={"maximum": maximum, "minimum": minimum, "pattern": pattern})


def record_hints(record_type: type[Any]) -> dict[str, Any]:
    """Resolve immutable record annotations without loading optional runtimes."""
    return get_type_hints(record_type)


def valid_type(value: Any, annotation: Any) -> bool:
    """Check strict JSON-compatible value types, including nested tuples."""
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is types.UnionType:
        return any(valid_type(value, item) for item in arguments)
    if origin is Literal:
        return any(type(value) is type(item) and value == item for item in arguments)
    if origin is tuple:
        return type(value) is tuple and all(valid_type(item, arguments[0]) for item in value)
    if annotation is type(None):
        return value is None
    return type(value) is annotation


@dataclass(frozen=True, slots=True, kw_only=True)
class MaintenanceValue:
    """Validate immutable structural fields on every construction path."""

    def __post_init__(self) -> None:
        hints = record_hints(type(self))
        for item in fields(self):
            value = getattr(self, item.name)
            if not valid_type(value, hints[item.name]):
                raise ValueError(f"invalid type for {item.name}")
            if value is None:
                continue
            if isinstance(value, str):
                try:
                    size = len(value.encode("utf-8"))
                except UnicodeError as error:
                    raise ValueError("invalid UTF-8 text") from error
                if size > 262_144:
                    raise ValueError("text exceeds record limit")
                if item.metadata.get("minimum", 0) and not value.strip():
                    raise ValueError(f"blank {item.name}")
                pattern = item.metadata.get("pattern")
                if pattern is not None and re.fullmatch(pattern, value) is None:
                    raise ValueError(f"invalid identifier for {item.name}")
            elif isinstance(value, tuple):
                size = len(value)
                for element in value:
                    if isinstance(element, str):
                        try:
                            encoded = element.encode("utf-8")
                        except UnicodeError as error:
                            raise ValueError("invalid tuple text") from error
                        if not element.strip() or len(encoded) > 2048:
                            raise ValueError("invalid or excessive tuple text")
                if size > 10_000:
                    raise ValueError("tuple exceeds record limit")
            elif type(value) is int:
                size = value
                if size < 0 or size > 2**63 - 1:
                    raise ValueError("integer outside supported range")
            else:
                continue
            if (
                not item.metadata.get("minimum", 0)
                <= size
                <= item.metadata.get("maximum", 2**63 - 1)
            ):
                raise ValueError(f"limit exceeded for {item.name}")


@dataclass(frozen=True, slots=True, kw_only=True)
class EvidenceSpan(MaintenanceValue):
    """Locate exact codepoints in a pinned original-source representation."""

    kind: Literal["markdown_span", "pdf_block_span"]
    source_id: str = bounded(78, pattern="source:sha256:" + _HASH)
    source_sha256: str = bounded(64, pattern=_HASH)
    representation_sha256: str = bounded(64, pattern=_HASH)
    start_char: int = bounded(2_000_000)
    end_char: int = bounded(2_000_000, minimum=1)
    exact_sha256: str = bounded(64, pattern=_HASH)
    page_index: int | None
    block_id: str | None

    def __post_init__(self) -> None:
        MaintenanceValue.__post_init__(self)
        if self.source_id != "source:sha256:" + self.source_sha256:
            raise ValueError("source identity mismatch")
        if not 0 < self.end_char - self.start_char <= 4096:
            raise ValueError("invalid evidence interval")
        if self.kind == "markdown_span":
            if self.page_index is not None or self.block_id is not None:
                raise ValueError("Markdown evidence cannot contain PDF coordinates")
        elif self.page_index is None or not self.block_id or len(self.block_id.encode()) > 256:
            raise ValueError("PDF evidence requires bounded block and page coordinates")


@dataclass(frozen=True, slots=True, kw_only=True)
class ManagedClaim(MaintenanceValue):
    """One assertion with source citations and explicit editorial status."""

    claim_id: str = bounded(70, pattern="claim:" + _HASH)
    text: str = bounded(2048, minimum=1)
    evidence: tuple[EvidenceSpan, ...] = bounded(16, minimum=1)
    status: Literal["current", "disputed", "superseded", "withdrawn"]
    knowledge: Literal["asserted", "inferred"]
    premise_ids: tuple[str, ...] = bounded(128)
    derivation: str | None = bounded(4096)

    def __post_init__(self) -> None:
        MaintenanceValue.__post_init__(self)
        if self.knowledge == "inferred" and (not self.premise_ids or not self.derivation):
            raise ValueError("inference requires premises and derivation")
        if self.knowledge == "asserted" and (self.premise_ids or self.derivation is not None):
            raise ValueError("assertion cannot declare derivation")
        if self.claim_id in self.premise_ids or len(set(self.premise_ids)) != len(self.premise_ids):
            raise ValueError("invalid premise identities")


@dataclass(frozen=True, slots=True, kw_only=True)
class Section(MaintenanceValue):
    """A safe heading and ordered claim references, without uncited prose."""

    heading: str = bounded(256, minimum=1)
    claim_ids: tuple[str, ...] = bounded(128)


@dataclass(frozen=True, slots=True, kw_only=True)
class Relation(MaintenanceValue):
    """A projection relationship requiring application-level resolution."""

    kind: Literal["related", "supersedes", "disputed"]
    target_id: str = bounded(256, minimum=1)
    reason: str = bounded(2048, minimum=1)


@dataclass(frozen=True, slots=True, kw_only=True)
class PageRevision(MaintenanceValue):
    """Host-addressed concept content, independent of a storage backend."""

    page_id: str = bounded(69, pattern=_PAGE)
    page_path: str = bounded(240, pattern=r"pages/concepts/[a-z0-9]+(?:-[a-z0-9]+)*\.md")
    parent_sha256: str | None = bounded(64, pattern=_HASH)
    title: str = bounded(512, minimum=1)
    kind: Literal["entity", "concept", "comparison", "query", "summary"]
    aliases: tuple[str, ...] = bounded(32)
    sections: tuple[Section, ...] = bounded(64, minimum=1)
    claims: tuple[ManagedClaim, ...] = bounded(128, minimum=1)
    relations: tuple[Relation, ...] = bounded(100)
    source_ids: tuple[str, ...] = bounded(128, minimum=1)
    rendering_version: Literal[1]


@dataclass(frozen=True, slots=True, kw_only=True)
class ClaimDraft(MaintenanceValue):
    """Untrusted Agent assertion using host-issued aliases, never source paths."""

    draft_id: str = bounded(64, pattern=r"[a-z][a-z0-9-]{0,63}")
    text: str = bounded(2048, minimum=1)
    evidence_aliases: tuple[str, ...] = bounded(16, minimum=1)
    status: Literal["current", "disputed", "superseded", "withdrawn"]
    knowledge: Literal["asserted", "inferred"]
    premise_ids: tuple[str, ...] = bounded(128)
    derivation: str | None = bounded(4096)

    def __post_init__(self) -> None:
        MaintenanceValue.__post_init__(self)
        if any(re.fullmatch(r"e[0-9]{1,6}", alias) is None for alias in self.evidence_aliases):
            raise ValueError("invalid evidence alias")
        if len(set(self.evidence_aliases)) != len(self.evidence_aliases):
            raise ValueError("duplicate evidence alias")
        if self.knowledge == "inferred" and (not self.premise_ids or not self.derivation):
            raise ValueError("inference requires premises and derivation")
        if self.knowledge == "asserted" and (self.premise_ids or self.derivation is not None):
            raise ValueError("assertion cannot declare derivation")


@dataclass(frozen=True, slots=True, kw_only=True)
class PageDraft(MaintenanceValue):
    """Agent page proposal; only the host can choose paths and revision hashes."""

    draft_id: str = bounded(64, pattern=r"[a-z][a-z0-9-]{0,63}")
    operation: Literal["create", "revise", "adopt"]
    page_id: str | None = bounded(69, pattern=_PAGE)
    title: str = bounded(512, minimum=1)
    kind: Literal["entity", "concept", "comparison", "query", "summary"]
    aliases: tuple[str, ...] = bounded(32)
    sections: tuple[Section, ...] = bounded(64, minimum=1)
    claims: tuple[ClaimDraft, ...] = bounded(128, minimum=1)
    relations: tuple[Relation, ...] = bounded(100)

    def __post_init__(self) -> None:
        MaintenanceValue.__post_init__(self)
        if (self.operation == "create") != (self.page_id is None):
            raise ValueError("only creation may omit existing page identity")
        if len({claim.draft_id for claim in self.claims}) != len(self.claims):
            raise ValueError("duplicate draft claim identity")


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceReference(MaintenanceValue):
    """Pin source and representation identities without filesystem authority."""

    source_id: str = bounded(78, pattern="source:sha256:" + _HASH)
    source_sha256: str = bounded(64, pattern=_HASH)
    representation_sha256: str = bounded(64, pattern=_HASH)

    def __post_init__(self) -> None:
        MaintenanceValue.__post_init__(self)
        if self.source_id != "source:sha256:" + self.source_sha256:
            raise ValueError("source identity mismatch")


@dataclass(frozen=True, slots=True, kw_only=True)
class EvidenceAlias(MaintenanceValue):
    """Bind a short host-issued label to an exact source span."""

    alias: str = bounded(32, pattern=r"e[0-9]{1,6}")
    span: EvidenceSpan
    text: str = bounded(16_384, minimum=1)


@dataclass(frozen=True, slots=True, kw_only=True)
class Candidate(MaintenanceValue):
    """Pin one candidate page at task creation."""

    page_id: str = bounded(69, pattern=_PAGE)
    revision_sha256: str = bounded(64, pattern=_HASH)
    reason: str = bounded(2048, minimum=1)
    title: str = ""
    body: str = ""
    context_complete: bool = False
    claim_context: tuple[ManagedClaim, ...] = field(default=(), metadata={"maximum": 128})


@dataclass(frozen=True, slots=True, kw_only=True)
class CompilationTask(MaintenanceValue):
    """Bounded caller-owned compilation input; confers no execution authority."""

    task_id: str = bounded(69, pattern="task:" + _HASH)
    task_version: int = bounded(2**31 - 1, minimum=1)
    workspace_id: str = bounded(128, minimum=1)
    wiki_revision: int
    policy_revision: int
    sources: tuple[SourceReference, ...] = bounded(8, minimum=1)
    evidence: tuple[EvidenceAlias, ...] = bounded(128, minimum=1)
    candidates: tuple[Candidate, ...] = bounded(20)
    permitted_page_ids: tuple[str, ...] = bounded(20)
    discovery_policy: str = bounded(128, minimum=1)
    coverage: Literal["complete_inventory", "partial"]
    max_output_bytes: int = bounded(MAX_RECORD_BYTES, minimum=1)
    previous_task_id: str | None = bounded(69, pattern="task:" + _HASH)
    affected_page_ids: tuple[str, ...] = field(default=(), metadata={"maximum": 100})
    deferred_page_count: int = field(default=0, metadata={"maximum": 10000})
    deferred_page_sha256: str | None = field(
        default=None, metadata={"maximum": 64, "pattern": _HASH}
    )

    def __post_init__(self) -> None:
        MaintenanceValue.__post_init__(self)
        if (
            sum(len(item.text.encode()) for item in self.evidence)
            + sum(
                len(json.dumps(asdict(item), ensure_ascii=False, separators=(",", ":")).encode())
                for item in self.candidates
            )
            > 262_144
        ):
            raise ValueError("task context exceeds 256 KiB")
        if len({item.alias for item in self.evidence}) != len(self.evidence):
            raise ValueError("duplicate evidence alias")


@dataclass(frozen=True, slots=True, kw_only=True)
class ImpactDecision(MaintenanceValue):
    """Account for one mechanically discovered affected page."""

    page_id: str = bounded(69, pattern=_PAGE)
    action: Literal["revise", "unchanged", "follow_up"]
    reason: str = bounded(2048, minimum=1)


@dataclass(frozen=True, slots=True, kw_only=True)
class Provenance(MaintenanceValue):
    """Informational generation provenance, never reviewer authority."""

    provider_id: str = bounded(256, minimum=1)
    model_revision: str = bounded(256, minimum=1)
    run_id: str = bounded(256, minimum=1)


@dataclass(frozen=True, slots=True, kw_only=True)
class CompilationSubmission(MaintenanceValue):
    """A proposed edit requiring source validation and host review."""

    task_id: str = bounded(69, pattern="task:" + _HASH)
    task_version: int = bounded(2**31 - 1, minimum=1)
    idempotency_key: str = bounded(128, minimum=1)
    dispositions: tuple[Literal["new", "update", "disputed", "no_material"], ...] = bounded(
        3, minimum=1
    )
    rationale: str = bounded(4096, minimum=1)
    pages: tuple[PageDraft, ...] = bounded(10)
    impact: tuple[ImpactDecision, ...] = bounded(100)
    provenance: Provenance | None

    def __post_init__(self) -> None:
        MaintenanceValue.__post_init__(self)
        if len(set(self.dispositions)) != len(self.dispositions):
            raise ValueError("duplicate disposition")
        if "no_material" in self.dispositions and (len(self.dispositions) != 1 or self.pages):
            raise ValueError("no_material is exclusive and cannot edit pages")
        if "no_material" not in self.dispositions and not self.pages:
            raise ValueError("material disposition requires page operations")
        if sum(len(page.claims) for page in self.pages) > 128:
            raise ValueError("submission exceeds claim limit")


@dataclass(frozen=True, slots=True, kw_only=True)
class ProposalRecord(MaintenanceValue):
    """Immutable proposal identity and its pinned publication base."""

    proposal_id: str = bounded(73, pattern="proposal:" + _HASH)
    workspace_id: str = bounded(128, minimum=1)
    submission: CompilationSubmission
    wiki_revision: int
    policy_revision: int
    rendered_sha256: str = bounded(64, pattern=_HASH)


@dataclass(frozen=True, slots=True, kw_only=True)
class ReviewDecision(MaintenanceValue):
    """Semantic judgment on a claim or disposition."""

    subject_id: str = bounded(256, minimum=1)
    verdict: Literal["supported", "unsupported", "uncertain"]
    reason: str = bounded(2048, minimum=1)


@dataclass(frozen=True, slots=True, kw_only=True)
class ReviewRecord(MaintenanceValue):
    """Host-issued review data; deserialization is not authentication."""

    review_id: str = bounded(71, pattern="review:" + _HASH)
    proposal_id: str = bounded(73, pattern="proposal:" + _HASH)
    proposal_sha256: str = bounded(64, pattern=_HASH)
    rendered_sha256: str = bounded(64, pattern=_HASH)
    wiki_revision: int
    policy_revision: int
    reviewer_id: str = bounded(256, minimum=1)
    decisions: tuple[ReviewDecision, ...] = bounded(256, minimum=1)
    accepted: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class FileChange(MaintenanceValue):
    """Before/after identity for a host-selected publication destination."""

    relative_path: str = bounded(512, minimum=1)
    before_sha256: str | None = bounded(64, pattern=_HASH)
    after_sha256: str | None = bounded(64, pattern=_HASH)

    def __post_init__(self) -> None:
        MaintenanceValue.__post_init__(self)
        if (
            self.relative_path.startswith("/")
            or "\\" in self.relative_path
            or any(part in {"", ".", ".."} for part in self.relative_path.split("/"))
            or ":" in self.relative_path
            or "\x00" in self.relative_path
        ):
            raise ValueError("unsafe relative destination")


@dataclass(frozen=True, slots=True, kw_only=True)
class PublicationRecord(MaintenanceValue):
    """Immutable publication lineage and observable destination inventory."""

    publication_id: str = bounded(76, pattern="publication:" + _HASH)
    parent_id: str | None = bounded(76, pattern="publication:" + _HASH)
    proposal_id: str | None = bounded(73, pattern="proposal:" + _HASH)
    review_id: str | None = bounded(71, pattern="review:" + _HASH)
    wiki_revision: int = bounded(2**63 - 1, minimum=1)
    policy_revision: int
    changes: tuple[FileChange, ...] = bounded(256)
    state_sha256: str = bounded(64, pattern=_HASH)
    status: Literal["complete", "partial", "aborted"]

    origin: Literal["reviewed", "migration", "ingest", "build", "book_section", "book_chunk"] = (
        "reviewed"
    )

    def __post_init__(self) -> None:
        MaintenanceValue.__post_init__(self)
        if self.origin == "reviewed":
            if self.proposal_id is None or self.review_id is None:
                raise ValueError("reviewed publication requires proposal and review")
        elif self.proposal_id is not None or self.review_id is not None:
            raise ValueError("system publication cannot impersonate semantic review")


@dataclass(frozen=True, slots=True, kw_only=True)
class SourcePolicyEvent(MaintenanceValue):
    """Monotonic source withdrawal, separate from page transaction state."""

    event_id: str = bounded(71, pattern="policy:" + _HASH)
    workspace_id: str = bounded(128, minimum=1)
    source_id: str = bounded(78, pattern="source:sha256:" + _HASH)
    policy_revision: int = bounded(2**63 - 1, minimum=1)
    reason: str = bounded(2048, minimum=1)
    idempotency_key: str = bounded(128, minimum=1)
    action: Literal["withdraw"]


@dataclass(frozen=True, slots=True, kw_only=True)
class AnswerTask(MaintenanceValue):
    """Read-only evidence context pinned to live policy and Wiki revisions."""

    task_id: str = bounded(69, pattern="task:" + _HASH)
    workspace_id: str = bounded(128, minimum=1)
    wiki_revision: int
    policy_revision: int
    question: str = bounded(8192, minimum=1)
    evidence: tuple[EvidenceAlias, ...] = bounded(128)
    claims: tuple[ManagedClaim, ...] = bounded(128)
    include_history: bool

    def __post_init__(self) -> None:
        MaintenanceValue.__post_init__(self)
        if sum(len(item.text.encode()) for item in self.evidence) > 262_144:
            raise ValueError("answer context exceeds 256 KiB")
        if len({item.alias for item in self.evidence}) != len(self.evidence):
            raise ValueError("duplicate answer alias")


@dataclass(frozen=True, slots=True, kw_only=True)
class AnswerCitation(MaintenanceValue):
    """Untrusted alias and quotation to resolve against authorized context."""

    alias: str = bounded(32, pattern=r"e[0-9]{1,6}")
    quote: str = bounded(16_384, minimum=1)


@dataclass(frozen=True, slots=True, kw_only=True)
class AnswerClaim(MaintenanceValue):
    """One generated statement with original-source citations."""

    text: str = bounded(2048, minimum=1)
    citations: tuple[AnswerCitation, ...] = bounded(8, minimum=1)
    status: Literal["current", "disputed", "superseded", "withdrawn"]


@dataclass(frozen=True, slots=True, kw_only=True)
class AnswerSubmission(MaintenanceValue):
    """Caller answer; no persisted editorial mutation is implied."""

    task_id: str = bounded(69, pattern="task:" + _HASH)
    claims: tuple[AnswerClaim, ...] = bounded(20)


@dataclass(frozen=True, slots=True, kw_only=True)
class AnswerResult(MaintenanceValue):
    """Validated answer shape, distinct from semantic qualification."""

    task_id: str = bounded(69, pattern="task:" + _HASH)
    wiki_revision: int
    policy_revision: int
    status: Literal["answered", "evidence_only", "no_evidence", "insufficient_evidence"]
    claims: tuple[AnswerClaim, ...] = bounded(20)
    semantic_status: Literal["not_verified", "supported", "unsupported", "uncertain"]
    omitted_claims: int
    evidence: tuple[EvidenceAlias, ...] = field(default=(), metadata={"maximum": 128})

    def __post_init__(self) -> None:
        MaintenanceValue.__post_init__(self)
        if (self.status == "answered") != bool(self.claims):
            raise ValueError("answer status does not match claim presence")


RECORD_TYPES = MappingProxyType(
    {
        "evidence_span": EvidenceSpan,
        "claim_draft": ClaimDraft,
        "page_draft": PageDraft,
        "managed_claim": ManagedClaim,
        "page_revision": PageRevision,
        "compilation_task": CompilationTask,
        "compilation_submission": CompilationSubmission,
        "proposal": ProposalRecord,
        "review": ReviewRecord,
        "publication": PublicationRecord,
        "source_policy_event": SourcePolicyEvent,
        "answer_task": AnswerTask,
        "answer_submission": AnswerSubmission,
        "answer_result": AnswerResult,
    }
)


@dataclass(frozen=True, slots=True, kw_only=True)
class MaintenanceRecord:
    """Explicitly versioned envelope; digest is external to canonical bytes."""

    schema_version: int
    record_type: str
    payload: MaintenanceValue

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported maintenance record version")
        if type(self.record_type) is not str or RECORD_TYPES.get(self.record_type) is not type(
            self.payload
        ):
            raise ValueError("record type does not match payload")
