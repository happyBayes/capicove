"""Immutable public values for the local Wiki capability."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from capicove.contracts import DocumentParseResult


class WikiAuthority(StrEnum):
    """Identify the service-dependency mode of one Wiki workspace."""

    STANDALONE = "standalone"


class WikiFindingSeverity(StrEnum):
    """Classify one deterministic Wiki validation finding."""

    ERROR = "error"
    WARNING = "warning"
    STYLE = "style"


@dataclass(frozen=True, slots=True)
class WikiInventory:
    """Summarize durable records in one local Wiki."""

    schema_version: int
    authority: WikiAuthority
    pages: int
    sources: int
    facts: int
    identity_events: int


@dataclass(frozen=True, slots=True)
class WikiIngestResult:
    """Describe one idempotent Markdown ingestion."""

    source_id: str
    page_id: str
    page_path: str
    created: bool


@dataclass(frozen=True, slots=True)
class WikiFinding:
    """Describe one stable Wiki validation issue."""

    severity: WikiFindingSeverity
    code: str
    path: str
    message: str


@dataclass(frozen=True, slots=True)
class WikiLintReport:
    """Return an ordered validation result."""

    findings: tuple[WikiFinding, ...]

    @property
    def valid(self) -> bool:
        """Return whether the report contains no errors."""
        return all(item.severity is not WikiFindingSeverity.ERROR for item in self.findings)


@dataclass(frozen=True, slots=True)
class WikiDoctorResult:
    """Describe one internally consistent Wiki health snapshot."""

    lint: WikiLintReport
    index_built: bool
    index_healthy: bool

    @property
    def healthy(self) -> bool:
        """Return whether validated content and a built index are healthy."""
        return self.lint.valid and self.index_built and self.index_healthy


@dataclass(frozen=True, slots=True)
class WikiBuildResult:
    """Describe one completed deterministic Wiki build."""

    pages: int
    index_path: str
    search_index_path: str


@dataclass(frozen=True, slots=True)
class WikiSearchHit:
    """Return one cited local Wiki search result."""

    page_id: str
    page_path: str
    title: str
    kind: str
    snippet: str
    source_ids: tuple[str, ...]
    score: float


@dataclass(frozen=True, slots=True)
class WikiSearchResult:
    """Return stable hits for one local Wiki query."""

    query: str
    hits: tuple[WikiSearchHit, ...]


class CitationValidationLevel(StrEnum):
    """Identify one deterministic citation-validation layer."""

    L0 = "L0"
    L1 = "L1"
    L2 = "L2"


@dataclass(frozen=True, slots=True)
class ClaimCitation:
    """Bind one atomic claim to immutable page and block evidence."""

    citation_id: str
    claim_id: str
    page_id: str
    book_id: str
    source_sha256: str
    representation_sha256: str
    page_index: int
    printed_page_label: str | None
    block_ids: tuple[str, ...]
    evidence_text_sha256: str
    quotation: str | None
    submission_version: int

    def __post_init__(self) -> None:
        for name in ("citation_id", "claim_id", "page_id", "book_id"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be blank")
        for name in ("source_sha256", "representation_sha256", "evidence_text_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if isinstance(self.page_index, bool) or self.page_index < 0:
            raise ValueError("page_index must be non-negative")
        if self.printed_page_label is not None and not self.printed_page_label.strip():
            raise ValueError("printed_page_label must not be blank")
        blocks = tuple(self.block_ids)
        if not blocks or len(set(blocks)) != len(blocks):
            raise ValueError("block_ids must contain unique evidence blocks")
        if self.quotation is not None and not self.quotation.strip():
            raise ValueError("quotation must not be blank")
        if (
            isinstance(self.submission_version, bool)
            or not isinstance(self.submission_version, int)
            or self.submission_version <= 0
        ):
            raise ValueError("submission_version must be positive")
        object.__setattr__(self, "block_ids", blocks)


@dataclass(frozen=True, slots=True)
class ChunkPublicationCitation:
    """Carry one exact accepted Book source slice to Wiki publication."""

    block_id: str
    page_index: int
    start_char: int
    end_char: int
    text_sha256: str


@dataclass(frozen=True, slots=True)
class ChunkPublicationClaim:
    """Carry one accepted factual claim and its source slices."""

    text: str
    citations: tuple[ChunkPublicationCitation, ...]
    result_id: str
    version: int


@dataclass(frozen=True, slots=True)
class ChunkPublicationPage:
    """Describe one section or book overview page to be rendered by Wiki."""

    section_id: str
    title: str
    claims: tuple[ChunkPublicationClaim, ...]


@dataclass(frozen=True, slots=True)
class ChunkPublicationPacket:
    """Pin one complete Book result DAG for the public Wiki publisher."""

    publication_id: str
    book_id: str
    run_id: str
    result_ids: tuple[str, ...]
    source_sha256: str
    representation_sha256: str
    wiki_schema_version: int
    render_policy_version: int
    source_pdf: Path
    parsed: DocumentParseResult
    pages: tuple[ChunkPublicationPage, ...]


@dataclass(frozen=True, slots=True)
class CitationValidationFinding:
    """Describe one stable deterministic citation failure."""

    level: CitationValidationLevel
    code: str
    claim_id: str
    message: str


@dataclass(frozen=True, slots=True)
class CitationValidationReport:
    """Return ordered L0-L2 findings."""

    validated_level: CitationValidationLevel
    findings: tuple[CitationValidationFinding, ...]

    @property
    def valid(self) -> bool:
        """Return whether every deterministic validation succeeded."""
        return not self.findings


@dataclass(frozen=True, slots=True)
class WikiMigrationPlan:
    """Describe exact explicit schema-v2 filesystem effects."""

    from_version: int
    to_version: int
    created_paths: tuple[str, ...]
    updated_paths: tuple[str, ...]
    applied: bool


@dataclass(frozen=True, slots=True)
class BookWikiPublishResult:
    """Describe one validated page-aware Wiki projection."""

    book_id: str
    page_paths: tuple[str, ...]
    citations: int
    representation_path: str
    source_path: str


__all__ = [
    "BookWikiPublishResult",
    "CitationValidationFinding",
    "CitationValidationLevel",
    "CitationValidationReport",
    "ClaimCitation",
    "WikiAuthority",
    "WikiBuildResult",
    "WikiDoctorResult",
    "WikiFinding",
    "WikiFindingSeverity",
    "WikiIngestResult",
    "WikiInventory",
    "WikiLintReport",
    "WikiMigrationPlan",
    "WikiSearchHit",
    "WikiSearchResult",
]


@dataclass(frozen=True, slots=True)
class ChunkPublicationPacketV3:
    """Explicit v3 envelope around immutable v2-compatible Book content.

    The nested content is validated only in private staging. This envelope pins
    the live Wiki and source policy; a bare v2 packet never authorizes v3 writes.
    """

    base_wiki_revision: int
    policy_revision: int
    content: ChunkPublicationPacket
    packet_version: int = 3

    def __post_init__(self) -> None:
        if (
            type(self.packet_version) is not int
            or self.packet_version != 3
            or type(self.base_wiki_revision) is not int
            or self.base_wiki_revision < 0
            or type(self.policy_revision) is not int
            or self.policy_revision < 0
            or type(self.content) is not ChunkPublicationPacket
            or self.content.wiki_schema_version != 2
        ):
            raise ValueError("invalid Book-v3 packet")
