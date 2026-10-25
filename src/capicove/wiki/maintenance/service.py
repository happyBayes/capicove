"""Host-root-bound maintenance operations on explicitly prepared v3 layouts."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from capicove.contracts.wiki_maintenance import (
    AnswerResult,
    AnswerTask,
    CompilationTask,
    ProposalRecord,
    PublicationRecord,
    ReviewDecision,
    ReviewRecord,
    SourcePolicyEvent,
)
from capicove.wiki.maintenance.codec import _forbid_number, _object
from capicove.wiki.maintenance.discovery import DiscoveryPage, DiscoveryResult
from capicove.wiki.maintenance.representations import (
    MAX_SOURCE_BYTES,
    MarkdownRepresentation,
    markdown_representation,
)
from capicove.wiki.maintenance.sources import SourceInterval
from capicove.wiki.maintenance.store import create_verified, maintenance_lock, read_authorized


if TYPE_CHECKING:
    from capicove.wiki.maintenance.follow_up import FollowUpActivation
    from capicove.wiki.maintenance.review import ReviewAuthority
    from capicove.wiki.maintenance.validation import ProposalPreview


@dataclass(frozen=True, slots=True)
class CaptureResult:
    """Identity of captured evidence, independent of page publication."""

    source_id: str
    source_sha256: str
    representation_sha256: str
    created: bool
    policy_revision: int


class MaintenanceService:
    """Bind authorized source roots once, outside untrusted request arguments."""

    def __init__(self, workspace: Path, *, source_roots: tuple[Path, ...] = ()) -> None:
        self.workspace = workspace.resolve(strict=True)
        self.wiki = self.workspace / "wiki"
        self.source_roots = tuple(root.resolve(strict=True) for root in source_roots)

    def _admit(self, *, allow_pending: bool = False) -> tuple[int, tuple[str, ...]]:
        manifest = tomllib.loads(
            read_authorized(self.wiki / "wiki.toml", (self.wiki,), 65536).decode()
        )
        if manifest.get("schema_version") != 3:
            raise ValueError("schema_upgrade_required")
        for name in ("raw", "representations", "maintenance"):
            directory = self.wiki / name
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError("evidence_unavailable: unsafe workspace layout")
        if not allow_pending and (self.wiki / "_migration" / "maintenance.pending").exists():
            raise ValueError("publication_pending")
        policy = json.loads(
            read_authorized(
                self.wiki / "maintenance" / "source-policy.json", (self.wiki,), 1048576
            ),
            object_pairs_hook=_object,
            parse_float=_forbid_number,
            parse_constant=_forbid_number,
        )
        if (
            not isinstance(policy, dict)
            or set(policy) != {"schema_version", "policy_revision", "withdrawn"}
            or type(policy["schema_version"]) is not int
            or policy["schema_version"] != 1
            or type(policy["policy_revision"]) is not int
            or policy["policy_revision"] < 0
            or not isinstance(policy["withdrawn"], list)
            or any(
                not isinstance(item, str)
                or re.fullmatch(r"source:sha256:[0-9a-f]{64}", item) is None
                for item in policy["withdrawn"]
            )
        ):
            raise ValueError("evidence_unavailable: invalid source policy")
        return policy["policy_revision"], tuple(policy["withdrawn"])

    def capture(self, source: Path) -> CaptureResult:
        """Capture Markdown without creating pages or invoking a generator."""
        raw = read_authorized(source, self.source_roots, MAX_SOURCE_BYTES)
        value = markdown_representation(raw)
        with maintenance_lock(self.wiki):
            policy_revision, withdrawn = self._admit()
            source_id = "source:sha256:" + value.source_sha256
            if source_id in withdrawn:
                raise ValueError("evidence_unavailable: source withdrawn")
            targets = (
                (self.wiki / "raw" / (value.source_sha256 + ".md"), raw),
                (
                    self.wiki / "representations" / (value.representation_sha256 + ".json"),
                    value.encoded,
                ),
            )
            # Validate all existing destinations before creating either artifact.
            for path, content in targets:
                if path.exists() or path.is_symlink():
                    if read_authorized(path, (self.wiki,), max(len(content), 1)) != content:
                        raise ValueError("evidence_digest_mismatch: destination collision")
            created = create_verified(targets[0][0], targets[0][1], self.wiki)
            create_verified(targets[1][0], targets[1][1], self.wiki)
            return CaptureResult(
                source_id,
                value.source_sha256,
                value.representation_sha256,
                created,
                policy_revision,
            )

    def read_markdown(self, source_id: str) -> MarkdownRepresentation:
        """Resolve a source under the common lock and current policy."""
        with maintenance_lock(self.wiki):
            _, withdrawn = self._admit()
            return self._read_markdown_unlocked(source_id, withdrawn)

    def _read_markdown_unlocked(
        self, source_id: str, withdrawn: tuple[str, ...]
    ) -> MarkdownRepresentation:
        if re.fullmatch(r"source:sha256:[0-9a-f]{64}", source_id) is None:
            raise ValueError("evidence_unavailable: invalid source identity")
        if source_id in withdrawn:
            raise ValueError("evidence_unavailable: source withdrawn")
        digest = source_id.removeprefix("source:sha256:")
        raw = read_authorized(self.wiki / "raw" / (digest + ".md"), (self.wiki,), MAX_SOURCE_BYTES)
        value = markdown_representation(raw)
        if value.source_sha256 != digest:
            raise ValueError("evidence_digest_mismatch")
        captured = read_authorized(
            self.wiki / "representations" / (value.representation_sha256 + ".json"),
            (self.wiki,),
            len(value.encoded),
        )
        if captured != value.encoded:
            raise ValueError("evidence_digest_mismatch")
        return value

    def _inventory(self) -> tuple[DiscoveryPage, ...]:
        import hashlib

        from capicove.wiki.filesystem import _split_markdown_text

        pages: list[DiscoveryPage] = []
        paths: dict[str, str] = {}
        root = self.wiki / "pages"
        if root.is_symlink() or not root.is_dir():
            raise ValueError("evidence_unavailable: unsafe pages")
        total_bytes = 0
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("evidence_unavailable: symlinked page")
            if path.is_dir() or path.suffix != ".md":
                continue
            if len(pages) >= 10000:
                raise ValueError("capacity_exceeded: inventory")
            raw = read_authorized(path, (self.wiki,), 16 * 1024 * 1024)
            total_bytes += len(raw)
            if total_bytes > 256 * 1024 * 1024:
                raise ValueError("capacity_exceeded: inventory")
            metadata, body = _split_markdown_text(
                path, raw.decode("utf-8"), require_frontmatter=True
            )
            identity = metadata.get("id")
            title = metadata.get("title")
            aliases = metadata.get("aliases", [])
            sources = metadata.get("sources", [])
            if (
                not isinstance(identity, str)
                or not isinstance(title, str)
                or not isinstance(aliases, list)
                or not isinstance(sources, list)
                or any(not isinstance(item, str) for item in [*aliases, *sources])
            ):
                raise ValueError("evidence_unavailable: invalid page metadata")
            relative = path.relative_to(self.wiki).as_posix()
            paths[relative] = identity
            pages.append(
                DiscoveryPage(
                    identity,
                    relative,
                    title,
                    body,
                    hashlib.sha256(raw).hexdigest(),
                    tuple(aliases),
                    (),
                    tuple(sources),
                    (),
                )
            )
        import posixpath
        from dataclasses import replace
        from pathlib import PurePosixPath
        from urllib.parse import unquote

        resolved = []
        for page in pages:
            targets = re.findall(r"\[[^\]]*\]\(([^ )]+)\)", page.body)
            links = []
            for target in targets:
                target = unquote(target.split("#", 1)[0])
                joined = posixpath.normpath(str(PurePosixPath(page.path).parent / target))
                if joined in paths:
                    links.append(paths[joined])
            resolved.append(replace(page, links=tuple(sorted(set(links)))))
        return tuple(resolved)

    def build_discovery(self) -> None:
        """Rebuild only the disposable maintenance discovery projection."""
        from capicove.wiki.maintenance.discovery import build_discovery

        with maintenance_lock(self.wiki):
            self._admit()
            derived = self.wiki / "_derived"
            if derived.is_symlink() or not derived.is_dir():
                raise ValueError("evidence_unavailable: unsafe derived directory")
            build_discovery(derived / "maintenance.sqlite", self._inventory())

    def discover(
        self, query: str, *, source_ids: tuple[str, ...] = (), page_ids: tuple[str, ...] = ()
    ) -> DiscoveryResult:
        """Discover current pages; source expansions must belong to this Wiki."""
        from capicove.wiki.maintenance.discovery import discover

        with maintenance_lock(self.wiki):
            _, withdrawn = self._admit()
            for source in source_ids:
                if (
                    re.fullmatch(r"source:sha256:[0-9a-f]{64}", source) is None
                    or source in withdrawn
                ):
                    raise ValueError("citation_out_of_scope: source")
                digest = source.removeprefix("source:sha256:")
                sources = [self.wiki / "raw" / (digest + suffix) for suffix in (".md", ".pdf")]
                sources = [path for path in sources if path.exists() or path.is_symlink()]
                if not sources:
                    raise ValueError("citation_out_of_scope: source")
                if any(
                    hashlib.sha256(
                        read_authorized(path, (self.wiki,), MAX_SOURCE_BYTES)
                    ).hexdigest()
                    != digest
                    for path in sources
                ):
                    raise ValueError("evidence_digest_mismatch")
            return discover(
                self.wiki / "_derived" / "maintenance.sqlite",
                self._inventory(),
                query,
                source_ids=source_ids,
                page_ids=page_ids,
            )

    def create_task(
        self,
        source_ids: tuple[str, ...],
        *,
        query: str,
        previous_task_id: str | None = None,
        intervals: tuple[SourceInterval, ...] | None = None,
        page_ids: tuple[str, ...] = (),
    ) -> CompilationTask:
        """Create a versioned bounded context without changing published pages."""
        from capicove.wiki.maintenance.tasks import create_task

        with maintenance_lock(self.wiki):
            return create_task(self, source_ids, query, previous_task_id, intervals, page_ids)

    def load_task(self, task_id: str) -> CompilationTask:
        """Read and validate the current immutable context."""
        from capicove.wiki.maintenance.tasks import load_task

        with maintenance_lock(self.wiki):
            return load_task(self, task_id)[0]

    def submit(self, raw: bytes) -> ProposalRecord:
        """Accept a closed proposal after complete source and structural checks."""
        from capicove.wiki.maintenance.validation import submit

        with maintenance_lock(self.wiki):
            return submit(self, raw)

    def preview(self, proposal_id: str) -> ProposalPreview:
        """Show an exact diff with source locators and semantic review subjects."""
        from capicove.wiki.maintenance.validation import preview

        with maintenance_lock(self.wiki):
            return preview(self, proposal_id)

    def review(
        self,
        proposal_id: str,
        decisions: tuple[ReviewDecision, ...],
        *,
        authority: ReviewAuthority,
        accept: bool,
    ) -> ReviewRecord:
        """Record a host-authorized review without publishing any page."""
        from capicove.wiki.maintenance.review import review

        with maintenance_lock(self.wiki):
            return review(self, proposal_id, decisions, authority, accept)

    def apply(self, proposal_id: str, review_id: str) -> PublicationRecord:
        """Publish a reviewed proposal on an experimental v3 workspace."""
        from capicove.wiki.maintenance.publication import apply

        with maintenance_lock(self.wiki):
            return apply(self, proposal_id, review_id)

    def withdraw(self, source_id: str, *, reason: str, idempotency_key: str) -> SourcePolicyEvent:
        """Withdraw evidence immediately, even during a pending publication."""
        from capicove.wiki.maintenance.policy import withdraw

        with maintenance_lock(self.wiki):
            return withdraw(self, source_id, reason, idempotency_key)

    def recover(self, *, abort: bool = False) -> None:
        """Resume or abort a pending transaction after complete hash verification."""
        from capicove.wiki.maintenance.transaction import recover

        recover(self.wiki, abort=abort)

    def answer_task(self, question: str, *, include_history: bool = False) -> AnswerTask:
        """Issue a read-only context from current source-backed managed claims."""
        from capicove.wiki.maintenance.answers import context

        with maintenance_lock(self.wiki):
            return context(self, question, include_history)[0]

    def answer_submit(self, task: AnswerTask, raw: bytes) -> AnswerResult:
        """Validate every returned citation against a freshly authorized context."""
        from capicove.wiki.maintenance.answers import submit

        with maintenance_lock(self.wiki):
            return submit(self, task, raw)

    def evidence_only(
        self, question: str, *, include_history: bool = False
    ) -> tuple[AnswerTask, AnswerResult]:
        """Answer without generation or editorial side effects."""
        from capicove.wiki.maintenance.answers import evidence_only

        with maintenance_lock(self.wiki):
            return evidence_only(self, question, include_history)

    def status(self) -> dict[str, Any]:
        """Read the active revision and live policy, including pending recovery."""
        from capicove.wiki.maintenance.tasks import state

        with maintenance_lock(self.wiki):
            policy, withdrawn = self._admit(allow_pending=True)
            return {
                **state(self),
                "policy_revision": policy,
                "withdrawn_sources": len(withdrawn),
                "publication_pending": (self.wiki / "_migration/maintenance.pending").exists(),
            }

    def activate_follow_up(
        self, identity: str, *, page_ids: tuple[str, ...] = ()
    ) -> FollowUpActivation:
        """Activate a bounded subset of committed deferred work at the live revision."""
        from capicove.wiki.maintenance.follow_up import activate

        with maintenance_lock(self.wiki):
            return activate(self, identity, page_ids)

    def rollback_proposal(
        self, page_id: str, publication_id: str, *, idempotency_key: str
    ) -> ProposalRecord:
        """Prepare a compensating proposal; a fresh host review is still required."""
        from capicove.wiki.maintenance.rollback import propose

        with maintenance_lock(self.wiki):
            return propose(self, page_id, publication_id, idempotency_key)
