"""Application operations for one local Wiki workspace."""

from __future__ import annotations

import os
import stat
from datetime import date
from pathlib import Path

from capicove.wiki.filesystem import (
    WikiError,
    WikiIndexUnavailableError,
    WikiUnsafePathError,
    WikiValidationError,
    append_log,
    atomic_write,
    ingest_markdown,
    initialize,
    scan,
    wiki_write_lock,
)
from capicove.wiki.indexing import bounded_snippet, load_documents, render_index
from capicove.wiki.models import (
    ChunkPublicationPacket,
    ChunkPublicationPacketV3,
    WikiBuildResult,
    WikiDoctorResult,
    WikiIngestResult,
    WikiInventory,
    WikiLintReport,
    WikiSearchHit,
    WikiSearchResult,
)
from capicove.wiki.validation import lint


class WikiService:
    """Operate one `wiki/` directory beneath a workspace."""

    def __init__(self, workspace: Path) -> None:
        if not isinstance(workspace, Path):
            raise TypeError("workspace must be a Path")
        try:
            self._workspace = workspace.resolve()
        except (OSError, RuntimeError) as error:
            raise WikiUnsafePathError("Wiki workspace path cannot be resolved safely") from error
        self._wiki = self._workspace / "wiki"

    @property
    def wiki_path(self) -> Path:
        """Return the absolute Wiki root."""
        return self._wiki

    def initialize(self, *, created_on: date) -> WikiInventory:
        """Initialize and inventory the standalone Wiki."""
        return initialize(self._wiki, created_on=created_on)

    def scan(self) -> WikiInventory:
        """Inventory the initialized Wiki."""
        return scan(self._wiki)

    def require_chunk_publication_ready(self) -> None:
        """Check Wiki-v2 publication prerequisites without following a changed POSIX path."""
        from capicove.wiki.filesystem import load_yaml_mapping

        if self.scan().schema_version not in {2, 3}:
            raise ValueError("Book result publication requires a version-2 or version-3 Wiki")
        ontology_path = self._wiki / "graph" / "ontology.yaml"
        if os.name == "posix":
            from capicove.wiki.chunk_publication import _read_wiki_bytes_posix

            load_yaml_mapping(
                ontology_path,
                content=_read_wiki_bytes_posix(self._wiki, ontology_path.relative_to(self._wiki)),
            )
        else:
            load_yaml_mapping(ontology_path)

    def ingest(self, source: Path, *, ingested_on: date) -> WikiIngestResult:
        """Ingest one UTF-8 Markdown source idempotently."""
        return ingest_markdown(self._wiki, source, ingested_on=ingested_on)

    def lint(self) -> WikiLintReport:
        """Validate the Wiki without changing it."""
        from capicove.wiki.maintenance.compatibility import execute, is_v3

        if is_v3(self._wiki):
            return execute(self._wiki, "lint", lambda stage: lint(stage))

        return lint(self._wiki)

    def build(self, *, built_on: date) -> WikiBuildResult:
        """Validate and rebuild deterministic human and FTS projections."""
        from capicove.wiki.maintenance.compatibility import execute, is_v3

        if is_v3(self._wiki):
            return execute(
                self._wiki,
                "build",
                lambda stage: WikiService(stage.parent).build(built_on=built_on),
            )

        from capicove.wiki._adapters.sqlite_index import rebuild_index

        try:
            with wiki_write_lock(self._wiki):
                report = self.lint()
                if not report.valid:
                    raise WikiValidationError(report)
                documents = load_documents(self._wiki)
                database_path = self._wiki / "_derived" / "search.sqlite3"
                pending = self._wiki / "_migration" / "build.pending"
                atomic_write(pending, "Build incomplete; rerun wiki build.\n")
                rebuild_index(database_path, documents)
                atomic_write(self._wiki / "index.md", render_index(documents, built_on=built_on))
                append_log(
                    self._wiki,
                    f"## [{built_on.isoformat()}] build | Indexed {len(documents)} pages",
                )
                pending.unlink()
                return WikiBuildResult(
                    pages=len(documents),
                    index_path="index.md",
                    search_index_path="_derived/search.sqlite3",
                )
        except WikiError:
            raise
        except (OSError, UnicodeError) as error:
            raise WikiError("Wiki build failed") from error

    def publish_chunk_results(
        self, packet: ChunkPublicationPacket, *, published_on: date
    ) -> dict[str, object]:
        """Publish verified Book chunk claims through Wiki's v2 visibility boundary."""
        from capicove.wiki.chunk_publication import publish_chunk_book

        return publish_chunk_book(self._wiki, packet, published_on=published_on)

    def publish_chunk_results_v3(
        self, packet: ChunkPublicationPacketV3, *, published_on: date
    ) -> dict[str, object]:
        """Publish an explicitly versioned Book packet through the v3 journal."""
        from capicove.wiki.chunk_publication import publish_chunk_book
        from capicove.wiki.maintenance.compatibility import execute, is_v3

        if type(packet) is not ChunkPublicationPacketV3 or not is_v3(self._wiki):
            raise ValueError("Book-v3 packet requires a version-3 Wiki")
        return execute(
            self._wiki,
            "book_chunk",
            lambda stage: publish_chunk_book(stage, packet.content, published_on=published_on),
            expected_revision=packet.base_wiki_revision,
            expected_policy=packet.policy_revision,
        )

    def query(self, text: str, *, top_k: int = 20) -> WikiSearchResult:
        """Search the current local FTS projection and return cited hits."""
        from capicove.wiki.maintenance.compatibility import execute, is_v3

        if is_v3(self._wiki):
            return execute(
                self._wiki,
                "query",
                lambda stage: WikiService(stage.parent).query(text, top_k=top_k),
            )

        from capicove.wiki._adapters.sqlite_index import check_index, search_index, tokenize_query

        if not isinstance(text, str) or not text.strip():
            raise ValueError("query must be nonblank")
        try:
            encoded = text.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("query must contain valid Unicode") from error
        if len(text) > 2048 or len(encoded) > 8192:
            raise ValueError("query must contain at most 2048 characters and 8192 UTF-8 bytes")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        query = text.strip()
        tokens = tokenize_query(query)
        if not tokens:
            raise ValueError("query must contain searchable text")
        if len(tokens) > 128:
            raise ValueError("query must contain at most 128 searchable tokens")
        with wiki_write_lock(self._wiki):
            self._require_complete_build()
            report = self.lint()
            if not report.valid:
                raise WikiValidationError(report)
            self._require_complete_build()
            check_index(self._wiki / "_derived" / "search.sqlite3", load_documents(self._wiki))
            documents = search_index(
                self._wiki / "_derived" / "search.sqlite3",
                query,
                tokens=tokens,
                top_k=top_k,
            )
        hits = tuple(
            WikiSearchHit(
                page_id=item.page_id,
                page_path=item.page_path,
                title=item.title,
                kind=item.kind,
                snippet=bounded_snippet(item.body, query),
                source_ids=item.source_ids,
                score=1.0 / rank,
            )
            for rank, item in enumerate(documents, 1)
        )
        return WikiSearchResult(query=query, hits=hits)

    def doctor(self) -> WikiDoctorResult:
        """Return one fail-closed health snapshot for the local Wiki."""
        from capicove.wiki.maintenance.compatibility import execute, is_v3

        if is_v3(self._wiki):
            return execute(self._wiki, "doctor", lambda stage: WikiService(stage.parent).doctor())

        database_path = self._wiki / "_derived" / "search.sqlite3"
        preflight = self.lint()
        if not preflight.valid:
            return WikiDoctorResult(
                lint=preflight,
                index_built=_is_regular_file_without_following(database_path),
                index_healthy=False,
            )
        try:
            with wiki_write_lock(self._wiki):
                report = self.lint()
                index_built = _is_regular_file_without_following(database_path)
                return WikiDoctorResult(
                    lint=report,
                    index_built=index_built,
                    index_healthy=index_built and self._index_is_healthy_without_lock(report),
                )
        except WikiError:
            return WikiDoctorResult(
                lint=preflight,
                index_built=_is_regular_file_without_following(database_path),
                index_healthy=False,
            )

    def index_is_healthy(self) -> bool:
        """Return whether the disposable index matches current validated pages."""
        try:
            with wiki_write_lock(self._wiki):
                return self._index_is_healthy_without_lock(self.lint())
        except WikiError:
            return False

    def _index_is_healthy_without_lock(self, report: WikiLintReport) -> bool:
        from capicove.wiki._adapters.sqlite_index import check_index

        if not report.valid:
            return False
        try:
            self._require_complete_build()
            check_index(self._wiki / "_derived" / "search.sqlite3", load_documents(self._wiki))
        except WikiError:
            return False
        return True

    def _require_complete_build(self) -> None:
        publication_pending = self._wiki / "_migration" / "chunk-publication.pending"
        if publication_pending.exists() or publication_pending.is_symlink():
            raise WikiIndexUnavailableError("chunk publication is incomplete; retry publication")
        if (self._wiki / "_migration" / "build.pending").exists():
            raise WikiIndexUnavailableError("Wiki build is incomplete; run wiki build")


def _is_regular_file_without_following(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


__all__ = ["WikiService"]
