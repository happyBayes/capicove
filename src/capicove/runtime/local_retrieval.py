"""Offline workspace retrieval: import, build, Search, and Evidence (RFC-0018, ADR-0024).

Configuration comes only from the version-2 host's ``capabilities.retrieval``
object. Import and build are explicit operator writes; Search and Evidence only
read the active, validated generation and never create storage.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import uuid
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from capicove.contracts import (
    CharacterRange,
    InvocationContext,
    ProcessingFingerprint,
    SearchableUnit,
    SearchRequest,
    SearchResult,
    TargetType,
    TextChunkLocator,
    TextOrigin,
    ToolHandler,
)
from capicove.runtime.mcp_config import WorkspaceHostConfiguration
from capicove.tools.evidence import CAPICOVE_EVIDENCE_SPEC, EvidenceToolHandler
from capicove.tools.search import CAPICOVE_SEARCH_SPEC, SearchToolHandler


if TYPE_CHECKING:
    from capicove.retrieval._adapters.sqlite_fts import Fts5TextBackend


# The shared Search contract allows networked providers; this binding is offline.
LOCAL_SEARCH_SPEC = replace(CAPICOVE_SEARCH_SPEC, network_access=False)
LOCAL_EVIDENCE_SPEC = CAPICOVE_EVIDENCE_SPEC

IMPORT_VERSION = "capicove-local-import-v1"
CHUNKING_VERSION = "capicove-markdown-paragraphs-v1"
MAX_SOURCE_ROOTS = 16
MAX_SOURCE_BYTES = 16 * 1024 * 1024
# Keep one chunk's JSON text far below the Evidence Tool's 4 KiB section limit.
MAX_CHUNK_JSON_BYTES = 2400
_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "source_roots",
        "index_root",
        "corpus_ref",
        "retrieval_profile_ref",
        "max_source_bytes",
    }
)
_ALIAS = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_EVIDENCE = re.compile(r"^ev-[0-9a-f]{64}$")
_MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})
_IMAGE_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)
_SUPPORTED_SECTIONS = frozenset({"metadata", "lineage"})
LEDGER_VERSION = 2
# Units written per index transaction; bounds memory independently of corpus size.
_WRITE_BATCH_UNITS = 512


class RetrievalError(ValueError):
    """Report a stable, path-free operator failure for import, build, or startup."""


class RetrievalDependencyError(ImportError):
    """Report that this Python's SQLite build lacks the required FTS5 module."""


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    """Validated, startup-owned retrieval authority."""

    workspace_id: str
    workspace_root: Path
    profile: str
    source_roots: Mapping[str, Path]
    index_root: Path
    corpus_ref: str
    retrieval_profile_ref: str
    max_source_bytes: int


def retrieval_settings(config: WorkspaceHostConfiguration) -> RetrievalSettings:
    """Validate the closed version-1 retrieval capability configuration."""
    if os.name != "posix":
        raise RetrievalError("retrieval requires POSIX no-follow file APIs in this release")
    from capicove.retrieval._adapters.sqlite_fts import fts5_available

    if not fts5_available():
        raise RetrievalDependencyError("retrieval requires SQLite with FTS5")
    raw = config.capabilities.get("retrieval")
    if raw is None:
        raise RetrievalError("capabilities.retrieval is not configured")
    if set(raw) != _CONFIG_FIELDS:
        raise RetrievalError("capabilities.retrieval requires exactly the documented fields")
    if raw["schema_version"] != 1:
        raise RetrievalError("capabilities.retrieval schema_version must be 1")
    workspace_root = config.workspace_root
    index_root = _absolute_path(raw["index_root"], "index_root")
    if index_root == workspace_root or not index_root.is_relative_to(workspace_root):
        raise RetrievalError("index_root must be inside workspace_root")
    _reject_symlink_components(index_root, "index_root")
    if index_root.exists():
        _require_private_directory(index_root, "index_root")
    roots = raw["source_roots"]
    if not isinstance(roots, Mapping) or not 1 <= len(roots) <= MAX_SOURCE_ROOTS:
        raise RetrievalError(f"source_roots must map 1 to {MAX_SOURCE_ROOTS} aliases")
    source_roots: dict[str, Path] = {}
    for alias, value in roots.items():
        if _ALIAS.fullmatch(alias) is None:
            raise RetrievalError("source_roots aliases must be lowercase identifiers")
        path = _absolute_path(value, "source_roots")
        _reject_symlink_components(path, "source_roots")
        if not path.is_dir():
            raise RetrievalError("source_roots must name existing local directories")
        if path.is_relative_to(index_root) or index_root.is_relative_to(path):
            raise RetrievalError("source_roots must be disjoint from index_root")
        source_roots[alias] = path
    if len(set(source_roots.values())) != len(source_roots):
        raise RetrievalError("source_roots must be unique")
    corpus_ref = raw["corpus_ref"]
    if not isinstance(corpus_ref, str) or not corpus_ref.strip() or len(corpus_ref) > 256:
        raise RetrievalError("corpus_ref must be a nonblank string of at most 256 characters")
    if raw["retrieval_profile_ref"] != "local-default":
        raise RetrievalError("retrieval_profile_ref must be local-default")
    limit = raw["max_source_bytes"]
    if type(limit) is not int or not 1 <= limit <= MAX_SOURCE_BYTES:
        raise RetrievalError("max_source_bytes must be an integer from 1 through 16 MiB")
    return RetrievalSettings(
        workspace_id=config.workspace_id,
        workspace_root=workspace_root,
        profile=config.profile,
        source_roots=source_roots,
        index_root=index_root,
        corpus_ref=corpus_ref,
        retrieval_profile_ref="local-default",
        max_source_bytes=limit,
    )


def import_source(
    config: WorkspaceHostConfiguration, *, alias: str, relative_path: str
) -> dict[str, object]:
    """Record one verified source snapshot identity without modifying the source."""
    settings = _writable_settings(config)
    root = _configured_root(settings, alias)
    parts = _relative_parts(relative_path)
    payload = _read_source(root, parts, settings.max_source_bytes)
    media_type = _media_type(parts[-1], payload)
    digest = hashlib.sha256(payload).hexdigest()
    source_ref = "/".join((alias, *parts))
    _ensure_index_root(settings)
    with _index_lock(settings.index_root), _import_ledger(settings.index_root) as ledger:
        latest = _latest_event(ledger, source_ref)
        changed = latest is None or latest["event"] != "import" or latest["sha256"] != digest
        if changed:
            _append_event(
                ledger,
                "import",
                source_ref,
                alias,
                parts,
                sha256=digest,
                size_bytes=len(payload),
                media_type=media_type,
            )
    return {
        "source_ref": source_ref,
        "sha256": digest,
        "size_bytes": len(payload),
        "media_type": media_type,
        "import_version": IMPORT_VERSION,
        "changed": changed,
    }


def forget_source(
    config: WorkspaceHostConfiguration, *, alias: str, relative_path: str
) -> dict[str, object]:
    """Retire one live source from the inventory; sources and generations are untouched."""
    settings = _writable_settings(config)
    _configured_root(settings, alias)
    parts = _relative_parts(relative_path)
    source_ref = "/".join((alias, *parts))
    if not (settings.index_root / "imports.sqlite3").is_file():
        return {"source_ref": source_ref, "forgotten": False}
    _ensure_index_root(settings)
    with _index_lock(settings.index_root), _import_ledger(settings.index_root) as ledger:
        latest = _latest_event(ledger, source_ref)
        forgotten = latest is not None and latest["event"] == "import"
        if forgotten:
            _append_event(ledger, "retire", source_ref, alias, parts)
    return {"source_ref": source_ref, "forgotten": forgotten}


@dataclass(frozen=True, slots=True)
class _PlannedSource:
    """Hold one live source's identity and its chunk ranges; never its bytes."""

    source_ref: str
    alias: str
    relative_path: str
    sha256: str
    size_bytes: int
    media_type: str
    import_version: str
    source_key: str
    # A digest of the chunk ranges, never the ranges: memory must not grow with the corpus.
    chunk_digest: str


def build_index(config: WorkspaceHostConfiguration) -> dict[str, object]:
    """Verify every live source, then build, validate, and activate one generation.

    Generation identity comes from the ledger inventory. Sources are then read one
    at a time twice: once to verify and chunk them, once to write bounded batches.
    """
    from capicove.retrieval._adapters.sqlite_fts import (
        INDEX_FORMAT_VERSION,
        Fts5TextBackend,
        IndexedSource,
    )
    from capicove.retrieval._adapters.sqlite_generations import SqliteTextIndexGenerationCatalog
    from capicove.retrieval.generation_service import TextIndexGenerationService
    from capicove.retrieval.generations import (
        TextIndexGenerationCatalogError,
        TextIndexGenerationManifest,
        TextIndexGenerationState,
    )

    settings = _writable_settings(config)
    if not (settings.index_root / "imports.sqlite3").is_file():
        raise RetrievalError("no sources have been imported")
    with _index_lock(settings.index_root):
        with _import_ledger(settings.index_root) as ledger:
            if ledger.execute("SELECT count(*) FROM ledger_events").fetchone()[0] == 0:
                raise RetrievalError("no sources have been imported")
            inventory = _live_inventory(ledger)
        identity = hashlib.sha256()
        for value in (
            "capicove.local-generation.v1",
            settings.workspace_id,
            settings.corpus_ref,
            INDEX_FORMAT_VERSION,
            CHUNKING_VERSION,
        ):
            identity.update(value.encode() + b"\0")
        for row in inventory:
            identity.update(
                json.dumps(
                    [row["source_ref"], row["sha256"], row["media_type"]], ensure_ascii=False
                ).encode()
                + b"\0"
            )
        generation_id = f"local-{identity.hexdigest()}"
        fingerprint = ProcessingFingerprint.from_manifest(
            {
                "import_version": IMPORT_VERSION,
                "chunking_version": CHUNKING_VERSION,
                "index_format_version": INDEX_FORMAT_VERSION,
            }
        )

        # Pass 1: verify and chunk one source at a time, keeping only digests and ranges.
        planned: list[_PlannedSource] = []
        # Byte-identical to the previous release's "\n".join / "\0".join digests.
        unit_digest, content_digest, unit_count = _JoinedDigest(b"\n"), _JoinedDigest(b"\0"), 0
        for row in inventory:
            source = _plan_source(row)
            if source.media_type != "text/markdown":
                _verified_payload(settings, source)
                planned.append(source)
                continue
            text = _verified_text(settings, source)
            chunks = chunk_markdown(text)
            source = replace(source, chunk_digest=_ranges_digest(chunks))
            planned.append(source)
            for start, end in chunks:
                unit_digest.update(_unit_id(generation_id, source, start, end).encode())
                content_digest.update(text[start:end].encode("utf-8"))
                unit_count += 1
            del text

        backend = Fts5TextBackend(settings.index_root)
        lifecycle = backend.lifecycle
        manifest = TextIndexGenerationManifest(
            generation_id=generation_id,
            workspace_ref=settings.workspace_id,
            corpus_refs=(settings.corpus_ref,),
            searchable_unit_manifest_sha256=unit_digest.hexdigest(),
            content_sha256=content_digest.hexdigest(),
            mapping_sha256=lifecycle.mapping_sha256(()),
            embedding_spaces=(),
            retriever_id=backend.provider_id,
            index_format_version=INDEX_FORMAT_VERSION,
            build_processing_fingerprint=fingerprint,
            expected_unit_count=unit_count,
        )
        sources = tuple(
            IndexedSource(
                source_key=source.source_key,
                source_ref=source.source_ref,
                sha256=source.sha256,
                size_bytes=source.size_bytes,
                media_type=source.media_type,
                import_version=source.import_version,
                evidence_id=_opaque("ev", generation_id, "source", source.source_key),
            )
            for source in planned
        )
        catalog = SqliteTextIndexGenerationCatalog(settings.index_root / "catalog.sqlite3")
        service = TextIndexGenerationService(lifecycle, catalog)
        try:
            service.reconcile_active(
                workspace_ref=settings.workspace_id, retriever_id=backend.provider_id
            )
        except (LookupError, TextIndexGenerationCatalogError):
            pass  # No prior activation, or nothing to recover.
        try:
            record = service.create(manifest)
            if record.state is TextIndexGenerationState.BUILDING:
                lifecycle.create(manifest)  # Discard any partial output from an interrupted build.
                backend.writer.add_sources(index_generation_id=generation_id, sources=sources)
                # Pass 2: re-verify each source and write its units in bounded batches.
                for source in planned:
                    if source.media_type == "text/markdown":
                        _write_units(backend, settings, generation_id, fingerprint, source)
                record = service.validate(generation_id)
            if record.state is TextIndexGenerationState.CORRUPT:
                raise RetrievalError("the generation failed validation; inspect the sources")
            service.activate(generation_id)
        except (TextIndexGenerationCatalogError, OSError, sqlite3.Error) as error:
            raise RetrievalError("the index build could not be completed") from error
    return {
        "generation_id": generation_id,
        "activated": True,
        "sources": len(sources),
        "text_units": unit_count,
        "source_evidence": [
            {
                "source_ref": item.source_ref,
                "media_type": item.media_type,
                "evidence_id": item.evidence_id,
            }
            for item in sources
        ],
    }


class _JoinedDigest:
    """Hash items as ``sha256(separator.join(items))`` without holding the items."""

    def __init__(self, separator: bytes) -> None:
        self._digest = hashlib.sha256()
        self._separator = separator
        self._first = True

    def update(self, item: bytes) -> None:
        if not self._first:
            self._digest.update(self._separator)
        self._digest.update(item)
        self._first = False

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _ranges_digest(chunks: tuple[tuple[int, int], ...]) -> str:
    digest = hashlib.sha256()
    for start, end in chunks:
        digest.update(f"{start}:{end};".encode())
    return digest.hexdigest()


def _plan_source(row: sqlite3.Row) -> _PlannedSource:
    source_ref, digest = str(row["source_ref"]), str(row["sha256"])
    return _PlannedSource(
        source_ref=source_ref,
        alias=str(row["alias"]),
        relative_path=str(row["relative_path"]),
        sha256=digest,
        size_bytes=int(row["size_bytes"]),
        media_type=str(row["media_type"]),
        import_version=str(row["import_version"]),
        source_key=hashlib.sha256(f"{source_ref}\0{digest}".encode()).hexdigest(),
        chunk_digest="",
    )


def _verified_payload(settings: RetrievalSettings, source: _PlannedSource) -> bytes:
    root = settings.source_roots.get(source.alias)
    if root is None:
        raise RetrievalError(f"imported source root alias {source.alias!r} is not configured")
    # A missing or unreadable live source is a change, even when its import was empty.
    try:
        payload = _read_source(
            root, _relative_parts(source.relative_path), settings.max_source_bytes
        )
    except RetrievalError:
        raise RetrievalError(
            f"source {source.source_ref} changed since import; re-import it"
        ) from None
    if hashlib.sha256(payload).hexdigest() != source.sha256:
        raise RetrievalError(f"source {source.source_ref} changed since import; re-import it")
    return payload


def _verified_text(settings: RetrievalSettings, source: _PlannedSource) -> str:
    return _verified_payload(settings, source).decode("utf-8")


def _unit_id(generation_id: str, source: _PlannedSource, start: int, end: int) -> str:
    return _opaque("unit", generation_id, f"{source.source_key}:{start}:{end}")


def _write_units(
    backend: Fts5TextBackend,
    settings: RetrievalSettings,
    generation_id: str,
    fingerprint: ProcessingFingerprint,
    source: _PlannedSource,
) -> None:
    from capicove.retrieval.models import TextIndexEntry

    text = _verified_text(settings, source)
    chunks = chunk_markdown(text)
    if _ranges_digest(chunks) != source.chunk_digest:
        raise RetrievalError(f"source {source.source_ref} changed during the build")
    for offset in range(0, len(chunks), _WRITE_BATCH_UNITS):
        batch = []
        for start, end in chunks[offset : offset + _WRITE_BATCH_UNITS]:
            unit_key = f"{source.source_key}:{start}:{end}"
            batch.append(
                TextIndexEntry(
                    unit=SearchableUnit(
                        unit_id=_unit_id(generation_id, source, start, end),
                        unit_type=TargetType.TEXT_CHUNK,
                        source_hash=source.sha256,
                        artifact_ref=f"artifact:sha256:{source.sha256}",
                        evidence_ref=_opaque("ev", generation_id, "unit", unit_key),
                        representation_refs=(f"representation:markdown:{source.source_key}",),
                        processing_run_ref=f"build:{generation_id}",
                        processing_version=CHUNKING_VERSION,
                        processing_fingerprint=fingerprint,
                        text_locator=TextChunkLocator(
                            TextOrigin.DOCUMENT, CharacterRange(start, end)
                        ),
                    ),
                    text=text[start:end],
                    corpus_refs=(settings.corpus_ref,),
                    media_type="text/markdown",
                )
            )
        backend.writer.add(index_generation_id=generation_id, entries=tuple(batch))


@dataclass(frozen=True, slots=True)
class LocalRetrievalRuntime:
    """Expose startup-bound canonical Search and Evidence handlers."""

    search_handler: ToolHandler
    evidence_handler: ToolHandler


def open_local_retrieval(
    config: WorkspaceHostConfiguration, settings: RetrievalSettings | None = None
) -> LocalRetrievalRuntime:
    """Compose read-only handlers; this never creates storage or builds an index."""
    resolved = settings or retrieval_settings(config)
    return LocalRetrievalRuntime(
        SearchToolHandler(_LocalSearchOperation(resolved)),
        EvidenceToolHandler(_LocalEvidenceReader(resolved)),
    )


def chunk_markdown(text: str) -> tuple[tuple[int, int], ...]:
    """Split Markdown into deterministic, bounded paragraph ranges of exact source text."""
    paragraphs: list[tuple[int, int]] = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=[ \t]*\n[ \t]*\n|\s*\Z)", text, flags=re.DOTALL):
        paragraphs.extend(_split_oversized(text, match.start(), match.end()))
    chunks: list[tuple[int, int]] = []
    for start, end in paragraphs:
        if chunks and _json_bytes(text[chunks[-1][0] : end]) <= MAX_CHUNK_JSON_BYTES:
            chunks[-1] = (chunks[-1][0], end)
        else:
            chunks.append((start, end))
    return tuple(chunks)


class _LocalSearchOperation:
    def __init__(self, settings: RetrievalSettings) -> None:
        self._settings = settings

    def search(self, request: SearchRequest) -> SearchResult:
        from capicove.retrieval._adapters.sqlite_fts import Fts5TextBackend
        from capicove.retrieval.text_profiles import (
            StaticTextRetrievalProfileRegistry,
            TextRetrievalProfile,
        )
        from capicove.retrieval.text_search import TextSearchError, TextSearchService

        catalog = self._settings.index_root / "catalog.sqlite3"
        if not _is_private_file(catalog):
            raise TextSearchError("index_generation_unavailable")
        from capicove.retrieval._adapters.sqlite_generations import (
            SqliteTextIndexGenerationCatalog,
        )

        service = TextSearchService(
            backend=Fts5TextBackend(self._settings.index_root),
            catalog=SqliteTextIndexGenerationCatalog(catalog),
            profiles=StaticTextRetrievalProfileRegistry(
                (
                    TextRetrievalProfile(
                        profile_id="local-default",
                        lexical_weight=1.0,
                        dense_weight=0.0,
                        lexical_required=True,
                        dense_required=False,
                        lexical_candidate_count=100,
                        dense_candidate_count=1,
                    ),
                )
            ),
            query_embedder=_NoQueryEmbedder(),
            query_id_factory=lambda: f"query-{uuid.uuid4().hex}",
        )
        return service.search(request)


class _NoQueryEmbedder:
    provider_id = "none"

    def embed_query(self, **_: object) -> tuple[float, ...]:
        raise RuntimeError("the local base profile has no dense query encoder")


class _LocalEvidenceReader:
    def __init__(self, settings: RetrievalSettings) -> None:
        self._settings = settings

    def read(
        self, *, context: InvocationContext, evidence_id: str, sections: tuple[str, ...]
    ) -> Mapping[str, object]:
        from capicove.retrieval._adapters.sqlite_fts import Fts5TextBackend
        from capicove.retrieval._adapters.sqlite_generations import (
            SqliteTextIndexGenerationCatalog,
        )
        from capicove.retrieval.generations import TextIndexGenerationState

        if context.workspace_id != self._settings.workspace_id:
            raise PermissionError("evidence belongs to another workspace")
        if _EVIDENCE.fullmatch(evidence_id) is None:
            raise LookupError("evidence is unavailable")
        if set(sections) - _SUPPORTED_SECTIONS:
            raise LookupError("unsupported evidence section")
        catalog_path = self._settings.index_root / "catalog.sqlite3"
        if not _is_private_file(catalog_path):
            raise LookupError("no active index generation")
        backend = Fts5TextBackend(self._settings.index_root)
        pin = backend.lifecycle.resolve_active()
        record = SqliteTextIndexGenerationCatalog(catalog_path).get(pin.generation_id)
        if (
            record is None
            or record.state is not TextIndexGenerationState.ACTIVE
            or record.manifest.manifest_sha256 != pin.manifest_sha256
            or record.physical_index_ref != pin.physical_index_ref
            or record.manifest.workspace_ref != self._settings.workspace_id
            or self._settings.corpus_ref not in record.manifest.corpus_refs
        ):
            raise LookupError("active generation is not consistent with the catalog")
        evidence = backend.read_evidence(pin, evidence_id)
        source = evidence.source
        result: dict[str, object] = {}
        if "metadata" in sections:
            metadata: dict[str, object] = {
                "kind": "source" if evidence.text is None else "text_chunk",
                "source_ref": source.source_ref,
                "media_type": source.media_type,
                "source_sha256": source.sha256,
                "size_bytes": source.size_bytes,
                "corpus_ref": self._settings.corpus_ref,
                "generation_id": evidence.generation_id,
            }
            if evidence.text is not None:
                metadata.update(
                    {
                        "start_char": evidence.start_char,
                        "end_char": evidence.end_char,
                        "text": evidence.text,
                    }
                )
            result["metadata"] = metadata
        if "lineage" in sections:
            result["lineage"] = {
                "generation_id": evidence.generation_id,
                "manifest_sha256": evidence.manifest_sha256,
                "source_sha256": source.sha256,
                "import_version": source.import_version,
                "chunking_version": CHUNKING_VERSION,
            }
        return result


def _writable_settings(config: WorkspaceHostConfiguration) -> RetrievalSettings:
    if config.profile not in {"write", "admin"}:
        raise RetrievalError("import and build require the write profile")
    settings = retrieval_settings(config)
    _require_private_directory(settings.workspace_root, "workspace_root")
    return settings


def _absolute_path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value or "\0" in value:
        raise RetrievalError(f"{field} must contain absolute local paths")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise RetrievalError(f"{field} must contain absolute local paths")
    return path


def _reject_symlink_components(path: Path, field: str) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise RetrievalError(f"{field} must not contain symlinks")


def _require_private_directory(path: Path, field: str) -> None:
    try:
        metadata = path.lstat()
    except OSError:
        raise RetrievalError(f"{field} must be an existing directory") from None
    if not stat.S_ISDIR(metadata.st_mode):
        raise RetrievalError(f"{field} must be a directory")
    if metadata.st_mode & 0o077 or metadata.st_uid != os.geteuid():
        raise RetrievalError(f"{field} must be owner-private")


def _ensure_index_root(settings: RetrievalSettings) -> None:
    missing: list[Path] = []
    for component in (settings.index_root, *settings.index_root.parents):
        if component == settings.workspace_root:
            break
        if not component.exists():
            missing.append(component)
    for component in reversed(missing):
        component.mkdir(mode=0o700)
    _reject_symlink_components(settings.index_root, "index_root")
    _require_private_directory(settings.index_root, "index_root")


@contextmanager
def _import_ledger(index_root: Path) -> Iterator[sqlite3.Connection]:
    """Open the version-2 event ledger, migrating a version-1 ledger in one transaction."""
    path = index_root / "imports.sqlite3"
    if not path.exists():
        os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600))
    if not _is_private_file(path):
        raise RetrievalError("the import ledger must be a private regular file")
    # Autocommit mode: every schema change below runs inside an explicit transaction.
    with closing(sqlite3.connect(path, isolation_level=None)) as connection:
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                _require_ledger_version(connection)
                yield connection
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            connection.execute("COMMIT")
        except sqlite3.Error as error:
            raise RetrievalError("the import ledger could not be updated") from error


def _require_ledger_version(connection: sqlite3.Connection) -> None:
    objects = {
        row["name"]: row["type"]
        for row in connection.execute("SELECT name, type FROM sqlite_master")
    }
    if "ledger_meta" in objects:
        row = connection.execute(
            "SELECT value FROM ledger_meta WHERE key = 'ledger_version'"
        ).fetchone()
        if row is None or row["value"] != str(LEDGER_VERSION):
            raise RetrievalError("the import ledger version is not supported by this release")
        return
    _create_ledger_tables(connection)
    if objects.get("imports") == "table":
        connection.execute(
            "INSERT INTO ledger_events (event, source_ref, alias, relative_path, sha256, "
            "size_bytes, media_type, import_version, recorded_at) "
            "SELECT 'import', source_ref, alias, relative_path, sha256, size_bytes, "
            "media_type, import_version, imported_at FROM imports ORDER BY sequence"
        )
        connection.execute("DROP TABLE imports")
    elif "imports" in objects:
        raise RetrievalError("the import ledger has an unknown layout")
    _create_sentinel_view(connection)
    connection.execute(
        "INSERT INTO ledger_meta (key, value) VALUES ('ledger_version', ?)", (str(LEDGER_VERSION),)
    )


def _create_ledger_tables(connection: sqlite3.Connection) -> None:
    connection.execute("CREATE TABLE ledger_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(
        "CREATE TABLE ledger_events ("
        "sequence INTEGER PRIMARY KEY AUTOINCREMENT, "
        "event TEXT NOT NULL CHECK (event IN ('import', 'retire')), "
        "source_ref TEXT NOT NULL, alias TEXT NOT NULL, relative_path TEXT NOT NULL, "
        "sha256 TEXT, size_bytes INTEGER, media_type TEXT, import_version TEXT, "
        "recorded_at TEXT NOT NULL)"
    )


def _create_sentinel_view(connection: sqlite3.Connection) -> None:
    # A version-1 reader selects and inserts columns this view lacks, so it fails closed.
    connection.execute(
        "CREATE VIEW imports AS SELECT 2 AS import_ledger_version_2_requires_newer_capicove"
    )


def _latest_event(connection: sqlite3.Connection, source_ref: str) -> sqlite3.Row | None:
    row: sqlite3.Row | None = connection.execute(
        "SELECT * FROM ledger_events WHERE source_ref = ? ORDER BY sequence DESC LIMIT 1",
        (source_ref,),
    ).fetchone()
    return row


def _append_event(
    connection: sqlite3.Connection,
    event: str,
    source_ref: str,
    alias: str,
    parts: tuple[str, ...],
    *,
    sha256: str | None = None,
    size_bytes: int | None = None,
    media_type: str | None = None,
) -> None:
    connection.execute(
        "INSERT INTO ledger_events (event, source_ref, alias, relative_path, sha256, "
        "size_bytes, media_type, import_version, recorded_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            event,
            source_ref,
            alias,
            "/".join(parts),
            sha256,
            size_bytes,
            media_type,
            IMPORT_VERSION if event == "import" else None,
            datetime.now(UTC).isoformat(),
        ),
    )


def _live_inventory(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        connection.execute(
            "SELECT * FROM ledger_events AS e WHERE event = 'import' AND sequence = ("
            "SELECT max(sequence) FROM ledger_events WHERE source_ref = e.source_ref) "
            "ORDER BY source_ref"
        ).fetchall()
    )


@contextmanager
def _index_lock(index_root: Path) -> Iterator[None]:
    """Serialize import, forget, and build across processes for one index root."""
    import fcntl  # POSIX-only; retrieval_settings has already rejected other platforms.

    try:
        # Non-blocking open: a FIFO planted at the lock path must fail, not hang.
        descriptor = os.open(
            index_root / ".build.lock",
            os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
            0o600,
        )
    except OSError:
        raise RetrievalError("the index lock could not be opened as a regular file") from None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RetrievalError("the index lock must be a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _configured_root(settings: RetrievalSettings, alias: str) -> Path:
    root = settings.source_roots.get(alias)
    if root is None:
        raise RetrievalError(f"source root alias {alias!r} is not configured")
    return root


def _relative_parts(relative_path: str) -> tuple[str, ...]:
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or len(relative_path.encode("utf-8")) > 1024
        or "\0" in relative_path
        or "\\" in relative_path
    ):
        raise RetrievalError("relative path must be a nonempty POSIX path below the root")
    path = PurePosixPath(relative_path)
    parts = path.parts
    if (
        path.is_absolute()
        or not parts
        or len(parts) > 32
        or any(part in {".", ".."} for part in parts)
        or "/".join(parts) != relative_path
    ):
        raise RetrievalError("relative path must be a normalized path below the root")
    return parts


def _read_source(root: Path, parts: tuple[str, ...], limit: int) -> bytes:
    """Read one regular file through pinned no-follow directory handles."""
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        directory = os.open(root.anchor, directory_flags)
    except OSError:
        raise RetrievalError("source root is unavailable") from None
    try:
        for component in (*root.parts[1:], *parts[:-1]):
            next_directory = os.open(component, directory_flags, dir_fd=directory)
            os.close(directory)
            directory = next_directory
        descriptor = os.open(
            parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
    except OSError:
        raise RetrievalError("source must be an existing file without symlinks") from None
    finally:
        os.close(directory)
    with os.fdopen(descriptor, "rb") as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise RetrievalError("source must be a regular file")
        if before.st_size > limit:
            raise RetrievalError("source exceeds max_source_bytes")
        payload = source.read(limit + 1)
        after = os.fstat(source.fileno())
    if len(payload) > limit:
        raise RetrievalError("source exceeds max_source_bytes")
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if (
        any(getattr(before, name) != getattr(after, name) for name in identity)
        or len(payload) != after.st_size
    ):
        raise RetrievalError("source changed while it was read")
    return payload


def _media_type(name: str, payload: bytes) -> str:
    for signature, media_type in _IMAGE_SIGNATURES:
        if payload.startswith(signature):
            return media_type
    if payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        return "image/webp"
    if PurePosixPath(name).suffix.lower() in _MARKDOWN_SUFFIXES:
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            raise RetrievalError("Markdown sources must be UTF-8") from None
        if "\0" in text:
            raise RetrievalError("Markdown sources must not contain NUL characters")
        return "text/markdown"
    raise RetrievalError("source media type is not supported; import Markdown or images")


def _split_oversized(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Split one paragraph into bounded pieces, preferring whitespace boundaries."""
    pieces: list[tuple[int, int]] = []
    while start < end:
        # Each character encodes to at least one byte, so only a short tail can fit.
        if end - start <= MAX_CHUNK_JSON_BYTES and (
            _json_bytes(text[start:end]) <= MAX_CHUNK_JSON_BYTES
        ):
            stop = end
        else:
            # The JSON width is monotonic in the slice length, so binary search the
            # longest bounded prefix with the C encoder instead of per-character calls.
            low, high = start + 1, min(end, start + MAX_CHUNK_JSON_BYTES)
            while low < high:
                middle = (low + high + 1) // 2
                if _json_bytes(text[start:middle]) <= MAX_CHUNK_JSON_BYTES:
                    low = middle
                else:
                    high = middle - 1
            stop = low
            space = max(text.rfind(" ", start, stop), text.rfind("\n", start, stop))
            if space > start + (stop - start) // 2:
                stop = space
        segment = text[start:stop]
        stripped = segment.strip()
        if stripped:
            offset = start + (len(segment) - len(segment.lstrip()))
            pieces.append((offset, offset + len(stripped)))
        start = stop
    return pieces


def _json_bytes(value: str) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def _opaque(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(("capicove.local-retrieval.v1", *parts)).encode())
    return f"{prefix}-{digest.hexdigest()}"


def _is_private_file(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(metadata.st_mode) and not metadata.st_mode & 0o077


__all__ = [
    "LOCAL_EVIDENCE_SPEC",
    "LOCAL_SEARCH_SPEC",
    "LocalRetrievalRuntime",
    "RetrievalDependencyError",
    "RetrievalError",
    "RetrievalSettings",
    "build_index",
    "chunk_markdown",
    "forget_source",
    "import_source",
    "open_local_retrieval",
    "retrieval_settings",
]
