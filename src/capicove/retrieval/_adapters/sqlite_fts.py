"""Private SQLite FTS5 backend for offline workspace text generations (ADR-0024).

Each generation is one immutable SQLite database under ``generations/<id>``. It is
written in a private staging directory, validated, renamed into place, and only
then published through an atomically replaced ``active.json`` alias.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path

from capicove.contracts import (
    CharacterRange,
    EmbeddingSpaceRef,
    SearchFilters,
    TextChunkLocator,
    TextOrigin,
    TextQuery,
)
from capicove.retrieval.generations import (
    PinnedTextIndexGeneration,
    TextIndexGenerationManifest,
    TextIndexValidationReport,
)
from capicove.retrieval.models import TextCandidate, TextIndexEntry
from capicove.retrieval.ports import (
    DenseCandidateRetriever,
    LexicalCandidateRetriever,
    TextIndexGenerationLifecycle,
    TextIndexWriter,
)


PROVIDER_ID = "sqlite-fts5-local-v1"
INDEX_FORMAT_VERSION = "capicove-local-fts5-v1"
SCORE_SEMANTICS = "fts5-bm25-negated"
_ACTIVE_ALIAS = "active.json"
_DATABASE = "index.sqlite3"
_MAX_ALIAS_BYTES = 4096
_MAX_QUERY_TOKENS = 32
_GENERATION = re.compile(r"^local-[0-9a-f]{64}$")
_SCHEMA = (
    "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    """CREATE TABLE sources (
        source_key TEXT PRIMARY KEY,
        source_ref TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        media_type TEXT NOT NULL,
        import_version TEXT NOT NULL,
        evidence_id TEXT NOT NULL UNIQUE
    )""",
    """CREATE TABLE units (
        rowid INTEGER PRIMARY KEY,
        unit_id TEXT NOT NULL UNIQUE,
        evidence_id TEXT NOT NULL UNIQUE,
        source_key TEXT NOT NULL REFERENCES sources(source_key),
        artifact_ref TEXT NOT NULL,
        corpus_ref TEXT NOT NULL,
        media_type TEXT NOT NULL,
        start_char INTEGER NOT NULL,
        end_char INTEGER NOT NULL,
        text TEXT NOT NULL,
        text_sha256 TEXT NOT NULL
    )""",
    "CREATE VIRTUAL TABLE units_fts USING fts5(text, tokenize='unicode61 remove_diacritics 2')",
)


class LocalIndexError(RuntimeError):
    """Report a stable local index failure without raw SQL or paths."""


@dataclass(frozen=True, slots=True)
class IndexedSource:
    """Describe one verified imported source recorded in a generation."""

    source_key: str
    source_ref: str
    sha256: str
    size_bytes: int
    media_type: str
    import_version: str
    evidence_id: str


@dataclass(frozen=True, slots=True)
class LocalEvidence:
    """Return one verified generation-bound evidence record."""

    evidence_id: str
    generation_id: str
    manifest_sha256: str
    source: IndexedSource
    corpus_ref: str | None = None
    start_char: int | None = None
    end_char: int | None = None
    text: str | None = None


def fts5_available() -> bool:
    """Return whether this Python's SQLite build provides FTS5."""
    try:
        with closing(sqlite3.connect(":memory:")) as connection:
            connection.execute("CREATE VIRTUAL TABLE probe USING fts5(text)")
    except sqlite3.Error:
        return False
    return True


def physical_index_ref(generation_id: str) -> str:
    """Return the stable physical reference recorded for one generation."""
    return f"fts5:{generation_id}"


class Fts5GenerationStore:
    """Own the on-disk layout of every generation below one private index root."""

    def __init__(self, index_root: Path) -> None:
        self._root = index_root
        self._generations = index_root / "generations"

    def staging_database(self, generation_id: str) -> Path:
        return self._generations / f".staging-{_require_generation(generation_id)}" / _DATABASE

    def final_database(self, generation_id: str) -> Path:
        return self._generations / _require_generation(generation_id) / _DATABASE

    @contextmanager
    def read(self, generation_id: str) -> Iterator[sqlite3.Connection]:
        """Open one published generation read-only."""
        path = self.final_database(generation_id)
        if not _is_private_regular_file(path):
            raise LookupError("index generation unavailable")
        try:
            connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
        except sqlite3.Error:
            raise LookupError("index generation unavailable") from None
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def write(self, generation_id: str) -> Iterator[sqlite3.Connection]:
        """Open one staging generation inside a single transaction."""
        path = self.staging_database(generation_id)
        if not _is_private_regular_file(path):
            raise LocalIndexError("staging generation is unavailable")
        connection = sqlite3.connect(path)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def read_alias(self) -> PinnedTextIndexGeneration:
        """Read the active alias without following a replaced link."""
        try:
            descriptor = os.open(
                self._root / _ACTIVE_ALIAS, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            )
        except OSError:
            raise LookupError("no active index generation") from None
        with os.fdopen(descriptor, "rb") as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise LookupError("active alias is not a regular file")
            payload = source.read(_MAX_ALIAS_BYTES + 1)
        try:
            value = json.loads(payload.decode("utf-8"))
            if len(payload) > _MAX_ALIAS_BYTES or set(value) != {
                "generation_id",
                "manifest_sha256",
                "physical_index_ref",
            }:
                raise ValueError("unexpected alias fields")
            pin = PinnedTextIndexGeneration(
                value["generation_id"], value["manifest_sha256"], value["physical_index_ref"]
            )
            _require_generation(pin.generation_id)
        except (UnicodeError, ValueError, TypeError):
            raise LookupError("active alias is invalid") from None
        return pin

    def write_alias(self, pin: PinnedTextIndexGeneration) -> None:
        """Replace the active alias atomically and durably."""
        payload = json.dumps(
            {
                "generation_id": pin.generation_id,
                "manifest_sha256": pin.manifest_sha256,
                "physical_index_ref": pin.physical_index_ref,
            },
            sort_keys=True,
        ).encode("utf-8")
        temporary = self._root / f".{_ACTIVE_ALIAS}.{os.getpid()}.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as target:
                target.write(payload)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, self._root / _ACTIVE_ALIAS)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        _fsync_directory(self._root)

    def reset_staging(self, generation_id: str) -> None:
        """Discard partial output for a building generation and create empty storage."""
        staging = self.staging_database(generation_id).parent
        final = self.final_database(generation_id).parent
        self._generations.mkdir(mode=0o700, exist_ok=True)
        for directory in (staging, final):
            if directory.is_symlink():
                raise LocalIndexError("generation storage must not be a symlink")
            if directory.exists():
                shutil.rmtree(directory)
        staging.mkdir(mode=0o700)
        database = staging / _DATABASE
        os.close(os.open(database, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600))

    def publish_staging(self, generation_id: str) -> None:
        """Move one validated staging generation into its immutable final location."""
        staging = self.staging_database(generation_id)
        final = self.final_database(generation_id)
        with open(staging, "rb") as source:
            os.fsync(source.fileno())
        os.rename(staging.parent, final.parent)
        _fsync_directory(self._generations)


class Fts5GenerationLifecycle:
    """Implement the provider-neutral generation lifecycle over local FTS5 files."""

    provider_id: str = PROVIDER_ID

    def __init__(self, store: Fts5GenerationStore) -> None:
        self._store = store

    def mapping_sha256(self, embedding_spaces: tuple[EmbeddingSpaceRef, ...]) -> str:
        if embedding_spaces:
            raise ValueError("the local FTS5 backend has no dense embedding spaces")
        digest = hashlib.sha256(INDEX_FORMAT_VERSION.encode())
        for statement in _SCHEMA:
            digest.update(b"\0" + statement.encode())
        return digest.hexdigest()

    def create(self, manifest: TextIndexGenerationManifest) -> str:
        self._store.reset_staging(manifest.generation_id)
        with self._store.write(manifest.generation_id) as connection:
            for statement in _SCHEMA:
                connection.execute(statement)
            connection.executemany(
                "INSERT INTO meta (key, value) VALUES (?, ?)",
                (
                    ("generation_id", manifest.generation_id),
                    ("manifest_sha256", manifest.manifest_sha256),
                    ("index_format_version", INDEX_FORMAT_VERSION),
                ),
            )
        return physical_index_ref(manifest.generation_id)

    def validate(self, manifest: TextIndexGenerationManifest) -> TextIndexValidationReport:
        issues: list[str] = []
        with self._store.write(manifest.generation_id) as connection:
            units = int(connection.execute("SELECT count(*) FROM units").fetchone()[0])
            indexed = int(connection.execute("SELECT count(*) FROM units_fts").fetchone()[0])
            meta = dict(connection.execute("SELECT key, value FROM meta").fetchall())
            try:
                connection.execute("INSERT INTO units_fts(units_fts) VALUES ('integrity-check')")
            except sqlite3.DatabaseError:
                issues.append("fts_integrity_failed")
        if units != manifest.expected_unit_count:
            issues.append("document_count_mismatch")
        if indexed != units:
            issues.append("fts_row_count_mismatch")
        if meta.get("manifest_sha256") != manifest.manifest_sha256:
            issues.append("manifest_mismatch")
        if not issues:
            self._store.publish_staging(manifest.generation_id)
        return TextIndexValidationReport(
            generation_id=manifest.generation_id,
            manifest_sha256=manifest.manifest_sha256,
            physical_index_ref=physical_index_ref(manifest.generation_id),
            mapping_sha256=manifest.mapping_sha256,
            expected_unit_count=manifest.expected_unit_count,
            observed_unit_count=units,
            issue_codes=tuple(issues),
        )

    def activate(
        self,
        manifest: TextIndexGenerationManifest,
        validation: TextIndexValidationReport,
    ) -> PinnedTextIndexGeneration:
        if not validation.valid or validation.manifest_sha256 != manifest.manifest_sha256:
            raise LocalIndexError("activation requires a valid generation")
        with self._store.read(manifest.generation_id) as connection:
            meta = dict(connection.execute("SELECT key, value FROM meta").fetchall())
        if meta.get("manifest_sha256") != manifest.manifest_sha256:
            raise LocalIndexError("published generation does not match its manifest")
        pin = PinnedTextIndexGeneration(
            manifest.generation_id,
            manifest.manifest_sha256,
            physical_index_ref(manifest.generation_id),
        )
        self._store.write_alias(pin)
        return pin

    def resolve_active(self) -> PinnedTextIndexGeneration:
        pin = self._store.read_alias()
        if pin.physical_index_ref != physical_index_ref(pin.generation_id):
            raise LookupError("active alias is invalid")
        if not _is_private_regular_file(self._store.final_database(pin.generation_id)):
            raise LookupError("active generation is unavailable")
        return pin


class Fts5IndexWriter:
    """Append verified sources and text units to one staging generation."""

    provider_id: str = PROVIDER_ID

    def __init__(self, store: Fts5GenerationStore) -> None:
        self._store = store

    def add_sources(self, *, index_generation_id: str, sources: tuple[IndexedSource, ...]) -> None:
        with self._store.write(index_generation_id) as connection:
            connection.executemany(
                "INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        item.source_key,
                        item.source_ref,
                        item.sha256,
                        item.size_bytes,
                        item.media_type,
                        item.import_version,
                        item.evidence_id,
                    )
                    for item in sources
                ),
            )

    def add(self, *, index_generation_id: str, entries: tuple[TextIndexEntry, ...]) -> None:
        with self._store.write(index_generation_id) as connection:
            for entry in entries:
                unit = entry.unit
                locator = unit.text_locator
                if locator is None or len(entry.corpus_refs) != 1 or entry.media_type is None:
                    raise ValueError("local text units require one corpus and a media type")
                cursor = connection.execute(
                    "INSERT INTO units (unit_id, evidence_id, source_key, artifact_ref, "
                    "corpus_ref, media_type, start_char, end_char, text, text_sha256) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        unit.unit_id,
                        unit.evidence_ref,
                        _source_key(unit.representation_refs),
                        unit.artifact_ref,
                        entry.corpus_refs[0],
                        entry.media_type,
                        locator.character_range.start_char,
                        locator.character_range.end_char,
                        entry.text,
                        hashlib.sha256(entry.text.encode("utf-8")).hexdigest(),
                    ),
                )
                connection.execute(
                    "INSERT INTO units_fts (rowid, text) VALUES (?, ?)",
                    (cursor.lastrowid, entry.text),
                )


class Fts5LexicalRetriever:
    """Rank one pinned generation with FTS5 BM25 and canonical filters."""

    provider_id: str = PROVIDER_ID

    def __init__(self, store: Fts5GenerationStore) -> None:
        self._store = store

    def retrieve(
        self,
        *,
        query: TextQuery,
        filters: SearchFilters,
        generation: PinnedTextIndexGeneration,
        top_k: int,
    ) -> tuple[TextCandidate, ...]:
        tokens = re.findall(r"\w+", query.text)[:_MAX_QUERY_TOKENS]
        if not tokens or filters.time_range is not None:
            return ()
        if filters.text_origins and TextOrigin.DOCUMENT not in filters.text_origins:
            return ()
        clauses = ["units_fts MATCH ?"]
        parameters: list[object] = [" OR ".join(f'"{token}"' for token in tokens)]
        for column, values in (
            ("corpus_ref", filters.corpus_refs),
            ("artifact_ref", filters.artifact_refs),
            ("media_type", filters.media_types),
        ):
            if values:
                clauses.append(f"u.{column} IN ({', '.join('?' for _ in values)})")
                parameters.extend(sorted(values))
        parameters.append(top_k)
        with self._store.read(generation.generation_id) as connection:
            _require_pin(connection, generation)
            rows = connection.execute(
                "SELECT u.unit_id, u.evidence_id, u.start_char, u.end_char, u.text, "  # nosec B608
                "snippet(units_fts, 0, '', '', '…', 24) AS snippet, "
                "bm25(units_fts) AS score "
                "FROM units_fts JOIN units AS u ON u.rowid = units_fts.rowid "
                f"WHERE {' AND '.join(clauses)} ORDER BY score, u.unit_id LIMIT ?",
                parameters,
            ).fetchall()
        return tuple(
            TextCandidate(
                unit_id=row["unit_id"],
                evidence_ref=row["evidence_id"],
                text_locator=TextChunkLocator(
                    TextOrigin.DOCUMENT, CharacterRange(row["start_char"], row["end_char"])
                ),
                snippet=(row["snippet"].strip() or row["text"][:240]),
                raw_score=-float(row["score"]),
                score_semantics=SCORE_SEMANTICS,
                rank=rank,
                provider_id=PROVIDER_ID,
                index_generation_id=generation.generation_id,
            )
            for rank, row in enumerate(rows, start=1)
        )


class UnavailableDenseRetriever:
    """Fail closed: the local base profile has no dense lane."""

    provider_id: str = PROVIDER_ID

    def retrieve(
        self,
        *,
        query_vector: tuple[float, ...],
        embedding_space: EmbeddingSpaceRef,
        filters: SearchFilters,
        generation: PinnedTextIndexGeneration,
        top_k: int,
    ) -> tuple[TextCandidate, ...]:
        raise LocalIndexError("the local FTS5 backend has no dense lane")


class Fts5TextBackend:
    """Group the local FTS5 writer, lanes, and lifecycle as one text backend."""

    provider_id: str = PROVIDER_ID

    def __init__(self, index_root: Path) -> None:
        self.store = Fts5GenerationStore(index_root)
        self.writer = Fts5IndexWriter(self.store)
        self.lifecycle = Fts5GenerationLifecycle(self.store)
        self.index_writer: TextIndexWriter = self.writer
        self.lexical_retriever: LexicalCandidateRetriever = Fts5LexicalRetriever(self.store)
        self.dense_retriever: DenseCandidateRetriever = UnavailableDenseRetriever()
        self.generation_lifecycle: TextIndexGenerationLifecycle = self.lifecycle

    def read_evidence(
        self, generation: PinnedTextIndexGeneration, evidence_id: str
    ) -> LocalEvidence:
        """Return one verified unit or source record from the pinned generation."""
        with self.store.read(generation.generation_id) as connection:
            _require_pin(connection, generation)
            unit = connection.execute(
                "SELECT * FROM units WHERE evidence_id = ?", (evidence_id,)
            ).fetchone()
            source_key = unit["source_key"] if unit is not None else None
            source = connection.execute(
                "SELECT * FROM sources WHERE "  # nosec B608 - fixed clauses, bound values
                + ("source_key = ?" if unit is not None else "evidence_id = ?"),
                (source_key if unit is not None else evidence_id,),
            ).fetchone()
        if source is None:
            raise LookupError("evidence is unavailable in the active generation")
        indexed = IndexedSource(
            source["source_key"],
            source["source_ref"],
            source["sha256"],
            source["size_bytes"],
            source["media_type"],
            source["import_version"],
            source["evidence_id"],
        )
        if unit is None:
            return LocalEvidence(
                evidence_id, generation.generation_id, generation.manifest_sha256, indexed
            )
        text = str(unit["text"])
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != unit["text_sha256"]:
            raise LookupError("evidence text does not match its recorded hash")
        if unit["artifact_ref"] != f"artifact:sha256:{indexed.sha256}":
            raise LookupError("evidence source does not match its recorded hash")
        return LocalEvidence(
            evidence_id,
            generation.generation_id,
            generation.manifest_sha256,
            indexed,
            corpus_ref=unit["corpus_ref"],
            start_char=unit["start_char"],
            end_char=unit["end_char"],
            text=text,
        )


def _require_pin(connection: sqlite3.Connection, generation: PinnedTextIndexGeneration) -> None:
    meta = dict(connection.execute("SELECT key, value FROM meta").fetchall())
    if (
        meta.get("generation_id") != generation.generation_id
        or meta.get("manifest_sha256") != generation.manifest_sha256
    ):
        raise LookupError("generation identity does not match the active pin")


def _require_generation(generation_id: str) -> str:
    if _GENERATION.fullmatch(generation_id) is None:
        raise LocalIndexError("generation identity is invalid")
    return generation_id


def _source_key(representation_refs: tuple[str, ...]) -> str:
    if len(representation_refs) != 1 or not representation_refs[0].startswith(
        "representation:markdown:"
    ):
        raise ValueError("local text units require one Markdown representation")
    return representation_refs[0].removeprefix("representation:markdown:")


def _is_private_regular_file(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(metadata.st_mode) and not metadata.st_mode & 0o077


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "INDEX_FORMAT_VERSION",
    "PROVIDER_ID",
    "Fts5TextBackend",
    "IndexedSource",
    "LocalEvidence",
    "LocalIndexError",
    "fts5_available",
    "physical_index_ref",
]
