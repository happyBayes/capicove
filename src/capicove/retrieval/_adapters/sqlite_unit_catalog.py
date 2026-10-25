"""Immutable SQLite unit catalog for personal reverse-image retrieval."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import BinaryIO, Self, cast

from capicove.contracts.retrieval import (
    DistanceMetric,
    EmbeddingSpaceRef,
    MatchKind,
    ProcessingFingerprint,
    QueryModality,
    SearchableUnit,
    SearchFilters,
    TargetType,
    TemporalRange,
    VectorNormalization,
)
from capicove.retrieval.search_models import (
    Candidate,
    ContentCandidateProvenance,
    RetrievalLane,
)


MAX_PERSONAL_UNITS = 10_000
MAX_METADATA_LOOKUP_UNITS = 100
SQLITE_UNIT_CATALOG_APPLICATION_ID = 0x43415055  # ``CAPU``
SQLITE_UNIT_CATALOG_SCHEMA_VERSION = 1
SQLITE_UNIT_CATALOG_FORMAT = "sqlite-unit-catalog-v1"

_PAGE_SIZE = 4096
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_METADATA_COLUMNS = {
    "singleton",
    "catalog_id",
    "activation_set_id",
    "corpus_ref",
    "unit_count",
    "unit_manifest_sha256",
    "catalog_format",
    "schema_version",
}
_EXPECTED_UNIT_COLUMNS = {
    "ordinal",
    "unit_id",
    "unit_type",
    "source_sha256",
    "corpus_ref",
    "artifact_ref",
    "media_type",
    "timestamp_ms",
    "time_start_ms",
    "time_end_ms",
    "unit_json",
}
_SCHEMA = """
CREATE TABLE catalog_metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    catalog_id TEXT NOT NULL,
    activation_set_id TEXT NOT NULL,
    corpus_ref TEXT NOT NULL,
    unit_count INTEGER NOT NULL CHECK (unit_count > 0 AND unit_count <= 10000),
    unit_manifest_sha256 TEXT NOT NULL CHECK (length(unit_manifest_sha256) = 64),
    catalog_format TEXT NOT NULL,
    schema_version INTEGER NOT NULL CHECK (schema_version = 1)
);

CREATE TABLE units (
    ordinal INTEGER PRIMARY KEY CHECK (ordinal >= 0 AND ordinal < 10000),
    unit_id TEXT NOT NULL UNIQUE,
    unit_type TEXT NOT NULL,
    source_sha256 TEXT NOT NULL CHECK (length(source_sha256) = 64),
    corpus_ref TEXT NOT NULL,
    artifact_ref TEXT NOT NULL,
    media_type TEXT NOT NULL,
    timestamp_ms INTEGER,
    time_start_ms INTEGER,
    time_end_ms INTEGER,
    unit_json TEXT NOT NULL
);

CREATE INDEX units_source_sha256_ordinal ON units (source_sha256, ordinal);
CREATE INDEX units_corpus_ordinal ON units (corpus_ref, ordinal);
CREATE INDEX units_artifact_ordinal ON units (artifact_ref, ordinal);
CREATE INDEX units_media_type_ordinal ON units (media_type, ordinal);
CREATE INDEX units_timestamp_ordinal ON units (timestamp_ms, ordinal);
CREATE INDEX units_time_range_ordinal ON units (time_start_ms, time_end_ms, ordinal);
"""


@dataclass(frozen=True, slots=True)
class IndexedUnitRecord:
    """Bind one canonical unit to its stable ordinal and filter-only metadata."""

    ordinal: int
    unit: SearchableUnit
    source_sha256: str
    media_type: str

    def __post_init__(self) -> None:
        _require_non_negative_int("ordinal", self.ordinal)
        if not isinstance(self.unit, SearchableUnit):
            raise TypeError("unit must be a SearchableUnit")
        _require_sha256("source_sha256", self.source_sha256)
        _require_non_blank("media_type", self.media_type)


@dataclass(frozen=True, slots=True)
class SqliteUnitCatalogDescriptor:
    """Describe one closed SQLite payload with its complete digest identity."""

    catalog_id: str
    activation_set_id: str
    corpus_ref: str
    unit_count: int
    unit_manifest_sha256: str
    path: Path
    sha256: str
    size_bytes: int
    payload_ref: str
    catalog_format: str = SQLITE_UNIT_CATALOG_FORMAT
    schema_version: int = SQLITE_UNIT_CATALOG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "catalog_id",
            "activation_set_id",
            "corpus_ref",
            "payload_ref",
            "catalog_format",
        ):
            _require_non_blank(field_name, getattr(self, field_name))
        _require_positive_int("unit_count", self.unit_count)
        if self.unit_count > MAX_PERSONAL_UNITS:
            raise ValueError("unit_count must be at most 10,000")
        _require_sha256("unit_manifest_sha256", self.unit_manifest_sha256)
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("path must be an absolute Path")
        _require_sha256("sha256", self.sha256)
        _require_positive_int("size_bytes", self.size_bytes)
        if self.catalog_format != SQLITE_UNIT_CATALOG_FORMAT:
            raise ValueError("catalog_format is unsupported")
        if self.schema_version != SQLITE_UNIT_CATALOG_SCHEMA_VERSION:
            raise ValueError("catalog schema version is unsupported")


@dataclass(slots=True)
class _OpenedCatalogPayload:
    path: Path
    handle: BinaryIO
    fingerprint: tuple[int, int, int, int, int]

    def close(self) -> None:
        self.handle.close()


@dataclass(frozen=True, slots=True)
class ContentCatalogMatch:
    """Identify one exact source-hash match by unit and stable ordinal."""

    unit_id: str
    ordinal: int

    def __post_init__(self) -> None:
        _require_non_blank("unit_id", self.unit_id)
        _require_non_negative_int("ordinal", self.ordinal)


@dataclass(frozen=True, slots=True)
class AllOrdinals:
    """Represent an unfiltered selection without materializing every ordinal."""

    unit_count: int
    filter_sha256: str
    ordinals: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _require_positive_int("unit_count", self.unit_count)
        _require_sha256("filter_sha256", self.filter_sha256)
        if self.ordinals is not None:
            raise ValueError("AllOrdinals.ordinals must be None")


@dataclass(frozen=True, slots=True)
class SparseOrdinals:
    """Represent one sorted, duplicate-free personal-profile selection."""

    unit_count: int
    filter_sha256: str
    ordinals: tuple[int, ...] | None

    def __post_init__(self) -> None:
        _require_positive_int("unit_count", self.unit_count)
        _require_sha256("filter_sha256", self.filter_sha256)
        if self.ordinals is None:
            raise ValueError("SparseOrdinals.ordinals must be a tuple")
        ordinals = tuple(self.ordinals)
        if any(
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or not 0 <= ordinal < self.unit_count
            for ordinal in ordinals
        ):
            raise ValueError("ordinals must be within the catalog unit range")
        if tuple(sorted(set(ordinals))) != ordinals:
            raise ValueError("ordinals must be sorted and unique")
        object.__setattr__(self, "ordinals", ordinals)


@dataclass(frozen=True, slots=True)
class BitmapOrdinals:
    """Represent a fixed-size dense ordinal selection without integer tuples."""

    unit_count: int
    filter_sha256: str
    bits: bytes
    matching_count: int

    def __post_init__(self) -> None:
        _require_positive_int("unit_count", self.unit_count)
        _require_sha256("filter_sha256", self.filter_sha256)
        if not isinstance(self.bits, bytes) or len(self.bits) != math.ceil(self.unit_count / 8):
            raise ValueError("bits must contain exactly one bit per catalog unit")
        _require_non_negative_int("matching_count", self.matching_count)
        if self.matching_count > self.unit_count:
            raise ValueError("matching_count cannot exceed unit_count")
        unused_bits = len(self.bits) * 8 - self.unit_count
        if unused_bits and self.bits[-1] >> (8 - unused_bits):
            raise ValueError("unused bitmap tail bits must be zero")
        if sum(byte.bit_count() for byte in self.bits) != self.matching_count:
            raise ValueError("matching_count must equal the number of selected bits")


OrdinalSelection = AllOrdinals | SparseOrdinals


@dataclass(frozen=True, slots=True)
class _PreparedRecord:
    ordinal: int
    unit_id: str
    unit_type: str
    source_sha256: str
    corpus_ref: str
    artifact_ref: str
    media_type: str
    timestamp_ms: int | None
    time_start_ms: int | None
    time_end_ms: int | None
    unit_json: str


class SqliteUnitCatalogWriter:
    """Build one closed, non-WAL personal SQLite catalog payload."""

    def __init__(self, max_units: int = MAX_PERSONAL_UNITS) -> None:
        _require_positive_int("max_units", max_units)
        if max_units > MAX_PERSONAL_UNITS:
            raise ValueError("max_units must be at most 10,000")
        self._max_units = max_units

    def build(
        self,
        rows: Iterable[IndexedUnitRecord],
        destination: Path,
        *,
        activation_set_id: str,
        corpus_ref: str,
        deadline: datetime,
    ) -> SqliteUnitCatalogDescriptor:
        """Validate all bounded inputs, then atomically expose a closed catalog."""
        _require_deadline(deadline)
        _require_non_blank("activation_set_id", activation_set_id)
        _require_non_blank("corpus_ref", corpus_ref)
        if not isinstance(destination, Path) or not destination.is_absolute():
            raise ValueError("destination must be an absolute Path")

        prepared = self._prepare_records(rows, corpus_ref=corpus_ref, deadline=deadline)
        unit_manifest_sha256 = _unit_manifest_sha256(prepared)
        identity = _canonical_json_bytes(
            {
                "activation_set_id": activation_set_id,
                "catalog_format": SQLITE_UNIT_CATALOG_FORMAT,
                "corpus_ref": corpus_ref,
                "unit_count": len(prepared),
                "unit_manifest_sha256": unit_manifest_sha256,
            }
        )
        catalog_id = f"catalog:{hashlib.sha256(identity).hexdigest()}"
        payload_digest = hashlib.sha256(b"catalog\0" + identity).hexdigest()
        payload_ref = f"payload:{payload_digest}"

        _require_deadline(deadline)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"catalog destination already exists: {destination}")
        destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        parent_stat = destination.parent.lstat()
        if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
            raise ValueError("destination parent must be a non-symlink directory")
        with tempfile.TemporaryDirectory(
            prefix=f".{destination.name}.", dir=destination.parent
        ) as staging_name:
            staging_directory = Path(staging_name)
            staging_directory.chmod(0o700)
            staging_path = staging_directory / "catalog.sqlite"
            self._write_database(
                staging_path,
                prepared,
                catalog_id=catalog_id,
                activation_set_id=activation_set_id,
                corpus_ref=corpus_ref,
                unit_manifest_sha256=unit_manifest_sha256,
                deadline=deadline,
            )
            staging_path.chmod(0o600)
            _require_deadline(deadline)
            size_bytes = staging_path.stat().st_size
            sha256 = self._digest_file_with_deadline(staging_path, deadline)
            descriptor = SqliteUnitCatalogDescriptor(
                catalog_id=catalog_id,
                activation_set_id=activation_set_id,
                corpus_ref=corpus_ref,
                unit_count=len(prepared),
                unit_manifest_sha256=unit_manifest_sha256,
                path=destination,
                sha256=sha256,
                size_bytes=size_bytes,
                payload_ref=payload_ref,
            )
            _require_deadline(deadline)
            _publish_catalog(staging_path, destination)
            return descriptor

    @staticmethod
    def digest_file(path: Path) -> str:
        """Return the SHA-256 digest of one closed regular file."""
        return _digest_file(path, deadline=None)

    @staticmethod
    def _digest_file_with_deadline(path: Path, deadline: datetime) -> str:
        return _digest_file(path, deadline=deadline)

    def _prepare_records(
        self,
        rows: Iterable[IndexedUnitRecord],
        *,
        corpus_ref: str,
        deadline: datetime,
    ) -> tuple[_PreparedRecord, ...]:
        rows_object: object = rows
        if isinstance(rows_object, (str, bytes)) or not isinstance(rows_object, Iterable):
            raise TypeError("rows must be an iterable of IndexedUnitRecord values")
        materialized: list[IndexedUnitRecord] = []
        for row in rows:
            _require_deadline(deadline)
            if not isinstance(row, IndexedUnitRecord):
                raise TypeError("rows must contain IndexedUnitRecord values")
            materialized.append(row)
            if len(materialized) > self._max_units:
                raise ValueError(f"catalog supports at most {self._max_units:,} units")
        if not materialized:
            raise ValueError("catalog must contain at least one unit")

        seen_unit_ids: set[str] = set()
        prepared: list[_PreparedRecord] = []
        for expected_ordinal, row in enumerate(materialized):
            _require_deadline(deadline)
            if row.ordinal != expected_ordinal:
                raise ValueError("ordinals must be contiguous from zero in input order")
            if row.unit.unit_id in seen_unit_ids:
                raise ValueError("unit_id values must be unique")
            seen_unit_ids.add(row.unit.unit_id)
            if row.source_sha256 != row.unit.source_hash:
                raise ValueError("source_sha256 must equal unit.source_hash")
            if row.unit.unit_type is not TargetType.IMAGE:
                raise ValueError("personal unit catalog supports image units only")
            unit_json = _encode_searchable_unit(row.unit)
            time_start_ms, time_end_ms = _unit_time_bounds(row.unit)
            prepared.append(
                _PreparedRecord(
                    ordinal=row.ordinal,
                    unit_id=row.unit.unit_id,
                    unit_type=row.unit.unit_type.value,
                    source_sha256=row.source_sha256,
                    corpus_ref=corpus_ref,
                    artifact_ref=row.unit.artifact_ref,
                    media_type=row.media_type,
                    timestamp_ms=row.unit.timestamp_ms,
                    time_start_ms=time_start_ms,
                    time_end_ms=time_end_ms,
                    unit_json=unit_json,
                )
            )
        return tuple(prepared)

    @staticmethod
    def _write_database(
        path: Path,
        records: tuple[_PreparedRecord, ...],
        *,
        catalog_id: str,
        activation_set_id: str,
        corpus_ref: str,
        unit_manifest_sha256: str,
        deadline: datetime,
    ) -> None:
        connection: sqlite3.Connection | None = None
        try:
            _require_deadline(deadline)
            connection = sqlite3.connect(path, isolation_level=None, timeout=0.0)
            connection.execute(f"PRAGMA page_size = {_PAGE_SIZE}")
            connection.execute("PRAGMA journal_mode = DELETE")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA trusted_schema = OFF")
            connection.execute(f"PRAGMA application_id = {SQLITE_UNIT_CATALOG_APPLICATION_ID}")
            connection.execute(f"PRAGMA user_version = {SQLITE_UNIT_CATALOG_SCHEMA_VERSION}")
            connection.executescript(_SCHEMA)
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO catalog_metadata (
                    singleton, catalog_id, activation_set_id, corpus_ref, unit_count,
                    unit_manifest_sha256, catalog_format, schema_version
                ) VALUES (1, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    catalog_id,
                    activation_set_id,
                    corpus_ref,
                    len(records),
                    unit_manifest_sha256,
                    SQLITE_UNIT_CATALOG_FORMAT,
                    SQLITE_UNIT_CATALOG_SCHEMA_VERSION,
                ),
            )
            for record in records:
                _require_deadline(deadline)
                connection.execute(
                    """
                    INSERT INTO units (
                        ordinal, unit_id, unit_type, source_sha256, corpus_ref,
                        artifact_ref, media_type, timestamp_ms, time_start_ms,
                        time_end_ms, unit_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.ordinal,
                        record.unit_id,
                        record.unit_type,
                        record.source_sha256,
                        record.corpus_ref,
                        record.artifact_ref,
                        record.media_type,
                        record.timestamp_ms,
                        record.time_start_ms,
                        record.time_end_ms,
                        record.unit_json,
                    ),
                )
            _require_deadline(deadline)
            connection.execute("COMMIT")
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            if integrity != ("ok",):
                raise ValueError("catalog integrity check failed")
        except BaseException:
            if connection is not None:
                try:
                    connection.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
            raise
        finally:
            if connection is not None:
                connection.close()
        with path.open("rb") as payload:
            os.fsync(payload.fileno())


class SqliteUnitCatalog:
    """Serve one checksum-verified SQLite catalog through a read-only handle."""

    def __init__(
        self,
        descriptor: SqliteUnitCatalogDescriptor,
        connection: sqlite3.Connection,
        opened_payload: _OpenedCatalogPayload,
    ) -> None:
        self._descriptor = descriptor
        self._connection = connection
        self._opened_payload = opened_payload
        self._connection_lock = RLock()
        self._connection_closed = False
        self._payload_closed = False
        self._closing = False
        self._closed = False

    @classmethod
    def open_readonly(
        cls,
        descriptor: SqliteUnitCatalogDescriptor,
        *,
        deadline: datetime,
    ) -> Self:
        """Verify the closed payload before opening an immutable SQLite URI."""
        _require_deadline(deadline)
        if not isinstance(descriptor, SqliteUnitCatalogDescriptor):
            raise TypeError("descriptor must be a SqliteUnitCatalogDescriptor")
        opened_payload = _open_verified_catalog_payload(descriptor, deadline=deadline)

        connection: sqlite3.Connection | None = None
        try:
            _require_deadline(deadline)
            try:
                connection = _connect_opened_catalog(opened_payload)
            except sqlite3.OperationalError:
                # Linux may report CANTOPEN for /dev/fd after the published path
                # is replaced. Recheck the verified handle/path binding so a
                # replacement remains an integrity failure on every platform.
                _verify_opened_catalog_unchanged(
                    opened_payload,
                    descriptor,
                    deadline=deadline,
                    verify_digest=True,
                )
                raise
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            connection.execute("PRAGMA trusted_schema = OFF")
            _set_busy_timeout(connection, deadline)
            cls._validate_open_database(connection, descriptor, deadline=deadline)
            _verify_opened_catalog_unchanged(
                opened_payload,
                descriptor,
                deadline=deadline,
                verify_digest=True,
            )
            return cls(descriptor, connection, opened_payload)
        except BaseException as primary_error:
            cleanup_failures: list[tuple[str, BaseException]] = []
            if connection is not None:
                try:
                    connection.close()
                except BaseException as cleanup_error:
                    cleanup_failures.append(("SQLite connection", cleanup_error))
            try:
                opened_payload.close()
            except BaseException as cleanup_error:
                cleanup_failures.append(("catalog payload", cleanup_error))
            for resource_name, recorded_cleanup_error in cleanup_failures:
                primary_error.add_note(
                    f"{resource_name} cleanup also failed: {type(recorded_cleanup_error).__name__}"
                )
            raise

    @property
    def is_query_only(self) -> bool:
        """Return whether SQLite has enforced query-only mode on this handle."""
        with self._connection_lock:
            self._require_open()
            row = self._connection.execute("PRAGMA query_only").fetchone()
            return row is not None and int(row[0]) == 1

    @property
    def unit_count(self) -> int:
        """Return the immutable catalog row count from the verified descriptor."""
        return self._descriptor.unit_count

    @property
    def unit_manifest_sha256(self) -> str:
        """Return the exact ordered unit-manifest identity used by lane indexes."""
        return self._descriptor.unit_manifest_sha256

    @property
    def activation_set_id(self) -> str:
        """Return the exact activation set pinned by this catalog."""
        return self._descriptor.activation_set_id

    def lookup_content_hash(
        self,
        source_sha256: str,
        filters: SearchFilters,
        limit: int,
        deadline: datetime,
    ) -> tuple[ContentCatalogMatch, ...]:
        """Return bounded exact source-hash matches in stable ordinal order."""
        _require_sha256("source_sha256", source_sha256)
        _require_limit(limit)
        where_sql, parameters = self._filter_sql(filters)
        # Clauses are fixed internally; filter values remain DB-API parameters.
        rows = self._fetchall(
            f"""
            SELECT unit_id, ordinal
            FROM units
            WHERE source_sha256 = ? AND {where_sql}
            ORDER BY ordinal
            LIMIT ?
            """,  # nosec B608
            (source_sha256, *parameters, limit),
            deadline=deadline,
        )
        return tuple(ContentCatalogMatch(str(row["unit_id"]), int(row["ordinal"])) for row in rows)

    def lookup_content_candidates(
        self,
        source_sha256: str,
        filters: SearchFilters,
        limit: int,
        deadline: datetime,
    ) -> tuple[Candidate, ...]:
        """Project exact catalog matches into the dedicated content lane."""
        matches = self.lookup_content_hash(
            source_sha256,
            filters,
            limit,
            deadline,
        )
        return tuple(
            Candidate(
                unit_id=match.unit_id,
                unit_ordinal=match.ordinal,
                lane=RetrievalLane.CONTENT,
                match_kind=MatchKind.NEAR_DUPLICATE,
                raw_score=1.0,
                score_semantics="sha256_equality",
                rank=rank,
                provenance=ContentCandidateProvenance(
                    activation_set_id=self._descriptor.activation_set_id,
                    unit_catalog_id=self._descriptor.catalog_id,
                ),
                exact_content=True,
            )
            for rank, match in enumerate(matches, start=1)
        )

    def select_ordinals(
        self,
        filters: SearchFilters,
        *,
        deadline: datetime,
    ) -> OrdinalSelection:
        """Return an O(1) all-selection or one bounded sparse selection."""
        _require_filters(filters)
        filter_sha256 = _filter_sha256(filters)
        if _filters_are_empty(filters):
            _require_deadline(deadline)
            return AllOrdinals(self.unit_count, filter_sha256)
        where_sql, parameters = self._filter_sql(filters)
        # Clauses are fixed internally; filter values remain DB-API parameters.
        rows = self._fetchall(
            f"SELECT ordinal FROM units WHERE {where_sql} ORDER BY ordinal",  # nosec B608
            parameters,
            deadline=deadline,
        )
        return SparseOrdinals(
            unit_count=self.unit_count,
            filter_sha256=filter_sha256,
            ordinals=tuple(int(row["ordinal"]) for row in rows),
        )

    def get_many(
        self,
        unit_ids: tuple[str, ...],
        deadline: datetime,
    ) -> tuple[SearchableUnit, ...]:
        """Resolve at most 100 unit IDs while preserving requested order."""
        identifiers = _bounded_string_tuple("unit_ids", unit_ids)
        if not identifiers:
            _require_deadline(deadline)
            return ()
        placeholders = ",".join("?" for _ in set(identifiers))
        # Only generated question-mark placeholders enter SQL; IDs remain bound.
        rows = self._fetchall(
            f"SELECT unit_id, unit_json FROM units WHERE unit_id IN ({placeholders})",  # nosec B608
            tuple(dict.fromkeys(identifiers)),
            deadline=deadline,
        )
        decoded = {
            str(row["unit_id"]): _decode_searchable_unit(str(row["unit_json"])) for row in rows
        }
        missing = tuple(identifier for identifier in identifiers if identifier not in decoded)
        if missing:
            raise KeyError(f"missing unit IDs: {', '.join(dict.fromkeys(missing))}")
        return tuple(decoded[identifier] for identifier in identifiers)

    def unit_ids_for_ordinals(
        self,
        ordinals: tuple[int, ...],
        *,
        deadline: datetime,
    ) -> tuple[str, ...]:
        """Resolve one bounded unique ordinal tuple while preserving input order."""
        values = _bounded_ordinal_tuple(ordinals, unit_count=self.unit_count)
        if not values:
            _require_deadline(deadline)
            return ()
        placeholders = ",".join("?" for _ in values)
        # Only generated question-mark placeholders enter SQL; ordinals remain bound.
        rows = self._fetchall(
            f"SELECT ordinal, unit_id FROM units WHERE ordinal IN ({placeholders})",  # nosec B608
            values,
            deadline=deadline,
        )
        resolved = {int(row["ordinal"]): str(row["unit_id"]) for row in rows}
        missing = tuple(ordinal for ordinal in values if ordinal not in resolved)
        if missing:
            raise KeyError(f"missing ordinals: {', '.join(str(value) for value in missing)}")
        return tuple(resolved[ordinal] for ordinal in values)

    def close(self) -> None:
        """Release the read-only SQLite handle idempotently."""
        with self._connection_lock:
            if self._closed:
                return
            self._closing = True
            first_error: BaseException | None = None
            if not self._connection_closed:
                try:
                    self._connection.close()
                except BaseException as error:
                    first_error = error
                else:
                    self._connection_closed = True
            if not self._payload_closed:
                try:
                    self._opened_payload.close()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
                else:
                    self._payload_closed = True
            self._closed = self._connection_closed and self._payload_closed
            if first_error is not None:
                raise first_error

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @staticmethod
    def _validate_open_database(
        connection: sqlite3.Connection,
        descriptor: SqliteUnitCatalogDescriptor,
        *,
        deadline: datetime,
    ) -> None:
        _require_deadline(deadline)
        application_id = _pragma_int(connection, "application_id")
        schema_version = _pragma_int(connection, "user_version")
        page_size = _pragma_int(connection, "page_size")
        if application_id != SQLITE_UNIT_CATALOG_APPLICATION_ID:
            raise ValueError("catalog schema application ID is unsupported")
        if schema_version != SQLITE_UNIT_CATALOG_SCHEMA_VERSION:
            raise ValueError("catalog schema version is unsupported")
        if page_size != _PAGE_SIZE:
            raise ValueError("catalog schema page-size policy is unsupported")
        if _pragma_int(connection, "query_only") != 1:
            raise ValueError("catalog reader is not query-only")
        if _pragma_int(connection, "trusted_schema") != 0:
            raise ValueError("catalog reader trusted-schema policy is unsafe")
        journal_mode_row = connection.execute("PRAGMA journal_mode").fetchone()
        if journal_mode_row is None or str(journal_mode_row[0]).lower() == "wal":
            raise ValueError("catalog schema must not use WAL")
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or str(integrity[0]) != "ok":
            raise ValueError("catalog integrity check failed")
        _require_deadline(deadline)

        if _table_columns(connection, "catalog_metadata") != _EXPECTED_METADATA_COLUMNS:
            raise ValueError("catalog metadata schema is unsupported")
        if _table_columns(connection, "units") != _EXPECTED_UNIT_COLUMNS:
            raise ValueError("catalog unit schema is unsupported")
        metadata = connection.execute(
            "SELECT * FROM catalog_metadata WHERE singleton = 1"
        ).fetchone()
        if metadata is None:
            raise ValueError("catalog metadata schema row is missing")
        expected_metadata: dict[str, object] = {
            "catalog_id": descriptor.catalog_id,
            "activation_set_id": descriptor.activation_set_id,
            "corpus_ref": descriptor.corpus_ref,
            "unit_count": descriptor.unit_count,
            "unit_manifest_sha256": descriptor.unit_manifest_sha256,
            "catalog_format": descriptor.catalog_format,
            "schema_version": descriptor.schema_version,
        }
        for field_name, expected in expected_metadata.items():
            if metadata[field_name] != expected:
                raise ValueError(f"catalog metadata schema mismatch: {field_name}")

        rows = connection.execute(
            """
            SELECT ordinal, unit_id, unit_type, source_sha256, corpus_ref, artifact_ref,
                   media_type, timestamp_ms, time_start_ms, time_end_ms, unit_json
            FROM units ORDER BY ordinal
            """
        ).fetchall()
        if len(rows) != descriptor.unit_count:
            raise ValueError("catalog integrity unit count mismatch")
        prepared: list[_PreparedRecord] = []
        for expected_ordinal, row in enumerate(rows):
            _require_deadline(deadline)
            if int(row["ordinal"]) != expected_ordinal:
                raise ValueError("catalog integrity ordinals are not contiguous")
            unit = _decode_searchable_unit(str(row["unit_json"]))
            source_sha256 = str(row["source_sha256"])
            if unit.unit_id != str(row["unit_id"]) or unit.source_hash != source_sha256:
                raise ValueError("catalog integrity unit identity mismatch")
            prepared.append(
                _PreparedRecord(
                    ordinal=expected_ordinal,
                    unit_id=str(row["unit_id"]),
                    unit_type=str(row["unit_type"]),
                    source_sha256=source_sha256,
                    corpus_ref=str(row["corpus_ref"]),
                    artifact_ref=str(row["artifact_ref"]),
                    media_type=str(row["media_type"]),
                    timestamp_ms=cast(int | None, row["timestamp_ms"]),
                    time_start_ms=cast(int | None, row["time_start_ms"]),
                    time_end_ms=cast(int | None, row["time_end_ms"]),
                    unit_json=str(row["unit_json"]),
                )
            )
        if _unit_manifest_sha256(tuple(prepared)) != descriptor.unit_manifest_sha256:
            raise ValueError("catalog integrity unit-manifest digest mismatch")

    def _filter_sql(self, filters: SearchFilters) -> tuple[str, tuple[object, ...]]:
        _require_filters(filters)
        if filters.corpus_refs and self._descriptor.corpus_ref not in filters.corpus_refs:
            return "0", ()
        clauses = ["1"]
        parameters: list[object] = []
        if filters.artifact_refs:
            values = tuple(sorted(filters.artifact_refs))
            clauses.append(f"artifact_ref IN ({','.join('?' for _ in values)})")
            parameters.extend(values)
        if filters.media_types:
            values = tuple(sorted(filters.media_types))
            clauses.append(f"media_type IN ({','.join('?' for _ in values)})")
            parameters.extend(values)
        if filters.time_range is not None:
            clauses.append(
                "((unit_type = ? AND timestamp_ms >= ? AND timestamp_ms < ?) "
                "OR (unit_type = ? AND time_start_ms < ? AND time_end_ms > ?))"
            )
            parameters.extend(
                (
                    TargetType.KEYFRAME.value,
                    filters.time_range.start_ms,
                    filters.time_range.end_ms,
                    TargetType.VIDEO_SEGMENT.value,
                    filters.time_range.end_ms,
                    filters.time_range.start_ms,
                )
            )
        return " AND ".join(clauses), tuple(parameters)

    def _fetchall(
        self,
        sql: str,
        parameters: tuple[object, ...],
        *,
        deadline: datetime,
    ) -> list[sqlite3.Row]:
        with self._connection_lock:
            self._require_open()
            _verify_opened_catalog_unchanged(
                self._opened_payload,
                self._descriptor,
                deadline=deadline,
                verify_digest=False,
            )
            with _sqlite_deadline(self._connection, deadline):
                try:
                    rows = cast(
                        list[sqlite3.Row],
                        self._connection.execute(sql, parameters).fetchall(),
                    )
                except sqlite3.OperationalError as error:
                    if datetime.now(UTC) >= deadline or "interrupted" in str(error).lower():
                        raise TimeoutError("unit catalog deadline exceeded") from error
                    raise ValueError("catalog query failed") from error
                except sqlite3.DatabaseError as error:
                    raise ValueError("catalog query failed") from error
            _verify_opened_catalog_unchanged(
                self._opened_payload,
                self._descriptor,
                deadline=deadline,
                verify_digest=False,
            )
            return rows

    def _require_open(self) -> None:
        if self._closing or self._closed:
            raise RuntimeError("unit catalog is closed")


@contextmanager
def _sqlite_deadline(
    connection: sqlite3.Connection,
    deadline: datetime,
) -> Iterator[None]:
    _require_deadline(deadline)
    _set_busy_timeout(connection, deadline)
    expired = False

    def interrupt_if_expired() -> int:
        nonlocal expired
        expired = datetime.now(UTC) >= deadline
        return int(expired)

    connection.set_progress_handler(interrupt_if_expired, 1_000)
    try:
        yield
        _require_deadline(deadline)
    finally:
        connection.set_progress_handler(None, 0)


def _set_busy_timeout(connection: sqlite3.Connection, deadline: datetime) -> None:
    remaining_ms = max(0, math.ceil((deadline - datetime.now(UTC)).total_seconds() * 1_000))
    connection.execute(f"PRAGMA busy_timeout = {min(remaining_ms, 5_000)}")


def _open_verified_catalog_payload(
    descriptor: SqliteUnitCatalogDescriptor,
    *,
    deadline: datetime,
) -> _OpenedCatalogPayload:
    _require_deadline(deadline)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    file_descriptor: int | None = None
    handle: BinaryIO | None = None
    try:
        file_descriptor = os.open(descriptor.path, flags)
        handle = os.fdopen(file_descriptor, "rb", closefd=True)
        file_descriptor = None
        file_stat = os.fstat(handle.fileno())
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("catalog checksum payload must be a regular file")
        if file_stat.st_size != descriptor.size_bytes:
            raise ValueError("catalog checksum size mismatch")
        opened = _OpenedCatalogPayload(
            path=descriptor.path,
            handle=handle,
            fingerprint=_catalog_file_fingerprint(file_stat),
        )
        _verify_catalog_path_binding(opened)
        _verify_catalog_sidecars_absent(descriptor.path)
        if _digest_file(handle, deadline=deadline) != descriptor.sha256:
            raise ValueError("catalog checksum mismatch")
        _verify_opened_catalog_unchanged(
            opened,
            descriptor,
            deadline=deadline,
            verify_digest=False,
        )
        handle.seek(0)
        return opened
    except OSError as error:
        if handle is not None:
            handle.close()
        raise ValueError("catalog checksum payload is unavailable") from error
    except BaseException:
        if handle is not None:
            handle.close()
        raise
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)


def _connect_opened_catalog(opened: _OpenedCatalogPayload) -> sqlite3.Connection:
    descriptor_uri = _opened_descriptor_uri(opened)
    if descriptor_uri is not None:
        return sqlite3.connect(
            descriptor_uri,
            uri=True,
            isolation_level=None,
            timeout=0.0,
            check_same_thread=False,
        )
    connection = sqlite3.connect(
        ":memory:",
        isolation_level=None,
        timeout=0.0,
        check_same_thread=False,
    )
    try:
        deserialize = getattr(connection, "deserialize", None)
        if not callable(deserialize):
            raise ValueError("descriptor-safe SQLite fallback is unavailable")
        opened.handle.seek(0)
        deserialize(opened.handle.read())
        opened.handle.seek(0)
        return connection
    except BaseException:
        connection.close()
        raise


def _opened_descriptor_uri(opened: _OpenedCatalogPayload) -> str | None:
    if os.name == "nt":
        return None
    for descriptor_root in (Path("/dev/fd"), Path("/proc/self/fd")):
        descriptor_path = descriptor_root / str(opened.handle.fileno())
        if descriptor_path.exists():
            return f"{descriptor_path.as_uri()}?mode=ro&immutable=1"
    return None


def _verify_opened_catalog_unchanged(
    opened: _OpenedCatalogPayload,
    descriptor: SqliteUnitCatalogDescriptor,
    *,
    deadline: datetime,
    verify_digest: bool,
) -> None:
    _require_deadline(deadline)
    try:
        if _catalog_file_fingerprint(os.fstat(opened.handle.fileno())) != opened.fingerprint:
            raise ValueError("catalog checksum payload changed")
        _verify_catalog_path_binding(opened)
        _verify_catalog_sidecars_absent(descriptor.path)
        if verify_digest and _digest_file(opened.handle, deadline=deadline) != descriptor.sha256:
            raise ValueError("catalog checksum payload changed")
        if _catalog_file_fingerprint(os.fstat(opened.handle.fileno())) != opened.fingerprint:
            raise ValueError("catalog checksum payload changed")
        _verify_catalog_path_binding(opened)
        _verify_catalog_sidecars_absent(descriptor.path)
    except OSError as error:
        raise ValueError("catalog checksum payload changed") from error
    _require_deadline(deadline)


def _verify_catalog_path_binding(opened: _OpenedCatalogPayload) -> None:
    path_stat = opened.path.lstat()
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise ValueError("catalog checksum payload changed")
    if _catalog_file_fingerprint(path_stat) != opened.fingerprint:
        raise ValueError("catalog checksum payload changed")


def _verify_catalog_sidecars_absent(path: Path) -> None:
    for suffix in ("-wal", "-shm"):
        if Path(f"{path}{suffix}").exists():
            raise ValueError("catalog integrity sidecar files are forbidden")


def _catalog_file_fingerprint(file_stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _digest_file(source: Path | BinaryIO, *, deadline: datetime | None) -> str:
    if isinstance(source, Path):
        with source.open("rb") as payload:
            return _digest_file(payload, deadline=deadline)
    digest = hashlib.sha256()
    source.seek(0)
    while chunk := source.read(1024 * 1024):
        if deadline is not None:
            _require_deadline(deadline)
        digest.update(chunk)
    source.seek(0)
    if deadline is not None:
        _require_deadline(deadline)
    return digest.hexdigest()


def _publish_catalog(staging_path: Path, destination: Path) -> None:
    try:
        os.link(staging_path, destination)
    except FileExistsError:
        raise FileExistsError(f"catalog destination already exists: {destination}") from None
    directory_descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    finally:
        os.close(directory_descriptor)


def _unit_manifest_sha256(records: tuple[_PreparedRecord, ...]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            _canonical_json_bytes(
                {
                    "artifact_ref": record.artifact_ref,
                    "corpus_ref": record.corpus_ref,
                    "media_type": record.media_type,
                    "ordinal": record.ordinal,
                    "source_sha256": record.source_sha256,
                    "unit": json.loads(record.unit_json),
                    "unit_id": record.unit_id,
                    "unit_type": record.unit_type,
                }
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _encode_searchable_unit(unit: SearchableUnit) -> str:
    return _canonical_json_bytes(_searchable_unit_document(unit)).decode("utf-8")


def _searchable_unit_document(unit: SearchableUnit) -> dict[str, object]:
    return {
        "artifact_ref": unit.artifact_ref,
        "embedding_spaces": tuple(
            _embedding_space_document(space) for space in unit.embedding_spaces
        ),
        "evidence_ref": unit.evidence_ref,
        "processing_fingerprint": unit.processing_fingerprint.sha256,
        "processing_run_ref": unit.processing_run_ref,
        "processing_version": unit.processing_version,
        "representation_refs": unit.representation_refs,
        "scene_range": _temporal_range_document(unit.scene_range),
        "source_hash": unit.source_hash,
        "time_range": _temporal_range_document(unit.time_range),
        "timestamp_ms": unit.timestamp_ms,
        "unit_id": unit.unit_id,
        "unit_type": unit.unit_type.value,
    }


def _embedding_space_document(space: EmbeddingSpaceRef) -> dict[str, object]:
    return {
        "dimension": space.dimension,
        "distance_metric": space.distance_metric.value,
        "model_revision": space.model_revision,
        "normalization": space.normalization.value,
        "preprocessing_version": space.preprocessing_version,
        "provider_id": space.provider_id,
        "query_modalities": tuple(sorted(value.value for value in space.query_modalities)),
        "schema_version": space.schema_version,
        "space_id": space.space_id,
        "target_modalities": tuple(sorted(value.value for value in space.target_modalities)),
    }


def _temporal_range_document(value: TemporalRange | None) -> dict[str, int] | None:
    if value is None:
        return None
    return {"end_ms": value.end_ms, "start_ms": value.start_ms}


def _decode_searchable_unit(payload: str) -> SearchableUnit:
    document = _load_json_object(payload)
    _require_exact_keys(
        document,
        {
            "artifact_ref",
            "embedding_spaces",
            "evidence_ref",
            "processing_fingerprint",
            "processing_run_ref",
            "processing_version",
            "representation_refs",
            "scene_range",
            "source_hash",
            "time_range",
            "timestamp_ms",
            "unit_id",
            "unit_type",
        },
        "unit",
    )
    embedding_documents = _object_sequence(document["embedding_spaces"], "embedding_spaces")
    representation_refs = tuple(
        _string(value, "representation_ref")
        for value in _object_sequence(document["representation_refs"], "representation_refs")
    )
    return SearchableUnit(
        unit_id=_string(document["unit_id"], "unit_id"),
        unit_type=TargetType(_string(document["unit_type"], "unit_type")),
        source_hash=_string(document["source_hash"], "source_hash"),
        artifact_ref=_string(document["artifact_ref"], "artifact_ref"),
        evidence_ref=_string(document["evidence_ref"], "evidence_ref"),
        representation_refs=representation_refs,
        processing_run_ref=_string(document["processing_run_ref"], "processing_run_ref"),
        processing_version=_string(document["processing_version"], "processing_version"),
        processing_fingerprint=ProcessingFingerprint(
            _string(document["processing_fingerprint"], "processing_fingerprint")
        ),
        embedding_spaces=tuple(_decode_embedding_space(value) for value in embedding_documents),
        timestamp_ms=_optional_int(document["timestamp_ms"], "timestamp_ms"),
        scene_range=_decode_temporal_range(document["scene_range"], "scene_range"),
        time_range=_decode_temporal_range(document["time_range"], "time_range"),
    )


def _decode_embedding_space(value: object) -> EmbeddingSpaceRef:
    document = _object_mapping(value, "embedding_space")
    _require_exact_keys(
        document,
        {
            "dimension",
            "distance_metric",
            "model_revision",
            "normalization",
            "preprocessing_version",
            "provider_id",
            "query_modalities",
            "schema_version",
            "space_id",
            "target_modalities",
        },
        "embedding_space",
    )
    query_modalities = frozenset(
        QueryModality(_string(item, "query_modality"))
        for item in _object_sequence(document["query_modalities"], "query_modalities")
    )
    target_modalities = frozenset(
        TargetType(_string(item, "target_modality"))
        for item in _object_sequence(document["target_modalities"], "target_modalities")
    )
    return EmbeddingSpaceRef(
        space_id=_string(document["space_id"], "space_id"),
        provider_id=_string(document["provider_id"], "provider_id"),
        model_revision=_string(document["model_revision"], "model_revision"),
        preprocessing_version=_string(
            document["preprocessing_version"],
            "preprocessing_version",
        ),
        dimension=_integer(document["dimension"], "dimension"),
        normalization=VectorNormalization(_string(document["normalization"], "normalization")),
        distance_metric=DistanceMetric(_string(document["distance_metric"], "distance_metric")),
        query_modalities=query_modalities,
        target_modalities=target_modalities,
        schema_version=_integer(document["schema_version"], "schema_version"),
    )


def _decode_temporal_range(value: object, field_name: str) -> TemporalRange | None:
    if value is None:
        return None
    document = _object_mapping(value, field_name)
    _require_exact_keys(document, {"end_ms", "start_ms"}, field_name)
    return TemporalRange(
        start_ms=_integer(document["start_ms"], f"{field_name}.start_ms"),
        end_ms=_integer(document["end_ms"], f"{field_name}.end_ms"),
    )


def _unit_time_bounds(unit: SearchableUnit) -> tuple[int | None, int | None]:
    if unit.time_range is None:
        return None, None
    return unit.time_range.start_ms, unit.time_range.end_ms


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        dict(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _load_json_object(payload: str) -> dict[str, object]:
    try:
        value: object = json.loads(payload)
    except (TypeError, ValueError) as error:
        raise ValueError("catalog unit JSON is invalid") from error
    return _object_mapping(value, "unit")


def _object_mapping(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} must be a JSON object")
    return cast(dict[str, object], value)


def _object_sequence(value: object, field_name: str) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a JSON array")
    return tuple(value)


def _require_exact_keys(
    document: Mapping[str, object],
    expected: set[str],
    field_name: str,
) -> None:
    if set(document) != expected:
        raise ValueError(f"{field_name} contains unsupported fields")


def _string(value: object, field_name: str) -> str:
    _require_non_blank(field_name, value)
    return cast(str, value)


def _integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _optional_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return _integer(value, field_name)


def _filter_sha256(filters: SearchFilters) -> str:
    _require_filters(filters)
    document: dict[str, object] = {
        "artifact_refs": tuple(sorted(filters.artifact_refs)),
        "corpus_refs": tuple(sorted(filters.corpus_refs)),
        "media_types": tuple(sorted(filters.media_types)),
        "schema_version": filters.schema_version,
        "time_range": _temporal_range_document(filters.time_range),
    }
    return hashlib.sha256(_canonical_json_bytes(document)).hexdigest()


def _filters_are_empty(filters: SearchFilters) -> bool:
    return not (
        filters.corpus_refs
        or filters.artifact_refs
        or filters.media_types
        or filters.time_range is not None
    )


def _require_filters(filters: object) -> None:
    if not isinstance(filters, SearchFilters):
        raise TypeError("filters must be SearchFilters")
    if filters.schema_version not in (1, 2):
        raise ValueError("filter schema version is unsupported")
    if filters.text_origins:
        raise ValueError("image catalog does not support text origin filters")


def _bounded_string_tuple(field_name: str, values: object) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{field_name} must be a tuple")
    if len(values) > MAX_METADATA_LOOKUP_UNITS:
        raise ValueError(f"{field_name} must contain at most 100 values")
    for value in values:
        _require_non_blank(field_name, value)
    return cast(tuple[str, ...], values)


def _bounded_ordinal_tuple(values: object, *, unit_count: int) -> tuple[int, ...]:
    if not isinstance(values, tuple):
        raise TypeError("ordinals must be a tuple")
    if len(values) > MAX_METADATA_LOOKUP_UNITS:
        raise ValueError("ordinals must contain at most 100 values")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < unit_count
        for value in values
    ):
        raise ValueError("ordinals must be within the catalog unit range")
    if len(set(values)) != len(values):
        raise ValueError("ordinals must be unique")
    return cast(tuple[int, ...], values)


def _require_limit(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10_000:
        raise ValueError("limit must be an integer from 1 to 10,000")


def _require_deadline(deadline: object) -> None:
    if (
        not isinstance(deadline, datetime)
        or deadline.tzinfo is None
        or deadline.utcoffset() is None
    ):
        raise ValueError("deadline must be timezone-aware")
    if datetime.now(UTC) >= deadline:
        raise TimeoutError("unit catalog deadline exceeded")


def _pragma_int(connection: sqlite3.Connection, name: str) -> int:
    row = connection.execute(f"PRAGMA {name}").fetchone()
    if row is None:
        raise ValueError(f"catalog schema PRAGMA {name} is unavailable")
    return int(row[0])


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}


def _require_non_blank(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")


def _require_sha256(field_name: str, value: object) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hash")


def _require_positive_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


def _require_non_negative_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


EMPTY_FILTER_SHA = _filter_sha256(SearchFilters())
StagedUnitCatalog = SqliteUnitCatalogDescriptor


__all__ = [
    "EMPTY_FILTER_SHA",
    "MAX_METADATA_LOOKUP_UNITS",
    "MAX_PERSONAL_UNITS",
    "SQLITE_UNIT_CATALOG_APPLICATION_ID",
    "SQLITE_UNIT_CATALOG_FORMAT",
    "SQLITE_UNIT_CATALOG_SCHEMA_VERSION",
    "AllOrdinals",
    "BitmapOrdinals",
    "ContentCatalogMatch",
    "IndexedUnitRecord",
    "OrdinalSelection",
    "SparseOrdinals",
    "SqliteUnitCatalog",
    "SqliteUnitCatalogDescriptor",
    "SqliteUnitCatalogWriter",
    "StagedUnitCatalog",
]
