"""Private SQLite adapter for authoritative text-index generation metadata."""

from __future__ import annotations

import json
import sqlite3
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from capicove.contracts import (
    DistanceMetric,
    EmbeddingSpaceRef,
    ProcessingFingerprint,
    QueryModality,
    TargetType,
    VectorNormalization,
)
from capicove.retrieval.generations import (
    PinnedTextIndexGeneration,
    TextIndexGenerationCatalogError,
    TextIndexGenerationConflictError,
    TextIndexGenerationEvent,
    TextIndexGenerationEventType,
    TextIndexGenerationManifest,
    TextIndexGenerationRecord,
    TextIndexGenerationState,
    TextIndexGenerationTransitionError,
    TextIndexValidationReport,
)


_SCHEMA_VERSION = 1


class SqliteTextIndexGenerationCatalog:
    """Persist generation projections and immutable event snapshots in SQLite."""

    def __init__(self, database_path: Path, *, busy_timeout_seconds: float = 5.0) -> None:
        if not isinstance(database_path, Path) or not database_path.is_absolute():
            raise ValueError("database_path must be an absolute Path")
        if busy_timeout_seconds <= 0:
            raise ValueError("busy_timeout_seconds must be positive")
        self._database_path = database_path
        self._busy_timeout_ms = round(busy_timeout_seconds * 1000)
        try:
            _reject_symlink_components(database_path)
            database_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            _reject_symlink_components(database_path)
        except OSError as error:
            raise TextIndexGenerationCatalogError(
                "generation catalog path initialization failed"
            ) from error
        self._initialize()

    def register(
        self,
        manifest: TextIndexGenerationManifest,
    ) -> TextIndexGenerationRecord:
        """Register one immutable manifest idempotently."""
        record = TextIndexGenerationRecord(
            manifest=manifest,
            state=TextIndexGenerationState.BUILDING,
            revision=0,
        )
        with self._transaction() as connection:
            row = self._select(connection, manifest.generation_id)
            if row is not None:
                existing = _decode_record(str(row["record_json"]))
                if existing.manifest != manifest:
                    raise TextIndexGenerationConflictError(
                        "generation ID identifies a different manifest"
                    )
                return existing
            connection.execute(
                """
                INSERT INTO text_index_generations (
                    generation_id, workspace_ref, retriever_id, manifest_sha256,
                    state, record_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    manifest.generation_id,
                    manifest.workspace_ref,
                    manifest.retriever_id,
                    manifest.manifest_sha256,
                    record.state.value,
                    _record_json(record),
                ),
            )
            self._insert_event(
                connection,
                TextIndexGenerationEventType.REGISTERED,
                record,
            )
        return record

    def get(self, generation_id: str) -> TextIndexGenerationRecord | None:
        """Return one generation projection by canonical identity."""
        try:
            with self._connect() as connection:
                row = self._select(connection, generation_id)
        except sqlite3.Error as error:
            raise TextIndexGenerationCatalogError("generation read failed") from error
        return None if row is None else _decode_record(str(row["record_json"]))

    def record_physical_index(
        self,
        generation_id: str,
        physical_index_ref: str,
    ) -> TextIndexGenerationRecord:
        """Attach one immutable provider physical identity to a building record."""
        with self._transaction() as connection:
            current = self._require_record(connection, generation_id)
            if current.physical_index_ref is not None:
                if current.physical_index_ref != physical_index_ref:
                    raise TextIndexGenerationConflictError(
                        "generation identifies a different physical index"
                    )
                return current
            if current.state is not TextIndexGenerationState.BUILDING:
                raise TextIndexGenerationTransitionError(
                    "physical index can only be recorded while building"
                )
            updated = TextIndexGenerationRecord(
                manifest=current.manifest,
                state=current.state,
                physical_index_ref=physical_index_ref,
                revision=current.revision + 1,
            )
            self._update(
                connection,
                current,
                updated,
                TextIndexGenerationEventType.PHYSICAL_INDEX_CREATED,
            )
        return updated

    def record_validation(
        self,
        report: TextIndexValidationReport,
    ) -> TextIndexGenerationRecord:
        """Record validation once and close the building generation."""
        with self._transaction() as connection:
            current = self._require_record(connection, report.generation_id)
            self._require_report_identity(current, report)
            if current.validation is not None:
                if current.validation != report:
                    raise TextIndexGenerationConflictError(
                        "generation identifies different validation evidence"
                    )
                return current
            if (
                current.state is not TextIndexGenerationState.BUILDING
                or current.physical_index_ref is None
            ):
                raise TextIndexGenerationTransitionError(
                    "validation requires a building generation with a physical index"
                )
            if report.valid and report.mapping_sha256 != current.manifest.mapping_sha256:
                raise TextIndexGenerationConflictError(
                    "valid validation mapping does not match the manifest"
                )
            target = (
                TextIndexGenerationState.READY if report.valid else TextIndexGenerationState.CORRUPT
            )
            event_type = (
                TextIndexGenerationEventType.VALIDATION_SUCCEEDED
                if report.valid
                else TextIndexGenerationEventType.VALIDATION_FAILED
            )
            updated = TextIndexGenerationRecord(
                manifest=current.manifest,
                state=target,
                physical_index_ref=current.physical_index_ref,
                validation=report,
                revision=current.revision + 1,
            )
            self._update(connection, current, updated, event_type)
        return updated

    def record_activation(
        self,
        pinned: PinnedTextIndexGeneration,
    ) -> TextIndexGenerationRecord:
        """Atomically activate one exact validated pin and supersede its predecessor."""
        with self._transaction() as connection:
            current = self._require_record(connection, pinned.generation_id)
            self._require_pin_identity(current, pinned)
            if current.state is TextIndexGenerationState.ACTIVE:
                return current
            if current.state not in {
                TextIndexGenerationState.READY,
                TextIndexGenerationState.SUPERSEDED,
            }:
                raise TextIndexGenerationTransitionError(
                    f"cannot activate generation in {current.state.value} state"
                )

            active_row = connection.execute(
                """
                SELECT record_json FROM text_index_generations
                WHERE workspace_ref = ? AND retriever_id = ? AND state = ?
                """,
                (
                    current.manifest.workspace_ref,
                    current.manifest.retriever_id,
                    TextIndexGenerationState.ACTIVE.value,
                ),
            ).fetchone()
            if active_row is not None:
                predecessor = _decode_record(str(active_row["record_json"]))
                superseded = TextIndexGenerationRecord(
                    manifest=predecessor.manifest,
                    state=TextIndexGenerationState.SUPERSEDED,
                    physical_index_ref=predecessor.physical_index_ref,
                    validation=predecessor.validation,
                    revision=predecessor.revision + 1,
                )
                self._update(
                    connection,
                    predecessor,
                    superseded,
                    TextIndexGenerationEventType.SUPERSEDED,
                )

            updated = TextIndexGenerationRecord(
                manifest=current.manifest,
                state=TextIndexGenerationState.ACTIVE,
                physical_index_ref=current.physical_index_ref,
                validation=current.validation,
                revision=current.revision + 1,
            )
            self._update(
                connection,
                current,
                updated,
                TextIndexGenerationEventType.ACTIVATED,
            )
        return updated

    def active(
        self,
        *,
        workspace_ref: str,
        retriever_id: str,
    ) -> TextIndexGenerationRecord | None:
        """Return the active projection for one workspace/retriever stream."""
        _require_non_blank("workspace_ref", workspace_ref)
        _require_non_blank("retriever_id", retriever_id)
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT record_json FROM text_index_generations
                    WHERE workspace_ref = ? AND retriever_id = ? AND state = ?
                    """,
                    (
                        workspace_ref,
                        retriever_id,
                        TextIndexGenerationState.ACTIVE.value,
                    ),
                ).fetchone()
        except sqlite3.Error as error:
            raise TextIndexGenerationCatalogError("active generation read failed") from error
        return None if row is None else _decode_record(str(row["record_json"]))

    def history(self, generation_id: str) -> tuple[TextIndexGenerationEvent, ...]:
        """Return append-only event snapshots in global commit order."""
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT event_sequence, event_type, record_json, recorded_at
                    FROM text_index_generation_events
                    WHERE generation_id = ? ORDER BY event_sequence
                    """,
                    (generation_id,),
                ).fetchall()
        except sqlite3.Error as error:
            raise TextIndexGenerationCatalogError("generation history read failed") from error
        return tuple(
            TextIndexGenerationEvent(
                sequence=int(row["event_sequence"]),
                event_type=TextIndexGenerationEventType(str(row["event_type"])),
                record=_decode_record(str(row["record_json"])),
                recorded_at=datetime.fromisoformat(str(row["recorded_at"])),
            )
            for row in rows
        )

    def _initialize(self) -> None:
        try:
            with self._connect() as connection:
                version = int(connection.execute("PRAGMA user_version").fetchone()[0])
                if version not in {0, _SCHEMA_VERSION}:
                    raise TextIndexGenerationCatalogError(
                        f"unsupported generation catalog schema version: {version}"
                    )
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS text_index_generations (
                        generation_id TEXT PRIMARY KEY,
                        workspace_ref TEXT NOT NULL,
                        retriever_id TEXT NOT NULL,
                        manifest_sha256 TEXT NOT NULL UNIQUE,
                        state TEXT NOT NULL,
                        record_json TEXT NOT NULL
                    );
                    CREATE UNIQUE INDEX IF NOT EXISTS text_index_one_active_stream
                        ON text_index_generations (workspace_ref, retriever_id)
                        WHERE state = 'active';
                    CREATE TABLE IF NOT EXISTS text_index_generation_events (
                        event_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        generation_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        record_json TEXT NOT NULL,
                        recorded_at TEXT NOT NULL,
                        FOREIGN KEY (generation_id)
                            REFERENCES text_index_generations(generation_id)
                    );
                    CREATE INDEX IF NOT EXISTS text_index_generation_history
                        ON text_index_generation_events (generation_id, event_sequence);
                    """
                )
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            self._database_path.chmod(0o600)
        except sqlite3.Error as error:
            raise TextIndexGenerationCatalogError(
                "generation catalog initialization failed"
            ) from error
        except OSError as error:
            raise TextIndexGenerationCatalogError(
                "generation catalog permissions could not be secured"
            ) from error

    def _connect(self) -> sqlite3.Connection:
        _reject_symlink_components(self._database_path)
        connection = sqlite3.connect(
            self._database_path.as_uri() + "?nofollow=1",
            timeout=self._busy_timeout_ms / 1000,
            isolation_level=None,
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except (
            TextIndexGenerationConflictError,
            TextIndexGenerationTransitionError,
        ):
            connection.rollback()
            raise
        except sqlite3.IntegrityError as error:
            connection.rollback()
            raise TextIndexGenerationConflictError(
                "generation catalog uniqueness constraint failed"
            ) from error
        except sqlite3.Error as error:
            connection.rollback()
            raise TextIndexGenerationCatalogError(
                "generation catalog transaction failed"
            ) from error
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _select(
        connection: sqlite3.Connection,
        generation_id: str,
    ) -> sqlite3.Row | None:
        return cast(
            sqlite3.Row | None,
            connection.execute(
                "SELECT record_json FROM text_index_generations WHERE generation_id = ?",
                (generation_id,),
            ).fetchone(),
        )

    def _require_record(
        self,
        connection: sqlite3.Connection,
        generation_id: str,
    ) -> TextIndexGenerationRecord:
        row = self._select(connection, generation_id)
        if row is None:
            raise TextIndexGenerationTransitionError("generation does not exist")
        return _decode_record(str(row["record_json"]))

    @staticmethod
    def _require_report_identity(
        record: TextIndexGenerationRecord,
        report: TextIndexValidationReport,
    ) -> None:
        if report.manifest_sha256 != record.manifest.manifest_sha256:
            raise TextIndexGenerationConflictError(
                "validation manifest identity does not match generation manifest"
            )
        if report.physical_index_ref != record.physical_index_ref:
            raise TextIndexGenerationConflictError(
                "validation physical index does not match generation"
            )
        if report.expected_unit_count != record.manifest.expected_unit_count:
            raise TextIndexGenerationConflictError(
                "validation expected count does not match generation manifest"
            )

    @staticmethod
    def _require_pin_identity(
        record: TextIndexGenerationRecord,
        pinned: PinnedTextIndexGeneration,
    ) -> None:
        if pinned.manifest_sha256 != record.manifest.manifest_sha256:
            raise TextIndexGenerationConflictError(
                "activation pin manifest does not match generation manifest"
            )
        if pinned.physical_index_ref != record.physical_index_ref:
            raise TextIndexGenerationConflictError(
                "activation pin physical index does not match generation"
            )

    def _update(
        self,
        connection: sqlite3.Connection,
        current: TextIndexGenerationRecord,
        updated: TextIndexGenerationRecord,
        event_type: TextIndexGenerationEventType,
    ) -> None:
        result = connection.execute(
            """
            UPDATE text_index_generations SET state = ?, record_json = ?
            WHERE generation_id = ? AND state = ? AND record_json = ?
            """,
            (
                updated.state.value,
                _record_json(updated),
                current.manifest.generation_id,
                current.state.value,
                _record_json(current),
            ),
        )
        if result.rowcount != 1:
            raise TextIndexGenerationTransitionError("generation projection changed concurrently")
        self._insert_event(connection, event_type, updated)

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection,
        event_type: TextIndexGenerationEventType,
        record: TextIndexGenerationRecord,
    ) -> None:
        connection.execute(
            """
            INSERT INTO text_index_generation_events (
                generation_id, event_type, record_json, recorded_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                record.manifest.generation_id,
                event_type.value,
                _record_json(record),
                datetime.now(UTC).isoformat(),
            ),
        )


def _record_json(record: TextIndexGenerationRecord) -> str:
    value: dict[str, object] = {
        "manifest": _encode_manifest(record.manifest),
        "state": record.state.value,
        "physical_index_ref": record.physical_index_ref,
        "validation": (
            None if record.validation is None else _encode_validation(record.validation)
        ),
        "revision": record.revision,
    }
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _decode_record(encoded: str) -> TextIndexGenerationRecord:
    value = _mapping(json.loads(encoded), "generation record")
    validation_value = value.get("validation")
    return TextIndexGenerationRecord(
        manifest=_decode_manifest(_mapping(value.get("manifest"), "manifest")),
        state=TextIndexGenerationState(_string(value.get("state"), "state")),
        physical_index_ref=_optional_string(
            value.get("physical_index_ref"),
            "physical_index_ref",
        ),
        validation=(
            None
            if validation_value is None
            else _decode_validation(_mapping(validation_value, "validation"))
        ),
        revision=_integer(value.get("revision"), "revision"),
    )


def _encode_manifest(manifest: TextIndexGenerationManifest) -> dict[str, object]:
    return {
        "generation_id": manifest.generation_id,
        "workspace_ref": manifest.workspace_ref,
        "corpus_refs": list(manifest.corpus_refs),
        "searchable_unit_manifest_sha256": manifest.searchable_unit_manifest_sha256,
        "content_sha256": manifest.content_sha256,
        "mapping_sha256": manifest.mapping_sha256,
        "embedding_spaces": [_encode_space(space) for space in manifest.embedding_spaces],
        "retriever_id": manifest.retriever_id,
        "index_format_version": manifest.index_format_version,
        "build_processing_fingerprint": manifest.build_processing_fingerprint.sha256,
        "expected_unit_count": manifest.expected_unit_count,
        "schema_version": manifest.schema_version,
    }


def _decode_manifest(value: Mapping[str, object]) -> TextIndexGenerationManifest:
    spaces = _sequence(value.get("embedding_spaces"), "embedding_spaces")
    return TextIndexGenerationManifest(
        generation_id=_string(value.get("generation_id"), "generation_id"),
        workspace_ref=_string(value.get("workspace_ref"), "workspace_ref"),
        corpus_refs=tuple(
            _string(item, "corpus_ref")
            for item in _sequence(value.get("corpus_refs"), "corpus_refs")
        ),
        searchable_unit_manifest_sha256=_string(
            value.get("searchable_unit_manifest_sha256"),
            "searchable_unit_manifest_sha256",
        ),
        content_sha256=_string(value.get("content_sha256"), "content_sha256"),
        mapping_sha256=_string(value.get("mapping_sha256"), "mapping_sha256"),
        embedding_spaces=tuple(
            _decode_space(_mapping(space, "embedding_space")) for space in spaces
        ),
        retriever_id=_string(value.get("retriever_id"), "retriever_id"),
        index_format_version=_string(
            value.get("index_format_version"),
            "index_format_version",
        ),
        build_processing_fingerprint=ProcessingFingerprint(
            _string(
                value.get("build_processing_fingerprint"),
                "build_processing_fingerprint",
            )
        ),
        expected_unit_count=_integer(
            value.get("expected_unit_count"),
            "expected_unit_count",
        ),
        schema_version=_integer(value.get("schema_version"), "schema_version"),
    )


def _encode_space(space: EmbeddingSpaceRef) -> dict[str, object]:
    return {
        "space_id": space.space_id,
        "provider_id": space.provider_id,
        "model_revision": space.model_revision,
        "preprocessing_version": space.preprocessing_version,
        "dimension": space.dimension,
        "normalization": space.normalization.value,
        "distance_metric": space.distance_metric.value,
        "query_modalities": sorted(value.value for value in space.query_modalities),
        "target_modalities": sorted(value.value for value in space.target_modalities),
        "schema_version": space.schema_version,
    }


def _decode_space(value: Mapping[str, object]) -> EmbeddingSpaceRef:
    return EmbeddingSpaceRef(
        space_id=_string(value.get("space_id"), "space_id"),
        provider_id=_string(value.get("provider_id"), "provider_id"),
        model_revision=_string(value.get("model_revision"), "model_revision"),
        preprocessing_version=_string(
            value.get("preprocessing_version"),
            "preprocessing_version",
        ),
        dimension=_integer(value.get("dimension"), "dimension"),
        normalization=VectorNormalization(_string(value.get("normalization"), "normalization")),
        distance_metric=DistanceMetric(_string(value.get("distance_metric"), "distance_metric")),
        query_modalities=frozenset(
            QueryModality(_string(item, "query_modality"))
            for item in _sequence(value.get("query_modalities"), "query_modalities")
        ),
        target_modalities=frozenset(
            TargetType(_string(item, "target_modality"))
            for item in _sequence(value.get("target_modalities"), "target_modalities")
        ),
        schema_version=_integer(value.get("schema_version"), "schema_version"),
    )


def _encode_validation(report: TextIndexValidationReport) -> dict[str, object]:
    return {
        "generation_id": report.generation_id,
        "manifest_sha256": report.manifest_sha256,
        "physical_index_ref": report.physical_index_ref,
        "mapping_sha256": report.mapping_sha256,
        "expected_unit_count": report.expected_unit_count,
        "observed_unit_count": report.observed_unit_count,
        "issue_codes": list(report.issue_codes),
    }


def _decode_validation(value: Mapping[str, object]) -> TextIndexValidationReport:
    return TextIndexValidationReport(
        generation_id=_string(value.get("generation_id"), "generation_id"),
        manifest_sha256=_string(value.get("manifest_sha256"), "manifest_sha256"),
        physical_index_ref=_string(
            value.get("physical_index_ref"),
            "physical_index_ref",
        ),
        mapping_sha256=_string(value.get("mapping_sha256"), "mapping_sha256"),
        expected_unit_count=_integer(
            value.get("expected_unit_count"),
            "expected_unit_count",
        ),
        observed_unit_count=_integer(
            value.get("observed_unit_count"),
            "observed_unit_count",
        ),
        issue_codes=tuple(
            _string(item, "issue_code")
            for item in _sequence(value.get("issue_codes"), "issue_codes")
        ),
    )


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TextIndexGenerationCatalogError(f"stored {field_name} is invalid")
    return cast(Mapping[str, object], value)


def _sequence(value: object, field_name: str) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise TextIndexGenerationCatalogError(f"stored {field_name} is invalid")
    return tuple(value)


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TextIndexGenerationCatalogError(f"stored {field_name} is invalid")
    return value


def _optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _string(value, field_name)


def _integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TextIndexGenerationCatalogError(f"stored {field_name} is invalid")
    return value


def _require_non_blank(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 500:
        raise ValueError(f"{field_name} must be non-blank and bounded")


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("database_path must not contain a symbolic link")
    if path.exists() and not path.is_file():
        raise ValueError("database_path must identify a regular file")


__all__ = ["SqliteTextIndexGenerationCatalog"]
