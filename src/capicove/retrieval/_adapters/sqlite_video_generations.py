"""Private SQLite catalog for atomic video generation sets and query leases."""

from __future__ import annotations

import json
import sqlite3
import stat
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

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
    TextIndexGenerationManifest,
)
from capicove.retrieval.video_models import (
    PinnedVideoGenerationSet,
    PinnedVisualIndexGeneration,
    VideoGenerationConflictError,
    VideoGenerationError,
    VideoGenerationSetEvent,
    VideoGenerationSetEventType,
    VideoGenerationSetManifest,
    VideoGenerationSetRecord,
    VideoGenerationSetState,
    VideoGenerationTransitionError,
    VisualIndexGenerationManifest,
)


_SCHEMA_VERSION = 1


class SqliteVideoGenerationSetCatalog:
    """Persist complete generation aliases and durable query lease counts."""

    def __init__(self, database_path: Path, *, busy_timeout_seconds: float = 5.0) -> None:
        if not isinstance(database_path, Path) or not database_path.is_absolute():
            raise ValueError("database_path must be an absolute Path")
        if busy_timeout_seconds <= 0:
            raise ValueError("busy_timeout_seconds must be positive")
        _reject_symlink_components(database_path)
        if database_path.exists():
            mode = database_path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError("database_path must not be a symlink")
            if not stat.S_ISREG(mode):
                raise ValueError("database_path must be a regular file")
        database_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _reject_symlink_components(database_path)
        self._database_path = database_path
        self._busy_timeout_ms = round(busy_timeout_seconds * 1_000)
        self._initialize()

    def register(self, record: VideoGenerationSetRecord) -> VideoGenerationSetRecord:
        """Register one fully pinned ready set idempotently."""
        if not isinstance(record, VideoGenerationSetRecord):
            raise TypeError("record must be a VideoGenerationSetRecord")
        if record.state is not VideoGenerationSetState.READY or record.revision != 0:
            raise VideoGenerationTransitionError(
                "registration requires a revision-zero ready generation set"
            )
        with self._transaction() as connection:
            existing = self._select(connection, record.manifest.set_id)
            if existing is not None:
                if existing.manifest != record.manifest or existing.pin != record.pin:
                    raise VideoGenerationConflictError(
                        "generation set ID identifies different immutable evidence"
                    )
                return existing
            connection.execute(
                """
                INSERT INTO video_generation_sets(
                    set_id, workspace_ref, profile_ref, state, manifest_sha256,
                    record_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.manifest.set_id,
                    record.manifest.workspace_ref,
                    record.manifest.profile_ref,
                    record.state.value,
                    record.manifest.manifest_sha256,
                    _record_json(record),
                ),
            )
            self._insert_event(connection, VideoGenerationSetEventType.REGISTERED, record)
        return record

    def get(self, set_id: str) -> VideoGenerationSetRecord | None:
        """Return one authoritative generation-set projection."""
        _require_non_blank("set_id", set_id)
        with self._read_connection() as connection:
            return self._select(connection, set_id)

    def record_activation(self, pin: PinnedVideoGenerationSet) -> VideoGenerationSetRecord:
        """Atomically switch the complete workspace/profile alias."""
        if not isinstance(pin, PinnedVideoGenerationSet):
            raise TypeError("pin must be a PinnedVideoGenerationSet")
        with self._transaction() as connection:
            current = self._require_record(connection, pin.set_id)
            if current.pin != pin:
                raise VideoGenerationTransitionError(
                    "activation pin does not match registered immutable evidence"
                )
            if current.state is VideoGenerationSetState.ACTIVE:
                active = self._active_in_connection(
                    connection,
                    workspace_ref=current.manifest.workspace_ref,
                    profile_ref=current.manifest.profile_ref,
                )
                if active is None or active.manifest.set_id != current.manifest.set_id:
                    raise VideoGenerationTransitionError(
                        "active record and complete alias are inconsistent"
                    )
                return current
            if current.state not in {
                VideoGenerationSetState.READY,
                VideoGenerationSetState.SUPERSEDED,
            }:
                raise VideoGenerationTransitionError(
                    f"cannot activate generation set in {current.state.value} state"
                )
            predecessor = self._active_in_connection(
                connection,
                workspace_ref=current.manifest.workspace_ref,
                profile_ref=current.manifest.profile_ref,
            )
            if predecessor is not None and predecessor.manifest.set_id != current.manifest.set_id:
                superseded = VideoGenerationSetRecord(
                    predecessor.manifest,
                    predecessor.pin,
                    VideoGenerationSetState.SUPERSEDED,
                    predecessor.revision + 1,
                )
                self._update(
                    connection,
                    predecessor,
                    superseded,
                    VideoGenerationSetEventType.SUPERSEDED,
                )
            activated = VideoGenerationSetRecord(
                current.manifest,
                current.pin,
                VideoGenerationSetState.ACTIVE,
                current.revision + 1,
            )
            self._update(
                connection,
                current,
                activated,
                VideoGenerationSetEventType.ACTIVATED,
            )
            connection.execute(
                """
                INSERT INTO active_video_generation_sets(
                    workspace_ref, profile_ref, set_id, manifest_sha256
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(workspace_ref, profile_ref) DO UPDATE SET
                    set_id = excluded.set_id,
                    manifest_sha256 = excluded.manifest_sha256
                """,
                (
                    activated.manifest.workspace_ref,
                    activated.manifest.profile_ref,
                    activated.manifest.set_id,
                    activated.manifest.manifest_sha256,
                ),
            )
        return activated

    def active(self, *, workspace_ref: str, profile_ref: str) -> VideoGenerationSetRecord | None:
        """Resolve one complete active alias, never individual member aliases."""
        _require_non_blank("workspace_ref", workspace_ref)
        _require_non_blank("profile_ref", profile_ref)
        with self._read_connection() as connection:
            return self._active_in_connection(
                connection, workspace_ref=workspace_ref, profile_ref=profile_ref
            )

    def history(self, set_id: str) -> tuple[VideoGenerationSetEvent, ...]:
        """Return append-only lifecycle snapshots for one set."""
        _require_non_blank("set_id", set_id)
        with self._read_connection() as connection:
            rows = connection.execute(
                """
                SELECT sequence, event_type, record_json, recorded_at
                FROM video_generation_events
                WHERE set_id = ? ORDER BY sequence
                """,
                (set_id,),
            ).fetchall()
        return tuple(
            VideoGenerationSetEvent(
                sequence=int(row["sequence"]),
                event_type=VideoGenerationSetEventType(str(row["event_type"])),
                record=_decode_record(str(row["record_json"])),
                recorded_at=datetime.fromisoformat(str(row["recorded_at"])),
            )
            for row in rows
        )

    def acquire_active_lease(
        self, *, workspace_ref: str, profile_ref: str
    ) -> tuple[str, PinnedVideoGenerationSet]:
        """Atomically resolve and retain the exact active complete-set pin."""
        _require_non_blank("workspace_ref", workspace_ref)
        _require_non_blank("profile_ref", profile_ref)
        with self._transaction() as connection:
            active = self._active_in_connection(
                connection, workspace_ref=workspace_ref, profile_ref=profile_ref
            )
            if active is None:
                raise VideoGenerationTransitionError(
                    "no active video generation set for workspace and profile"
                )
            lease_id = uuid.uuid4().hex
            connection.execute(
                """
                INSERT INTO video_generation_leases(
                    lease_id, set_id, manifest_sha256, acquired_at, released_at
                ) VALUES (?, ?, ?, ?, NULL)
                """,
                (
                    lease_id,
                    active.manifest.set_id,
                    active.manifest.manifest_sha256,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return lease_id, active.pin

    def release_lease(self, lease_id: str) -> None:
        """Idempotently release one exact lease."""
        _require_non_blank("lease_id", lease_id)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT released_at FROM video_generation_leases WHERE lease_id = ?",
                (lease_id,),
            ).fetchone()
            if row is None:
                raise VideoGenerationTransitionError("video generation lease does not exist")
            if row["released_at"] is not None:
                return
            connection.execute(
                """
                UPDATE video_generation_leases SET released_at = ?
                WHERE lease_id = ? AND released_at IS NULL
                """,
                (datetime.now(UTC).isoformat(), lease_id),
            )

    def lease_count(self, set_id: str) -> int:
        """Return the number of active leases retaining one set."""
        _require_non_blank("set_id", set_id)
        with self._read_connection() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) FROM video_generation_leases
                WHERE set_id = ? AND released_at IS NULL
                """,
                (set_id,),
            ).fetchone()
        return 0 if row is None else int(row[0])

    def can_evict(self, set_id: str) -> bool:
        """Return whether a non-active set has no remaining query leases."""
        record = self.get(set_id)
        if record is None:
            raise VideoGenerationTransitionError("video generation set does not exist")
        return (
            record.state
            in {
                VideoGenerationSetState.SUPERSEDED,
                VideoGenerationSetState.CORRUPT,
            }
            and self.lease_count(set_id) == 0
        )

    def record_eviction(self, set_id: str) -> VideoGenerationSetRecord:
        """Record logical eviction only after every query lease has released."""
        with self._transaction() as connection:
            current = self._require_record(connection, set_id)
            row = connection.execute(
                """
                SELECT COUNT(*) FROM video_generation_leases
                WHERE set_id = ? AND released_at IS NULL
                """,
                (set_id,),
            ).fetchone()
            if current.state is VideoGenerationSetState.EVICTED:
                return current
            if (
                current.state
                not in {
                    VideoGenerationSetState.SUPERSEDED,
                    VideoGenerationSetState.CORRUPT,
                }
                or row is None
                or int(row[0]) != 0
            ):
                raise VideoGenerationTransitionError(
                    "generation set is active, ready, or retained by a query lease"
                )
            evicted = VideoGenerationSetRecord(
                current.manifest,
                current.pin,
                VideoGenerationSetState.EVICTED,
                current.revision + 1,
            )
            self._update(
                connection,
                current,
                evicted,
                VideoGenerationSetEventType.EVICTED,
            )
        return evicted

    def close(self) -> None:
        """Release resources; connections are transaction-scoped."""

    def _initialize(self) -> None:
        try:
            with self._transaction(initialize=True) as connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS video_generation_sets (
                        set_id TEXT PRIMARY KEY,
                        workspace_ref TEXT NOT NULL,
                        profile_ref TEXT NOT NULL,
                        state TEXT NOT NULL,
                        manifest_sha256 TEXT NOT NULL,
                        record_json TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS video_generation_stream_idx
                        ON video_generation_sets(workspace_ref, profile_ref, state);
                    CREATE TABLE IF NOT EXISTS active_video_generation_sets (
                        workspace_ref TEXT NOT NULL,
                        profile_ref TEXT NOT NULL,
                        set_id TEXT NOT NULL,
                        manifest_sha256 TEXT NOT NULL,
                        PRIMARY KEY (workspace_ref, profile_ref),
                        FOREIGN KEY (set_id) REFERENCES video_generation_sets(set_id)
                    ) WITHOUT ROWID;
                    CREATE TABLE IF NOT EXISTS video_generation_events (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        set_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        record_json TEXT NOT NULL,
                        recorded_at TEXT NOT NULL,
                        FOREIGN KEY (set_id) REFERENCES video_generation_sets(set_id)
                    );
                    CREATE INDEX IF NOT EXISTS video_generation_events_set_idx
                        ON video_generation_events(set_id, sequence);
                    CREATE TABLE IF NOT EXISTS video_generation_leases (
                        lease_id TEXT PRIMARY KEY,
                        set_id TEXT NOT NULL,
                        manifest_sha256 TEXT NOT NULL,
                        acquired_at TEXT NOT NULL,
                        released_at TEXT,
                        FOREIGN KEY (set_id) REFERENCES video_generation_sets(set_id)
                    );
                    CREATE INDEX IF NOT EXISTS video_generation_leases_active_idx
                        ON video_generation_leases(set_id, released_at);
                    CREATE TABLE IF NOT EXISTS metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    ) WITHOUT ROWID;
                    """
                )
                row = connection.execute(
                    "SELECT value FROM metadata WHERE key = 'schema_version'"
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO metadata(key, value) VALUES ('schema_version', ?)",
                        (str(_SCHEMA_VERSION),),
                    )
                elif str(row[0]) != str(_SCHEMA_VERSION):
                    raise VideoGenerationError(
                        "unsupported video generation catalog schema version"
                    )
        except sqlite3.Error as error:
            raise VideoGenerationError("video generation catalog initialization failed") from error

    @contextmanager
    def _transaction(self, *, initialize: bool = False) -> Iterator[sqlite3.Connection]:
        connection = self._open_connection()
        try:
            if initialize:
                connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def _read_connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._open_connection()
        try:
            yield connection
        finally:
            connection.close()

    def _open_connection(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(
                self._database_path,
                timeout=self._busy_timeout_ms / 1_000,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
            connection.execute("PRAGMA synchronous = FULL")
            return connection
        except sqlite3.Error as error:
            raise VideoGenerationError("video generation catalog open failed") from error

    def _select(
        self, connection: sqlite3.Connection, set_id: str
    ) -> VideoGenerationSetRecord | None:
        row = connection.execute(
            "SELECT record_json FROM video_generation_sets WHERE set_id = ?",
            (set_id,),
        ).fetchone()
        return None if row is None else _decode_record(str(row["record_json"]))

    def _require_record(
        self, connection: sqlite3.Connection, set_id: str
    ) -> VideoGenerationSetRecord:
        record = self._select(connection, set_id)
        if record is None:
            raise VideoGenerationTransitionError("video generation set does not exist")
        return record

    def _active_in_connection(
        self,
        connection: sqlite3.Connection,
        *,
        workspace_ref: str,
        profile_ref: str,
    ) -> VideoGenerationSetRecord | None:
        row = connection.execute(
            """
            SELECT g.record_json, a.manifest_sha256
            FROM active_video_generation_sets AS a
            JOIN video_generation_sets AS g ON g.set_id = a.set_id
            WHERE a.workspace_ref = ? AND a.profile_ref = ?
            """,
            (workspace_ref, profile_ref),
        ).fetchone()
        if row is None:
            return None
        record = _decode_record(str(row["record_json"]))
        if (
            record.state is not VideoGenerationSetState.ACTIVE
            or str(row["manifest_sha256"]) != record.manifest.manifest_sha256
        ):
            raise VideoGenerationTransitionError(
                "active complete-set alias does not match its record"
            )
        return record

    def _update(
        self,
        connection: sqlite3.Connection,
        current: VideoGenerationSetRecord,
        updated: VideoGenerationSetRecord,
        event_type: VideoGenerationSetEventType,
    ) -> None:
        cursor = connection.execute(
            """
            UPDATE video_generation_sets
            SET state = ?, record_json = ?
            WHERE set_id = ? AND state = ? AND record_json = ?
            """,
            (
                updated.state.value,
                _record_json(updated),
                current.manifest.set_id,
                current.state.value,
                _record_json(current),
            ),
        )
        if cursor.rowcount != 1:
            raise VideoGenerationTransitionError("video generation set changed during transition")
        self._insert_event(connection, event_type, updated)

    def _insert_event(
        self,
        connection: sqlite3.Connection,
        event_type: VideoGenerationSetEventType,
        record: VideoGenerationSetRecord,
    ) -> None:
        connection.execute(
            """
            INSERT INTO video_generation_events(
                set_id, event_type, record_json, recorded_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                record.manifest.set_id,
                event_type.value,
                _record_json(record),
                datetime.now(UTC).isoformat(),
            ),
        )


def _record_json(record: VideoGenerationSetRecord) -> str:
    return json.dumps(
        {
            "manifest": _set_manifest_dict(record.manifest),
            "pin": _set_pin_dict(record.pin),
            "state": record.state.value,
            "revision": record.revision,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _decode_record(payload: str) -> VideoGenerationSetRecord:
    try:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise TypeError("record must be an object")
        manifest_value = _mapping(value, "manifest")
        pin_value = _mapping(value, "pin")
        return VideoGenerationSetRecord(
            manifest=_set_manifest_from_dict(manifest_value),
            pin=_set_pin_from_dict(pin_value),
            state=VideoGenerationSetState(_string(value, "state")),
            revision=_integer(value, "revision"),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise VideoGenerationError("stored video generation record is invalid") from error


def _set_manifest_dict(manifest: VideoGenerationSetManifest) -> dict[str, object]:
    return {
        "set_id": manifest.set_id,
        "workspace_ref": manifest.workspace_ref,
        "profile_ref": manifest.profile_ref,
        "corpus_refs": list(manifest.corpus_refs),
        "unit_catalog_ref": manifest.unit_catalog_ref,
        "unit_manifest_sha256": manifest.unit_manifest_sha256,
        "keyframe_visual": _visual_manifest_dict(manifest.keyframe_visual),
        "segment_visual": _visual_manifest_dict(manifest.segment_visual),
        "asr_text": _text_manifest_dict(manifest.asr_text),
        "ocr_text": (None if manifest.ocr_text is None else _text_manifest_dict(manifest.ocr_text)),
        "fusion_profile_ref": manifest.fusion_profile_ref,
        "schema_version": manifest.schema_version,
        "manifest_sha256": manifest.manifest_sha256,
    }


def _set_manifest_from_dict(value: dict[str, object]) -> VideoGenerationSetManifest:
    ocr_value = value.get("ocr_text")
    if ocr_value is not None and not isinstance(ocr_value, dict):
        raise TypeError("ocr_text must be an object or null")
    manifest = VideoGenerationSetManifest(
        set_id=_string(value, "set_id"),
        workspace_ref=_string(value, "workspace_ref"),
        profile_ref=_string(value, "profile_ref"),
        corpus_refs=tuple(_strings(value, "corpus_refs")),
        unit_catalog_ref=_string(value, "unit_catalog_ref"),
        unit_manifest_sha256=_string(value, "unit_manifest_sha256"),
        keyframe_visual=_visual_manifest_from_dict(_mapping(value, "keyframe_visual")),
        segment_visual=_visual_manifest_from_dict(_mapping(value, "segment_visual")),
        asr_text=_text_manifest_from_dict(_mapping(value, "asr_text")),
        ocr_text=None if ocr_value is None else _text_manifest_from_dict(ocr_value),
        fusion_profile_ref=_string(value, "fusion_profile_ref"),
        schema_version=_integer(value, "schema_version"),
    )
    if value.get("manifest_sha256") != manifest.manifest_sha256:
        raise ValueError("stored generation-set manifest digest is invalid")
    return manifest


def _set_pin_dict(pin: PinnedVideoGenerationSet) -> dict[str, object]:
    return {
        "set_id": pin.set_id,
        "manifest_sha256": pin.manifest_sha256,
        "workspace_ref": pin.workspace_ref,
        "profile_ref": pin.profile_ref,
        "unit_catalog_ref": pin.unit_catalog_ref,
        "unit_manifest_sha256": pin.unit_manifest_sha256,
        "keyframe_visual": _visual_pin_dict(pin.keyframe_visual),
        "segment_visual": _visual_pin_dict(pin.segment_visual),
        "asr_text": _text_pin_dict(pin.asr_text),
        "ocr_text": None if pin.ocr_text is None else _text_pin_dict(pin.ocr_text),
        "fusion_profile_ref": pin.fusion_profile_ref,
    }


def _set_pin_from_dict(value: dict[str, object]) -> PinnedVideoGenerationSet:
    ocr_value = value.get("ocr_text")
    if ocr_value is not None and not isinstance(ocr_value, dict):
        raise TypeError("OCR pin must be an object or null")
    return PinnedVideoGenerationSet(
        set_id=_string(value, "set_id"),
        manifest_sha256=_string(value, "manifest_sha256"),
        workspace_ref=_string(value, "workspace_ref"),
        profile_ref=_string(value, "profile_ref"),
        unit_catalog_ref=_string(value, "unit_catalog_ref"),
        unit_manifest_sha256=_string(value, "unit_manifest_sha256"),
        keyframe_visual=_visual_pin_from_dict(_mapping(value, "keyframe_visual")),
        segment_visual=_visual_pin_from_dict(_mapping(value, "segment_visual")),
        asr_text=_text_pin_from_dict(_mapping(value, "asr_text")),
        ocr_text=None if ocr_value is None else _text_pin_from_dict(ocr_value),
        fusion_profile_ref=_string(value, "fusion_profile_ref"),
    )


def _visual_manifest_dict(manifest: VisualIndexGenerationManifest) -> dict[str, object]:
    return {
        "generation_id": manifest.generation_id,
        "workspace_ref": manifest.workspace_ref,
        "corpus_refs": list(manifest.corpus_refs),
        "target_type": manifest.target_type.value,
        "searchable_unit_manifest_sha256": manifest.searchable_unit_manifest_sha256,
        "content_sha256": manifest.content_sha256,
        "embedding_space": _space_dict(manifest.embedding_space),
        "retriever_id": manifest.retriever_id,
        "index_format_version": manifest.index_format_version,
        "build_processing_fingerprint": manifest.build_processing_fingerprint.sha256,
        "expected_unit_count": manifest.expected_unit_count,
        "schema_version": manifest.schema_version,
        "manifest_sha256": manifest.manifest_sha256,
    }


def _visual_manifest_from_dict(value: dict[str, object]) -> VisualIndexGenerationManifest:
    manifest = VisualIndexGenerationManifest(
        generation_id=_string(value, "generation_id"),
        workspace_ref=_string(value, "workspace_ref"),
        corpus_refs=tuple(_strings(value, "corpus_refs")),
        target_type=TargetType(_string(value, "target_type")),
        searchable_unit_manifest_sha256=_string(value, "searchable_unit_manifest_sha256"),
        content_sha256=_string(value, "content_sha256"),
        embedding_space=_space_from_dict(_mapping(value, "embedding_space")),
        retriever_id=_string(value, "retriever_id"),
        index_format_version=_string(value, "index_format_version"),
        build_processing_fingerprint=ProcessingFingerprint(
            _string(value, "build_processing_fingerprint")
        ),
        expected_unit_count=_integer(value, "expected_unit_count"),
        schema_version=_integer(value, "schema_version"),
    )
    if value.get("manifest_sha256") != manifest.manifest_sha256:
        raise ValueError("stored visual manifest digest is invalid")
    return manifest


def _text_manifest_dict(manifest: TextIndexGenerationManifest) -> dict[str, object]:
    return {
        "generation_id": manifest.generation_id,
        "workspace_ref": manifest.workspace_ref,
        "corpus_refs": list(manifest.corpus_refs),
        "searchable_unit_manifest_sha256": manifest.searchable_unit_manifest_sha256,
        "content_sha256": manifest.content_sha256,
        "mapping_sha256": manifest.mapping_sha256,
        "embedding_spaces": [_space_dict(space) for space in manifest.embedding_spaces],
        "retriever_id": manifest.retriever_id,
        "index_format_version": manifest.index_format_version,
        "build_processing_fingerprint": manifest.build_processing_fingerprint.sha256,
        "expected_unit_count": manifest.expected_unit_count,
        "schema_version": manifest.schema_version,
        "manifest_sha256": manifest.manifest_sha256,
    }


def _text_manifest_from_dict(value: dict[str, object]) -> TextIndexGenerationManifest:
    spaces = value.get("embedding_spaces")
    if not isinstance(spaces, list) or any(not isinstance(space, dict) for space in spaces):
        raise TypeError("embedding_spaces must be a list of objects")
    manifest = TextIndexGenerationManifest(
        generation_id=_string(value, "generation_id"),
        workspace_ref=_string(value, "workspace_ref"),
        corpus_refs=tuple(_strings(value, "corpus_refs")),
        searchable_unit_manifest_sha256=_string(value, "searchable_unit_manifest_sha256"),
        content_sha256=_string(value, "content_sha256"),
        mapping_sha256=_string(value, "mapping_sha256"),
        embedding_spaces=tuple(_space_from_dict(space) for space in spaces),
        retriever_id=_string(value, "retriever_id"),
        index_format_version=_string(value, "index_format_version"),
        build_processing_fingerprint=ProcessingFingerprint(
            _string(value, "build_processing_fingerprint")
        ),
        expected_unit_count=_integer(value, "expected_unit_count"),
        schema_version=_integer(value, "schema_version"),
    )
    if value.get("manifest_sha256") != manifest.manifest_sha256:
        raise ValueError("stored text manifest digest is invalid")
    return manifest


def _visual_pin_dict(pin: PinnedVisualIndexGeneration) -> dict[str, object]:
    return {
        "generation_id": pin.generation_id,
        "manifest_sha256": pin.manifest_sha256,
        "physical_index_ref": pin.physical_index_ref,
        "target_type": pin.target_type.value,
        "embedding_space": _space_dict(pin.embedding_space),
    }


def _visual_pin_from_dict(value: dict[str, object]) -> PinnedVisualIndexGeneration:
    return PinnedVisualIndexGeneration(
        generation_id=_string(value, "generation_id"),
        manifest_sha256=_string(value, "manifest_sha256"),
        physical_index_ref=_string(value, "physical_index_ref"),
        target_type=TargetType(_string(value, "target_type")),
        embedding_space=_space_from_dict(_mapping(value, "embedding_space")),
    )


def _text_pin_dict(pin: PinnedTextIndexGeneration) -> dict[str, object]:
    return {
        "generation_id": pin.generation_id,
        "manifest_sha256": pin.manifest_sha256,
        "physical_index_ref": pin.physical_index_ref,
    }


def _text_pin_from_dict(value: dict[str, object]) -> PinnedTextIndexGeneration:
    return PinnedTextIndexGeneration(
        generation_id=_string(value, "generation_id"),
        manifest_sha256=_string(value, "manifest_sha256"),
        physical_index_ref=_string(value, "physical_index_ref"),
    )


def _space_dict(space: EmbeddingSpaceRef) -> dict[str, object]:
    return {
        "space_id": space.space_id,
        "provider_id": space.provider_id,
        "model_revision": space.model_revision,
        "preprocessing_version": space.preprocessing_version,
        "dimension": space.dimension,
        "normalization": space.normalization.value,
        "distance_metric": space.distance_metric.value,
        "query_modalities": sorted(item.value for item in space.query_modalities),
        "target_modalities": sorted(item.value for item in space.target_modalities),
        "schema_version": space.schema_version,
    }


def _space_from_dict(value: dict[str, object]) -> EmbeddingSpaceRef:
    return EmbeddingSpaceRef(
        space_id=_string(value, "space_id"),
        provider_id=_string(value, "provider_id"),
        model_revision=_string(value, "model_revision"),
        preprocessing_version=_string(value, "preprocessing_version"),
        dimension=_integer(value, "dimension"),
        normalization=VectorNormalization(_string(value, "normalization")),
        distance_metric=DistanceMetric(_string(value, "distance_metric")),
        query_modalities=frozenset(
            QueryModality(item) for item in _strings(value, "query_modalities")
        ),
        target_modalities=frozenset(
            TargetType(item) for item in _strings(value, "target_modalities")
        ),
        schema_version=_integer(value, "schema_version"),
    )


def _reject_symlink_components(path: Path) -> None:
    current = path
    while current != current.parent:
        if current.exists() and current.is_symlink():
            raise ValueError("database_path must not contain symlink components")
        current = current.parent


def _require_non_blank(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")


def _mapping(value: Mapping[str, object], key: str) -> dict[str, object]:
    item = value[key]
    if not isinstance(item, dict):
        raise TypeError(f"{key} must be an object")
    return item


def _string(value: Mapping[str, object], key: str) -> str:
    item = value[key]
    if not isinstance(item, str):
        raise TypeError(f"{key} must be a string")
    return item


def _integer(value: Mapping[str, object], key: str) -> int:
    item = value[key]
    if isinstance(item, bool) or not isinstance(item, int):
        raise TypeError(f"{key} must be an integer")
    return item


def _strings(value: Mapping[str, object], key: str) -> list[str]:
    item = value[key]
    if not isinstance(item, list) or any(not isinstance(element, str) for element in item):
        raise TypeError(f"{key} must be a list of strings")
    return item


__all__ = ["SqliteVideoGenerationSetCatalog"]
