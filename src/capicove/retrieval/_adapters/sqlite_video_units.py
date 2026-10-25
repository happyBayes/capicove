"""Private immutable SQLite catalog for video retrieval units and relations."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import uuid
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from capicove.contracts import (
    CharacterRange,
    DistanceMetric,
    EmbeddingSpaceRef,
    ProcessingFingerprint,
    QueryModality,
    SearchableUnit,
    TargetType,
    TemporalRange,
    TextChunkLocator,
    TextOrigin,
    VectorNormalization,
)
from capicove.retrieval.video_models import searchable_units_manifest_sha256


_SCHEMA_VERSION = 1
_MAX_BULK_IDS = 1_000
_SQL_CHUNK_SIZE = 500


class VideoUnitCatalogError(RuntimeError):
    """Report a bounded unit-catalog storage or integrity failure."""


class SqliteVideoUnitCatalogWriter:
    """Publish a frozen unit catalog through an atomic sibling staging file."""

    def __init__(self, target_path: Path) -> None:
        self._target_path = _validated_absolute_path(target_path, allow_missing=True)
        self._target_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _reject_symlink_components(self._target_path)

    def publish(
        self,
        *,
        units: tuple[SearchableUnit, ...],
        unit_manifest_sha256: str,
        keyframe_to_segments: Mapping[str, tuple[str, ...]],
        text_to_segments: Mapping[str, tuple[tuple[str, int], ...]],
    ) -> str:
        """Validate and atomically publish one immutable catalog."""
        materialized = tuple(units)
        if any(not isinstance(unit, SearchableUnit) for unit in materialized):
            raise TypeError("units must contain SearchableUnit values")
        observed_manifest = searchable_units_manifest_sha256(materialized)
        if unit_manifest_sha256 != observed_manifest:
            raise ValueError("unit manifest SHA-256 does not match units")
        keyframe_relations = _validate_keyframe_relations(materialized, keyframe_to_segments)
        text_relations = _validate_text_relations(materialized, text_to_segments)
        catalog_sha256 = _catalog_digest(
            materialized,
            unit_manifest_sha256,
            keyframe_relations,
            text_relations,
        )
        if self._target_path.exists():
            reader = SqliteVideoUnitCatalogReader(self._target_path)
            if reader.catalog_sha256 != catalog_sha256:
                raise VideoUnitCatalogError(
                    "existing catalog path identifies different immutable content"
                )
            return str(self._target_path)

        staging = self._target_path.with_name(
            f".{self._target_path.name}.{uuid.uuid4().hex}.staging"
        )
        try:
            self._write_staging(
                staging,
                materialized,
                unit_manifest_sha256,
                catalog_sha256,
                keyframe_relations,
                text_relations,
            )
            staged_reader = SqliteVideoUnitCatalogReader(staging)
            if staged_reader.catalog_sha256 != catalog_sha256:
                raise VideoUnitCatalogError("staged catalog identity validation failed")
            try:
                os.link(staging, self._target_path)
            except FileExistsError as error:
                existing = SqliteVideoUnitCatalogReader(self._target_path)
                if existing.catalog_sha256 != catalog_sha256:
                    raise VideoUnitCatalogError(
                        "catalog publication raced with different immutable content"
                    ) from error
            os.chmod(self._target_path, stat.S_IRUSR | stat.S_IRGRP)
            _fsync_directory(self._target_path.parent)
            return str(self._target_path)
        except (OSError, sqlite3.Error) as error:
            raise VideoUnitCatalogError("unit catalog publication failed") from error
        finally:
            try:
                staging.unlink(missing_ok=True)
            except OSError:
                pass

    def _write_staging(
        self,
        path: Path,
        units: tuple[SearchableUnit, ...],
        unit_manifest_sha256: str,
        catalog_sha256: str,
        keyframe_relations: tuple[tuple[str, str, int], ...],
        text_relations: tuple[tuple[str, str, int, int], ...],
    ) -> None:
        connection = sqlite3.connect(path)
        try:
            connection.execute("PRAGMA journal_mode = DELETE")
            connection.execute("PRAGMA synchronous = FULL")
            connection.executescript(
                """
                CREATE TABLE metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                ) WITHOUT ROWID;
                CREATE TABLE units (
                    ordinal INTEGER PRIMARY KEY,
                    unit_id TEXT NOT NULL UNIQUE,
                    unit_type TEXT NOT NULL,
                    artifact_ref TEXT NOT NULL,
                    start_ms INTEGER,
                    end_ms INTEGER,
                    unit_json TEXT NOT NULL
                );
                CREATE TABLE keyframe_segments (
                    keyframe_id TEXT NOT NULL,
                    segment_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    PRIMARY KEY (keyframe_id, ordinal),
                    UNIQUE (keyframe_id, segment_id),
                    FOREIGN KEY (keyframe_id) REFERENCES units(unit_id),
                    FOREIGN KEY (segment_id) REFERENCES units(unit_id)
                ) WITHOUT ROWID;
                CREATE TABLE text_segments (
                    text_id TEXT NOT NULL,
                    segment_id TEXT NOT NULL,
                    overlap_ms INTEGER NOT NULL CHECK (overlap_ms > 0),
                    ordinal INTEGER NOT NULL,
                    PRIMARY KEY (text_id, ordinal),
                    UNIQUE (text_id, segment_id),
                    FOREIGN KEY (text_id) REFERENCES units(unit_id),
                    FOREIGN KEY (segment_id) REFERENCES units(unit_id)
                ) WITHOUT ROWID;
                CREATE INDEX units_artifact_type_idx
                    ON units(artifact_ref, unit_type);
                CREATE INDEX keyframe_segments_segment_idx
                    ON keyframe_segments(segment_id);
                CREATE INDEX text_segments_segment_idx
                    ON text_segments(segment_id);
                """
            )
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                (
                    ("schema_version", str(_SCHEMA_VERSION)),
                    ("unit_manifest_sha256", unit_manifest_sha256),
                    ("catalog_sha256", catalog_sha256),
                    ("unit_count", str(len(units))),
                ),
            )
            connection.executemany(
                """
                INSERT INTO units(
                    ordinal, unit_id, unit_type, artifact_ref, start_ms, end_ms,
                    unit_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(
                    (
                        ordinal,
                        unit.unit_id,
                        unit.unit_type.value,
                        unit.artifact_ref,
                        _unit_start_ms(unit),
                        _unit_end_ms(unit),
                        _unit_json(unit),
                    )
                    for ordinal, unit in enumerate(units)
                ),
            )
            connection.executemany(
                """
                INSERT INTO keyframe_segments(keyframe_id, segment_id, ordinal)
                VALUES (?, ?, ?)
                """,
                keyframe_relations,
            )
            connection.executemany(
                """
                INSERT INTO text_segments(text_id, segment_id, overlap_ms, ordinal)
                VALUES (?, ?, ?, ?)
                """,
                text_relations,
            )
            connection.commit()
            check = connection.execute("PRAGMA quick_check").fetchone()
            if check is None or str(check[0]) != "ok":
                raise VideoUnitCatalogError("staged catalog integrity check failed")
        finally:
            connection.close()
        _fsync_file(path)


class SqliteVideoUnitCatalogReader:
    """Read one validated immutable video unit catalog with bounded queries."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = _validated_absolute_path(database_path, allow_missing=False)
        self._signature = self._file_signature()
        metadata = self._read_metadata()
        if metadata.get("schema_version") != str(_SCHEMA_VERSION):
            raise VideoUnitCatalogError("unsupported unit catalog schema version")
        self._unit_manifest_sha256 = _metadata_digest(metadata, "unit_manifest_sha256")
        self._catalog_sha256 = _metadata_digest(metadata, "catalog_sha256")
        with self._read_connection() as connection:
            check = connection.execute("PRAGMA quick_check").fetchone()
            count = connection.execute("SELECT COUNT(*) FROM units").fetchone()
        if check is None or str(check[0]) != "ok":
            raise VideoUnitCatalogError("unit catalog integrity check failed")
        if count is None or int(count[0]) != int(metadata.get("unit_count", "-1")):
            raise VideoUnitCatalogError("unit catalog count does not match metadata")

    @property
    def unit_manifest_sha256(self) -> str:
        """Return the frozen searchable-unit manifest identity."""
        return self._unit_manifest_sha256

    def _file_signature(self) -> tuple[int, ...]:
        try:
            stat = self._database_path.stat(follow_symlinks=False)
        except OSError as error:
            raise VideoUnitCatalogError("immutable unit catalog changed or disappeared") from error
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_mode,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )

    def verify_unchanged(self) -> None:
        """Reject replacement, mutation or erasure of this opened immutable catalog."""
        if self._file_signature() != self._signature:
            raise VideoUnitCatalogError("immutable unit catalog changed")

    @property
    def catalog_sha256(self) -> str:
        """Return the complete catalog and relation identity."""
        return self._catalog_sha256

    def get(self, unit_id: str) -> SearchableUnit | None:
        """Return one canonical unit by ID."""
        _require_non_blank("unit_id", unit_id)
        with self._read_connection() as connection:
            row = connection.execute(
                "SELECT unit_json FROM units WHERE unit_id = ?", (unit_id,)
            ).fetchone()
        return None if row is None else _unit_from_json(str(row[0]))

    def get_many(self, unit_ids: tuple[str, ...]) -> tuple[SearchableUnit, ...]:
        """Return known units in caller order, bounded to 1,000 IDs."""
        ids = _bounded_ids(unit_ids)
        if not ids:
            return ()
        found: dict[str, SearchableUnit] = {}
        with self._read_connection() as connection:
            for chunk in _chunks(ids):
                placeholders = ",".join("?" for _ in chunk)
                # Only fixed parameter markers are interpolated; IDs are bound below.
                rows = connection.execute(
                    f"SELECT unit_id, unit_json FROM units WHERE unit_id IN ({placeholders})",  # nosec B608
                    chunk,
                ).fetchall()
                found.update((str(row[0]), _unit_from_json(str(row[1]))) for row in rows)
        return tuple(found[unit_id] for unit_id in ids if unit_id in found)

    def segments_for_keyframe(self, unit_id: str) -> tuple[str, ...]:
        """Return ordered candidate segments containing one keyframe."""
        return self.segments_for_keyframes((unit_id,)).get(unit_id, ())

    def segments_for_keyframes(self, unit_ids: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
        """Return bounded keyframe-to-segment relations."""
        ids = _bounded_ids(unit_ids)
        grouped: dict[str, list[str]] = {unit_id: [] for unit_id in ids}
        if not ids:
            return {}
        with self._read_connection() as connection:
            for chunk in _chunks(ids):
                placeholders = ",".join("?" for _ in chunk)
                # Only fixed parameter markers are interpolated; IDs are bound below.
                rows = connection.execute(
                    f"""
                    SELECT keyframe_id, segment_id FROM keyframe_segments
                    WHERE keyframe_id IN ({placeholders})
                    ORDER BY keyframe_id, ordinal
                    """,  # nosec B608
                    chunk,
                ).fetchall()
                for row in rows:
                    grouped[str(row[0])].append(str(row[1]))
        return {unit_id: tuple(grouped[unit_id]) for unit_id in ids}

    def segments_for_text(self, unit_id: str) -> tuple[tuple[str, int], ...]:
        """Return ordered overlapping segments for one text unit."""
        return self.segments_for_texts((unit_id,)).get(unit_id, ())

    def segments_for_texts(
        self, unit_ids: tuple[str, ...]
    ) -> dict[str, tuple[tuple[str, int], ...]]:
        """Return bounded text-to-segment overlap relations."""
        ids = _bounded_ids(unit_ids)
        grouped: dict[str, list[tuple[str, int]]] = {unit_id: [] for unit_id in ids}
        if not ids:
            return {}
        with self._read_connection() as connection:
            for chunk in _chunks(ids):
                placeholders = ",".join("?" for _ in chunk)
                # Only fixed parameter markers are interpolated; IDs are bound below.
                rows = connection.execute(
                    f"""
                    SELECT text_id, segment_id, overlap_ms FROM text_segments
                    WHERE text_id IN ({placeholders})
                    ORDER BY text_id, ordinal
                    """,  # nosec B608
                    chunk,
                ).fetchall()
                for row in rows:
                    grouped[str(row[0])].append((str(row[1]), int(row[2])))
        return {unit_id: tuple(grouped[unit_id]) for unit_id in ids}

    def _read_metadata(self) -> dict[str, str]:
        try:
            with self._read_connection() as connection:
                rows = connection.execute("SELECT key, value FROM metadata").fetchall()
        except sqlite3.Error as error:
            raise VideoUnitCatalogError("unit catalog metadata read failed") from error
        return {str(row[0]): str(row[1]) for row in rows}

    @contextmanager
    def _read_connection(self) -> Iterator[sqlite3.Connection]:
        connection: sqlite3.Connection | None = None
        try:
            self.verify_unchanged()
            connection = sqlite3.connect(
                f"{self._database_path.as_uri()}?mode=ro&immutable=1",
                uri=True,
            )
            yield connection
            self.verify_unchanged()
        except sqlite3.Error as error:
            raise VideoUnitCatalogError("unit catalog open failed") from error
        finally:
            if connection is not None:
                connection.close()


def _validate_keyframe_relations(
    units: tuple[SearchableUnit, ...],
    values: Mapping[str, tuple[str, ...]],
) -> tuple[tuple[str, str, int], ...]:
    by_id = {unit.unit_id: unit for unit in units}
    rows: list[tuple[str, str, int]] = []
    for keyframe_id in sorted(values):
        keyframe = by_id.get(keyframe_id)
        if keyframe is None:
            raise ValueError("keyframe relation references an unknown unit")
        if keyframe.unit_type is not TargetType.KEYFRAME:
            raise ValueError("keyframe relation source must be a keyframe")
        segment_ids = tuple(values[keyframe_id])
        if len(set(segment_ids)) != len(segment_ids):
            raise ValueError("keyframe relation must not contain duplicate segments")
        for ordinal, segment_id in enumerate(segment_ids):
            segment = by_id.get(segment_id)
            if segment is None:
                raise ValueError("keyframe relation references an unknown segment")
            if segment.unit_type is not TargetType.VIDEO_SEGMENT:
                raise ValueError("keyframe relation target must be a video segment")
            rows.append((keyframe_id, segment_id, ordinal))
    return tuple(rows)


def _validate_text_relations(
    units: tuple[SearchableUnit, ...],
    values: Mapping[str, tuple[tuple[str, int], ...]],
) -> tuple[tuple[str, str, int, int], ...]:
    by_id = {unit.unit_id: unit for unit in units}
    rows: list[tuple[str, str, int, int]] = []
    for text_id in sorted(values):
        text = by_id.get(text_id)
        if text is None:
            raise ValueError("text relation references an unknown unit")
        if text.unit_type is not TargetType.TEXT_CHUNK:
            raise ValueError("text relation source must be a text chunk")
        relations = tuple(values[text_id])
        segment_ids = tuple(item[0] for item in relations)
        if len(set(segment_ids)) != len(segment_ids):
            raise ValueError("text relation must not contain duplicate segments")
        for ordinal, relation in enumerate(relations):
            if not isinstance(relation, tuple) or len(relation) != 2:
                raise TypeError("text relation must contain (segment_id, overlap_ms)")
            segment_id, overlap_ms = relation
            segment = by_id.get(segment_id)
            if segment is None:
                raise ValueError("text relation references an unknown segment")
            if segment.unit_type is not TargetType.VIDEO_SEGMENT:
                raise ValueError("text relation target must be a video segment")
            if isinstance(overlap_ms, bool) or not isinstance(overlap_ms, int) or overlap_ms < 1:
                raise ValueError("text relation overlap_ms must be positive")
            rows.append((text_id, segment_id, overlap_ms, ordinal))
    return tuple(rows)


def _catalog_digest(
    units: tuple[SearchableUnit, ...],
    unit_manifest_sha256: str,
    keyframe_relations: tuple[tuple[str, str, int], ...],
    text_relations: tuple[tuple[str, str, int, int], ...],
) -> str:
    encoded = json.dumps(
        {
            "schema_version": _SCHEMA_VERSION,
            "unit_manifest_sha256": unit_manifest_sha256,
            "unit_json": [_unit_json(unit) for unit in units],
            "keyframe_relations": keyframe_relations,
            "text_relations": text_relations,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unit_json(unit: SearchableUnit) -> str:
    payload: dict[str, object] = {
        "unit_id": unit.unit_id,
        "unit_type": unit.unit_type.value,
        "source_hash": unit.source_hash,
        "artifact_ref": unit.artifact_ref,
        "evidence_ref": unit.evidence_ref,
        "representation_refs": list(unit.representation_refs),
        "processing_run_ref": unit.processing_run_ref,
        "processing_version": unit.processing_version,
        "processing_fingerprint": unit.processing_fingerprint.sha256,
        "embedding_spaces": [_space_dict(space) for space in unit.embedding_spaces],
        "timestamp_ms": unit.timestamp_ms,
        "scene_range": _range_dict(unit.scene_range),
        "time_range": _range_dict(unit.time_range),
        "text_locator": _locator_dict(unit.text_locator),
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _unit_from_json(payload: str) -> SearchableUnit:
    try:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("unit JSON must be an object")
        return SearchableUnit(
            unit_id=_string(value, "unit_id"),
            unit_type=TargetType(_string(value, "unit_type")),
            source_hash=_string(value, "source_hash"),
            artifact_ref=_string(value, "artifact_ref"),
            evidence_ref=_string(value, "evidence_ref"),
            representation_refs=tuple(_string_list(value, "representation_refs")),
            processing_run_ref=_string(value, "processing_run_ref"),
            processing_version=_string(value, "processing_version"),
            processing_fingerprint=ProcessingFingerprint(_string(value, "processing_fingerprint")),
            embedding_spaces=tuple(
                _space_from_dict(item) for item in _mapping_list(value, "embedding_spaces")
            ),
            timestamp_ms=_optional_int(value.get("timestamp_ms")),
            scene_range=_range_from_dict(value.get("scene_range")),
            time_range=_range_from_dict(value.get("time_range")),
            text_locator=_locator_from_dict(value.get("text_locator")),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise VideoUnitCatalogError("stored unit JSON is invalid") from error


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


def _space_from_dict(value: Mapping[str, object]) -> EmbeddingSpaceRef:
    return EmbeddingSpaceRef(
        space_id=_string(value, "space_id"),
        provider_id=_string(value, "provider_id"),
        model_revision=_string(value, "model_revision"),
        preprocessing_version=_string(value, "preprocessing_version"),
        dimension=_integer(value, "dimension"),
        normalization=VectorNormalization(_string(value, "normalization")),
        distance_metric=DistanceMetric(_string(value, "distance_metric")),
        query_modalities=frozenset(
            QueryModality(item) for item in _string_list(value, "query_modalities")
        ),
        target_modalities=frozenset(
            TargetType(item) for item in _string_list(value, "target_modalities")
        ),
        schema_version=_integer(value, "schema_version"),
    )


def _locator_dict(locator: TextChunkLocator | None) -> dict[str, object] | None:
    if locator is None:
        return None
    return {
        "origin": locator.origin.value,
        "character_range": {
            "start_char": locator.character_range.start_char,
            "end_char": locator.character_range.end_char,
        },
        "page_number": locator.page_number,
        "time_range": _range_dict(locator.time_range),
    }


def _locator_from_dict(value: object) -> TextChunkLocator | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("text_locator must be an object")
    character = value.get("character_range")
    if not isinstance(character, dict):
        raise TypeError("character_range must be an object")
    return TextChunkLocator(
        origin=TextOrigin(_string(value, "origin")),
        character_range=CharacterRange(
            _integer(character, "start_char"), _integer(character, "end_char")
        ),
        page_number=_optional_int(value.get("page_number")),
        time_range=_range_from_dict(value.get("time_range")),
    )


def _range_dict(value: TemporalRange | None) -> dict[str, int] | None:
    if value is None:
        return None
    return {"start_ms": value.start_ms, "end_ms": value.end_ms}


def _range_from_dict(value: object) -> TemporalRange | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("temporal range must be an object")
    return TemporalRange(_integer(value, "start_ms"), _integer(value, "end_ms"))


def _unit_start_ms(unit: SearchableUnit) -> int | None:
    if unit.timestamp_ms is not None:
        return unit.timestamp_ms
    if unit.time_range is not None:
        return unit.time_range.start_ms
    if unit.text_locator is not None and unit.text_locator.time_range is not None:
        return unit.text_locator.time_range.start_ms
    return None


def _unit_end_ms(unit: SearchableUnit) -> int | None:
    if unit.timestamp_ms is not None:
        return unit.timestamp_ms + 1
    if unit.time_range is not None:
        return unit.time_range.end_ms
    if unit.text_locator is not None and unit.text_locator.time_range is not None:
        return unit.text_locator.time_range.end_ms
    return None


def _validated_absolute_path(path: Path, *, allow_missing: bool) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("catalog path must be an absolute Path")
    _reject_symlink_components(path)
    if path.exists():
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError("catalog path must not be a symlink")
        if not stat.S_ISREG(mode):
            raise ValueError("catalog path must be a regular file")
    elif not allow_missing:
        raise ValueError("catalog path must identify an existing regular file")
    return path


def _reject_symlink_components(path: Path) -> None:
    current = path
    while current != current.parent:
        if current.exists() and current.is_symlink():
            raise ValueError("catalog path must not contain symlink components")
        current = current.parent


def _bounded_ids(unit_ids: Iterable[str]) -> tuple[str, ...]:
    if isinstance(unit_ids, (str, bytes)):
        raise TypeError("unit_ids must be a tuple of strings")
    ids = tuple(unit_ids)
    if len(ids) > _MAX_BULK_IDS:
        raise ValueError("bulk reads accept at most 1,000 unit IDs")
    if len(set(ids)) != len(ids):
        raise ValueError("unit_ids must not contain duplicates")
    for unit_id in ids:
        _require_non_blank("unit_id", unit_id)
    return ids


def _chunks(values: tuple[str, ...]) -> Iterable[tuple[str, ...]]:
    for offset in range(0, len(values), _SQL_CHUNK_SIZE):
        yield values[offset : offset + _SQL_CHUNK_SIZE]


def _metadata_digest(metadata: Mapping[str, str], key: str) -> str:
    value = metadata.get(key)
    if value is None or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise VideoUnitCatalogError(f"unit catalog {key} metadata is invalid")
    return value


def _require_non_blank(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")


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


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("optional integer value is invalid")
    return value


def _string_list(value: Mapping[str, object], key: str) -> list[str]:
    item = value[key]
    if not isinstance(item, list) or any(not isinstance(element, str) for element in item):
        raise TypeError(f"{key} must be a list of strings")
    return item


def _mapping_list(value: Mapping[str, object], key: str) -> list[Mapping[str, object]]:
    item = value[key]
    if not isinstance(item, list) or any(not isinstance(element, dict) for element in item):
        raise TypeError(f"{key} must be a list of objects")
    return item


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "SqliteVideoUnitCatalogReader",
    "SqliteVideoUnitCatalogWriter",
    "VideoUnitCatalogError",
]
