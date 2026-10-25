"""Private disk-backed aggregation used during bounded trace export."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from capicove.export.media import media_reference_identity_sha256
from capicove.traces import EpisodeCompleted, ToolResultObserved, TraceRecord


class ExportStateError(RuntimeError):
    """Report conflicting media identity during temporary aggregation."""


class ExportAggregation:
    """Aggregate scalar counts and unique media without unbounded Python sets."""

    def __init__(self, database_path: Path) -> None:
        self._connection = sqlite3.connect(database_path)
        self._connection.executescript(
            """
            CREATE TABLE counts (
                category TEXT NOT NULL,
                key TEXT NOT NULL,
                count INTEGER NOT NULL,
                PRIMARY KEY (category, key)
            );
            CREATE TABLE media_ids (
                media_id TEXT PRIMARY KEY,
                identity_sha256 TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                missing INTEGER NOT NULL
            );
            CREATE TABLE media_hashes (
                sha256 TEXT PRIMARY KEY,
                size_bytes INTEGER NOT NULL,
                media_type TEXT NOT NULL
            );
            """
        )

    def observe(self, record: TraceRecord, *, assume_media_available: bool = False) -> None:
        """Update all manifest aggregations for one encoded record."""
        self._increment("record_type", record.record_type.value)
        for modality in record.modalities:
            self._increment("modality", modality.value)
        if isinstance(record.payload, ToolResultObserved):
            self._increment("tool_status", record.payload.status.value)
        if isinstance(record.payload, EpisodeCompleted):
            self._increment("episode_status", record.payload.status.value)
        for reference in record.media:
            existing_id = self._connection.execute(
                """
                SELECT identity_sha256, sha256, size_bytes, missing
                FROM media_ids WHERE media_id = ?
                """,
                (reference.media_id,),
            ).fetchone()
            missing = int(not assume_media_available and reference.locator is None)
            identity_sha256 = media_reference_identity_sha256(reference)
            if existing_id is None:
                self._connection.execute(
                    """
                    INSERT INTO media_ids (
                        media_id, identity_sha256, sha256, size_bytes, missing
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        reference.media_id,
                        identity_sha256,
                        reference.sha256,
                        reference.size_bytes,
                        missing,
                    ),
                )
            elif existing_id != (
                identity_sha256,
                reference.sha256,
                reference.size_bytes,
                missing,
            ):
                raise ExportStateError("one media ID has conflicting export identity")
            existing_hash = self._connection.execute(
                "SELECT size_bytes, media_type FROM media_hashes WHERE sha256 = ?",
                (reference.sha256,),
            ).fetchone()
            if existing_hash is None:
                self._connection.execute(
                    """
                    INSERT INTO media_hashes (sha256, size_bytes, media_type)
                    VALUES (?, ?, ?)
                    """,
                    (reference.sha256, reference.size_bytes, reference.media_type),
                )
            elif existing_hash != (reference.size_bytes, reference.media_type):
                raise ExportStateError("one media hash has conflicting content identity")

    def counts(self, category: str) -> dict[str, int]:
        """Return one small scalar count map in stable key order."""
        return {
            key: count
            for key, count in self._connection.execute(
                "SELECT key, count FROM counts WHERE category = ? ORDER BY key",
                (category,),
            )
        }

    def media_summary(self) -> tuple[int, int, int]:
        """Return unique ID count, unique-content bytes, and missing-ID count."""
        media_count, missing_count = self._connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(missing), 0) FROM media_ids"
        ).fetchone()
        media_size = self._connection.execute(
            "SELECT COALESCE(SUM(size_bytes), 0) FROM media_hashes"
        ).fetchone()[0]
        return media_count, media_size, missing_count

    def close(self) -> None:
        """Close the private temporary database."""
        self._connection.close()

    def _increment(self, category: str, key: str) -> None:
        self._connection.execute(
            """
            INSERT INTO counts (category, key, count) VALUES (?, ?, 1)
            ON CONFLICT (category, key) DO UPDATE SET count = count + 1
            """,
            (category, key),
        )
