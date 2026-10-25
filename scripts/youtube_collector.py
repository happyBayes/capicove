#!/usr/bin/env python3
"""Safely discover, review, and download YouTube media by topic."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Protocol, TextIO


VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
REVIEW_COLUMNS = (
    "video_id",
    "source_url",
    "title",
    "channel_id",
    "channel",
    "upload_date",
    "duration",
    "filter_status",
    "filter_reason",
    "review_status",
    "reviewer_note",
)
IMMUTABLE_REVIEW_COLUMNS = REVIEW_COLUMNS[:-2]


class Runner(Protocol):
    """Run one bounded external command."""

    def run(self, args: Sequence[str], timeout: int) -> CommandResult:
        """Return captured process output."""


class DiskUsageResult(Protocol):
    """Expose the free-space value consumed by storage checks."""

    free: int


class SingleInstanceLock:
    """Prevent overlapping state-changing collector processes."""

    def __init__(self, path: Path):
        self.path = path
        self._handle: TextIO | None = None

    def __enter__(self) -> SingleInstanceLock:
        self.path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
        self._handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._handle.close()
            self._handle = None
            raise RuntimeError("collector is already running for this root") from error
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(f"pid={os.getpid()}\n")
        self._handle.flush()
        return self

    def __exit__(self, *_: object) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


@dataclass(frozen=True, slots=True)
class Topic:
    """Describe one named collection topic."""

    topic_id: str
    queries: tuple[str, ...]
    max_results: int
    date_after: date
    date_before: date


@dataclass(frozen=True, slots=True)
class Config:
    """Contain one normalized collection configuration."""

    root: Path
    date_after: date
    date_before: date
    discovery_workers: int
    download_workers: int
    media: str
    write_subtitles: bool
    write_auto_subtitles: bool
    write_thumbnail: bool
    yt_dlp: str
    ffmpeg: str
    cookie_file: Path | None
    command_timeout_seconds: int
    socket_timeout_seconds: int
    min_duration_seconds: int
    max_duration_seconds: int
    max_retries: int
    max_height: int
    min_free_bytes: int
    max_media_bytes: int
    max_file_bytes: int
    topics: tuple[Topic, ...]


def _parse_date(value: object, field_name: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be an ISO date") from error


def load_config(
    path: Path,
    *,
    environ: Mapping[str, str] | None = None,
    overrides: Mapping[str, object] | None = None,
) -> Config:
    """Load and normalize one TOML collection configuration."""
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    collection = raw.get("collection")
    topics_raw = raw.get("topics")
    if not isinstance(collection, dict) or not isinstance(topics_raw, list):
        raise ValueError("configuration requires [collection] and [[topics]]")
    collection = dict(collection)
    environment = os.environ if environ is None else environ
    integer_fields = (
        "discovery_workers",
        "download_workers",
        "command_timeout_seconds",
        "socket_timeout_seconds",
        "min_duration_seconds",
        "max_duration_seconds",
        "max_retries",
        "max_height",
        "min_free_gib",
        "max_media_gib",
        "max_file_gib",
    )
    for field_name in integer_fields:
        environment_value = environment.get(f"YTCOLLECT_{field_name.upper()}")
        if environment_value is not None:
            try:
                collection[field_name] = int(environment_value)
            except ValueError as error:
                raise ValueError(f"YTCOLLECT_{field_name.upper()} must be an integer") from error
    for field_name in (
        "root",
        "date_after",
        "date_before",
        "media",
        "yt_dlp",
        "ffmpeg",
        "cookie_file",
    ):
        environment_value = environment.get(f"YTCOLLECT_{field_name.upper()}")
        if environment_value is not None:
            collection[field_name] = environment_value
    for field_name in ("write_subtitles", "write_auto_subtitles", "write_thumbnail"):
        environment_value = environment.get(f"YTCOLLECT_{field_name.upper()}")
        if environment_value is None:
            continue
        normalized = environment_value.strip().lower()
        if normalized not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError(f"YTCOLLECT_{field_name.upper()} must be a boolean")
        collection[field_name] = normalized in {"true", "1", "yes"}
    for field_name, value in (overrides or {}).items():
        if value is not None:
            collection[field_name] = value

    date_after = _parse_date(collection.get("date_after"), "date_after")
    date_before = _parse_date(collection.get("date_before"), "date_before")
    if date_after > date_before:
        raise ValueError("date_after must not be later than date_before")
    root_value = collection.get("root")
    if not isinstance(root_value, str) or not root_value.strip():
        raise ValueError("collection.root must not be blank")
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = path.parent / root
    root = root.resolve()
    if root in {Path("/"), Path.home().resolve()} or len(root.parts) < 3:
        raise ValueError(f"unsafe collection root: {root}")

    topics: list[Topic] = []
    seen_ids: set[str] = set()
    for item in topics_raw:
        if not isinstance(item, dict):
            raise ValueError("each topic must be a TOML table")
        topic_id = item.get("id")
        queries = item.get("queries")
        max_results = item.get("max_results")
        if not isinstance(topic_id, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", topic_id) is None:
            raise ValueError("topic id must use lowercase letters, digits, '_' or '-'")
        if topic_id in seen_ids:
            raise ValueError(f"duplicate topic id: {topic_id}")
        if (
            not isinstance(queries, list)
            or not queries
            or not all(isinstance(query, str) and query.strip() for query in queries)
        ):
            raise ValueError(f"topic {topic_id} requires non-empty queries")
        if not isinstance(max_results, int) or isinstance(max_results, bool) or max_results < 1:
            raise ValueError(f"topic {topic_id} max_results must be positive")
        topic_after = _parse_date(item.get("date_after", date_after.isoformat()), "date_after")
        topic_before = _parse_date(item.get("date_before", date_before.isoformat()), "date_before")
        if topic_after > topic_before:
            raise ValueError(f"topic {topic_id} has an inverted date range")
        topics.append(
            Topic(
                topic_id=topic_id,
                queries=tuple(query.strip() for query in queries),
                max_results=max_results,
                date_after=topic_after,
                date_before=topic_before,
            )
        )
        seen_ids.add(topic_id)

    discovery_workers = collection.get("discovery_workers", 2)
    download_workers = collection.get("download_workers", 2)
    for field_name, value in (
        ("discovery_workers", discovery_workers),
        ("download_workers", download_workers),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 8:
            raise ValueError(f"{field_name} must be between 1 and 8")
    media = collection.get("media", "both")
    if media not in {"video", "audio", "both"}:
        raise ValueError("media must be 'video', 'audio', or 'both'")

    gib = 1024**3
    return Config(
        root=root,
        date_after=date_after,
        date_before=date_before,
        discovery_workers=discovery_workers,
        download_workers=download_workers,
        media=str(media),
        write_subtitles=bool(collection.get("write_subtitles", True)),
        write_auto_subtitles=bool(collection.get("write_auto_subtitles", False)),
        write_thumbnail=bool(collection.get("write_thumbnail", True)),
        yt_dlp=str(collection.get("yt_dlp", "yt-dlp")),
        ffmpeg=str(collection.get("ffmpeg", "ffmpeg")),
        cookie_file=_optional_path(collection.get("cookie_file"), path.parent),
        command_timeout_seconds=int(collection.get("command_timeout_seconds", 900)),
        socket_timeout_seconds=int(collection.get("socket_timeout_seconds", 30)),
        min_duration_seconds=int(collection.get("min_duration_seconds", 1)),
        max_duration_seconds=int(collection.get("max_duration_seconds", 14_400)),
        max_retries=int(collection.get("max_retries", 3)),
        max_height=int(collection.get("max_height", 720)),
        min_free_bytes=int(collection.get("min_free_gib", 1)) * gib,
        max_media_bytes=int(collection.get("max_media_gib", 500)) * gib,
        max_file_bytes=int(collection.get("max_file_gib", 4)) * gib,
        topics=tuple(topics),
    )


def _optional_path(value: object, base: Path) -> Path | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError("cookie_file must be a path string")
    result = Path(value).expanduser()
    if not result.is_absolute():
        result = base / result
    return result.resolve()


def open_database(path: Path) -> sqlite3.Connection:
    """Open the collector catalog with concurrent-safe local settings."""
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


def open_database_read_only(path: Path) -> sqlite3.Connection:
    """Open an existing catalog without permitting database mutation."""
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


def _initialize_runtime(config: Config) -> None:
    (config.root / "catalog").mkdir(mode=0o750, parents=True, exist_ok=True)
    (config.root / "metadata").mkdir(mode=0o750, parents=True, exist_ok=True)
    with open_database(config.root / "catalog" / "state.sqlite3") as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS candidates (
                video_id TEXT PRIMARY KEY,
                source_url TEXT NOT NULL,
                title TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                upload_date TEXT NOT NULL,
                duration REAL NOT NULL,
                filter_status TEXT NOT NULL CHECK(filter_status IN ('accepted','rejected')),
                filter_reason TEXT NOT NULL DEFAULT '',
                review_status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(review_status IN ('pending','approved','rejected')),
                reviewer_note TEXT NOT NULL DEFAULT '',
                download_state TEXT NOT NULL DEFAULT 'pending'
                    CHECK(download_state IN
                        ('pending','running','retry','complete','failed','interrupted')),
                attempts INTEGER NOT NULL DEFAULT 0,
                next_retry_at TEXT,
                last_error TEXT NOT NULL DEFAULT '',
                output_path TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS discoveries (
                video_id TEXT NOT NULL REFERENCES candidates(video_id) ON DELETE CASCADE,
                topic_id TEXT NOT NULL,
                query TEXT NOT NULL,
                discovered_at TEXT NOT NULL,
                PRIMARY KEY(video_id, topic_id, query)
            );
            """
        )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o640)
    os.replace(temporary, path)


def _regular_file_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return total
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names[:] = [
            name for name in directory_names if not (Path(directory) / name).is_symlink()
        ]
        for name in file_names:
            path = Path(directory) / name
            try:
                if path.is_file() and not path.is_symlink():
                    total += path.stat().st_size
            except FileNotFoundError:
                continue
    return total


def check_storage(
    config: Config,
    reserved_bytes: int = 0,
    *,
    disk_usage: Callable[[Path], DiskUsageResult] | None = None,
) -> None:
    """Reject a download that would cross configured storage limits."""
    if reserved_bytes < 0:
        raise ValueError("reserved_bytes must not be negative")
    media_bytes = _regular_file_bytes(config.root / "media")
    usage_path = config.root if config.root.exists() else config.root.parent
    usage = shutil.disk_usage(usage_path) if disk_usage is None else disk_usage(usage_path)
    free = int(usage.free)
    if media_bytes + reserved_bytes > config.max_media_bytes:
        raise RuntimeError("media storage budget would be exceeded")
    if free - reserved_bytes < config.min_free_bytes:
        raise RuntimeError("minimum free-space floor would be crossed")


def _cookie_args(config: Config) -> list[str]:
    return [] if config.cookie_file is None else ["--cookies", str(config.cookie_file)]


def build_search_command(config: Config, topic: Topic, query: str) -> list[str]:
    """Build one metadata-only YouTube topic search command."""
    return [
        config.yt_dlp,
        "--ignore-config",
        *_cookie_args(config),
        "--flat-playlist",
        "--dump-json",
        "--playlist-end",
        str(topic.max_results),
        "--dateafter",
        topic.date_after.strftime("%Y%m%d"),
        "--datebefore",
        topic.date_before.strftime("%Y%m%d"),
        "--socket-timeout",
        str(config.socket_timeout_seconds),
        "--no-warnings",
        f"ytsearch{topic.max_results}:{query}",
    ]


def build_metadata_command(config: Config, video_id: str) -> list[str]:
    """Build one full metadata extraction command without media download."""
    if VIDEO_ID_PATTERN.fullmatch(video_id) is None:
        raise ValueError(f"invalid YouTube video id: {video_id!r}")
    return [
        config.yt_dlp,
        "--ignore-config",
        *_cookie_args(config),
        "--dump-single-json",
        "--skip-download",
        "--no-playlist",
        "--socket-timeout",
        str(config.socket_timeout_seconds),
        f"https://www.youtube.com/watch?v={video_id}",
    ]


def _json_objects(text: str) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError("yt-dlp JSON output must contain objects")
        result.append(value)
    return result


def _classify_metadata(
    metadata: Mapping[str, object], topic: Topic, config: Config
) -> tuple[str, str]:
    video_id = str(metadata.get("id") or "")
    if VIDEO_ID_PATTERN.fullmatch(video_id) is None:
        return "rejected", "invalid_video_id"
    upload_date = str(metadata.get("upload_date") or "")
    try:
        uploaded = datetime.strptime(upload_date, "%Y%m%d").date()
    except ValueError:
        return "rejected", "upload_date_invalid"
    if not topic.date_after <= uploaded <= topic.date_before:
        return "rejected", "upload_date_out_of_range"
    if metadata.get("live_status") != "not_live" or bool(metadata.get("is_live")):
        return "rejected", "live_or_upcoming"
    try:
        age_limit = int(str(metadata.get("age_limit")))
    except (TypeError, ValueError):
        return "rejected", "age_restriction_unknown"
    if age_limit > 0:
        return "rejected", "age_restricted"
    if metadata.get("availability") not in {"public", "unlisted"}:
        return "rejected", "access_restricted"
    try:
        duration = float(str(metadata.get("duration") or 0))
    except (TypeError, ValueError):
        return "rejected", "duration_invalid"
    if not config.min_duration_seconds <= duration <= config.max_duration_seconds:
        return "rejected", "duration_out_of_range"
    return "accepted", ""


def _upsert_discovery(
    config: Config,
    metadata: Mapping[str, object],
    topic: Topic,
    query: str,
    filter_status: str,
    filter_reason: str,
) -> None:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    video_id = str(metadata["id"])
    with open_database(config.root / "catalog" / "state.sqlite3") as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            INSERT INTO candidates(
                video_id, source_url, title, channel_id, channel, upload_date,
                duration, filter_status, filter_reason, metadata_json,
                created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(video_id) DO UPDATE SET
                metadata_json=excluded.metadata_json,
                updated_at=excluded.updated_at
            """,
            (
                video_id,
                f"https://www.youtube.com/watch?v={video_id}",
                str(metadata.get("title") or ""),
                str(metadata.get("channel_id") or ""),
                str(metadata.get("channel") or ""),
                str(metadata.get("upload_date") or ""),
                float(str(metadata.get("duration") or 0)),
                filter_status,
                filter_reason,
                json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                now,
                now,
            ),
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO discoveries(video_id, topic_id, query, discovered_at)
            VALUES (?,?,?,?)
            """,
            (video_id, topic.topic_id, query, now),
        )


def run_discovery(config: Config, runner: Runner) -> dict[str, int]:
    """Discover and persist metadata candidates without downloading media."""
    _initialize_runtime(config)
    search_inputs = [(topic, query) for topic in config.topics for query in topic.queries]

    def search_one(item: tuple[Topic, str]) -> tuple[Topic, str, list[str]]:
        topic, query = item
        search = runner.run(
            build_search_command(config, topic, query),
            config.command_timeout_seconds,
        )
        if search.returncode != 0:
            raise RuntimeError(search.stderr or f"search failed for {query!r}")
        video_ids = [str(metadata.get("id") or "") for metadata in _json_objects(search.stdout)]
        return (
            topic,
            query,
            [
                video_id
                for video_id in video_ids
                if VIDEO_ID_PATTERN.fullmatch(video_id) is not None
            ],
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.discovery_workers) as executor:
        search_results = list(executor.map(search_one, search_inputs))

    selected_by_topic: dict[str, set[str]] = {topic.topic_id: set() for topic in config.topics}
    assignments: dict[str, list[tuple[Topic, str]]] = {}
    for topic, query, video_ids in search_results:
        selected = selected_by_topic[topic.topic_id]
        for video_id in video_ids:
            if video_id not in selected and len(selected) >= topic.max_results:
                continue
            selected.add(video_id)
            relationship = (topic, query)
            if relationship not in assignments.setdefault(video_id, []):
                assignments[video_id].append(relationship)

    def fetch_one(item: tuple[str, list[tuple[Topic, str]]]) -> tuple[str, str]:
        video_id, relationships = item
        result = runner.run(
            build_metadata_command(config, video_id),
            config.command_timeout_seconds,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr or f"metadata failed for {video_id}")
        objects = _json_objects(result.stdout)
        if len(objects) != 1:
            raise RuntimeError(f"expected one metadata object for {video_id}")
        metadata = objects[0]
        primary_topic = relationships[0][0]
        filter_status, filter_reason = _classify_metadata(metadata, primary_topic, config)
        _atomic_write_text(
            config.root / "metadata" / video_id / "info.json",
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )
        for topic, query in relationships:
            _upsert_discovery(
                config,
                metadata,
                topic,
                query,
                filter_status,
                filter_reason,
            )
        return filter_status, filter_reason

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.discovery_workers) as executor:
        decisions = list(executor.map(fetch_one, assignments.items()))
    return {
        "seen": len(assignments),
        "accepted": sum(status == "accepted" for status, _ in decisions),
        "rejected": sum(status == "rejected" for status, _ in decisions),
    }


def set_review_status(
    config: Config,
    video_id: str,
    review_status: str,
    reviewer_note: str = "",
) -> None:
    """Apply one operator review decision to a known candidate."""
    if VIDEO_ID_PATTERN.fullmatch(video_id) is None:
        raise ValueError(f"invalid YouTube video id: {video_id!r}")
    if review_status not in {"approved", "rejected", "pending"}:
        raise ValueError(f"invalid review status: {review_status}")
    database = config.root / "catalog" / "state.sqlite3"
    if not database.is_file():
        raise ValueError("collector state is not initialized")
    with open_database(database) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT filter_status FROM candidates WHERE video_id=?", (video_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown candidate: {video_id}")
        if review_status == "approved" and row["filter_status"] != "accepted":
            raise ValueError("an automatically rejected candidate cannot be approved")
        connection.execute(
            """
            UPDATE candidates
            SET review_status=?, reviewer_note=?, updated_at=?
            WHERE video_id=?
            """,
            (
                review_status,
                reviewer_note.strip(),
                datetime.now(UTC).isoformat(timespec="seconds"),
                video_id,
            ),
        )


def build_download_command(config: Config, video_id: str, output_dir: Path) -> list[str]:
    """Build one resumable media download command."""
    if VIDEO_ID_PATTERN.fullmatch(video_id) is None:
        raise ValueError(f"invalid YouTube video id: {video_id!r}")
    command = [
        config.yt_dlp,
        "--ignore-config",
        *_cookie_args(config),
        "--no-playlist",
        "--continue",
        "--part",
        "--no-mtime",
        "--write-info-json",
        "--socket-timeout",
        str(config.socket_timeout_seconds),
        "--retries",
        "3",
        "--fragment-retries",
        "3",
        "--max-filesize",
        str(config.max_file_bytes),
    ]
    if config.media in {"video", "both"}:
        command.extend(
            [
                "--format",
                (
                    f"bestvideo[height<={config.max_height}]+bestaudio/"
                    f"best[height<={config.max_height}]"
                ),
                "--merge-output-format",
                "mp4",
            ]
        )
    else:
        command.extend(["--extract-audio", "--audio-format", "m4a"])
    if config.write_thumbnail:
        command.append("--write-thumbnail")
    if config.write_subtitles:
        command.extend(["--write-subs", "--sub-langs", "zh.*,en.*"])
    if config.write_auto_subtitles:
        command.extend(["--write-auto-subs", "--sub-langs", "zh.*,en.*"])
    command.extend(
        [
            "-o",
            str(output_dir / "%(id)s.%(ext)s"),
            f"https://www.youtube.com/watch?v={video_id}",
        ]
    )
    return command


def build_audio_command(config: Config, video_path: Path, audio_path: Path) -> list[str]:
    """Build a lossless-if-compatible standalone audio extraction command."""
    return [
        config.ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-c:a",
        "copy",
        str(audio_path),
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(config: Config, video_id: str, output_dir: Path) -> int:
    video_files = [
        path
        for path in output_dir.iterdir()
        if path.is_file()
        and not path.is_symlink()
        and path.stem == video_id
        and path.suffix.lower() in {".mp4", ".mkv", ".webm"}
    ]
    if config.media in {"video", "both"} and len(video_files) != 1:
        raise RuntimeError(f"expected one video output for {video_id}")
    audio = output_dir / "audio.m4a"
    if config.media in {"audio", "both"} and not audio.is_file():
        raise RuntimeError(f"missing audio output for {video_id}")
    metadata = output_dir / f"{video_id}.info.json"
    if not metadata.is_file():
        raise RuntimeError(f"missing yt-dlp metadata output for {video_id}")
    files: list[dict[str, object]] = []
    total = 0
    for path in sorted(output_dir.iterdir()):
        if path.name == "manifest.json" or path.name.endswith(".part"):
            continue
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            continue
        size = path.stat().st_size
        total += size
        files.append({"path": path.name, "bytes": size, "sha256": _sha256_file(path)})
    manifest = {
        "video_id": video_id,
        "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "bytes": total,
        "files": files,
    }
    _atomic_write_text(
        output_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    return total


def _claim_downloads(config: Config) -> list[str]:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with open_database(config.root / "catalog" / "state.sqlite3") as connection:
        connection.execute("BEGIN IMMEDIATE")
        rows = connection.execute(
            """
            SELECT video_id FROM candidates
            WHERE filter_status='accepted'
              AND review_status='approved'
              AND download_state IN ('pending','retry','interrupted')
              AND attempts < ?
              AND (next_retry_at IS NULL OR next_retry_at <= ?)
            ORDER BY video_id
            """,
            (config.max_retries, now),
        ).fetchall()
        video_ids = [str(row["video_id"]) for row in rows]
        connection.executemany(
            """
            UPDATE candidates
            SET download_state='running', attempts=attempts+1, updated_at=?
            WHERE video_id=?
            """,
            [(now, video_id) for video_id in video_ids],
        )
    return video_ids


def _manifest_is_complete(config: Config, video_id: str, path: Path) -> bool:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("video_id") != video_id or not isinstance(manifest.get("files"), list):
            return False
        video_found = audio_found = metadata_found = False
        total = 0
        for item in manifest["files"]:
            relative = item["path"]
            size = item["bytes"]
            expected_hash = item["sha256"]
            if not isinstance(relative, str) or Path(relative).name != relative:
                return False
            asset = path.parent / relative
            if (
                asset.is_symlink()
                or not asset.is_file()
                or not isinstance(size, int)
                or size <= 0
                or asset.stat().st_size != size
                or _sha256_file(asset) != expected_hash
            ):
                return False
            video_found |= asset.stem == video_id and asset.suffix.lower() in {
                ".mp4",
                ".mkv",
                ".webm",
            }
            audio_found |= relative == "audio.m4a"
            metadata_found |= relative == f"{video_id}.info.json"
            total += size
        if config.media in {"video", "both"} and not video_found:
            return False
        if config.media in {"audio", "both"} and not audio_found:
            return False
        return metadata_found and manifest.get("bytes") == total
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _reconcile_running(config: Config) -> int:
    recovered = 0
    database = config.root / "catalog" / "state.sqlite3"
    with open_database_read_only(database) as connection:
        rows = connection.execute(
            "SELECT video_id FROM candidates WHERE download_state='running'"
        ).fetchall()
    for row in rows:
        video_id = str(row["video_id"])
        output_dir = config.root / "media" / video_id
        manifest = output_dir / "manifest.json"
        if _manifest_is_complete(config, video_id, manifest):
            _set_download_result(config, video_id, "complete", output_path=str(output_dir))
            recovered += 1
        else:
            _set_download_result(
                config,
                video_id,
                "interrupted",
                error="previous process ended before a valid manifest was committed",
            )
    return recovered


def _set_download_result(
    config: Config,
    video_id: str,
    state: str,
    *,
    error: str = "",
    output_path: str = "",
) -> None:
    with open_database(config.root / "catalog" / "state.sqlite3") as connection:
        connection.execute(
            """
            UPDATE candidates
            SET download_state=?, last_error=?, output_path=?, updated_at=?
            WHERE video_id=?
            """,
            (
                state,
                error[:2000],
                output_path,
                datetime.now(UTC).isoformat(timespec="seconds"),
                video_id,
            ),
        )


def _is_transient_download_error(message: str) -> bool:
    normalized = message.lower()
    return any(
        marker in normalized
        for marker in (
            "http error 429",
            "http error 500",
            "http error 502",
            "http error 503",
            "http error 504",
            "timed out",
            "timeout",
            "connection reset",
            "temporary",
            "truncated",
        )
    )


def _schedule_retry(config: Config, video_id: str, error: str, delay: float) -> bool:
    database = config.root / "catalog" / "state.sqlite3"
    with open_database(database) as connection:
        connection.execute("BEGIN IMMEDIATE")
        attempts = int(
            connection.execute(
                "SELECT attempts FROM candidates WHERE video_id=?", (video_id,)
            ).fetchone()[0]
        )
        if attempts >= config.max_retries:
            return False
        connection.execute(
            """
            UPDATE candidates SET download_state='retry', last_error=?, next_retry_at=?,
                updated_at=? WHERE video_id=?
            """,
            (
                error[:2000],
                (datetime.now(UTC) + timedelta(seconds=delay)).isoformat(timespec="seconds"),
                datetime.now(UTC).isoformat(timespec="seconds"),
                video_id,
            ),
        )
    return True


def _begin_retry_attempt(config: Config, video_id: str) -> None:
    with open_database(config.root / "catalog" / "state.sqlite3") as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            UPDATE candidates SET download_state='running', attempts=attempts+1,
                next_retry_at=NULL, updated_at=? WHERE video_id=?
            """,
            (datetime.now(UTC).isoformat(timespec="seconds"), video_id),
        )


def _download_one(
    config: Config,
    runner: Runner,
    video_id: str,
    wait: Callable[[float], object],
) -> str:
    output_dir = config.root / "media" / video_id
    output_dir.mkdir(mode=0o750, parents=True, exist_ok=True)
    while True:
        try:
            result = runner.run(
                build_download_command(config, video_id, output_dir),
                config.command_timeout_seconds,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr or f"yt-dlp failed for {video_id}")
            if config.media == "audio":
                downloaded_audio = output_dir / f"{video_id}.m4a"
                if downloaded_audio.is_file():
                    os.replace(downloaded_audio, output_dir / "audio.m4a")
            if config.media == "both":
                video_files = [
                    path
                    for path in output_dir.iterdir()
                    if path.is_file()
                    and path.stem == video_id
                    and path.suffix.lower() in {".mp4", ".mkv", ".webm"}
                ]
                if len(video_files) != 1:
                    raise RuntimeError(f"expected one video output for {video_id}")
                audio_result = runner.run(
                    build_audio_command(config, video_files[0], output_dir / "audio.m4a"),
                    config.command_timeout_seconds,
                )
                if audio_result.returncode != 0:
                    raise RuntimeError(audio_result.stderr or f"ffmpeg failed for {video_id}")
            _write_manifest(config, video_id, output_dir)
            _set_download_result(config, video_id, "complete", output_path=str(output_dir))
            return "complete"
        except (OSError, RuntimeError, ValueError) as error:
            message = str(error)
            with open_database(config.root / "catalog" / "state.sqlite3") as connection:
                attempts = int(
                    connection.execute(
                        "SELECT attempts FROM candidates WHERE video_id=?", (video_id,)
                    ).fetchone()[0]
                )
            delay = min(480.0, 30.0 * (2 ** max(0, attempts - 1)))
            if _is_transient_download_error(message) and _schedule_retry(
                config, video_id, message, delay
            ):
                wait(delay)
                _begin_retry_attempt(config, video_id)
                continue
            _set_download_result(config, video_id, "failed", error=message)
            return "failed"


def run_downloads(
    config: Config,
    runner: Runner,
    *,
    wait: Callable[[float], object] = time.sleep,
) -> dict[str, int]:
    """Download every approved candidate with bounded concurrency."""
    _initialize_runtime(config)
    check_storage(config, config.max_file_bytes * config.download_workers)
    recovered = _reconcile_running(config)
    video_ids = _claim_downloads(config)
    complete = recovered
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.download_workers) as executor:
        futures = [
            executor.submit(_download_one, config, runner, video_id, wait) for video_id in video_ids
        ]
        for future in concurrent.futures.as_completed(futures):
            outcome = future.result()
            complete += int(outcome == "complete")
            failed += int(outcome == "failed")
    return {"complete": complete, "failed": failed}


def _review_rows(config: Config) -> list[dict[str, str]]:
    database = config.root / "catalog" / "state.sqlite3"
    if not database.is_file():
        return []
    with open_database(database) as connection:
        rows = connection.execute(
            """
            SELECT video_id, source_url, title, channel_id, channel, upload_date,
                   duration, filter_status, filter_reason, review_status, reviewer_note
            FROM candidates ORDER BY video_id
            """
        ).fetchall()
    return [
        {
            "video_id": str(row["video_id"]),
            "source_url": str(row["source_url"]),
            "title": str(row["title"]),
            "channel_id": str(row["channel_id"]),
            "channel": str(row["channel"]),
            "upload_date": str(row["upload_date"]),
            "duration": format(float(row["duration"]), "g"),
            "filter_status": str(row["filter_status"]),
            "filter_reason": str(row["filter_reason"]),
            "review_status": str(row["review_status"]),
            "reviewer_note": str(row["reviewer_note"]),
        }
        for row in rows
    ]


def export_review(config: Config, target: Path) -> int:
    """Atomically export candidates for spreadsheet review."""
    rows = _review_rows(config)
    target.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=target.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o640)
    os.replace(temporary, target)
    return len(rows)


def import_review(config: Config, source: Path) -> int:
    """Import operator fields while preserving discovered evidence fields."""
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REVIEW_COLUMNS:
            raise ValueError("review CSV columns do not match the exported schema")
        rows = list(reader)
    video_ids = [row["video_id"] for row in rows]
    if len(set(video_ids)) != len(video_ids):
        raise ValueError("review CSV contains duplicate video IDs")
    originals = {row["video_id"]: row for row in _review_rows(config)}
    updates: list[tuple[str, str, str, str]] = []
    for row in rows:
        video_id = row["video_id"]
        original = originals.get(video_id)
        if original is None:
            raise ValueError(f"unknown candidate: {video_id}")
        for column in IMMUTABLE_REVIEW_COLUMNS:
            if row[column] != original[column]:
                raise ValueError(f"immutable review field changed: {video_id}.{column}")
        review_status = row["review_status"]
        if review_status not in {"pending", "approved", "rejected"}:
            raise ValueError(f"invalid review status: {review_status}")
        if review_status == "approved" and row["filter_status"] != "accepted":
            raise ValueError(f"automatically rejected candidate cannot be approved: {video_id}")
        updates.append(
            (
                review_status,
                row["reviewer_note"].strip(),
                datetime.now(UTC).isoformat(timespec="seconds"),
                video_id,
            )
        )
    with open_database(config.root / "catalog" / "state.sqlite3") as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.executemany(
            """
            UPDATE candidates SET review_status=?, reviewer_note=?, updated_at=?
            WHERE video_id=?
            """,
            updates,
        )
    return len(updates)


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Contain one bounded subprocess result."""

    returncode: int
    stdout: str
    stderr: str


class CommandRunner:
    """Run external tools without invoking a shell."""

    def run(self, args: Sequence[str], timeout: int) -> CommandResult:
        if not args or not all(isinstance(argument, str) and argument for argument in args):
            raise ValueError("command arguments must be non-empty strings")
        try:
            completed = subprocess.run(
                list(args),
                check=False,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"failed to run {args[0]}: {error}") from error
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def doctor_report(config: Config) -> dict[str, object]:
    """Return side-effect-free local dependency diagnostics."""
    tools = {
        "yt-dlp": shutil.which(config.yt_dlp),
        "ffmpeg": shutil.which(config.ffmpeg),
    }
    errors = [name for name, path in tools.items() if path is None]
    cookie: dict[str, object] = {"configured": config.cookie_file is not None}
    if config.cookie_file is not None:
        cookie["path"] = str(config.cookie_file)
        cookie["readable"] = config.cookie_file.is_file()
        if not cookie["readable"]:
            errors.append("cookie_file")
    return {"ok": not errors, "tools": tools, "cookie": cookie, "errors": errors}


def status_report(config: Config) -> dict[str, object]:
    """Return persisted candidate and download counts without mutation."""
    database = config.root / "catalog" / "state.sqlite3"
    if not database.is_file():
        return {"initialized": False, "candidates": 0, "states": {}, "reviews": {}}
    with open_database_read_only(database) as connection:
        total = int(connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
        states = {
            str(row[0]): int(row[1])
            for row in connection.execute(
                "SELECT download_state, COUNT(*) FROM candidates GROUP BY download_state"
            )
        }
        reviews = {
            str(row[0]): int(row[1])
            for row in connection.execute(
                "SELECT review_status, COUNT(*) FROM candidates GROUP BY review_status"
            )
        }
    return {"initialized": True, "candidates": total, "states": states, "reviews": reviews}


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone collector CLI parser."""
    parser = argparse.ArgumentParser(
        prog="yt-collect",
        description=__doc__,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--discovery-workers", type=int)
    parser.add_argument("--download-workers", type=int)
    parser.add_argument("--media", choices=("video", "audio", "both"))
    parser.add_argument("--date-after")
    parser.add_argument("--date-before")
    parser.add_argument("--topic", action="append", dest="topic_ids")
    subparsers = parser.add_subparsers(dest="command", required=True)
    discover_parser = subparsers.add_parser("discover", help="discover candidate metadata")
    discover_parser.add_argument("--execute", action="store_true")
    download_parser = subparsers.add_parser("download", help="download approved candidates")
    download_parser.add_argument("--execute", action="store_true")
    subparsers.add_parser("doctor", help="check local runtime dependencies")
    plan_parser = subparsers.add_parser("plan", help="print a read-only execution plan")
    plan_parser.add_argument(
        "stage", choices=("discover", "download", "all"), default="all", nargs="?"
    )
    subparsers.add_parser("status", help="show collection status")
    review_parser = subparsers.add_parser("review", help="manage candidate reviews")
    review_subparsers = review_parser.add_subparsers(dest="review_command", required=True)
    for decision in ("approve", "reject"):
        decision_parser = review_subparsers.add_parser(decision)
        decision_parser.add_argument("video_id")
        decision_parser.add_argument("--note", default="")
    export_parser = review_subparsers.add_parser("export")
    export_parser.add_argument("path", type=Path)
    import_parser = review_subparsers.add_parser("import")
    import_parser.add_argument("path", type=Path)
    return parser


def _dry_run_discovery_plan(config: Config) -> dict[str, object]:
    return {
        "mode": "dry-run",
        "command": "discover",
        "network": False,
        "workers": config.discovery_workers,
        "topics": [
            {
                "id": topic.topic_id,
                "queries": list(topic.queries),
                "max_results": topic.max_results,
                "date_after": topic.date_after.isoformat(),
                "date_before": topic.date_before.isoformat(),
            }
            for topic in config.topics
        ],
    }


def _dry_run_download_plan(config: Config) -> dict[str, object]:
    items: list[dict[str, str]] = []
    database = config.root / "catalog" / "state.sqlite3"
    if database.is_file():
        with open_database_read_only(database) as connection:
            rows = connection.execute(
                """
                SELECT video_id FROM candidates
                WHERE filter_status='accepted'
                  AND review_status='approved'
                  AND download_state IN ('pending','retry','interrupted')
                ORDER BY video_id
                """
            ).fetchall()
        items = [
            {
                "video_id": str(row["video_id"]),
                "media": config.media,
                "output_dir": str(config.root / "media" / str(row["video_id"])),
            }
            for row in rows
        ]
    return {
        "mode": "dry-run",
        "command": "download",
        "network": False,
        "workers": config.download_workers,
        "items": items,
    }


def _select_topics(config: Config, topic_ids: Sequence[str] | None) -> Config:
    if not topic_ids:
        return config
    requested = set(topic_ids)
    available = {topic.topic_id for topic in config.topics}
    unknown = requested - available
    if unknown:
        raise ValueError(f"unknown topic ids: {', '.join(sorted(unknown))}")
    return replace(
        config,
        topics=tuple(topic for topic in config.topics if topic.topic_id in requested),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the collector command-line interface."""
    arguments = build_parser().parse_args(argv)
    config = load_config(
        arguments.config,
        overrides={
            "root": str(arguments.root) if arguments.root is not None else None,
            "discovery_workers": arguments.discovery_workers,
            "download_workers": arguments.download_workers,
            "media": arguments.media,
            "date_after": arguments.date_after,
            "date_before": arguments.date_before,
        },
    )
    config = _select_topics(config, arguments.topic_ids)
    if arguments.command == "discover" and not arguments.execute:
        print(json.dumps(_dry_run_discovery_plan(config), ensure_ascii=False))
        return 0
    if arguments.command == "download" and not arguments.execute:
        print(json.dumps(_dry_run_download_plan(config), ensure_ascii=False))
        return 0
    if arguments.command == "discover":
        with SingleInstanceLock(config.root / ".collector.lock"):
            summary = run_discovery(config, CommandRunner())
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    if arguments.command == "download":
        with SingleInstanceLock(config.root / ".collector.lock"):
            summary = run_downloads(config, CommandRunner())
        print(json.dumps(summary, ensure_ascii=False))
        return 1 if summary["failed"] else 0
    if arguments.command == "plan":
        result: dict[str, object] = {"mode": "dry-run", "network": False}
        if arguments.stage in {"discover", "all"}:
            result["discover"] = _dry_run_discovery_plan(config)
        if arguments.stage in {"download", "all"}:
            result["download"] = _dry_run_download_plan(config)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if arguments.command == "doctor":
        report = doctor_report(config)
        print(json.dumps(report, ensure_ascii=False))
        return 0 if report["ok"] else 2
    if arguments.command == "status":
        print(json.dumps(status_report(config), ensure_ascii=False))
        return 0
    if arguments.command == "review":
        with SingleInstanceLock(config.root / ".collector.lock"):
            if arguments.review_command in {"approve", "reject"}:
                review_status = "approved" if arguments.review_command == "approve" else "rejected"
                set_review_status(config, arguments.video_id, review_status, arguments.note)
                print(json.dumps({"video_id": arguments.video_id, "review_status": review_status}))
                return 0
            if arguments.review_command == "export":
                count = export_review(config, arguments.path)
                print(json.dumps({"exported": count, "path": str(arguments.path)}))
                return 0
            if arguments.review_command == "import":
                count = import_review(config, arguments.path)
                print(json.dumps({"imported": count, "path": str(arguments.path)}))
                return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
