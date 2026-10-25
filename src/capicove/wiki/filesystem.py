"""Safe filesystem primitives for local Wiki workspaces."""

from __future__ import annotations

import errno
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import threading
import tomllib
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from capicove.wiki.models import (
    WikiAuthority,
    WikiIngestResult,
    WikiInventory,
    WikiLintReport,
    WikiMigrationPlan,
)


_LAYOUT_ROOTS = ("raw", "pages", "graph", "_derived", "_migration")
_MANAGED_FILES = (
    "wiki.toml",
    "SCHEMA.md",
    "index.md",
    "log.md",
    "graph/ontology.yaml",
    "graph/facts.jsonl",
    "graph/identity-events.jsonl",
    "_migration/writer.lock",
)
_MAX_PAGE_SLUG_LENGTH = 200
_WINDOWS_RESERVED_PAGE_STEMS = frozenset(
    {"aux", "con", "nul", "prn"}
    | {f"com{number}" for number in range(1, 10)}
    | {f"lpt{number}" for number in range(1, 10)}
)
_WRITER_LOCKS_GUARD = threading.Lock()
_WRITER_LOCKS: dict[str, threading.RLock] = {}


class WikiError(RuntimeError):
    """Base class for stable local Wiki failures."""


class WikiExistsError(WikiError):
    """Raised when initialization would overwrite an existing Wiki."""


class WikiNotFoundError(WikiError):
    """Raised when an operation requires a missing Wiki."""


class WikiUnsafePathError(WikiError):
    """Raised when a managed path escapes the Wiki or uses a symlink."""


class WikiFormatError(WikiError):
    """Raised when a durable Wiki file cannot be parsed."""


class WikiDependencyError(WikiFormatError):
    """Raised when an explicitly requested optional Wiki dependency is absent."""


class WikiIndexUnavailableError(WikiError):
    """Raised when query requires an absent or unreadable derived index."""


class WikiValidationError(WikiError):
    """Raised when a build is blocked by validation errors."""

    def __init__(self, report: WikiLintReport) -> None:
        super().__init__("Wiki validation failed")
        self.report = report


@dataclass(frozen=True, slots=True)
class MarkdownRecord:
    """Represent parsed YAML frontmatter and exact Markdown body text."""

    path: Path
    metadata: dict[str, Any]
    body: str


def initialize(wiki: Path, *, created_on: date) -> WikiInventory:
    """Create a minimal standalone Wiki without overwriting existing content."""
    if wiki.exists() or wiki.is_symlink():
        raise WikiExistsError(f"Wiki already exists: {wiki}")
    try:
        wiki.parent.mkdir(parents=True, exist_ok=True)
        staging_parent = Path(tempfile.mkdtemp(prefix=f".{wiki.name}.init-", dir=wiki.parent))
    except OSError as error:
        raise WikiError("Wiki initialization failed") from error
    staging = staging_parent / wiki.name
    try:
        staging.mkdir()
        for name in _LAYOUT_ROOTS:
            (staging / name).mkdir(parents=True, exist_ok=False)
        _atomic_write(staging / "wiki.toml", 'schema_version = 1\nauthority = "standalone"\n')
        _atomic_write(
            staging / "SCHEMA.md",
            "# Capicove Wiki Schema\n\n"
            "Schema version: 1\n"
            "Mode: standalone (service-independent local projection)\n",
        )
        _atomic_write(
            staging / "index.md",
            f"# Wiki Index\n\n> Last built: {created_on.isoformat()} | Total pages: 0\n",
        )
        _atomic_write(
            staging / "log.md",
            f"# Wiki Log\n\n## [{created_on.isoformat()}] init | Wiki initialized\n",
        )
        _atomic_write(
            staging / "graph" / "ontology.yaml",
            "schemaVersion: 1\nentityTypes: {}\npredicates: {}\nlabels:\n  namespaces: []\n",
        )
        _atomic_write(staging / "graph" / "facts.jsonl", b"")
        _atomic_write(staging / "graph" / "identity-events.jsonl", b"")
        _atomic_write(staging / "_migration" / "writer.lock", b"\0")
        inventory = scan(staging)
        if wiki.exists() or wiki.is_symlink():
            raise WikiExistsError(f"Wiki already exists: {wiki}")
        staging.rename(wiki)
        return inventory
    except WikiExistsError:
        raise
    except OSError as error:
        if wiki.exists() or wiki.is_symlink():
            raise WikiExistsError(f"Wiki already exists: {wiki}") from error
        raise WikiError("Wiki initialization failed") from error
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)


@contextmanager
def wiki_write_lock(wiki: Path) -> Iterator[None]:
    """Serialize and validate legacy Wiki mutations."""
    with _wiki_filesystem_lock(wiki):
        _require_safe_layout(wiki)
        _read_manifest(wiki)
        _cleanup_stale_create_files(wiki)
        yield


@contextmanager
def _wiki_filesystem_lock(wiki: Path) -> Iterator[None]:
    """Acquire the common lock; each schema adapter validates its own layout."""
    key = str(wiki.resolve())
    with _WRITER_LOCKS_GUARD:
        thread_lock = _WRITER_LOCKS.setdefault(key, threading.RLock())
    with thread_lock:
        lock_path = wiki / "_migration" / "writer.lock"
        flags = os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(lock_path, flags)
        except OSError as error:
            raise WikiError("Wiki writer lock is unavailable") from error
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise WikiUnsafePathError("Wiki writer lock must be a regular file")
        acquired = False
        try:
            try:
                if sys.platform == "win32":  # pragma: no cover - exercised on Windows
                    import msvcrt

                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
                else:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                acquired = True
            except OSError as error:
                raise WikiError("Wiki writer lock is unavailable") from error
            yield
        finally:
            if acquired:
                if sys.platform == "win32":  # pragma: no cover - exercised on Windows
                    import msvcrt

                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def scan(wiki: Path) -> WikiInventory:
    """Return a deterministic inventory of one initialized Wiki."""
    from capicove.wiki.maintenance.compatibility import is_v3, scan_v3

    if is_v3(wiki):
        return scan_v3(wiki)
    _require_safe_layout(wiki)
    manifest = _read_manifest(wiki)
    return WikiInventory(
        schema_version=manifest[0],
        authority=manifest[1],
        pages=len(_regular_files(wiki / "pages", suffix=".md")),
        sources=len(_regular_files(wiki / "raw")),
        facts=_jsonl_count(wiki / "graph" / "facts.jsonl"),
        identity_events=_jsonl_count(wiki / "graph" / "identity-events.jsonl"),
    )


def migrate_to_v2(wiki: Path, *, preview: bool) -> WikiMigrationPlan:
    """Preview or explicitly apply the page-aware Wiki schema-v2 migration."""
    _require_safe_layout(wiki)
    version, _ = _read_manifest(wiki)
    if version == 2:
        return WikiMigrationPlan(2, 2, (), (), False)
    created = ("representations", "pages/books", "graph/citations.jsonl")
    updated = ("wiki.toml", "SCHEMA.md")
    if preview:
        return WikiMigrationPlan(1, 2, created, updated, False)
    (wiki / "representations").mkdir(exist_ok=False)
    (wiki / "pages" / "books").mkdir(exist_ok=False)
    _atomic_write(wiki / "graph" / "citations.jsonl", b"")
    _atomic_write(wiki / "wiki.toml", 'schema_version = 2\nauthority = "standalone"\n')
    _atomic_write(
        wiki / "SCHEMA.md",
        "# Capicove Wiki Schema\n\n"
        "Schema version: 2\n"
        "Mode: standalone page-aware book projection\n",
    )
    _require_safe_layout(wiki)
    return WikiMigrationPlan(1, 2, created, updated, True)


def ingest_markdown(wiki: Path, source: Path, *, ingested_on: date) -> WikiIngestResult:
    """Preserve one UTF-8 Markdown source and create a cited durable page."""
    from capicove.wiki.maintenance.compatibility import execute, is_v3

    if is_v3(wiki):
        return execute(
            wiki, "ingest", lambda stage: ingest_markdown(stage, source, ingested_on=ingested_on)
        )
    _require_safe_layout(wiki)
    if source.is_symlink() or not source.is_file():
        raise WikiFormatError("ingest source must be a regular file")
    try:
        _reject_symlinked_ancestors(source)
        raw = _read_stable_regular_file(source, "ingest source must be a regular file")
        text = raw.decode("utf-8-sig")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    except (OSError, UnicodeError) as error:
        raise WikiFormatError("ingest source must be readable UTF-8 Markdown") from error
    import hashlib

    digest = hashlib.sha256(raw).hexdigest()
    source_id = f"source:sha256:{digest}"
    page_id = f"page:{digest[:26].upper()}"
    raw_path = wiki / "raw" / f"{digest}.md"
    metadata, body = _split_markdown_text(source, text)
    title = _source_title(source, metadata, body)
    try:
        title.encode("utf-8")
    except UnicodeEncodeError as error:
        raise WikiFormatError("page title must contain valid Unicode") from error
    slug = _slug(title, digest)
    rendered = _render_page(
        page_id=page_id,
        title=title,
        source_id=source_id,
        body=body,
        page_date=ingested_on,
    )
    try:
        rendered.encode("utf-8")
    except UnicodeEncodeError as error:  # pragma: no cover - title check guards this
        raise WikiFormatError("generated page must contain valid Unicode") from error

    try:
        with wiki_write_lock(wiki):
            _require_safe_layout(wiki)
            read_log(wiki)
            _ensure_content_addressed_source(raw_path, raw, wiki=wiki)
            existing = _page_for_source_id(wiki, page_id, source_id)
            if existing is not None:
                existing_page, created_on = existing
                relative = existing_page.relative_to(wiki).as_posix()
                _ensure_ingest_log(
                    wiki,
                    relative,
                    source_id=source_id,
                    ingested_on=created_on,
                )
                return WikiIngestResult(
                    source_id=source_id,
                    page_id=page_id,
                    page_path=relative,
                    created=False,
                )

            page_path: Path | None = None
            base_path = wiki / "pages" / f"{slug}.md"
            collision_paths = tuple(
                wiki / "pages" / f"{_bounded_digest_slug(slug, digest, length)}.md"
                for length in range(8, 65, 4)
            )
            candidates = (base_path, *collision_paths)
            for candidate in dict.fromkeys(candidates):
                if _atomic_create(candidate, rendered, temporary_directory=wiki / "_migration"):
                    page_path = candidate
                    break
                concurrent = _page_for_source_id(wiki, page_id, source_id)
                if concurrent is not None:
                    concurrent_page, created_on = concurrent
                    relative = concurrent_page.relative_to(wiki).as_posix()
                    _ensure_ingest_log(
                        wiki,
                        relative,
                        source_id=source_id,
                        ingested_on=created_on,
                    )
                    return WikiIngestResult(
                        source_id=source_id,
                        page_id=page_id,
                        page_path=relative,
                        created=False,
                    )
            if page_path is None:
                raise WikiFormatError("stable page destination already exists")
            relative = page_path.relative_to(wiki).as_posix()
            _ensure_ingest_log(
                wiki,
                relative,
                source_id=source_id,
                ingested_on=ingested_on,
            )
            return WikiIngestResult(
                source_id=source_id,
                page_id=page_id,
                page_path=relative,
                created=True,
            )
    except WikiError:
        raise
    except (OSError, UnicodeError) as error:
        raise WikiError("Wiki ingest failed") from error


def read_markdown(path: Path) -> MarkdownRecord:
    """Parse one UTF-8 Markdown file with required YAML frontmatter."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise WikiFormatError("Markdown file is not readable UTF-8") from error
    metadata, body = _split_markdown_text(path, text, require_frontmatter=True)
    return MarkdownRecord(path=path, metadata=metadata, body=body)


def _require_safe_layout(wiki: Path) -> None:
    if not wiki.exists():
        raise WikiNotFoundError(f"Wiki does not exist: {wiki}")
    if wiki.is_symlink() or not wiki.is_dir():
        raise WikiUnsafePathError("Wiki root must be a real directory, not a symbolic link")
    root = wiki.resolve()
    version, _ = _read_manifest(wiki)
    for name in _LAYOUT_ROOTS:
        path = wiki / name
        if path.is_symlink():
            raise WikiUnsafePathError(f"managed Wiki root is a symbolic link: {name}")
        if not path.is_dir():
            raise WikiFormatError(f"managed Wiki root is missing: {name}")
        resolved = path.resolve()
        if resolved != root and root not in resolved.parents:
            raise WikiUnsafePathError(f"managed Wiki root escapes the Wiki: {name}")
        for managed in path.rglob("*"):
            relative = managed.relative_to(wiki).as_posix()
            if managed.is_symlink():
                raise WikiUnsafePathError(f"managed Wiki path is a symbolic link: {relative}")
            resolved_managed = managed.resolve()
            if resolved_managed != root and root not in resolved_managed.parents:
                raise WikiUnsafePathError(f"managed Wiki path escapes the Wiki: {relative}")
    for relative in _MANAGED_FILES:
        path = wiki / relative
        if path.is_symlink():
            raise WikiUnsafePathError(f"managed Wiki file is a symbolic link: {relative}")
        if not path.is_file():
            raise WikiFormatError(f"managed Wiki file is missing: {relative}")
        resolved = path.resolve()
        if root not in resolved.parents:
            raise WikiUnsafePathError(f"managed Wiki file escapes the Wiki: {relative}")
    if version == 2:
        for relative in ("representations", "pages/books"):
            path = wiki / relative
            if path.is_symlink() or not path.is_dir():
                raise WikiFormatError(f"managed Wiki v2 root is missing: {relative}")
        citations = wiki / "graph" / "citations.jsonl"
        if citations.is_symlink() or not citations.is_file():
            raise WikiFormatError("managed Wiki v2 citation ledger is missing")


def _read_manifest(wiki: Path) -> tuple[int, WikiAuthority]:
    try:
        value = tomllib.loads((wiki / "wiki.toml").read_text(encoding="utf-8"))
        version = value["schema_version"]
        authority = WikiAuthority(value["authority"])
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, KeyError, ValueError) as error:
        raise WikiFormatError("wiki.toml is invalid") from error
    if type(version) is not int or version not in {1, 2}:
        raise WikiFormatError("unsupported Wiki schema version")
    return version, authority


def _split_markdown_text(
    path: Path,
    text: str,
    *,
    require_frontmatter: bool = False,
) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        if require_frontmatter:
            raise WikiFormatError(f"Markdown frontmatter is missing: {path.name}")
        return {}, text
    frontmatter_end: int | None = None
    body_start: int | None = None
    line_start = 4
    while line_start <= len(text):
        newline = text.find("\n", line_start)
        line_end = len(text) if newline < 0 else newline
        if text[line_start:line_end] == "---":
            frontmatter_end = line_start
            if line_start > 4 and text[line_start - 1] == "\n":
                frontmatter_end -= 1
            body_start = line_end if newline < 0 else line_end + 1
            break
        if newline < 0:
            break
        line_start = newline + 1
    if frontmatter_end is None or body_start is None:
        raise WikiFormatError(f"Markdown frontmatter is unclosed: {path.name}")
    try:
        import yaml  # type: ignore[import-untyped]

        loaded = yaml.safe_load(text[4:frontmatter_end]) or {}
    except ImportError as error:
        raise WikiDependencyError(
            "Wiki YAML support is unavailable; install capicove[wiki]"
        ) from error
    except Exception as error:
        raise WikiFormatError(f"Markdown frontmatter is invalid: {path.name}") from error
    if not isinstance(loaded, dict):
        raise WikiFormatError(f"Markdown frontmatter must be a mapping: {path.name}")
    return loaded, text[body_start:]


def load_yaml_mapping(path: Path, *, content: bytes | None = None) -> dict[str, Any]:
    """Load a YAML mapping without importing YAML during package import."""
    try:
        import yaml

        text = content.decode("utf-8") if content is not None else path.read_text(encoding="utf-8")
        value = yaml.safe_load(text)
    except ImportError as error:
        raise WikiDependencyError(
            "Wiki YAML support is unavailable; install capicove[wiki]"
        ) from error
    except Exception as error:
        raise WikiFormatError(f"YAML file is invalid: {path.name}") from error
    if not isinstance(value, dict):
        raise WikiFormatError(f"YAML file must contain a mapping: {path.name}")
    return value


def page_paths(wiki: Path) -> tuple[Path, ...]:
    """Return Wiki page paths in canonical relative-path order."""
    _require_safe_layout(wiki)
    return tuple(_regular_files(wiki / "pages", suffix=".md"))


def jsonl_records(path: Path) -> tuple[dict[str, Any], ...]:
    """Parse JSONL objects or raise a stable format error."""
    try:
        return jsonl_records_from_bytes(path.read_bytes(), name=path.name)
    except OSError as error:
        raise WikiFormatError(f"{path.name} is invalid") from error


def jsonl_records_from_bytes(content: bytes, *, name: str) -> tuple[dict[str, Any], ...]:
    """Parse already-opened JSONL bytes without reopening an untrusted path."""
    records: list[dict[str, Any]] = []
    try:
        for number, line in enumerate(content.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise WikiFormatError(f"{name} line {number} must be an object")
            records.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WikiFormatError(f"{name} is invalid") from error
    return tuple(records)


def _page_for_id(wiki: Path, page_id: str) -> Path | None:
    matches: list[Path] = []
    for path in _regular_files(wiki / "pages", suffix=".md"):
        try:
            if read_markdown(path).metadata.get("id") == page_id:
                matches.append(path)
        except WikiDependencyError:
            raise
        except WikiFormatError:
            continue
    if len(matches) > 1:
        raise WikiFormatError("page identity is duplicated")
    return matches[0] if matches else None


def _page_for_source_id(wiki: Path, page_id: str, source_id: str) -> tuple[Path, date] | None:
    path = _page_for_id(wiki, page_id)
    if path is None:
        return None
    metadata = read_markdown(path).metadata
    sources = metadata.get("sources")
    if not isinstance(sources, list) or source_id not in sources:
        raise WikiFormatError("page identity disagrees with the full source digest")
    created = metadata.get("created")
    if isinstance(created, date):
        created_on = created
    elif isinstance(created, str):
        try:
            created_on = date.fromisoformat(created)
        except ValueError as error:
            raise WikiFormatError("existing page created date is invalid") from error
    else:
        raise WikiFormatError("existing page created date is invalid")
    return path, created_on


def _source_title(source: Path, metadata: dict[str, Any], body: str) -> str:
    configured = metadata.get("title")
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    match = re.search(r"^#\s+(.+?)\s*$", body, re.MULTILINE)
    if match is not None:
        return match.group(1).strip()
    return source.stem.strip() or "Untitled"


def _reject_symlinked_ancestors(path: Path) -> None:
    if os.pardir in path.parts:
        raise WikiFormatError("ingest source path must not contain parent traversal")
    absolute = Path(os.path.abspath(path))
    try:
        for ancestor in absolute.parents:
            if stat.S_ISLNK(os.lstat(ancestor).st_mode):
                raise WikiFormatError("ingest source path contains a symbolic link")
    except OSError as error:
        raise WikiFormatError("ingest source path cannot be inspected safely") from error


def _read_stable_regular_file(path: Path, error_message: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        before = os.lstat(path)
        if not stat.S_ISREG(before.st_mode):
            raise WikiFormatError(error_message)
        descriptor = os.open(path, flags)
    except WikiFormatError:
        raise
    except OSError as error:
        raise WikiFormatError(error_message) from error
    try:
        with os.fdopen(descriptor, "rb") as stream:
            opened_before = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened_before.st_mode) or _file_identity(
                opened_before
            ) != _file_identity(before):
                raise WikiFormatError(error_message)
            content = stream.read()
            opened_after = os.fstat(stream.fileno())
        after = os.lstat(path)
    except WikiFormatError:
        raise
    except OSError as error:
        raise WikiFormatError(error_message) from error
    if _file_snapshot(opened_before) != _file_snapshot(opened_after) or _file_snapshot(
        opened_after
    ) != _file_snapshot(after):
        raise WikiFormatError(error_message)
    return content


def _file_identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _file_snapshot(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _ensure_content_addressed_source(raw_path: Path, raw: bytes, *, wiki: Path) -> None:
    exists = raw_path.exists()
    if not exists and _atomic_create(
        raw_path,
        raw,
        temporary_directory=wiki / "_migration",
    ):
        return
    captured = _read_stable_regular_file(
        raw_path,
        "content-addressed source is unreadable",
    )
    if captured != raw:
        raise WikiFormatError("content-addressed source bytes disagree")


def _cleanup_stale_create_files(wiki: Path) -> None:
    migration = wiki / "_migration"
    for path in sorted(migration.glob(".create-*"), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file():
            raise WikiUnsafePathError("stale Wiki publication path is unsafe")
        try:
            path.unlink()
        except OSError as error:
            raise WikiError("stale Wiki publication cleanup failed") from error


def _bounded_digest_slug(slug: str, digest: str, length: int = 8) -> str:
    suffix = f"-{digest[:length]}"
    prefix = slug[: _MAX_PAGE_SLUG_LENGTH - len(suffix)].rstrip("-") or "page"
    return f"{prefix}{suffix}"


def _slug(title: str, digest: str) -> str:
    normalized = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-")
    if not slug or slug in _WINDOWS_RESERVED_PAGE_STEMS:
        return f"page-{digest[:8]}"
    if len(slug) > _MAX_PAGE_SLUG_LENGTH:
        return _bounded_digest_slug(slug, digest)
    return slug


def _render_page(
    *,
    page_id: str,
    title: str,
    source_id: str,
    body: str,
    page_date: date,
) -> str:
    rendered_title = json.dumps(title, ensure_ascii=False)
    rendered_sources = json.dumps([source_id], ensure_ascii=False, separators=(",", ":"))
    normalized_body = body.lstrip("\n")
    return (
        "---\n"
        f"id: {page_id}\n"
        f"title: {rendered_title}\n"
        "kind: concept\n"
        f"created: {page_date.isoformat()}\n"
        f"updated: {page_date.isoformat()}\n"
        f"sources: {rendered_sources}\n"
        "---\n\n"
        f"{normalized_body}"
    )


def _append_log(wiki: Path, entry: str) -> None:
    path = wiki / "log.md"
    current = read_log(wiki)
    separator = "" if current.endswith("\n\n") else "\n"
    _atomic_write(path, f"{current}{separator}{entry}\n")


def _ensure_ingest_log(
    wiki: Path,
    page_path: str,
    *,
    source_id: str,
    ingested_on: date,
) -> None:
    lines = read_log(wiki).splitlines()
    marker = f"ingest | {page_path} | {source_id}"
    if any(line.startswith("## [") and line.endswith(marker) for line in lines):
        return
    _append_log(wiki, f"## [{ingested_on.isoformat()}] {marker}")


def atomic_write(path: Path, content: str | bytes) -> None:
    """Atomically replace one capability-owned file."""
    _atomic_write(path, content)


def read_log(wiki: Path) -> str:
    """Read the durable Wiki log as UTF-8 or raise a stable format error."""
    try:
        return (wiki / "log.md").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise WikiFormatError("Wiki log is unreadable") from error


def append_log(wiki: Path, entry: str) -> None:
    """Append one durable operation through an atomic replacement."""
    _append_log(wiki, entry)


def _regular_files(root: Path, *, suffix: str | None = None) -> list[Path]:
    files = [path for path in root.rglob("*") if path.is_file() and not path.is_symlink()]
    if suffix is not None:
        files = [path for path in files if path.suffix == suffix]
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _jsonl_count(path: Path) -> int:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        count = 0
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise WikiFormatError(f"{path.name} line {number} must be an object")
            count += 1
        return count
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WikiFormatError(f"{path.name} is invalid") from error


def _atomic_write(path: Path, content: str | bytes) -> None:
    data = content.encode("utf-8") if isinstance(content, str) else content
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_create(
    path: Path,
    content: str | bytes,
    *,
    temporary_directory: Path,
) -> bool:
    """Publish one complete regular file only when the destination is absent."""
    data = content.encode("utf-8") if isinstance(content, str) else content
    descriptor, temporary_name = tempfile.mkstemp(prefix=".create-", dir=temporary_directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError:
            return False
        except OSError as error:
            if error.errno in (
                errno.ENOTSUP,
                errno.EOPNOTSUPP,
                errno.ENOSYS,
                errno.EXDEV,
            ):
                raise WikiError(
                    "local Wiki filesystem does not support safe create-only "
                    "publication with hard links"
                ) from error
            raise
        return True
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


__all__ = [
    "MarkdownRecord",
    "WikiDependencyError",
    "WikiError",
    "WikiExistsError",
    "WikiFormatError",
    "WikiIndexUnavailableError",
    "WikiNotFoundError",
    "WikiUnsafePathError",
    "WikiValidationError",
    "append_log",
    "atomic_write",
    "ingest_markdown",
    "initialize",
    "jsonl_records",
    "load_yaml_mapping",
    "migrate_to_v2",
    "page_paths",
    "read_log",
    "read_markdown",
    "scan",
    "wiki_write_lock",
]
