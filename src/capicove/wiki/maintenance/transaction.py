"""Recoverable local file transactions; cooperating readers honor the marker.

This primitive is internal. Callers must validate their entire candidate view
before invoking it. Process-crash recovery does not imply power-loss safety.
"""

import hashlib
import os
import re
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from capicove.wiki.maintenance.store import (
    canonical_object,
    create_verified,
    ensure_directory,
    maintenance_lock,
    read_authorized,
    read_object,
    unlink_authorized,
    write_authorized,
)


_LIMIT = 768 * 1024 * 1024
_TOTAL_LIMIT = 2 * 1024 * 1024 * 1024


def image_limit(relative: str) -> int:
    """Permit bounded binary indexes without enlarging editorial record limits."""
    if relative.startswith("_derived/"):
        return _LIMIT
    if relative.startswith("raw/"):
        return 256 * 1024 * 1024
    return 16 * 1024 * 1024


_ALLOWED = (
    "pages/",
    "maintenance/managed/",
    "maintenance/publications/",
    "maintenance/follow-ups/",
    "maintenance/book-snapshots/",
    "raw/",
    "representations/",
    "graph/",
    "_derived/",
)
_STATE = "maintenance/state.json"
_FIELDS = {"schema_version", "transaction_id", "policy_revision", "changes", "read_set"}


def _read_set(wiki: Path, excluded: set[str]) -> str:
    """Pin unchanged source and page inputs, including directory membership."""
    result = {}
    for root in (
        "raw",
        "representations",
        "pages",
        "graph",
        "maintenance/managed",
        "maintenance/book-snapshots",
    ):
        for path in sorted((wiki / root).rglob("*")):
            relative = path.relative_to(wiki).as_posix()
            if path.is_symlink():
                raise ValueError("recovery_conflict: redirected input")
            if path.is_file() and relative not in excluded:
                result[relative] = hashlib.sha256(
                    read_authorized(path, (wiki,), 256 * 1024 * 1024)
                ).hexdigest()
                if len(result) > 50000:
                    raise ValueError("capacity_exceeded: transaction inputs")
    digest = hashlib.sha256()
    for relative, value in sorted(result.items()):
        digest.update(relative.encode() + b"\0" + value.encode() + b"\n")
    return digest.hexdigest()


def _path(wiki: Path, relative: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or any(p in {"", ".", ".."} for p in relative.split("/"))
        or "\\" in relative
        or ":" in relative
        or not (
            relative in {_STATE, "wiki.toml", "SCHEMA.md", "index.md", "log.md"}
            or relative.startswith(_ALLOWED)
        )
    ):
        raise ValueError("unsafe transaction destination")
    path = wiki / relative
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ValueError("recovery_conflict: redirected destination")
        if ancestor == wiki:
            break
    return path


def _content(wiki: Path, relative: str) -> bytes | None:
    path = _path(wiki, relative)
    return read_authorized(path, (wiki,), image_limit(relative)) if path.exists() else None


def _digest(data: bytes | None) -> str | None:
    return hashlib.sha256(data).hexdigest() if data is not None else None


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _replace(wiki: Path, relative: str, content: bytes | None) -> None:
    path = _path(wiki, relative)
    if content is None:
        unlink_authorized(path, wiki)
    else:
        write_authorized(path, content, wiki)


def _policy(wiki: Path) -> int:
    value = read_object(
        wiki / "maintenance/source-policy.json",
        wiki,
        {"schema_version", "policy_revision", "withdrawn"},
    )
    if type(value["policy_revision"]) is not int or value["policy_revision"] < 0:
        raise ValueError("invalid source policy")
    return int(value["policy_revision"])


def _blob(wiki: Path, directory: Path, digest: str | None) -> bytes | None:
    if digest is None:
        return None
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("invalid transaction blob identity")
    data = read_authorized(directory / digest, (wiki,), _LIMIT)
    if _digest(data) != digest:
        raise ValueError("evidence_digest_mismatch: journal blob")
    return data


def _finish(
    wiki: Path,
    journal: dict[str, Any],
    directory: Path,
    *,
    abort: bool,
    barrier: Callable[[str], None],
) -> None:
    changes = journal["changes"]
    if (
        type(journal["schema_version"]) is not int
        or journal["schema_version"] != 1
        or type(journal["policy_revision"]) is not int
        or not isinstance(changes, list)
        or not 1 <= len(changes) <= 256
    ):
        raise ValueError("invalid transaction journal")
    if not abort and _policy(wiki) != journal["policy_revision"]:
        raise ValueError("policy_changed")
    targets: dict[str, bytes | None] = {}
    total = 0
    for change in changes:
        if not isinstance(change, dict) or set(change) != {"path", "before", "after"}:
            raise ValueError("invalid transaction change")
        relative = change["path"]
        _path(wiki, relative)
        if relative in targets:
            raise ValueError("duplicate transaction destination")
        before = _blob(wiki, directory, change["before"])
        after = _blob(wiki, directory, change["after"])
        total += len(before or b"") + len(after or b"")
        if total > _TOTAL_LIMIT or any(
            len(value) > image_limit(relative) for value in (before, after) if value
        ):
            raise ValueError("capacity_exceeded: transaction images")
        if _digest(_content(wiki, relative)) not in {change["before"], change["after"]}:
            raise ValueError("recovery_conflict: destination changed")
        targets[relative] = before if abort else after
    if _STATE not in targets or changes[-1]["path"] != _STATE:
        raise ValueError("invalid transaction state ordering")
    if not abort and _read_set(wiki, set(targets)) != journal["read_set"]:
        raise ValueError("recovery_conflict: transaction input changed")
    # Validate every destination and both images before the first write.
    for relative, content in targets.items():
        if relative == _STATE:
            barrier("after_receipt")
            barrier("before_state")
        if _content(wiki, relative) != content:
            _replace(wiki, relative, content)
        barrier("after_state" if relative == _STATE else "after_destination")
    for relative, content in targets.items():
        if _content(wiki, relative) != content:
            raise ValueError("recovery_conflict: published verification")
    barrier("before_marker_removal")
    create_verified(
        directory / "outcome.json",
        canonical_object(
            {
                "transaction_id": journal["transaction_id"],
                "status": "aborted" if abort else "committed",
            }
        ),
        wiki,
    )
    marker = wiki / "_migration/maintenance.pending"
    unlink_authorized(marker, wiki)
    _sync_directory(marker.parent)


def transact(
    wiki: Path,
    transaction_id: str,
    changes: Mapping[str, bytes | None],
    *,
    state_bytes: bytes,
    policy_revision: int,
    barrier: Callable[[str], None] = lambda _: None,
) -> None:
    """Commit host-validated bytes under the common writer lock."""
    with maintenance_lock(wiki):
        transact_locked(
            wiki,
            transaction_id,
            changes,
            state_bytes=state_bytes,
            policy_revision=policy_revision,
            barrier=barrier,
        )


def transact_locked(
    wiki: Path,
    transaction_id: str,
    changes: Mapping[str, bytes | None],
    *,
    state_bytes: bytes,
    policy_revision: int,
    barrier: Callable[[str], None] = lambda _: None,
) -> None:
    """Internal variant for callers already holding the common writer lock."""
    if re.fullmatch(r"[0-9a-f]{64}", transaction_id) is None:
        raise ValueError("invalid transaction identity")
    marker = wiki / "_migration/maintenance.pending"
    if marker.exists() or marker.is_symlink():
        raise ValueError("publication_pending")
    if _policy(wiki) != policy_revision:
        raise ValueError("policy_changed")
    if _STATE in changes or not 0 <= len(changes) < 256:
        raise ValueError("invalid transaction changes")
    all_changes = {**changes, _STATE: state_bytes}
    images = []
    total = 0
    for relative, after in all_changes.items():
        before = _content(wiki, relative)
        if after is not None and (
            not isinstance(after, bytes) or len(after) > image_limit(relative)
        ):
            raise ValueError("capacity_exceeded: transaction image")
        total += len(before or b"") + len(after or b"")
        if total > _TOTAL_LIMIT:
            raise ValueError("capacity_exceeded: transaction")
        images.append((relative, before, after))
    if (
        sum(len(data) for _, before, after in images for data in (before, after) if data)
        > _TOTAL_LIMIT
    ):
        raise ValueError("capacity_exceeded: transaction")
    required = sum(len(data) for _, before, after in images for data in (before, after) if data)
    if shutil.disk_usage(wiki).free < required * 2 + 16 * 1024 * 1024:
        raise ValueError("capacity_exceeded: insufficient journal disk space")
    read_set = _read_set(wiki, set(all_changes))
    barrier("before_journal")
    directory = wiki / "_migration" / ("maintenance-" + transaction_id)
    if directory.is_symlink():
        raise ValueError("unsafe journal directory")
    ensure_directory(directory, wiki)
    entries = []
    for relative, before, after in images:
        for data in (before, after):
            if data is not None:
                create_verified(directory / str(_digest(data)), data, wiki)
        entries.append({"path": relative, "before": _digest(before), "after": _digest(after)})
    journal = {
        "schema_version": 1,
        "transaction_id": transaction_id,
        "policy_revision": policy_revision,
        "changes": entries,
        "read_set": read_set,
    }
    encoded = canonical_object(journal)
    create_verified(directory / "journal.json", encoded, wiki)
    _sync_directory(directory)
    create_verified(
        marker,
        canonical_object({"transaction_id": transaction_id, "journal_sha256": _digest(encoded)}),
        wiki,
    )
    _sync_directory(marker.parent)
    _finish(wiki, journal, directory, abort=False, barrier=barrier)


def recover(wiki: Path, *, abort: bool = False) -> None:
    """Resume or restore verified before-images without changing live policy."""
    with maintenance_lock(wiki):
        marker = wiki / "_migration/maintenance.pending"
        if not marker.exists() and not marker.is_symlink():
            return
        pointer = read_object(marker, wiki, {"transaction_id", "journal_sha256"})
        identity = pointer["transaction_id"]
        if not isinstance(identity, str) or re.fullmatch(r"[0-9a-f]{64}", identity) is None:
            raise ValueError("invalid transaction identity")
        directory = wiki / "_migration" / ("maintenance-" + identity)
        raw = read_authorized(directory / "journal.json", (wiki,), 16 * 1024 * 1024)
        if _digest(raw) != pointer["journal_sha256"]:
            raise ValueError("evidence_digest_mismatch: journal")
        journal = read_object(directory / "journal.json", wiki, _FIELDS)
        if journal["transaction_id"] != identity:
            raise ValueError("invalid transaction identity")
        _finish(wiki, journal, directory, abort=abort, barrier=lambda _: None)
