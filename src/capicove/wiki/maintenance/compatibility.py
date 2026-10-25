"""Private legacy projections for version-fenced v3 operations.

Legacy algorithms run only in a private v2 staging view. Their validated byte
changes enter the same v3 journal and truthful system-origin publication chain.
"""

import hashlib
import json
import re
import tempfile
import tomllib
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, TypeVar

from capicove.contracts.wiki_maintenance import FileChange, MaintenanceRecord, PublicationRecord
from capicove.wiki.maintenance.codec import decode_record, encode_record
from capicove.wiki.maintenance.service import MaintenanceService
from capicove.wiki.maintenance.store import (
    canonical_object,
    maintenance_lock,
    read_authorized,
    record_path,
)
from capicove.wiki.maintenance.tasks import advance_state, state
from capicove.wiki.maintenance.transaction import image_limit, transact_locked
from capicove.wiki.models import WikiInventory


_T = TypeVar("_T")
_ROOTS = {"raw", "pages", "graph", "representations", "_derived"}
_FILES = {"index.md", "log.md", "wiki.toml", "SCHEMA.md"}


def is_v3(wiki: Path) -> bool:
    """Probe only a bounded safe manifest; ordinary missing Wiki errors stay legacy."""
    if not (wiki / "wiki.toml").exists():
        return False
    try:
        version = tomllib.loads(read_authorized(wiki / "wiki.toml", (wiki,), 65536).decode()).get(
            "schema_version"
        )
    except (ValueError, OSError, UnicodeError):
        return False
    return version == 3 and (wiki / "maintenance/state.json").is_file()


@contextmanager
def projection(wiki: Path) -> Iterator[Path]:
    """Create a bounded private view without touching published bytes."""
    with tempfile.TemporaryDirectory(prefix="capicove-wiki-view-") as root:
        stage = Path(root).resolve() / "wiki"
        stage.mkdir()
        total = 0
        for path in sorted(wiki.rglob("*")):
            relative = path.relative_to(wiki)
            if relative.parts[0] in {"maintenance", "_migration"}:
                continue
            if path.is_symlink():
                raise ValueError("unsafe compatibility source")
            target = stage / relative
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            data = read_authorized(path, (wiki,), image_limit(relative.as_posix()))
            total += len(data)
            if total > 2 * 1024 * 1024 * 1024:
                raise ValueError("capacity_exceeded: compatibility view")
            if (
                relative.parts[0] == "representations"
                and json.loads(data).get("transform") == "wiki-markdown-v1"
            ):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        for name in ("_migration", "_derived", "representations", "pages/books"):
            (stage / name).mkdir(parents=True, exist_ok=True)
        (stage / "_migration/writer.lock").touch()
        if not (stage / "graph/citations.jsonl").exists():
            (stage / "graph/citations.jsonl").write_bytes(b"")
        manifest = stage / "wiki.toml"
        manifest.write_text(
            re.sub(
                r"(?m)^(\s*schema_version\s*=\s*)[13](?=\s*(?:#.*)?$)",
                r"\g<1>2",
                manifest.read_text(),
            )
        )
        yield stage


def snapshot(wiki: Path) -> dict[str, bytes]:
    """Read bounded candidate files that may participate in a system publication."""
    result = {}
    for path in sorted(wiki.rglob("*")):
        relative = path.relative_to(wiki).as_posix()
        if relative not in _FILES and relative.split("/", 1)[0] not in _ROOTS:
            continue
        if path.is_symlink():
            raise ValueError("unsafe candidate path")
        if path.is_file():
            result[relative] = read_authorized(path, (wiki,), image_limit(relative))
    return result


def evidence_records(
    wiki: Path, *, preferred: set[str] | None = None, current: Path | None = None
) -> dict[str, bytes]:
    """Prepare host-authored representation and Book snapshot commitments."""
    from capicove.documents import document_parse_result_from_json, document_parse_result_to_json
    from capicove.wiki.maintenance.representations import markdown_representation

    result = {}
    for path in sorted((wiki / "raw").glob("*.md")):
        value = markdown_representation(read_authorized(path, (wiki,), 16777216))
        if path.stem != value.source_sha256:
            raise ValueError("evidence_digest_mismatch")
        result["representations/" + value.representation_sha256 + ".json"] = value.encoded
    snapshots: dict[str, dict[str, bytes]] = {}
    for path in sorted((wiki / "representations").glob("*.json")):
        raw = read_authorized(path, (wiki,), 16777216)
        value_json = json.loads(raw)
        if value_json.get("transform") == "wiki-markdown-v1":
            continue
        parsed = document_parse_result_from_json(value_json)
        encoded = json.dumps(
            document_parse_result_to_json(parsed),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        commitment = canonical_object(
            {
                "schema_version": 1,
                "source_sha256": parsed.source_sha256,
                "representation_sha256": parsed.representation_sha256,
                "snapshot_sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
        snapshots.setdefault(parsed.source_sha256, {})[parsed.representation_sha256] = commitment
        result[
            "maintenance/book-snapshots/"
            + parsed.source_sha256
            + "-"
            + parsed.representation_sha256
            + ".json"
        ] = commitment
    for source, representations in snapshots.items():
        selected = set(representations) & (preferred or set())
        pointer = "maintenance/book-snapshots/" + source + ".json"
        if not selected and current is not None and (current / pointer).exists():
            previous = json.loads(read_authorized(current / pointer, (current,), 1048576))
            selected = {previous["representation_sha256"]} & representations.keys()
        if not selected and len(representations) == 1:
            selected = set(representations)
        if len(selected) != 1:
            raise ValueError("evidence_unavailable: ambiguous Book representation")
        result[pointer] = representations[next(iter(selected))]
    return result


def publish_system(
    service: MaintenanceService, origin: Any, changes: dict[str, bytes | None]
) -> PublicationRecord:
    """Journal a validated non-Agent operation without fabricating a review."""
    current = state(service)
    policy, _ = service._admit()
    entries = []
    changes = dict(changes)
    for relative, data in tuple(changes.items()):
        path = service.wiki / relative
        before = (
            read_authorized(path, (service.wiki,), image_limit(relative)) if path.exists() else None
        )
        if before == data:
            del changes[relative]
            continue
        entries.append(
            FileChange(
                relative_path=relative,
                before_sha256=hashlib.sha256(before).hexdigest() if before is not None else None,
                after_sha256=hashlib.sha256(data).hexdigest() if data is not None else None,
            )
        )
    if not entries and current["publication_id"] is not None:
        prior = decode_record(
            read_authorized(
                record_path(service.wiki, "publications", current["publication_id"], "publication"),
                (service.wiki,),
                1048576,
            )
        ).payload
        if isinstance(prior, PublicationRecord) and prior.origin == origin:
            return prior
    identity = (
        "publication:"
        + hashlib.sha256(
            canonical_object(
                {
                    "origin": origin,
                    "base": current,
                    "changes": [
                        [c.relative_path, c.before_sha256, c.after_sha256] for c in entries
                    ],
                }
            )
        ).hexdigest()
    )
    next_state = advance_state(service, current, identity, policy)
    state_bytes = canonical_object(next_state)
    receipt = PublicationRecord(
        publication_id=identity,
        parent_id=current["publication_id"],
        proposal_id=None,
        review_id=None,
        wiki_revision=next_state["wiki_revision"],
        policy_revision=policy,
        changes=tuple(entries),
        state_sha256=hashlib.sha256(state_bytes).hexdigest(),
        status="complete",
        origin=origin,
    )
    path = record_path(service.wiki, "publications", identity, "publication")
    changes[path.relative_to(service.wiki).as_posix()] = encode_record(
        MaintenanceRecord(schema_version=1, record_type="publication", payload=receipt)
    )
    transact_locked(
        service.wiki, identity[12:], changes, state_bytes=state_bytes, policy_revision=policy
    )
    return receipt


def execute(
    wiki: Path,
    operation: str,
    callback: Callable[[Path], _T],
    *,
    expected_revision: int | None = None,
    expected_policy: int | None = None,
) -> _T:
    """Fence every compatibility operation and publish only validated differences."""
    from capicove.wiki.maintenance.discovery import build_discovery
    from capicove.wiki.maintenance.publication import _candidate_lint
    from capicove.wiki.validation import lint

    service = MaintenanceService(wiki.parent)
    with maintenance_lock(wiki):
        policy, _ = service._admit()
        current = state(service)
        if expected_revision is not None and current["wiki_revision"] != expected_revision:
            raise ValueError("proposal_stale: Book publication base")
        if expected_policy is not None and policy != expected_policy:
            raise ValueError("policy_changed")
        _candidate_lint(service, {})
        with projection(wiki) as stage:
            before = snapshot(stage)
            result = callback(stage)
            if operation in {"scan", "lint", "query", "doctor"}:
                return result
            if not lint(stage).valid:
                raise ValueError("candidate_invalid: legacy operation")
            _refresh_search_projection(stage, before)
            after = snapshot(stage)
            changes: dict[str, bytes | None] = {
                path: after.get(path)
                for path in before.keys() | after.keys()
                if before.get(path) != after.get(path)
            }
            changes.pop("wiki.toml", None)
            owned = {
                page.path
                for page in service._inventory()
                if (wiki / "maintenance/managed" / (page.page_id[5:] + ".json")).exists()
            }
            if owned & changes.keys():
                raise ValueError("manual_edit_conflict: legacy writer cannot revise managed pages")
            preferred = {
                Path(path).stem
                for path in changes
                if path.startswith("representations/") and path.endswith(".json")
            }
            changes.update(evidence_records(stage, preferred=preferred, current=wiki))
            build_discovery(
                stage / "_derived/maintenance.sqlite", MaintenanceService(stage.parent)._inventory()
            )
            changes["_derived/maintenance.sqlite"] = (
                stage / "_derived/maintenance.sqlite"
            ).read_bytes()
            receipt = publish_system(service, operation, changes)
            if operation == "book_chunk" and isinstance(result, dict):
                result["legacy_projection_id"] = result["publication_id"]
                result["publication_id"] = receipt.publication_id
                result["wiki_schema_version"] = 3
            return result


def _refresh_search_projection(stage: Path, before: Mapping[str, bytes]) -> None:
    """Keep the FTS projection consistent when a v3 writer changed pages.

    Legacy v2 writers left the index to an explicit ``wiki build``; under v3 every
    published revision carries a search projection matching its pages, so queries
    after ingest or Book publication never see a stale or missing index.
    """
    from capicove.wiki._adapters.sqlite_index import rebuild_index
    from capicove.wiki.indexing import load_documents

    current = snapshot(stage)
    index = "_derived/search.sqlite3"
    pages_changed = any(
        path.startswith("pages/") and before.get(path) != current.get(path)
        for path in before.keys() | current.keys()
    )
    if pages_changed and before.get(index) == current.get(index):
        rebuild_index(stage / index, load_documents(stage))


def scan_v3(wiki: Path) -> WikiInventory:
    """Return current-schema inventory through the bounded compatibility view."""
    from capicove.wiki.filesystem import scan

    return replace(execute(wiki, "scan", scan), schema_version=3)
