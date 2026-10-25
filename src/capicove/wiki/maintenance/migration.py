"""Explicit v3 migration with original-byte preservation and journal recovery."""

import hashlib
import re
import uuid
from pathlib import Path

from capicove.contracts.wiki_maintenance import FileChange, MaintenanceRecord, PublicationRecord
from capicove.wiki.filesystem import scan
from capicove.wiki.maintenance.codec import encode_record
from capicove.wiki.maintenance.compatibility import evidence_records, is_v3, projection
from capicove.wiki.maintenance.discovery import build_discovery
from capicove.wiki.maintenance.service import MaintenanceService
from capicove.wiki.maintenance.store import (
    canonical_object,
    create_verified,
    ensure_directory,
    maintenance_lock,
    read_authorized,
)
from capicove.wiki.maintenance.transaction import transact_locked
from capicove.wiki.models import WikiMigrationPlan
from capicove.wiki.validation import lint


def migrate_v3(wiki: Path, *, preview: bool = True) -> WikiMigrationPlan:
    """Preview exact schema changes; applying is an explicit operator action."""
    wiki = wiki.resolve(strict=True)
    with maintenance_lock(wiki):
        if (wiki / "_migration/maintenance.pending").exists():
            raise ValueError("publication_pending")
        if is_v3(wiki):
            return WikiMigrationPlan(3, 3, (), (), False)
        inventory = scan(wiki)
        if not lint(wiki).valid:
            raise ValueError("candidate_invalid: migrate only a valid legacy Wiki")
        with projection(wiki) as stage:
            additions = evidence_records(stage)
            additions["graph/citations.jsonl"] = (stage / "graph/citations.jsonl").read_bytes()
            # Publish the disposable discovery projection in the same journal so a
            # migrated Wiki accepts tasks without an out-of-band private rebuild.
            ensure_directory(stage / "_derived", stage)
            build_discovery(
                stage / "_derived/maintenance.sqlite", MaintenanceService(stage.parent)._inventory()
            )
            additions["_derived/maintenance.sqlite"] = (
                stage / "_derived/maintenance.sqlite"
            ).read_bytes()
        plan = WikiMigrationPlan(
            inventory.schema_version,
            3,
            tuple(
                sorted(
                    [
                        "maintenance/state.json",
                        "maintenance/source-policy.json",
                        *[path for path in additions if not (wiki / path).exists()],
                    ]
                )
            ),
            ("wiki.toml",),
            not preview,
        )
        if preview:
            return plan
        for name in ("maintenance", "representations", "pages/books"):
            ensure_directory(wiki / name, wiki)
        initial = {
            "schema_version": 1,
            "workspace_id": "workspace:" + uuid.uuid5(uuid.NAMESPACE_URL, str(wiki)).hex,
            "wiki_revision": 0,
            "publication_id": None,
            "policy_revision": 0,
            "record_references": {"proposal": None, "review": None, "parent_publication": None},
        }
        create_verified(wiki / "maintenance/state.json", canonical_object(initial), wiki)
        create_verified(
            wiki / "maintenance/source-policy.json",
            canonical_object({"schema_version": 1, "policy_revision": 0, "withdrawn": []}),
            wiki,
        )
        changes: dict[str, bytes | None] = dict(additions)
        manifest = read_authorized(wiki / "wiki.toml", (wiki,), 65536).decode()
        changes["wiki.toml"] = re.sub(
            r"(?m)^(\s*schema_version\s*=\s*)[12](?=\s*(?:#.*)?$)", r"\g<1>3", manifest
        ).encode()
        entries = []
        for relative, after in changes.items():
            before = (
                read_authorized(wiki / relative, (wiki,), 16777216)
                if (wiki / relative).exists()
                else None
            )
            entries.append(
                FileChange(
                    relative_path=relative,
                    before_sha256=hashlib.sha256(before).hexdigest()
                    if before is not None
                    else None,
                    after_sha256=hashlib.sha256(after).hexdigest() if after is not None else None,
                )
            )
        identity = (
            "publication:"
            + hashlib.sha256(
                canonical_object(
                    {
                        "migration": initial,
                        "manifest": hashlib.sha256(changes["wiki.toml"] or b"").hexdigest(),
                    }
                )
            ).hexdigest()
        )
        next_state = canonical_object({**initial, "wiki_revision": 1, "publication_id": identity})
        receipt = PublicationRecord(
            publication_id=identity,
            parent_id=None,
            proposal_id=None,
            review_id=None,
            wiki_revision=1,
            policy_revision=0,
            changes=tuple(entries),
            state_sha256=hashlib.sha256(next_state).hexdigest(),
            status="complete",
            origin="migration",
        )
        changes["maintenance/publications/" + identity[12:] + ".json"] = encode_record(
            MaintenanceRecord(schema_version=1, record_type="publication", payload=receipt)
        )
        transact_locked(wiki, identity[12:], changes, state_bytes=next_state, policy_revision=0)
        return plan
