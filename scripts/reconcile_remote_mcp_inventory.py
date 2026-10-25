"""Append newly discovered remote MCP candidates without rewriting reviewed rows.

Discovery rows come from ``check_remote_mcp_coverage.expected_rows``. New candidates
are appended exactly as discovered: pending classification, no contract mapping,
and listed as unresolved in the core (C01-C10, C25, C26) or extended (C11-C24)
partition. A changed source digest of an existing row is applied only when the
operator names that row as reviewed. Vanished candidates are reported, never
silently deleted. Manifest hashes and linkage digests are refreshed last.

This helper grants no planning, contract or qualification state.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = "docs/development/remote-mcp-operations.json"
CORE = "docs/development/remote-mcp-contracts-core.json"
EXTENDED = "docs/development/remote-mcp-contracts-extended.json"
CORE_FAMILIES = frozenset(
    {"C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10", "C25", "C26"}
)
EXTENDED_FAMILIES = frozenset(f"C{number:02d}" for number in range(11, 25))
Key = tuple[str, str]


def _key(row: dict[str, Any]) -> Key:
    return row["operation_id"], row["provider_profile"]


def reconcile(
    inventory: dict[str, Any],
    core: dict[str, Any],
    extended: dict[str, Any],
    expected: list[dict[str, Any]],
    *,
    reviewed: set[Key],
) -> dict[str, list[str]]:
    """Mutate the three documents in place; raise before any change on violations."""
    discovered = {_key(row): row for row in expected}
    current = {_key(row): row for row in inventory["operations"]}
    vanished = sorted(set(current) - set(discovered))
    if vanished:
        raise ValueError(f"vanished candidates require explicit retirement review: {vanished}")
    drifted: dict[Key, list[str]] = {}
    for key, row in current.items():
        fields = [
            name
            for name, value in discovered[key].items()
            if name != "planning" and row.get(name) != value
        ]
        if fields:
            drifted[key] = fields
    unreviewed = sorted(set(drifted) - reviewed)
    if unreviewed:
        raise ValueError(f"unreviewed drift: {unreviewed}")
    stale = sorted(reviewed - set(drifted))
    if stale:
        raise ValueError(f"reviewed rows have not drifted: {stale}")
    for key, fields in drifted.items():
        for name in fields:
            current[key][name] = discovered[key][name]
    added = []
    for key in sorted(set(discovered) - set(current)):
        row = dict(discovered[key])
        row.setdefault("planning", None)
        row["contract_mapping"] = None
        row["extended_contract_mapping"] = None
        inventory["operations"].append(row)
        family = row["capability_id"]
        if family in CORE_FAMILIES:
            core["unresolved_candidates"].append(
                {
                    "operation_id": key[0],
                    "provider_profile": key[1],
                    "capability_id": family,
                    "source_entrypoints": row["source_entrypoints"],
                }
            )
        elif family in EXTENDED_FAMILIES:
            extended["unresolved_candidates"].append(
                {"operation_id": key[0], "provider_profile": key[1], "capability_id": family}
            )
        added.append(key[0])
    return {"added": added, "reviewed": sorted(key[0] for key in drifted)}


def _load_checker() -> Any:
    spec = importlib.util.spec_from_file_location(
        "check_remote_mcp_coverage", ROOT / "scripts/check_remote_mcp_coverage.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    """Reconcile the repository inventory; print the appended and reviewed rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reviewed",
        action="append",
        default=[],
        metavar="OPERATION_ID|PROVIDER",
        help="Existing row whose source drift was reviewed (repeatable).",
    )
    parser.add_argument(
        "--reviewed-contract-source",
        action="append",
        default=[],
        metavar="PATH",
        help="Core/extended contract source pin whose change was reviewed (repeatable).",
    )
    args = parser.parse_args()
    checker = _load_checker()
    paths = {name: ROOT / name for name in (INVENTORY, CORE, EXTENDED)}
    inventory, core, extended = (json.loads(paths[name].read_text()) for name in paths)
    reviewed = set()
    for item in args.reviewed:
        operation, separator, provider = item.rpartition("|")
        if not separator:
            parser.error("--reviewed requires OPERATION_ID|PROVIDER")
        reviewed.add((operation, provider))
    report = reconcile(inventory, core, extended, checker.expected_rows(ROOT), reviewed=reviewed)
    for manifest in (core, extended):
        for relative in args.reviewed_contract_source:
            if relative in manifest["source_manifest"]:
                manifest["source_manifest"][relative] = _sha256(ROOT / relative)
    for name, document in ((CORE, core), (EXTENDED, extended)):
        paths[name].write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    inventory["source_manifest"] = checker.source_manifest(ROOT)
    inventory["core_contract_manifest"] = {"path": CORE, "sha256": _sha256(paths[CORE])}
    inventory["extended_contract_manifest"] = {
        "path": EXTENDED,
        "sha256": _sha256(paths[EXTENDED]),
    }
    paths[INVENTORY].write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n")
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
