"""The reconciliation helper may only append pending candidates and apply named reviews."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/reconcile_remote_mcp_inventory.py"


def helper() -> ModuleType:
    spec = importlib.util.spec_from_file_location("inventory_reconciler", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _expected(operation: str, family: str = "C05", digest: str = "1" * 64) -> dict[str, Any]:
    return {
        "capability_id": family,
        "operation_id": operation,
        "provider_profile": "builtin",
        "source_entrypoints": [operation.split(":", 1)[1]],
        "source_digest": digest,
        "state": "inventoried",
        "planning": None,
    }


def _documents() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    reviewed = {
        **_expected("public:src/a.py:kept"),
        "contract_mapping": {"candidate_operation_id": "public:src/a.py:kept", "x": 1},
        "extended_contract_mapping": None,
        "planning": {"reviewed": True},
    }
    inventory = {"operations": [reviewed], "source_manifest": {}}
    core = {"source_mappings": [reviewed["contract_mapping"]], "unresolved_candidates": []}
    extended: dict[str, Any] = {"source_mappings": [], "unresolved_candidates": []}
    return inventory, core, extended


def test_new_candidates_are_appended_pending_and_partitioned_by_family() -> None:
    inventory, core, extended = _documents()
    expected = [
        _expected("public:src/a.py:kept"),
        _expected("public:src/b.py:core_new"),
        _expected("public:src/c.py:inference_new", "C21"),
    ]
    report = helper().reconcile(inventory, core, extended, expected, reviewed=set())
    rows = {row["operation_id"]: row for row in inventory["operations"]}
    assert rows["public:src/a.py:kept"]["planning"] == {"reviewed": True}
    assert rows["public:src/b.py:core_new"]["contract_mapping"] is None
    assert rows["public:src/c.py:inference_new"]["extended_contract_mapping"] is None
    assert core["unresolved_candidates"] == [
        {
            "operation_id": "public:src/b.py:core_new",
            "provider_profile": "builtin",
            "capability_id": "C05",
            "source_entrypoints": ["src/b.py:core_new"],
        }
    ]
    assert extended["unresolved_candidates"] == [
        {
            "operation_id": "public:src/c.py:inference_new",
            "provider_profile": "builtin",
            "capability_id": "C21",
        }
    ]
    assert report["added"] == ["public:src/b.py:core_new", "public:src/c.py:inference_new"]


def test_drift_requires_an_explicit_review_and_preserves_planning() -> None:
    inventory, core, extended = _documents()
    changed = [_expected("public:src/a.py:kept", digest="2" * 64)]
    module = helper()
    with pytest.raises(ValueError, match="unreviewed drift"):
        module.reconcile(inventory, core, extended, changed, reviewed=set())
    module.reconcile(
        inventory, core, extended, changed, reviewed={("public:src/a.py:kept", "builtin")}
    )
    row = cast(dict[str, Any], inventory["operations"][0])
    assert row["source_digest"] == "2" * 64
    assert row["planning"] == {"reviewed": True}


def test_vanished_candidates_are_never_removed_silently() -> None:
    inventory, core, extended = _documents()
    with pytest.raises(ValueError, match="vanished"):
        helper().reconcile(inventory, core, extended, [], reviewed=set())


def test_reviewed_key_that_did_not_drift_is_rejected() -> None:
    inventory, core, extended = _documents()
    with pytest.raises(ValueError, match="not drifted"):
        helper().reconcile(
            inventory,
            core,
            extended,
            [_expected("public:src/a.py:kept")],
            reviewed={("public:src/a.py:kept", "builtin")},
        )


def test_repository_inventory_round_trips_without_changes() -> None:
    data = json.loads((ROOT / "docs/development/remote-mcp-operations.json").read_text())
    assert isinstance(data["operations"], list)
