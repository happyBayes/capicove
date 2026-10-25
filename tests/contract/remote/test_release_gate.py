"""R6.05 release gate: the full-capability qualification check cannot be satisfied falsely."""

from __future__ import annotations

from typing import Any

from tests.contract.remote.test_operation_coverage import ROOT, checker, inventory


def _errors(receipts: Any) -> list[str]:
    return list(checker().validate(inventory(), ROOT, "qualified", receipts))


def test_repository_inventory_is_not_release_qualified() -> None:
    errors = _errors(None)
    assert any(error.startswith("not qualified: ") for error in errors)
    assert "missing qualification receipts" in errors
    assert any("verifier is not configured" in error for error in errors)


def test_fabricated_or_skipped_receipts_do_not_qualify_a_release() -> None:
    fabricated = {
        "receipts": [
            {"operation_id": "tool:capicove.search@1", "state": "passed", "kind": "mock"},
            {"operation_id": "tool:capicove.get_evidence@1", "state": "skipped"},
        ]
    }
    errors = _errors(fabricated)
    assert "missing qualification receipts" not in errors
    assert any("verifier is not configured" in error for error in errors)
    assert any(error.startswith("not qualified: ") for error in errors)
