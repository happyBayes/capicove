"""A release candidate needs frozen contracts and cannot inherit planning approval."""

import importlib
from types import ModuleType

import pytest


def _api() -> ModuleType:
    module = importlib.import_module("tests.remote_support.receipts")
    assert hasattr(module, "validate_candidate"), "candidate validation is missing"
    return module


def _candidate() -> dict[str, object]:
    return {
        "source_revision": "a" * 40,
        "source_dirty": False,
        "wheel_sha256": "b" * 64,
        "lock_sha256": "c" * 64,
        "protocol_version": "2025-11-25",
        "sdk_version": "1.30.0",
        "contracts": {"RFC-0028": "accepted", "ADR-0032": "accepted"},
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_dirty", True),
        ("source_dirty", "false"),
        ("source_revision", "HEAD"),
        ("wheel_sha256", ""),
        ("lock_sha256", "unknown"),
        ("protocol_version", "latest"),
        ("sdk_version", "latest"),
        ("contracts", {"RFC-0028": "proposed", "ADR-0032": "accepted"}),
        ("contracts", {}),
    ],
)
def test_candidate_rejects_unfrozen_identity(field: str, value: object) -> None:
    api = _api()
    candidate = _candidate()
    candidate[field] = value
    with pytest.raises(ValueError):
        api.validate_candidate(candidate, required_contracts={"RFC-0028", "ADR-0032"})


def test_candidate_accepts_only_complete_explicit_contract_set() -> None:
    api = _api()
    api.validate_candidate(_candidate(), required_contracts={"RFC-0028", "ADR-0032"})
    with pytest.raises(ValueError):
        api.validate_candidate(
            _candidate(), required_contracts={"RFC-0028", "ADR-0032", "knowledge-mutation-v1"}
        )
