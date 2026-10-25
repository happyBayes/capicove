"""Resource policy must bind exact versions, lineage and complete authority."""

import importlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from capicove.contracts.remote import AuthorityKey, PrincipalKey
from capicove.runtime.remote_epochs import EpochStore


def authority(subject: str = "alice") -> AuthorityKey:
    return AuthorityKey(
        PrincipalKey("https://issuer.test", "tenant", "app", "human", subject),
        "workspace",
        "grant",
        "https://mcp.test/mcp",
    )


def values(tmp_path: Path) -> tuple[Any, Any, dict[str, Any], Any]:
    assert importlib.util.find_spec("capicove.runtime.remote_resources") is not None
    module = importlib.import_module("capicove.runtime.remote_resources")
    source = module.ResourceRef("source:one", 1, "a" * 64)
    derived = module.ResourceRef("derived:one", 1, "b" * 64)
    records = {
        source.resource_id: module.ResourceRecord(source, authority(), frozenset({"read"}), (), 0),
        derived.resource_id: module.ResourceRecord(
            derived, authority(), frozenset({"read", "export"}), (source,), 0
        ),
    }
    broker = module.ResourceBroker(
        lambda caller, ref: records.get(ref.resource_id), EpochStore(tmp_path / "epochs.sqlite")
    )
    return broker, derived, records, module


def test_derived_permissions_cannot_exceed_source_lineage(tmp_path: Path) -> None:
    broker, derived, _, _ = values(tmp_path)
    assert broker.authorize(authority(), derived, expected_epoch=0) == frozenset({"read"})


@pytest.mark.parametrize(
    "mutation",
    ["subject", "grant", "audience", "version", "hash", "cycle", "source_owner", "revoked"],
)
def test_invalid_resource_resolution_fails_closed(tmp_path: Path, mutation: str) -> None:
    broker, derived, records, _ = values(tmp_path)
    caller = authority()
    if mutation == "subject":
        caller = authority("bob")
    elif mutation == "grant":
        caller = replace(caller, grant_id="other")
    elif mutation == "audience":
        caller = replace(caller, audience="https://other.test/mcp")
    elif mutation == "version":
        derived = replace(derived, version=2)
    elif mutation == "hash":
        derived = replace(derived, sha256="c" * 64)
    elif mutation == "cycle":
        records["source:one"] = replace(records["source:one"], sources=(derived,))
    elif mutation == "source_owner":
        records["source:one"] = replace(records["source:one"], owner=authority("bob"))
    elif mutation == "revoked":
        broker.epochs.revoke("workspace")
    with pytest.raises(PermissionError):
        broker.authorize(caller, derived, expected_epoch=0)


def test_dispatch_checks_resource_policy_before_tool_effect(tmp_path: Path) -> None:
    from tests.unit.runtime.remote.test_remote_dispatch import dispatcher

    broker, reference, records, _ = values(tmp_path)
    dispatch, principal, calls = dispatcher(tmp_path)
    for key, record in records.items():
        records[key] = replace(record, owner=dispatch.authority)
    dispatch._resources = broker.policy_for(
        dispatch.authority,
        expected_epoch=0,
        selectors={"capicove.test.echo": lambda arguments: (reference,)},
    )
    assert dispatch.invoke(principal, "capicove_test_echo", {})["status"] == "succeeded"
    assert len(calls) == 1
    records["source:one"] = replace(records["source:one"], owner=authority("bob"))
    with pytest.raises(PermissionError):
        dispatch.invoke(principal, "capicove_test_echo", {})
    assert len(calls) == 1


def test_empty_resource_acl_cannot_authorize_default_permission_tool(tmp_path: Path) -> None:
    from tests.unit.runtime.remote.test_remote_dispatch import dispatcher

    broker, reference, records, _ = values(tmp_path)
    dispatch, principal, calls = dispatcher(tmp_path)
    for key, record in records.items():
        records[key] = replace(record, owner=dispatch.authority, permissions=frozenset())
    binding = dispatch._bindings["capicove_test_echo"]
    dispatch._bindings = {
        binding.name: replace(binding, spec=replace(binding.spec, required_permissions=frozenset()))
    }
    dispatch._resources = broker.policy_for(
        dispatch.authority,
        expected_epoch=0,
        selectors={binding.spec.tool_id: lambda arguments: (reference,)},
    )
    with pytest.raises(PermissionError):
        dispatch.invoke(principal, binding.name, {})
    assert calls == []
