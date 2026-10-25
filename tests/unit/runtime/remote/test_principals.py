"""Remote identity boundaries must preserve every authority dimension."""

import importlib
from dataclasses import FrozenInstanceError
from types import ModuleType

import pytest


def contracts() -> ModuleType:
    """Fail behaviorally when the remote contract has not been implemented."""
    assert importlib.util.find_spec("capicove.contracts.remote") is not None
    return importlib.import_module("capicove.contracts.remote")


def test_subject_and_grant_partition_authority() -> None:
    module = contracts()
    alice = module.PrincipalKey("https://issuer.test", "tenant", "app", "human", "alice")
    bob = module.PrincipalKey("https://issuer.test", "tenant", "app", "human", "bob")
    keys = {
        module.AuthorityKey(subject, "workspace", grant, "mcp")
        for subject in (alice, bob)
        for grant in ("grant-a", "grant-b")
    }
    assert len(keys) == 4
    with pytest.raises(FrozenInstanceError):
        alice.subject = "bob"


@pytest.mark.parametrize("kind", ["", "user", "admin", None])
def test_subject_kind_is_explicit(kind: object) -> None:
    with pytest.raises(ValueError):
        contracts().PrincipalKey("https://issuer.test", "tenant", "app", kind, "alice")


@pytest.mark.parametrize("value", ["", " ", "x\n", 3, None])
def test_identity_components_reject_ambiguous_values(value: object) -> None:
    with pytest.raises(ValueError):
        contracts().PrincipalKey("https://issuer.test", value, "app", "human", "alice")


def test_authorization_ceiling_cannot_be_mutated() -> None:
    module = contracts()
    key = module.PrincipalKey("https://issuer.test", "tenant", "app", "service", "service")
    authority = module.AuthorityKey(key, "workspace", "grant", "mcp")
    scopes = {"read"}
    decision = module.AuthorizationDecision(authority, scopes, 1, 1)
    scopes.add("write")
    assert decision.permissions == frozenset({"read"})


@pytest.mark.parametrize("epoch", [-1, True, 1.5])
def test_invalid_epoch_is_rejected(epoch: object) -> None:
    module = contracts()
    key = module.PrincipalKey("https://issuer.test", "tenant", "app", "service", "service")
    with pytest.raises(ValueError):
        module.AuthorizationDecision(
            module.AuthorityKey(key, "workspace", "grant", "mcp"), {"read"}, epoch, 1
        )
