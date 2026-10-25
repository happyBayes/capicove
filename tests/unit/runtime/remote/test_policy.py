"""No policy tier may silently supply missing authority."""

import importlib

import pytest

from capicove.contracts.remote import AuthorityKey, AuthorizationDecision, PrincipalKey
from capicove.runtime.remote_policy import authorize as authorize_policy


def authorize(**overrides: frozenset[str]) -> AuthorizationDecision:
    assert importlib.util.find_spec("capicove.runtime.remote_policy") is not None
    tiers = {
        name: frozenset({"read", "write"})
        for name in ("deployment", "application", "token", "user", "workspace", "resource")
    }
    tiers.update(overrides)
    key = AuthorityKey(
        PrincipalKey("https://issuer.test", "tenant", "app", "human", "alice"), "ws", "grant", "mcp"
    )
    return authorize_policy(
        key, required=frozenset({"read"}), policy_epoch=1, source_epoch=1, **tiers
    )


@pytest.mark.parametrize(
    "tier", ["deployment", "application", "token", "user", "workspace", "resource"]
)
def test_every_tier_can_deny(tier: str) -> None:
    with pytest.raises(PermissionError):
        authorize(**{tier: frozenset({"write"})})


def test_ceiling_is_intersection() -> None:
    decision = authorize(resource=frozenset({"read"}))
    assert decision.permissions == frozenset({"read"})
