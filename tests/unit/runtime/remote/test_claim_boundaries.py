"""Verified issuer responses must still satisfy access-token claim boundaries."""

import time
from typing import Any

import pytest

from capicove.runtime._remote.claims import principal_from_claims
from capicove.runtime.remote_config import EnrolledApplication, IdentityEnrollment


@pytest.mark.parametrize(
    "replacement",
    [
        {"nbf": "tomorrow"},
        {"nbf": True},
        {"nbf": float("inf")},
        {"scope": "read\nwrite"},
        {"scope": "read\twrite"},
        {"scope": "read\\write"},
        {"scope": 'read"write'},
        {"scope": "读取"},
    ],
)
def test_malformed_verified_claims_fail_closed(replacement: dict[str, Any]) -> None:
    now = int(time.time())
    claims: dict[str, Any] = dict(
        iss="https://issuer.test",
        aud="https://mcp.test/mcp",
        client_id="client",
        sub="alice",
        iat=now,
        exp=now + 120,
        scope="read",
    )
    claims.update(replacement)
    enrollment = IdentityEnrollment(
        "https://issuer.test",
        "https://mcp.test/mcp",
        (EnrolledApplication("client", "tenant", "app"),),
    )
    with pytest.raises(ValueError):
        principal_from_claims(claims, enrollment, subject_kind="human")


def test_not_yet_valid_introspected_claims_fail_closed() -> None:
    now = int(time.time())
    claims = dict(
        iss="https://issuer.test",
        aud="https://mcp.test/mcp",
        client_id="client",
        sub="alice",
        iat=now,
        exp=now + 120,
        nbf=now + 90,
        scope="read",
    )
    enrollment = IdentityEnrollment(
        "https://issuer.test",
        "https://mcp.test/mcp",
        (EnrolledApplication("client", "tenant", "app"),),
    )
    with pytest.raises(ValueError):
        principal_from_claims(claims, enrollment, subject_kind="human")
