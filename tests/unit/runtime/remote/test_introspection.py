"""Opaque access tokens require a live bounded issuer decision."""

import importlib
import time
from typing import Any

import pytest

from capicove.runtime.remote_config import EnrolledApplication, IdentityEnrollment


@pytest.mark.anyio
async def test_introspection_cache_expiry_and_admin_bypass() -> None:
    assert importlib.util.find_spec("capicove.runtime._remote.introspection") is not None
    module = importlib.import_module("capicove.runtime._remote.introspection")
    now = [0.0]
    issued = int(time.time())
    claims: dict[str, Any] = dict(
        active=True,
        token_type="Bearer",
        iss="https://issuer.test",
        aud="https://mcp.test/mcp",
        client_id="client",
        sub="alice",
        iat=issued,
        exp=issued + 120,
        scope="read",
        auth_time=issued,
        amr=["mfa"],
    )
    calls: list[dict[str, str]] = []

    class Endpoint:
        async def introspect(self, fields: dict[str, str]) -> dict[str, object]:
            calls.append(fields)
            return dict(claims)

    enrollment = IdentityEnrollment(
        "https://issuer.test",
        "https://mcp.test/mcp",
        (EnrolledApplication("client", "tenant", "app"),),
    )
    verifier = module.IntrospectionVerifier(
        enrollment,
        Endpoint(),
        client_id="gateway",
        secret=lambda: "fixture-secret",
        subject_kind="human",
        clock=lambda: now[0],
    )
    assert (await verifier.verify("opaque")).key.subject == "alice"
    await verifier.verify("opaque")
    assert len(calls) == 1
    await verifier.verify("opaque", administrative=True)
    assert len(calls) == 2
    claims["active"] = False
    now[0] = 5
    with pytest.raises(ValueError):
        await verifier.verify("opaque")
    assert len(calls) == 3
