"""JWKS cache expiry must not preserve removed or unavailable keys."""

import importlib
import json
from typing import Any

import pytest

from capicove.runtime.remote_config import EnrolledApplication, IdentityEnrollment
from tests.unit.runtime.remote.test_jwt_verifier import signing_key, token


__all__ = ["signing_key"]


@pytest.mark.anyio
async def test_jwks_rotation_and_outage_fail_closed(signing_key: Any) -> None:
    assert importlib.util.find_spec("capicove.runtime._remote.jwks") is not None
    module = importlib.import_module("capicove.runtime._remote.jwks")
    jwt = importlib.import_module("jwt")
    public = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(signing_key.public_key()))
    public.update(kid="key-1", alg="RS256", use="sig")
    state: dict[str, Any] = {"keys": [public]}
    calls = []

    class Endpoint:
        async def get(self) -> dict[str, object]:
            calls.append(1)
            if state.get("outage"):
                raise ValueError("outage")
            return state

    now = [0.0]
    enrollment = IdentityEnrollment(
        "https://issuer.test",
        "https://mcp.test/mcp",
        (EnrolledApplication("client", "tenant", "app"),),
    )
    verifier = module.JwksVerifier(
        enrollment, Endpoint(), subject_kind="human", clock=lambda: now[0]
    )
    credential = token(signing_key)
    assert (await verifier.verify(credential)).key.subject == "alice"
    await verifier.verify(credential)
    assert len(calls) == 1
    state["keys"] = []
    now[0] = 5.0
    with pytest.raises(ValueError):
        await verifier.verify(credential)
    state["outage"] = True
    now[0] = 10.0
    with pytest.raises(ValueError):
        await verifier.verify(credential)
