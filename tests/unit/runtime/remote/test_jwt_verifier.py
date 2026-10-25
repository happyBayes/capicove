"""Signed token verification rejects identity and token-class confusion."""

import importlib
import time
from typing import Any

import pytest

from capicove.runtime.remote_config import EnrolledApplication, IdentityEnrollment


@pytest.fixture
def signing_key() -> Any:
    rsa = importlib.import_module("cryptography.hazmat.primitives.asymmetric.rsa")
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def verifier(key: Any) -> Any:
    assert importlib.util.find_spec("capicove.runtime._remote.jwt") is not None
    module = importlib.import_module("capicove.runtime._remote.jwt")
    enrollment = IdentityEnrollment(
        "https://issuer.test",
        "https://mcp.test/mcp",
        (EnrolledApplication("client", "tenant", "app"),),
    )
    return module.JwtVerifier(enrollment, {"key-1": key.public_key()}, subject_kind="human")


def token(key: Any, **changes: Any) -> str:
    jwt = importlib.import_module("jwt")
    now = int(time.time())
    claims = dict(
        iss="https://issuer.test",
        aud="https://mcp.test/mcp",
        sub="alice",
        client_id="client",
        scope="read",
        iat=now,
        exp=now + 120,
    )
    if changes.get("iat") == "future":
        changes["iat"] = now + 90
    claims.update(changes)
    return str(
        jwt.encode(claims, key, algorithm="RS256", headers={"typ": "at+jwt", "kid": "key-1"})
    )


@pytest.mark.anyio
async def test_signed_access_token_maps_server_owned_identity(signing_key: Any) -> None:
    principal = await verifier(signing_key).verify(token(signing_key, tenant_id="spoof"))
    assert principal.key.tenant_id == "tenant"
    assert principal.key.subject == "alice"
    assert principal.token_permissions == frozenset({"read"})


@pytest.mark.anyio
@pytest.mark.parametrize(
    "changes",
    [
        {"aud": "other"},
        {"iss": "https://other.test"},
        {"client_id": "other"},
        {"exp": 1},
        {"exp": int(time.time()) + 3600},
        {"iat": "future"},
        {"scope": ["read"]},
        {"exp": True},
    ],
)
async def test_invalid_claims_rejected(signing_key: Any, changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        await verifier(signing_key).verify(token(signing_key, **changes))


@pytest.mark.anyio
async def test_forged_signature_and_id_token_rejected(signing_key: Any) -> None:
    jwt = importlib.import_module("jwt")
    credential = token(signing_key)
    claims = jwt.decode(credential, options={"verify_signature": False})
    wrong = importlib.import_module(
        "cryptography.hazmat.primitives.asymmetric.rsa"
    ).generate_private_key(public_exponent=65537, key_size=2048)
    with pytest.raises(ValueError):
        await verifier(signing_key).verify(token(wrong))
    id_token = jwt.encode(
        claims, signing_key, algorithm="RS256", headers={"kid": "key-1", "typ": "JWT"}
    )
    with pytest.raises(ValueError):
        await verifier(signing_key).verify(id_token)


@pytest.mark.anyio
async def test_offline_jwt_cannot_authorize_administration(signing_key: Any) -> None:
    with pytest.raises(ValueError):
        await verifier(signing_key).verify(
            token(signing_key, auth_time=int(time.time()), amr=["mfa"]), administrative=True
        )


@pytest.mark.anyio
async def test_untrusted_deep_jwt_header_is_bounded(signing_key: Any) -> None:
    import base64

    header = (
        base64.urlsafe_b64encode(b'{"extra":' + b"[" * 1500 + b"0" + b"]" * 1500 + b"}")
        .rstrip(b"=")
        .decode()
    )
    with pytest.raises(ValueError):
        await verifier(signing_key).verify(header + ".e30.invalid")
