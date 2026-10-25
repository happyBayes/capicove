"""JWT access-token verifier for an explicitly provisioned RSA trust profile.

Keys are provisioned by the host, never selected from token-controlled URLs.
Remote JWKS refresh and opaque introspection are separate adapters.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal

from capicove.contracts.remote import Principal
from capicove.runtime._remote.claims import principal_from_claims
from capicove.runtime.remote_config import IdentityEnrollment


class JwtVerifier:
    """Verify a bounded RS256 access token against pinned issuer/client policy."""

    def __init__(
        self,
        enrollment: IdentityEnrollment,
        keys: Mapping[str, object],
        *,
        subject_kind: Literal["human", "service"],
    ) -> None:
        if not isinstance(enrollment, IdentityEnrollment) or subject_kind not in {
            "human",
            "service",
        }:
            raise ValueError("An enrolled identity profile is required")
        if not keys or len(keys) > 32 or any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("A bounded pinned key set is required")
        self._enrollment = enrollment
        self._keys = MappingProxyType(dict(keys))
        self._subject_kind = subject_kind

    async def verify(self, credential: str, *, administrative: bool = False) -> Principal:
        """Verify signatures before mapping identity; never expose token material."""
        if administrative:
            raise ValueError("Administrative access requires fresh online introspection")
        jwt: Any = importlib.import_module("jwt")
        try:
            if not isinstance(credential, str) or len(credential) > 16384:
                raise ValueError("Invalid credential")
            header = jwt.get_unverified_header(credential)
            if (
                header.get("typ") != "at+jwt"
                or header.get("alg") != "RS256"
                or any(field in header for field in ("crit", "jku", "x5u", "jwk"))
                or header.get("kid") not in self._keys
            ):
                raise ValueError("Unapproved token profile")
            claims = jwt.decode(
                credential,
                self._keys[header["kid"]],
                algorithms=["RS256"],
                audience=self._enrollment.audience,
                issuer=self._enrollment.issuer,
                leeway=30,
                options={
                    "require": ["iss", "aud", "sub", "client_id", "iat", "exp"],
                    "strict_aud": True,
                },
            )
            return principal_from_claims(
                claims,
                self._enrollment,
                subject_kind=self._subject_kind,
                administrative=administrative,
            )
        except (ValueError, TypeError, KeyError, jwt.PyJWTError):
            raise ValueError("Access token rejected") from None
