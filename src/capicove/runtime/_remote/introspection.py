"""Replaceable opaque-token verification with bounded nonadministrative caching."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Callable
from typing import Literal, Protocol

from capicove.contracts.remote import Principal
from capicove.runtime._remote.claims import principal_from_claims
from capicove.runtime.remote_config import IdentityEnrollment


class IntrospectionEndpoint(Protocol):
    """Send credentials only to the explicitly enrolled authorization server."""

    async def introspect(self, fields: dict[str, str]) -> dict[str, object]:
        """Return a bounded issuer response or fail closed."""
        ...


class IntrospectionVerifier:
    """Cache verified results for at most five seconds, never raw credentials."""

    def __init__(
        self,
        enrollment: IdentityEnrollment,
        endpoint: IntrospectionEndpoint,
        *,
        client_id: str,
        secret: Callable[[], str],
        subject_kind: Literal["human", "service"],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not client_id or not callable(secret) or subject_kind not in {"human", "service"}:
            raise ValueError("An explicit identity adapter profile is required")
        self._enrollment = enrollment
        self._endpoint = endpoint
        self._client_id = client_id
        self._secret = secret
        self._subject_kind = subject_kind
        self._clock = clock
        self._cache: dict[bytes, tuple[float, Principal]] = {}
        self._lock = asyncio.Lock()

    async def verify(self, credential: str, *, administrative: bool = False) -> Principal:
        """Bypass the cache for every administrative authorization decision."""
        if not isinstance(credential, str) or not 1 <= len(credential) <= 16384:
            raise ValueError("Invalid opaque credential")
        digest = hashlib.sha256(credential.encode()).digest()
        async with self._lock:
            cached = self._cache.get(digest)
            if (
                not administrative
                and cached is not None
                and cached[0] > self._clock()
                and cached[1].expires_at > time.time()
            ):
                return cached[1]
            self._cache.pop(digest, None)
            started = self._clock()
            try:
                secret = self._secret()
                if not isinstance(secret, str) or not secret:
                    raise ValueError("Identity credential unavailable")
                claims = await self._endpoint.introspect(
                    {
                        "token": credential,
                        "token_type_hint": "access_token",
                        "client_id": self._client_id,
                        "client_secret": secret,
                    }
                )
                if claims.get("active") is not True or claims.get("token_type") != "Bearer":
                    raise ValueError("Inactive or non-access token")
                principal = principal_from_claims(
                    claims,
                    self._enrollment,
                    subject_kind=self._subject_kind,
                    administrative=administrative,
                )
                if not administrative and principal.expires_at > time.time():
                    if len(self._cache) >= 1024:
                        self._cache.clear()
                    self._cache[digest] = (
                        min(started + 5, self._clock() + principal.expires_at - time.time()),
                        principal,
                    )
                return principal
            except (ValueError, TypeError, KeyError):
                raise ValueError("Access token rejected") from None
