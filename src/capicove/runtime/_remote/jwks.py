"""Fail-closed JWKS refresh bounded to one attempt per five-second window."""

from __future__ import annotations

import asyncio
import importlib
import time
from collections.abc import Callable
from typing import Any, Literal, Protocol

from capicove.contracts.remote import Principal
from capicove.runtime._remote.jwt import JwtVerifier
from capicove.runtime.remote_config import IdentityEnrollment


class KeyEndpoint(Protocol):
    """Fetch an enrolled bounded key document."""

    async def get(self) -> dict[str, object]:
        """Return the key document or fail closed."""
        ...


class JwksVerifier:
    """Refresh issuer keys without retaining stale keys after an outage."""

    def __init__(
        self,
        enrollment: IdentityEnrollment,
        endpoint: KeyEndpoint,
        *,
        subject_kind: Literal["human", "service"],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._enrollment = enrollment
        self._endpoint = endpoint
        self._subject_kind = subject_kind
        self._clock = clock
        self._expires = float("-inf")
        self._verifier: JwtVerifier | None = None
        self._lock = asyncio.Lock()

    async def verify(self, credential: str, *, administrative: bool = False) -> Principal:
        """Revalidate every token; unknown keys never trigger an unbounded fetch."""
        async with self._lock:
            if self._clock() >= self._expires:
                self._verifier = None
                self._expires = self._clock() + 5
                document = await self._endpoint.get()
                keys = document.get("keys")
                if not isinstance(keys, list) or not 1 <= len(keys) <= 32:
                    raise ValueError("Invalid issuer key set")
                jwt: Any = importlib.import_module("jwt")
                trusted: dict[str, object] = {}
                for entry in keys:
                    if (
                        not isinstance(entry, dict)
                        or entry.get("kty") != "RSA"
                        or entry.get("alg", "RS256") != "RS256"
                        or entry.get("use", "sig") != "sig"
                        or entry.get("key_ops", ["verify"]) != ["verify"]
                        or "d" in entry
                        or not isinstance(entry.get("kid"), str)
                        or not entry["kid"]
                        or entry["kid"] in trusted
                    ):
                        raise ValueError("Unapproved issuer key")
                    try:
                        key = jwt.PyJWK.from_dict(entry, algorithm="RS256").key
                        if key.key_size < 2048:
                            raise ValueError("RSA issuer key is too small")
                        trusted[entry["kid"]] = key
                    except (ValueError, TypeError, KeyError, jwt.PyJWTError):
                        raise ValueError("Invalid issuer public key") from None
                self._verifier = JwtVerifier(
                    self._enrollment, trusted, subject_kind=self._subject_kind
                )
            verifier = self._verifier
        if verifier is None:
            raise ValueError("Issuer key refresh unavailable")
        return await verifier.verify(credential, administrative=administrative)
