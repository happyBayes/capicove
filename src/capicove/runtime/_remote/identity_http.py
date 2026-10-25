"""Bounded HTTPS requests to explicitly pinned identity-provider endpoints."""

from __future__ import annotations

import asyncio
import importlib
import json
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Ambiguous identity document")
        result[key] = value
    return result


def _bounded_depth(payload: bytearray) -> None:
    depth = 0
    quoted = False
    escaped = False
    for character in payload.decode("utf-8"):
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character in "[{":
            depth += 1
            if depth > 32:
                raise ValueError("Identity document nesting exceeds its limit")
        elif character in "]}":
            depth -= 1


class IdentityHttp:
    """Pin the endpoint; do not inherit proxies, redirects or ambient cookies."""

    def __init__(
        self, endpoint: str, *, network_enabled: bool = False, transport: Any = None
    ) -> None:
        parsed = urlsplit(endpoint)
        if (
            network_enabled is not True
            or parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or (parsed.port is not None and not 1 <= parsed.port <= 65535)
        ):
            raise ValueError("Identity network access requires an explicitly pinned HTTPS endpoint")
        self._endpoint = endpoint
        self._transport = transport

    async def get(self) -> dict[str, object]:
        """Read one bounded discovery or key document."""
        return await self._request("GET", None)

    async def introspect(self, fields: Mapping[str, str]) -> dict[str, object]:
        """Send explicit credentials only to this enrolled introspection endpoint."""
        return await self._request("POST", fields)

    async def _request(self, method: str, fields: Mapping[str, str] | None) -> dict[str, object]:
        httpx: Any = importlib.import_module("httpx")
        try:
            async with asyncio.timeout(2):
                async with httpx.AsyncClient(
                    timeout=2,
                    follow_redirects=False,
                    trust_env=False,
                    transport=self._transport,
                    headers={"Accept-Encoding": "identity"},
                ) as client:
                    async with client.stream(method, self._endpoint, data=fields) as response:
                        if response.status_code != 200:
                            raise ValueError("Identity endpoint unavailable")
                        if (
                            response.headers.get("content-encoding", "identity").lower()
                            != "identity"
                        ):
                            raise ValueError("Compressed identity documents are not accepted")
                        length = response.headers.get("content-length")
                        if length is not None and not 0 <= int(length) <= 65536:
                            raise ValueError("Identity document exceeds its byte limit")
                        payload = bytearray()
                        async for chunk in response.aiter_bytes(chunk_size=4096):
                            if len(payload) + len(chunk) > 65536:
                                raise ValueError("Identity document exceeds its byte limit")
                            payload.extend(chunk)
                        _bounded_depth(payload)
                        value = json.loads(payload, object_pairs_hook=_unique)
                        if not isinstance(value, dict):
                            raise ValueError("Identity document must be an object")
                        return value
        except (TimeoutError, ValueError, RecursionError, httpx.HTTPError):
            raise ValueError("Identity endpoint request rejected") from None
