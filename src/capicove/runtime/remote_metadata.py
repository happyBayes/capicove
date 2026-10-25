"""Bounded public OAuth resource discovery without private enrollment details."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

from capicove.runtime.remote_config import _https


@dataclass(frozen=True, slots=True)
class ProtectedResourceMetadata:
    """Advertise one pinned resource and issuer, without initiating discovery."""

    resource: str
    authorization_server: str

    def __post_init__(self) -> None:
        for endpoint in (self.resource, self.authorization_server):
            _https(endpoint)
            if any(ord(char) < 33 or ord(char) > 126 or char in '\\"' for char in endpoint):
                raise ValueError("Public metadata requires ASCII HTTPS endpoints")
        if urlsplit(self.resource).path != "/mcp":
            raise ValueError("Resource metadata requires the exact configured MCP endpoint")

    @property
    def host(self) -> str:
        """Return the explicit authority required in the HTTP Host header."""
        return urlsplit(self.resource).netloc.lower()

    @property
    def metadata_url(self) -> str:
        """Return the path-specific RFC 9728 discovery location."""
        endpoint = urlsplit(self.resource)
        return f"https://{endpoint.netloc}/.well-known/oauth-protected-resource/mcp"

    def payload(self) -> dict[str, object]:
        """Return a fresh public document containing no registrations or roots."""
        return {
            "resource": self.resource,
            "authorization_servers": [self.authorization_server],
            "bearer_methods_supported": ["header"],
        }

    def challenge(self) -> bytes:
        """Build the bounded discovery challenge from validated configuration."""
        return f'Bearer resource_metadata="{self.metadata_url}"'.encode("ascii")
