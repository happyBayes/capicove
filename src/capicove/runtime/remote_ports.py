"""Public ports separating remote policy from identity provider adapters."""

from typing import Protocol

from capicove.contracts.remote import Principal


class TokenVerifier(Protocol):
    """Verify a credential against pinned identity-provider policy."""

    async def verify(self, credential: str, *, administrative: bool = False) -> Principal:
        """Return verified claims or raise; never retain or forward the token."""
        ...
