"""Credential transport admission for the separately enabled remote host."""

import re
from collections.abc import Sequence


_BEARER = re.compile(rb"Bearer ([A-Za-z0-9._~+/-]+=*)", re.IGNORECASE)


def bearer_token(headers: Sequence[tuple[bytes, bytes]], *, query: bytes = b"") -> str:
    """Require exactly one bounded bearer header and no alternative credentials.

    Remote MCP endpoints accept no query parameters or cookies. The returned
    credential is unverified and must be passed to the configured verifier before
    any principal, session, catalog or Tool is constructed.
    """
    if query or any(name.lower() == b"cookie" for name, _ in headers):
        raise ValueError("Query and cookie credentials are not accepted")
    values = [value for name, value in headers if name.lower() == b"authorization"]
    if len(values) != 1 or len(values[0]) > 16391:
        raise ValueError("Exactly one bounded bearer credential is required")
    match = _BEARER.fullmatch(values[0])
    if match is None:
        raise ValueError("Invalid bearer credential")
    return match.group(1).decode("ascii")
