"""Shared validation of verified issuer claims, without transport dependencies."""

import time
from collections.abc import Mapping
from typing import Any, Literal

from capicove.contracts.remote import Principal
from capicove.runtime.remote_config import IdentityEnrollment


def principal_from_claims(
    claims: Mapping[str, Any],
    enrollment: IdentityEnrollment,
    *,
    subject_kind: Literal["human", "service"],
    administrative: bool = False,
) -> Principal:
    """Map already-verified issuer claims to a bounded principal."""
    if claims.get("iss") != enrollment.issuer or claims.get("aud") != enrollment.audience:
        raise ValueError("Issuer or audience mismatch")
    now = time.time()
    issued, expires = claims["iat"], claims["exp"]
    if (
        type(issued) is not int
        or type(expires) is not int
        or not 0 < expires - issued <= 300
        or issued > now + 30
        or issued < now - 330
        or expires <= now - 30
    ):
        raise ValueError("Invalid token lifetime")
    not_before = claims.get("nbf", issued)
    if type(not_before) is not int or not_before > now + 30 or not_before >= expires:
        raise ValueError("Invalid token activation time")
    scope = claims.get("scope", "")
    if (
        not isinstance(scope, str)
        or len(scope) > 8192
        or any(ord(char) < 32 or ord(char) > 126 or char in '"\\' for char in scope)
    ):
        raise ValueError("Invalid token scopes")
    principal_key = enrollment.map_subject(claims["client_id"], subject_kind, claims["sub"])
    if subject_kind == "service" and claims["sub"] != claims["client_id"]:
        raise ValueError("Service token subject must identify the enrolled client")
    if administrative:
        auth_time = claims.get("auth_time")
        amr = claims.get("amr")
        if (
            subject_kind != "human"
            or type(auth_time) is not int
            or not 0 <= now - auth_time <= 300
            or not isinstance(amr, list)
            or "mfa" not in amr
        ):
            raise ValueError("Fresh human MFA required")
    return Principal(principal_key, frozenset(scope.split()), float(expires), enrollment.audience)
