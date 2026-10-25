"""Validate table registration independently from table row counts."""

from collections.abc import Iterable

from capicove.contracts.graph_build.snapshots import TABLE_ROLES


def required_table_roles() -> frozenset[str]:
    """Return roles required even when their corresponding tables are empty."""
    return TABLE_ROLES


def validate_table_roles(roles: Iterable[str]) -> tuple[str, ...]:
    """Reject unknown or repeated roles and return canonical role order."""
    if isinstance(roles, (str, bytes)):
        raise TypeError("Table roles must be a collection, not a string")
    seen: set[str] = set()
    for role in roles:
        if not isinstance(role, str) or role not in TABLE_ROLES:
            raise ValueError("Unknown graph table role")
        if role in seen:
            raise ValueError("Duplicate graph table role")
        seen.add(role)
    return tuple(sorted(seen))


def missing_table_roles(roles: frozenset[str]) -> frozenset[str]:
    """Return absent required roles after validating the supplied roles."""
    return TABLE_ROLES.difference(validate_table_roles(roles))
