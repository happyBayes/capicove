"""Authority-owned immutable resource metadata and source-lineage authorization.

Domain adapters supply current metadata through an authority-aware lookup port.
This broker returns permissions only: it never exposes a host path or opens media.
Shared domain sources require explicit authority-owned binding records; common
hashes or a common application never imply ownership.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from capicove.contracts import ToolSpec
from capicove.contracts.remote import AuthorityKey
from capicove.runtime.remote_epochs import EpochStore


@dataclass(frozen=True, slots=True)
class ResourceRef:
    """Pin the complete identity of an immutable domain resource."""

    resource_id: str
    version: int
    sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.resource_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}", self.resource_id) is None
            or type(self.version) is not int
            or not 1 <= self.version <= 2147483647
            or not isinstance(self.sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None
        ):
            raise ValueError("A bounded immutable resource reference is required")


@dataclass(frozen=True, slots=True)
class ResourceRecord:
    """Snapshot one server-owned binding and all immediate source dependencies."""

    reference: ResourceRef
    owner: AuthorityKey
    permissions: frozenset[str]
    sources: tuple[ResourceRef, ...]
    source_epoch: int

    def __post_init__(self) -> None:
        if not isinstance(self.reference, ResourceRef) or not isinstance(self.owner, AuthorityKey):
            raise ValueError("Typed resource identity and ownership required")
        raw_permissions: object = self.permissions
        if isinstance(raw_permissions, (str, bytes)):
            raise ValueError("Explicit resource permissions required")
        permissions = frozenset(self.permissions)
        if len(permissions) > 256 or any(
            not isinstance(item, str)
            or not 1 <= len(item) <= 256
            or any(ord(char) < 33 or ord(char) > 126 or char in '"\\' for char in item)
            for item in permissions
        ):
            raise ValueError("Resource permissions must be bounded identifiers")
        sources = tuple(self.sources)
        if (
            len(sources) > 256
            or any(not isinstance(item, ResourceRef) for item in sources)
            or len(set(sources)) != len(sources)
        ):
            raise ValueError("Resource dependencies must be bounded unique typed references")
        if type(self.source_epoch) is not int or self.source_epoch < 0:
            raise ValueError("Resource epoch must be a nonnegative integer")
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "permissions", permissions)


ResourceLookup = Callable[[AuthorityKey, ResourceRef], ResourceRecord | None]


class ResourceBroker:
    """Intersect all source ACLs under the workspace's acknowledged epoch fence."""

    def __init__(self, lookup: ResourceLookup, epochs: EpochStore) -> None:
        if not callable(lookup) or not isinstance(epochs, EpochStore):
            raise ValueError("Explicit authority-aware metadata lookup and epoch store required")
        self._lookup = lookup
        self.epochs = epochs

    def authorize(
        self, authority: AuthorityKey, reference: ResourceRef, *, expected_epoch: int
    ) -> frozenset[str]:
        """Deny stale, foreign, missing or cyclic lineage without revealing existence.

        Lookup adapters must be bounded metadata reads and must not acquire a
        nested epoch permit. Domain readers and publishers still require their
        own execution/release permits; this decision does not authorize a later
        read after revocation.
        """
        if not isinstance(authority, AuthorityKey) or not isinstance(reference, ResourceRef):
            raise PermissionError("Resource is unavailable")
        visiting: set[ResourceRef] = set()
        resolved: dict[ResourceRef, frozenset[str]] = {}

        def visit(current: ResourceRef, depth: int) -> frozenset[str]:
            if current in visiting or depth > 32:
                raise PermissionError("Resource is unavailable")
            if current in resolved:
                return resolved[current]
            if len(visiting) + len(resolved) >= 256:
                raise PermissionError("Resource is unavailable")
            visiting.add(current)
            record = self._lookup(authority, current)
            if (
                not isinstance(record, ResourceRecord)
                or record.reference != current
                or record.owner != authority
                or record.source_epoch != expected_epoch
            ):
                raise PermissionError("Resource is unavailable")
            permissions = record.permissions
            for source in record.sources:
                permissions = permissions.intersection(visit(source, depth + 1))
            visiting.remove(current)
            resolved[current] = permissions
            return permissions

        try:
            with self.epochs.permit(authority.workspace_binding_id, expected_epoch, phase="start"):
                permissions = visit(reference, 0)
                if not permissions:
                    raise PermissionError("Resource is unavailable")
                return permissions
        except Exception:
            raise PermissionError("Resource is unavailable") from None

    def policy_for(
        self,
        authority: AuthorityKey,
        *,
        expected_epoch: int,
        selectors: Mapping[str, Callable[[Mapping[str, object]], tuple[ResourceRef, ...]]],
    ) -> Callable[[AuthorityKey, ToolSpec, Mapping[str, object]], frozenset[str]]:
        """Compose explicitly selected resource-bearing Tools into dispatch policy.

        Each selector is server-owned and maps domain arguments to exact
        immutable resource references. Unregistered Tools and empty selections
        are denied; resource-free diagnostic Tools need a separate explicit policy.
        """
        if (
            not isinstance(authority, AuthorityKey)
            or type(expected_epoch) is not int
            or expected_epoch < 0
        ):
            raise ValueError("Complete authority and a nonnegative epoch required")
        registry = MappingProxyType(dict(selectors))
        if not registry or any(
            not isinstance(key, str) or not key or not callable(value)
            for key, value in registry.items()
        ):
            raise ValueError("Explicit bounded Tool resource selectors required")
        if len(registry) > 1024:
            raise ValueError("Too many resource selectors")

        def policy(
            caller: AuthorityKey, spec: ToolSpec, arguments: Mapping[str, object]
        ) -> frozenset[str]:
            if caller != authority or spec.tool_id not in registry or not spec.required_permissions:
                raise PermissionError("Resource is unavailable")
            try:
                references = registry[spec.tool_id](arguments)
                if not isinstance(references, tuple) or not 1 <= len(references) <= 256:
                    raise PermissionError("Resource is unavailable")
                permissions = self.authorize(
                    authority, references[0], expected_epoch=expected_epoch
                )
                for reference in references[1:]:
                    permissions = permissions.intersection(
                        self.authorize(authority, reference, expected_epoch=expected_epoch)
                    )
                return permissions
            except Exception:
                raise PermissionError("Resource is unavailable") from None

        return policy
