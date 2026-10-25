"""Strict enrollment decoding, independent of local-host configuration formats."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlsplit

from capicove.contracts.remote import PrincipalKey


if TYPE_CHECKING:
    from capicove.runtime.remote_dispatch import WorkspaceGrant


@dataclass(frozen=True, slots=True)
class EnrolledApplication:
    """Map a trusted issuer client identifier to server-owned tenancy."""

    client_id: str
    tenant_id: str
    application_id: str

    def __post_init__(self) -> None:
        for value in (self.client_id, self.tenant_id, self.application_id):
            _text(value)


@dataclass(frozen=True, slots=True)
class IdentityEnrollment:
    """Pin one issuer, audience and immutable application enrollment table."""

    issuer: str
    audience: str
    applications: tuple[EnrolledApplication, ...]

    def __post_init__(self) -> None:
        _https(self.issuer)
        _https(self.audience)
        applications = tuple(self.applications)
        if not 1 <= len(applications) <= 1024 or any(
            not isinstance(item, EnrolledApplication) for item in applications
        ):
            raise ValueError("Enrollment requires bounded typed applications")
        if len({item.client_id for item in applications}) != len(applications):
            raise ValueError("Duplicate enrolled client")
        object.__setattr__(self, "applications", applications)

    def map_subject(
        self, client_id: str, subject_kind: Literal["human", "service"], subject: str
    ) -> PrincipalKey:
        """Map verified claims only; caller tenant claims are never consulted."""
        for application in self.applications:
            if application.client_id == client_id:
                return PrincipalKey(
                    self.issuer,
                    application.tenant_id,
                    application.application_id,
                    subject_kind,
                    subject,
                )
        raise ValueError("Client is not enrolled")


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate configuration key")
        result[key] = value
    return result


def _text(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value.encode("utf-8")) > 2048
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("Invalid configuration identifier")
    return value


def _https(value: object) -> str:
    result = _text(value)
    parsed = urlsplit(result)
    port = parsed.port
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("Invalid identity endpoint port")
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or parsed.query
    ):
        raise ValueError("Identity endpoints require pinned HTTPS URLs without credentials")
    return result


def decode_enrollment(document: str) -> IdentityEnrollment:
    """Decode a bounded enrollment document, rejecting unknown or duplicate keys."""
    if len(document.encode("utf-8")) > 262144:
        raise ValueError("Enrollment document exceeds its byte limit")
    value = json.loads(document, object_pairs_hook=_unique)
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "issuer",
        "audience",
        "applications",
    }:
        raise ValueError("Unexpected enrollment fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("Unsupported enrollment version")
    raw_applications = value["applications"]
    if not isinstance(raw_applications, list) or not 1 <= len(raw_applications) <= 1024:
        raise ValueError("Enrollment requires a bounded, nonempty application list")
    applications: list[EnrolledApplication] = []
    for application in raw_applications:
        if not isinstance(application, dict) or set(application) != {
            "client_id",
            "tenant_id",
            "application_id",
        }:
            raise ValueError("Unexpected application fields")
        applications.append(
            EnrolledApplication(
                _text(application["client_id"]),
                _text(application["tenant_id"]),
                _text(application["application_id"]),
            )
        )
    if len({item.client_id for item in applications}) != len(applications):
        raise ValueError("Duplicate enrolled client")
    return IdentityEnrollment(
        _https(value["issuer"]), _https(value["audience"]), tuple(applications)
    )


def _permission_set(value: object) -> frozenset[str]:
    """Validate bounded OAuth permission identifiers without whitespace aliases."""
    if not isinstance(value, (list, tuple, frozenset)) or len(value) > 256:
        raise ValueError("Permissions must be a bounded explicit collection")
    permissions: list[str] = []
    for item in value:
        permission = _text(item)
        if any(ord(char) < 33 or ord(char) > 126 or char in '"\\' for char in permission):
            raise ValueError("Invalid permission identifier")
        permissions.append(permission)
    if len(set(permissions)) != len(permissions):
        raise ValueError("Duplicate permission")
    return frozenset(permissions)


@dataclass(frozen=True, slots=True)
class RemotePolicyConfiguration:
    """Bind one endpoint to explicit server-owned principal and workspace grants.

    One principal selects one authority at this endpoint. Multiple workspaces
    require separately composed endpoint configurations; no caller root selector
    or implicit first-match grant is supported.
    """

    environment_id: str
    public_url: str
    identity: IdentityEnrollment
    deployment_permissions: frozenset[str]
    grants: tuple[WorkspaceGrant, ...]

    def __post_init__(self) -> None:
        from capicove.runtime.remote_dispatch import WorkspaceGrant

        _text(self.environment_id)
        _https(self.public_url)
        if (
            not isinstance(self.identity, IdentityEnrollment)
            or self.public_url != self.identity.audience
            or urlsplit(self.public_url).path != "/mcp"
        ):
            raise ValueError("Endpoint must match its enrolled resource audience")
        grants = tuple(self.grants)
        if not 1 <= len(grants) <= 1024 or any(
            not isinstance(grant, WorkspaceGrant) for grant in grants
        ):
            raise ValueError("Explicit bounded workspace grants required")
        principals: set[PrincipalKey] = set()
        for grant in grants:
            key = grant.authority.principal_key
            if (
                key in principals
                or key.issuer != self.identity.issuer
                or grant.authority.audience != self.public_url
                or grant.environment_id != self.environment_id
                or not any(
                    application.tenant_id == key.tenant_id
                    and application.application_id == key.application_id
                    and (key.subject_kind == "human" or key.subject == application.client_id)
                    for application in self.identity.applications
                )
            ):
                raise ValueError("Duplicate or foreign workspace grant")
            for permissions in (grant.application, grant.user, grant.workspace):
                _permission_set(permissions)
            principals.add(key)
        object.__setattr__(self, "grants", grants)
        object.__setattr__(
            self, "deployment_permissions", _permission_set(self.deployment_permissions)
        )


def decode_remote_policy(document: str) -> RemotePolicyConfiguration:
    """Decode endpoint policy without opening state, credentials or sockets."""
    from capicove.contracts.remote import AuthorityKey
    from capicove.runtime.remote_dispatch import WorkspaceGrant

    if len(document.encode("utf-8")) > 262144:
        raise ValueError("Remote policy document exceeds its byte limit")
    value = json.loads(document, object_pairs_hook=_unique)
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "environment_id",
        "public_url",
        "identity",
        "deployment_permissions",
        "grants",
    }:
        raise ValueError("Unexpected remote policy fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("Unsupported remote policy version")
    identity = decode_enrollment(json.dumps(value["identity"]))
    environment = _text(value["environment_id"])
    endpoint = _https(value["public_url"])
    raw_grants = value["grants"]
    if not isinstance(raw_grants, list) or not 1 <= len(raw_grants) <= 1024:
        raise ValueError("Explicit bounded workspace grants required")
    grants: list[WorkspaceGrant] = []
    for raw in raw_grants:
        if not isinstance(raw, dict) or set(raw) != {
            "client_id",
            "subject_kind",
            "subject",
            "workspace_binding_id",
            "grant_id",
            "epoch",
            "application_permissions",
            "user_permissions",
            "workspace_permissions",
        }:
            raise ValueError("Unexpected grant fields")
        kind = raw["subject_kind"]
        if kind not in ("human", "service"):
            raise ValueError("Explicit subject kind required")
        key = identity.map_subject(_text(raw["client_id"]), kind, _text(raw["subject"]))
        if kind == "service" and key.subject != raw["client_id"]:
            raise ValueError("Service grants must name their enrolled client")
        authority = AuthorityKey(
            key, _text(raw["workspace_binding_id"]), _text(raw["grant_id"]), endpoint
        )
        grants.append(
            WorkspaceGrant(
                authority,
                environment,
                _permission_set(raw["application_permissions"]),
                _permission_set(raw["user_permissions"]),
                _permission_set(raw["workspace_permissions"]),
                raw["epoch"],
            )
        )
    return RemotePolicyConfiguration(
        environment,
        endpoint,
        identity,
        _permission_set(value["deployment_permissions"]),
        tuple(grants),
    )


def _check_private_acl(descriptor: int) -> None:
    """Reject ACL grants that can broaden mode-bit permissions on an open object."""
    import errno
    import os
    import sys

    # A plain str keeps both platform branches type-checked on every host OS.
    platform: str = sys.platform
    if platform == "linux":
        for name in ("system.posix_acl_access", "system.posix_acl_default"):
            try:
                os.getxattr(descriptor, name)
            except OSError as error:
                if error.errno != errno.ENODATA:
                    raise ValueError("Policy ACL inspection failed") from error
            else:
                raise ValueError("Extended policy ACLs are not supported")
        return
    if platform != "darwin":
        raise ValueError("Policy ACL inspection is unsupported on this platform")

    import ctypes

    library = ctypes.CDLL(None, use_errno=True)
    library.acl_get_fd_np.argtypes = [ctypes.c_int, ctypes.c_int]
    library.acl_get_fd_np.restype = ctypes.c_void_p
    library.acl_valid.argtypes = [ctypes.c_void_p]
    library.acl_valid.restype = ctypes.c_int
    library.acl_get_entry.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.acl_get_entry.restype = ctypes.c_int
    library.acl_get_tag_type.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    library.acl_get_tag_type.restype = ctypes.c_int
    library.acl_free.argtypes = [ctypes.c_void_p]
    library.acl_free.restype = ctypes.c_int
    # Constants and return conventions come from the Darwin sys/acl.h API.
    ctypes.set_errno(0)
    acl = library.acl_get_fd_np(descriptor, 0x100)
    if not acl:
        # Darwin reports ENOENT for an absent extended ACL on a valid fd.
        if ctypes.get_errno() == errno.ENOENT and os.fstat(descriptor).st_nlink > 0:
            return
        raise ValueError("Policy ACL inspection failed")
    try:
        if library.acl_valid(acl) != 0:
            raise ValueError("Invalid policy ACL")
        for index in range(170):
            entry = ctypes.c_void_p()
            ctypes.set_errno(0)
            status = library.acl_get_entry(acl, index, ctypes.byref(entry))
            if status != 0:
                if ctypes.get_errno() == errno.EINVAL:
                    return
                raise ValueError("Policy ACL entry inspection failed")
            tag = ctypes.c_int()
            if library.acl_get_tag_type(entry, ctypes.byref(tag)) != 0 or tag.value != 2:
                raise ValueError("Policy ACL permits an additional identity")
        raise ValueError("Policy ACL exceeds its entry limit")
    finally:
        library.acl_free(acl)


def load_remote_policy(path: Path) -> RemotePolicyConfiguration:
    """Read a private POSIX policy through no-follow directory descriptors.

    This loader opens only the requested policy file. Workspace databases,
    credential files and network endpoints remain unopened until composition.
    """
    return decode_remote_policy(read_private_document(path))


def read_private_document(path: Path) -> str:
    """Read one bounded owner-only UTF-8 file through no-follow directory descriptors."""
    import os
    import stat

    if os.name != "posix" or not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("Private policy loading requires POSIX no-follow descriptors")
    if not isinstance(path, Path) or not path.is_absolute() or ".." in path.parts:
        raise ValueError("Policy path must be absolute without parent traversal")
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    descriptor: int | None = None
    try:
        _check_private_acl(directory)
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
            _check_private_acl(directory)
            metadata = os.fstat(directory)
            if metadata.st_uid not in {0, os.geteuid()} or (
                metadata.st_mode & 0o022
                and not (metadata.st_uid == 0 and metadata.st_mode & stat.S_ISVTX)
            ):
                raise ValueError("Policy directory is writable by another identity")
        descriptor = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
        _check_private_acl(descriptor)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_mode & 0o077
            or before.st_nlink != 1
            or before.st_size > 262144
        ):
            raise ValueError("Policy must be a bounded private owner-only regular file")
        payload = bytearray()
        while len(payload) <= 262144:
            chunk = os.read(descriptor, min(65536, 262145 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        _check_private_acl(descriptor)
        after = os.fstat(descriptor)
        if len(payload) > 262144 or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("Policy changed during bounded loading")
        return payload.decode("utf-8")
    except (OSError, UnicodeError, RecursionError) as error:
        raise ValueError("Private remote document could not be loaded") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)
