"""Trusted local startup for image-search CLI and single-principal stdio MCP."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
from uuid import uuid4

from capicove.contracts import InvocationContext, PrivacyClass, SearchFailureCode
from capicove.retrieval.search_models import InvocationAuthorizationContext, SearchFailure


if TYPE_CHECKING:
    from capicove.runtime.image_query import ResolvedImageArtifact
    from capicove.runtime.reverse_image import ReverseImageRuntime

_FIELDS = frozenset(
    {
        "schema_version",
        "workspace_id",
        "actor_id",
        "generation_root",
        "lifecycle_database_path",
        "model_root",
        "provisioning_manifest",
        "provisioning_sha256",
        "source_roots",
        "qualification_root",
        "qualification_key_file",
        "cursor_key_file",
        "permissions",
        "cpu_threads",
    }
)


@dataclass(frozen=True, slots=True)
class ImageSearchHost:
    """Own a local runtime and its startup-authorized invocation identity."""

    runtime: ReverseImageRuntime
    workspace_id: str
    actor_id: str
    permissions: frozenset[str]

    def create(self, *, tool_id: str, deadline_cap: datetime) -> InvocationContext:
        """Create fresh IDs without accepting authority from request arguments."""
        if tool_id != "capicove.search.image":
            raise ValueError("unsupported image-search Tool")
        return InvocationContext(
            request_id=f"image-{uuid4().hex}",
            trace_id=f"image-trace-{uuid4().hex}",
            workspace_id=self.workspace_id,
            actor_id=self.actor_id,
            environment_id="local-image-search",
            deadline=deadline_cap,
            permissions=self.permissions,
            privacy_class=PrivacyClass.CONFIDENTIAL,
        )

    def __enter__(self) -> ImageSearchHost:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.runtime.close()


class _NoArtifacts:
    def resolve(
        self,
        artifact_ref: str,
        *,
        authorization: InvocationAuthorizationContext,
        deadline: datetime,
    ) -> ResolvedImageArtifact:
        # This host exposes inline bytes and authorized local paths, not an
        # implicit artifact catalog or a network fallback.
        raise SearchFailure(SearchFailureCode.PERMISSION_DENIED)


def load_image_search_host(config_path: Path) -> ImageSearchHost:
    """Load an owner-controlled closed configuration; never build or activate data."""
    from capicove.runtime.retrieval import CpuExecutionBudget
    from capicove.runtime.retrieval_qualification import LocalPreviewQualificationStore
    from capicove.runtime.reverse_image import (
        ReverseImageModelProvisioning,
        build_reverse_image_runtime,
    )
    from capicove.tools.image_search import StaticSearchAuthorizationFactory

    config = read_json_object(config_path, maximum_bytes=65536, trusted=True)
    if set(config) != _FIELDS or type(config["schema_version"]) is not int:
        raise ValueError("invalid image-search configuration schema")
    if config["schema_version"] != 1:
        raise ValueError("unsupported image-search configuration version")
    workspace = _string(config["workspace_id"])
    actor = _string(config["actor_id"])
    if config["permissions"] != ["retrieval:search"]:
        raise ValueError("explicit retrieval:search permission is required")
    permissions = frozenset({"retrieval:search"})
    threads = config["cpu_threads"]
    if type(threads) is not int or not 1 <= threads <= 32:
        raise ValueError("cpu_threads must be between 1 and 32")
    roots_value = config["source_roots"]
    if not isinstance(roots_value, dict) or not 1 <= len(roots_value) <= 32:
        raise ValueError("source_roots must be a bounded object")
    roots = {_string(key): _path(value) for key, value in roots_value.items()}
    if any(not root.is_dir() for root in roots.values()):
        raise ValueError("source roots must be existing directories")
    generation_root = _path(config["generation_root"])
    lifecycle_path = _path(config["lifecycle_database_path"])
    qualification_root = _path(config["qualification_root"])
    model_root = _path(config["model_root"])
    manifest = _string(config["provisioning_manifest"])
    digest = _string(config["provisioning_sha256"])
    provisioning = ReverseImageModelProvisioning(model_root, PurePosixPath(manifest), digest)
    cursor_key = _read_file(_path(config["cursor_key_file"]), 64, secret=True)
    qualification_key = _read_file(_path(config["qualification_key_file"]), 64, secret=True)
    if len(cursor_key) < 32 or len(qualification_key) < 32:
        raise ValueError("signing keys must contain 32 to 64 raw bytes")
    if cursor_key == qualification_key:
        raise ValueError("cursor and qualification keys must be distinct")
    # Query startup must not create an empty deployment or silently qualify it.
    if not generation_root.is_dir() or not qualification_root.is_dir():
        raise ValueError("initialize generation and qualification storage before search")
    if not lifecycle_path.is_file():
        raise ValueError("initialize the lifecycle database before search")
    verifier = LocalPreviewQualificationStore(qualification_root, hmac_key=qualification_key)
    parallel = min(2, threads)
    runtime = build_reverse_image_runtime(
        workspace_id=workspace,
        generation_root=generation_root,
        lifecycle_database_path=lifecycle_path,
        model_provisioning=provisioning,
        artifact_resolver=_NoArtifacts(),
        local_roots_by_policy={"local-only": roots},
        preview_activation_verifier=verifier,
        cursor_signing_key=cursor_key,
        cpu_budget=CpuExecutionBudget(threads, parallel, threads // parallel),
        authorization_policy=StaticSearchAuthorizationFactory(
            authorization_scope_ref=f"local-image:{workspace}",
            permissions=permissions,
            allowed_data_policy_refs=frozenset({"local-only"}),
            allowed_source_root_refs=frozenset(roots),
        ),
        owner_id=f"image-host-{uuid4().hex}",
        owner_session_epoch=uuid4().hex,
        clock=lambda: datetime.now(UTC),
    )
    return ImageSearchHost(runtime, workspace, actor, permissions)


def read_json_object(
    path: Path,
    *,
    maximum_bytes: int,
    trusted: bool = False,
) -> dict[str, object]:
    """Read bounded regular-file JSON, rejecting duplicate keys and non-objects."""
    value = json.loads(
        _read_file(path.absolute(), maximum_bytes, trusted=trusted),
        object_pairs_hook=_unique_object,
    )
    if not isinstance(value, dict):
        raise ValueError("JSON must contain an object")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _string(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.encode()) > 256:
        raise ValueError("expected a bounded nonblank string")
    return value


def _path(value: object) -> Path:
    if not isinstance(value, str) or len(value.encode()) > 4096:
        raise ValueError("expected an absolute local path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("expected an absolute local path without traversal")
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("symlink paths are not supported")
    return path


def _read_file(
    path: Path,
    maximum_bytes: int,
    *,
    trusted: bool = False,
    secret: bool = False,
) -> bytes:
    path = _path(str(path))
    if not all(hasattr(os, name) for name in ("O_NOFOLLOW", "O_NONBLOCK", "O_CLOEXEC")):
        raise ValueError("secure local startup requires POSIX file-descriptor support")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    with os.fdopen(descriptor, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("expected a regular file")
        forbidden = 0o077 if secret else (0o022 if trusted else 0)
        if info.st_mode & forbidden:
            raise ValueError("configuration or signing-key permissions are too broad")
        payload = handle.read(maximum_bytes + 1)
    if len(payload) > maximum_bytes:
        raise ValueError("local input exceeds its size limit")
    return payload
