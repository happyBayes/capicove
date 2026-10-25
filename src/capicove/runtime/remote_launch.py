"""Launch configuration, readiness and ASGI composition for the remote read host (R2.04).

A closed, private ``capicove.remote-mcp.launch`` version-1 document names the
endpoint policy, the version-2 workspace host whose read Tools are exposed, the
epoch store, a loopback listener and one identity verifier. TLS, public Host
routing and streaming limits belong to the reverse proxy in front of the
loopback listener. ``remote_check`` validates everything without opening sockets
or contacting the identity provider; it reports no registrations, roots or secrets.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import ExitStack, asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from capicove.contracts import InvocationContext, SideEffect, ToolResult, ToolSpec, ToolStatus
from capicove.mcp.bindings import McpToolBinding
from capicove.mcp.remote_catalog import READINESS_MCP_NAME
from capicove.runtime.remote_config import (
    RemotePolicyConfiguration,
    decode_remote_policy,
    read_private_document,
)
from capicove.tools.remote_readiness import READINESS_SPEC


LAUNCH_SCHEMA_VERSION = 1
PROTOCOL_VERSION = "2025-11-25"
_LOOPBACK = frozenset({"127.0.0.1", "::1"})
_FIELDS = frozenset(
    {
        "schema_version",
        "policy_path",
        "workspace_config",
        "epoch_store",
        "listen",
        "verifier",
        "json_response",
        "publish_resource_metadata",
    }
)
# Optional fields; omitting them keeps the read-only preview surface.
_OPTIONAL_FIELDS = frozenset({"control_store", "revocation_journal"})


@dataclass(frozen=True, slots=True)
class VerifierSettings:
    """Select one identity verifier; credentials stay in private files."""

    kind: str
    url: str
    subject_kind: str
    client_id: str | None = None
    secret_file: Path | None = None


@dataclass(frozen=True, slots=True)
class RemoteLaunch:
    """Validated launch authority for one remote MCP endpoint."""

    policy_path: Path
    workspace_config: Path
    epoch_store: Path
    listen_host: str
    listen_port: int
    verifier: VerifierSettings
    json_response: bool
    publish_resource_metadata: bool
    document_sha256: str = field(default="")
    control_store: Path | None = None
    revocation_journal: Path | None = None


def _absolute(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError(f"{name} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be an absolute path without traversal")
    return path


def _https(value: object, name: str) -> str:
    parsed = urlsplit(value) if isinstance(value, str) else None
    if (
        parsed is None
        or parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{name} must be a pinned HTTPS URL")
    return str(value)


def decode_launch(document: str) -> RemoteLaunch:
    """Decode the closed launch document; unknown fields and inline secrets fail."""
    if len(document.encode("utf-8")) > 65536:
        raise ValueError("Launch document exceeds its byte limit")
    raw = json.loads(document, object_pairs_hook=_unique)
    if not isinstance(raw, dict) or not _FIELDS <= set(raw) <= _FIELDS | _OPTIONAL_FIELDS:
        raise ValueError("Launch document requires exactly the documented fields")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != LAUNCH_SCHEMA_VERSION:
        raise ValueError("Unsupported launch schema version")
    listen = raw["listen"]
    if not isinstance(listen, dict) or set(listen) != {"host", "port"}:
        raise ValueError("listen requires host and port")
    if listen["host"] not in _LOOPBACK:
        raise ValueError("The listener must be loopback; terminate TLS in a reverse proxy")
    if type(listen["port"]) is not int or not 1 <= listen["port"] <= 65535:
        raise ValueError("listen.port must be an integer from 1 to 65535")
    verifier = raw["verifier"]
    if not isinstance(verifier, dict):
        raise ValueError("verifier must be an object")
    kind = verifier.get("kind")
    if kind == "jwks":
        if set(verifier) != {"kind", "url", "subject_kind"}:
            raise ValueError("jwks verifier requires kind, url and subject_kind")
        settings = VerifierSettings(
            "jwks", _https(verifier["url"], "verifier.url"), str(verifier["subject_kind"])
        )
    elif kind == "introspection":
        if set(verifier) != {"kind", "url", "subject_kind", "client_id", "secret_file"}:
            raise ValueError("introspection verifier requires a client_id and private secret_file")
        client = verifier["client_id"]
        if not isinstance(client, str) or not client.strip() or len(client) > 256:
            raise ValueError("verifier.client_id must be bounded text")
        settings = VerifierSettings(
            "introspection",
            _https(verifier["url"], "verifier.url"),
            str(verifier["subject_kind"]),
            client,
            _absolute(verifier["secret_file"], "verifier.secret_file"),
        )
    else:
        raise ValueError("verifier.kind must be jwks or introspection")
    if settings.subject_kind not in {"human", "service"}:
        raise ValueError("verifier.subject_kind must be human or service")
    for flag in ("json_response", "publish_resource_metadata"):
        if type(raw[flag]) is not bool:
            raise ValueError(f"{flag} must be a boolean")
    return RemoteLaunch(
        policy_path=_absolute(raw["policy_path"], "policy_path"),
        workspace_config=_absolute(raw["workspace_config"], "workspace_config"),
        epoch_store=_absolute(raw["epoch_store"], "epoch_store"),
        listen_host=listen["host"],
        listen_port=listen["port"],
        verifier=settings,
        json_response=raw["json_response"],
        publish_resource_metadata=raw["publish_resource_metadata"],
        document_sha256=hashlib.sha256(document.encode("utf-8")).hexdigest(),
        control_store=_absolute(raw["control_store"], "control_store")
        if "control_store" in raw
        else None,
        revocation_journal=_absolute(raw["revocation_journal"], "revocation_journal")
        if "revocation_journal" in raw
        else None,
    )


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate launch field")
        result[key] = value
    return result


def load_launch(path: Path) -> RemoteLaunch:
    """Read the launch document with the same private no-follow rules as policy."""
    return decode_launch(read_private_document(path))


@dataclass(frozen=True, slots=True)
class _Prepared:
    policy: RemotePolicyConfiguration
    policy_sha256: str
    workspace: Any


def _prepare(launch: RemoteLaunch) -> _Prepared:
    from capicove.runtime.mcp_config import WorkspaceHostConfiguration, load_mcp_host_config

    document = read_private_document(launch.policy_path)
    policy = decode_remote_policy(document)
    workspace = load_mcp_host_config(launch.workspace_config)
    if not isinstance(workspace, WorkspaceHostConfiguration):
        raise ValueError("workspace_config must be a version-2 workspace host")
    if workspace.environment_id != policy.environment_id:
        raise ValueError("policy environment does not match the workspace host")
    if workspace.network["enabled"]:
        raise ValueError("remote read endpoints require a workspace without network Tools")
    if any(
        grant.authority.workspace_binding_id != workspace.workspace_id for grant in policy.grants
    ):
        raise ValueError("every grant must bind the configured workspace")
    return _Prepared(policy, hashlib.sha256(document.encode("utf-8")).hexdigest(), workspace)


def _catalog_entries(bindings: tuple[McpToolBinding, ...]) -> list[dict[str, object]]:
    return sorted(
        (
            {
                "name": item.name,
                "tool_id": item.spec.tool_id,
                "contract_version": item.spec.contract_version,
                "input_schema_sha256": hashlib.sha256(
                    json.dumps(item.spec.input_schema, sort_keys=True, default=dict).encode()
                ).hexdigest(),
            }
            for item in bindings
            if item.spec.side_effect in {SideEffect.NONE, SideEffect.READ}
            and not item.spec.network_access
        ),
        key=lambda entry: str(entry["name"]),
    )


def _readiness_binding(report: Callable[[], dict[str, object]]) -> McpToolBinding:
    def handler(context: InvocationContext, arguments: Mapping[str, object]) -> ToolResult:
        started = datetime.now(UTC)
        return ToolResult(
            request_id=context.request_id,
            trace_id=context.trace_id,
            tool_id=READINESS_SPEC.tool_id,
            tool_version=READINESS_SPEC.contract_version,
            status=ToolStatus.SUCCEEDED,
            result=report(),
            started_at=started,
            completed_at=datetime.now(UTC),
        )

    def context(spec: ToolSpec) -> InvocationContext:
        raise PermissionError("remote readiness has no startup identity")

    return McpToolBinding(READINESS_MCP_NAME, READINESS_SPEC, handler, context)


@contextmanager
def _bindings(prepared: _Prepared, *, catalog_only: bool) -> Iterator[tuple[McpToolBinding, ...]]:
    from capicove.runtime.mcp_host import compose_mcp_host

    with compose_mcp_host(prepared.workspace, catalog_only=catalog_only) as bindings:
        yield tuple(bindings)


def _report(
    launch: RemoteLaunch, prepared: _Prepared, catalog: list[dict[str, object]]
) -> dict[str, object]:
    return {
        "status": "ready",
        "protocol_version": PROTOCOL_VERSION,
        "sdk_version": importlib.metadata.version("mcp"),
        "launch_sha256": launch.document_sha256,
        "policy_sha256": prepared.policy_sha256,
        "catalog_sha256": hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest(),
        "catalog": catalog,
        "grants": len(prepared.policy.grants),
        "verifier": launch.verifier.kind,
        "listener": "loopback",
        "resource_metadata": launch.publish_resource_metadata,
        "control_plane": launch.control_store is not None,
    }


def _workspace_resources(
    permissions: frozenset[str],
) -> Callable[[object, ToolSpec, Mapping[str, object]], frozenset[str]]:
    def resources(
        authority: object, spec: ToolSpec, arguments: Mapping[str, object]
    ) -> frozenset[str]:
        return permissions

    return resources


def remote_check(launch: RemoteLaunch) -> dict[str, object]:
    """Validate policy, workspace, grants and catalog; no socket, model or IdP call."""
    prepared = _prepare(launch)
    with _bindings(prepared, catalog_only=True) as bindings:
        catalog = _catalog_entries((*bindings, _readiness_binding(dict)))
    return _report(launch, prepared, catalog)


class RemoteApp:
    """ASGI application owning the SDK session lifecycle and composed runtimes."""

    def __init__(self, host: Any, stack: ExitStack) -> None:
        self._host = host
        self._stack = stack

    @asynccontextmanager
    async def lifespan_context(self) -> AsyncIterator[None]:
        """Run SDK sessions, then close composed workspace runtimes."""
        try:
            async with self._host.run():
                yield
        finally:
            self._stack.close()

    async def __call__(self, scope: Mapping[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") == "lifespan":
            message = await receive()
            if message["type"] != "lifespan.startup":
                return
            context = self.lifespan_context()
            try:
                await context.__aenter__()
            except Exception:
                await send({"type": "lifespan.startup.failed", "message": "startup failed"})
                return
            await send({"type": "lifespan.startup.complete"})
            message = await receive()
            await context.__aexit__(None, None, None)
            await send({"type": "lifespan.shutdown.complete"})
            return
        await self._host(scope, receive, send)


def _verifier(launch: RemoteLaunch, policy: RemotePolicyConfiguration, transport: Any) -> Any:
    from capicove.runtime._remote.identity_http import IdentityHttp

    endpoint = IdentityHttp(launch.verifier.url, network_enabled=True, transport=transport)
    if launch.verifier.kind == "jwks":
        from capicove.runtime._remote.jwks import JwksVerifier

        return JwksVerifier(policy.identity, endpoint, subject_kind=launch.verifier.subject_kind)  # type: ignore[arg-type]
    from capicove.runtime._remote.introspection import IntrospectionVerifier

    secret_file = launch.verifier.secret_file
    assert secret_file is not None and launch.verifier.client_id is not None
    return IntrospectionVerifier(
        policy.identity,
        endpoint,
        client_id=launch.verifier.client_id,
        secret=lambda: read_private_document(secret_file).strip(),
        subject_kind=launch.verifier.subject_kind,  # type: ignore[arg-type]
    )


def _control_bindings(
    control: Path, grants: Any, epochs: Any
) -> Callable[[Any], tuple[tuple[McpToolBinding, ...], frozenset[str]]]:
    """Open private artifact/operation stores and bind their Tools per grant."""
    from capicove.mcp.remote_catalog import REMOTE_CONTROL_MCP_NAMES
    from capicove.runtime.remote_artifacts import ArtifactService
    from capicove.runtime.remote_audit import AuditLog
    from capicove.runtime.remote_operations import OperationService
    from capicove.runtime.remote_recovery import TombstoneJournal
    from capicove.tools.remote_control import (
        ARTIFACT_SPECS,
        OPERATION_SPECS,
        artifact_handlers,
        operation_handlers,
    )

    control.mkdir(mode=0o700, parents=True, exist_ok=True)
    status = control.lstat()
    if control.is_symlink() or status.st_mode & 0o077:
        raise ValueError("control_store must be an owner-only real directory")
    by_authority = {grant.authority: grant for grant in grants}

    def authorize(authority: Any) -> None:
        # Every store call rechecks the grant's live revocation epoch.
        grant = by_authority.get(authority)
        if grant is None or epochs.current(authority.workspace_binding_id) != grant.epoch:
            raise PermissionError("authority is revoked or unknown")

    artifacts = ArtifactService(
        control / "artifacts",
        authorize=authorize,
        tombstones=TombstoneJournal(control / "tombstones" / "journal.jsonl"),
    )
    # No operation targets are registered until R4/R5 adapters exist; status,
    # result and cancel still apply to any operation admitted by server code.
    operations = OperationService(
        control / "operations" / "control.sqlite3",
        (),
        authorize=authorize,
        audit=AuditLog(control / "audit" / "audit.jsonl"),
    )
    specs = {spec.tool_id: spec for spec in (*ARTIFACT_SPECS.values(), *OPERATION_SPECS.values())}

    def context(spec: ToolSpec) -> InvocationContext:
        raise PermissionError("remote control Tools have no startup identity")

    def bind(grant: Any) -> tuple[tuple[McpToolBinding, ...], frozenset[str]]:
        handlers = {
            **artifact_handlers(artifacts, grant.authority),
            **operation_handlers(operations, grant.authority),
        }
        bindings = tuple(
            McpToolBinding(REMOTE_CONTROL_MCP_NAMES[tool_id], specs[tool_id], handler, context)
            for tool_id, handler in sorted(handlers.items())
        )
        writes = frozenset(
            binding.name
            for binding in bindings
            if binding.spec.side_effect not in {SideEffect.NONE, SideEffect.READ}
        )
        return bindings, writes

    return bind


def compose_remote_app(launch: RemoteLaunch, *, identity_transport: Any = None) -> RemoteApp:
    """Compose verifier, dispatchers and host; network calls happen only per request."""
    from capicove.mcp.remote import RemoteReadHost
    from capicove.runtime.remote_dispatch import RemoteDispatcher
    from capicove.runtime.remote_epochs import EpochStore
    from capicove.runtime.remote_metadata import ProtectedResourceMetadata

    prepared = _prepare(launch)
    stack = ExitStack()
    try:
        bindings = stack.enter_context(_bindings(prepared, catalog_only=False))
        catalog_holder: list[dict[str, object]] = []
        readiness = _readiness_binding(lambda: _report(launch, prepared, catalog_holder))
        served = (*bindings, readiness)
        catalog_holder.extend(_catalog_entries(served))
        launch.epoch_store.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        epochs = EpochStore(launch.epoch_store, revocations=launch.revocation_journal)
        policy = prepared.policy
        bind = (
            _control_bindings(launch.control_store, policy.grants, epochs)
            if launch.control_store is not None
            else None
        )
        dispatchers = {}
        for grant in policy.grants:
            control, writes = bind(grant) if bind is not None else ((), frozenset())
            dispatchers[grant.authority.principal_key] = RemoteDispatcher(
                grant,
                (*served, *control),
                deployment=policy.deployment_permissions,
                # Workspace Tools resolve sources inside the single bound workspace;
                # per-source ACL lineage is not yet composed for these Tools.
                resources=_workspace_resources(grant.workspace),
                epochs=epochs,
                admitted_mutations=writes,
            )
        metadata = (
            ProtectedResourceMetadata(policy.public_url, policy.identity.issuer)
            if launch.publish_resource_metadata
            else None
        )
        host = RemoteReadHost(
            _verifier(launch, policy, identity_transport),
            dispatchers,
            allowed_hosts=(urlsplit(policy.public_url).netloc.lower(),),
            json_response=launch.json_response,
            resource_metadata=metadata,
        )
    except BaseException:
        stack.close()
        raise
    return RemoteApp(host, stack)


def serve(launch: RemoteLaunch) -> None:
    """Serve on the loopback listener until interrupted."""
    import uvicorn

    app = compose_remote_app(launch)
    uvicorn.run(
        app,
        host=launch.listen_host,
        port=launch.listen_port,
        lifespan="on",
        proxy_headers=False,
        server_header=False,
        date_header=False,
        log_level="warning",
        access_log=False,
    )


__all__ = [
    "READINESS_SPEC",
    "RemoteApp",
    "RemoteLaunch",
    "VerifierSettings",
    "compose_remote_app",
    "decode_launch",
    "load_launch",
    "remote_check",
    "serve",
]
