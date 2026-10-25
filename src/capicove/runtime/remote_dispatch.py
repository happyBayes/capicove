"""Authority-bound read dispatch; durable effects require the execution layer."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import astuple, dataclass, replace
from datetime import UTC, datetime
from types import MappingProxyType

from capicove.contracts import InvocationContext, SideEffect, ToolResult, ToolSpec
from capicove.contracts.remote import AuthorityKey, Principal
from capicove.mcp.bindings import McpToolBinding, invoke_binding, validate_bindings
from capicove.runtime.remote_epochs import EpochStore
from capicove.runtime.remote_policy import authorize


@dataclass(frozen=True, slots=True)
class WorkspaceGrant:
    """Pin server-owned permissions to one authority and policy epoch."""

    authority: AuthorityKey
    environment_id: str
    application: frozenset[str]
    user: frozenset[str]
    workspace: frozenset[str]
    epoch: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.authority, AuthorityKey) or not self.environment_id.strip():
            raise ValueError("A complete server-owned workspace grant is required")
        if type(self.epoch) is not int or self.epoch < 0:
            raise ValueError("A nonnegative grant epoch is required")
        for name in ("application", "user", "workspace"):
            value = getattr(self, name)
            if not isinstance(value, frozenset) or any(
                not isinstance(item, str) or not item.strip() for item in value
            ):
                raise ValueError("Grant permissions must be immutable identifiers")


ResourcePolicy = Callable[[AuthorityKey, ToolSpec, Mapping[str, object]], frozenset[str]]


class RemoteDispatcher:
    """Apply all policy tiers before invoking an explicitly registered read Tool."""

    def __init__(
        self,
        grant: WorkspaceGrant,
        bindings: tuple[McpToolBinding, ...],
        *,
        deployment: frozenset[str],
        resources: ResourcePolicy,
        epochs: EpochStore,
        admitted_mutations: frozenset[str] = frozenset(),
    ) -> None:
        validate_bindings(bindings)
        if not isinstance(deployment, frozenset) or not callable(resources):
            raise ValueError("Explicit deployment and resource policies are required")
        names = {item.name for item in bindings}
        if not isinstance(admitted_mutations, frozenset) or not admitted_mutations <= names:
            raise ValueError("Admitted mutations must name registered bindings")
        # Only control-plane bindings with their own durable idempotency records
        # (artifact upload control, operation cancellation) may write remotely.
        self._admitted = admitted_mutations
        self._grant = grant
        self._bindings = MappingProxyType({item.name: item for item in bindings})
        self._deployment = deployment
        self._resources = resources
        self._epochs = epochs

    @property
    def authority(self) -> AuthorityKey:
        """Expose the immutable enrolled authority for session binding."""
        return self._grant.authority

    def validate_principal(self, principal: Principal) -> None:
        """Reject expired, foreign or revoked authority before discovery or calls."""
        if (
            principal.key != self._grant.authority.principal_key
            or principal.audience != self._grant.authority.audience
            or principal.expires_at <= time.time()
            or self._epochs.current(self._grant.authority.workspace_binding_id) != self._grant.epoch
        ):
            raise PermissionError("Operation is not authorized")

    @contextmanager
    def release_permit(self, principal: Principal) -> Iterator[None]:
        """Fence the actual transport send against acknowledged revocation."""
        if (
            principal.key != self._grant.authority.principal_key
            or principal.audience != self._grant.authority.audience
            or principal.expires_at <= time.time()
        ):
            raise PermissionError("Operation is not authorized")
        with self._epochs.permit(
            self._grant.authority.workspace_binding_id, self._grant.epoch, phase="release"
        ):
            yield

    def catalog(self, principal: Principal) -> tuple[McpToolBinding, ...]:
        """List only explicitly registered, permitted read bindings."""
        self.validate_principal(principal)
        ceiling = self._deployment.intersection(
            self._grant.application,
            principal.token_permissions,
            self._grant.user,
            self._grant.workspace,
        )
        return tuple(
            binding
            for binding in self._bindings.values()
            if (
                binding.spec.side_effect in {SideEffect.NONE, SideEffect.READ}
                or binding.name in self._admitted
            )
            and not binding.spec.network_access
            and binding.spec.required_permissions.issubset(ceiling)
        )

    def invoke(
        self, principal: Principal, name: str, arguments: Mapping[str, object]
    ) -> dict[str, object]:
        """Bind context and reject mutations until durable admission is composed."""
        grant = self._grant
        self.validate_principal(principal)
        binding = self._bindings.get(name)
        if (
            binding is None
            or binding.spec.network_access
            or (
                binding.spec.side_effect not in {SideEffect.NONE, SideEffect.READ}
                and name not in self._admitted
            )
        ):
            raise PermissionError("Operation is not authorized")
        ceiling = self._deployment.intersection(
            grant.application, principal.token_permissions, grant.user, grant.workspace
        )
        if not binding.spec.required_permissions <= ceiling:
            raise PermissionError("Operation is not authorized")
        resource_permissions = self._resources(grant.authority, binding.spec, arguments)
        decision = authorize(
            grant.authority,
            required=binding.spec.required_permissions,
            deployment=self._deployment,
            application=grant.application,
            token=principal.token_permissions,
            user=grant.user,
            workspace=grant.workspace,
            resource=resource_permissions,
            policy_epoch=grant.epoch,
            source_epoch=grant.epoch,
        )
        actor = (
            "remote:"
            + hashlib.sha256(
                json.dumps(astuple(principal.key), separators=(",", ":")).encode()
            ).hexdigest()
        )
        context = InvocationContext(
            str(uuid.uuid4()),
            str(uuid.uuid4()),
            grant.authority.workspace_binding_id,
            actor,
            grant.environment_id,
            datetime.fromtimestamp(
                min(principal.expires_at, time.time() + binding.spec.timeout_seconds), UTC
            ),
            decision.permissions,
            binding.spec.privacy_class,
        )

        def admitted_handler(
            invocation: InvocationContext, decoded: Mapping[str, object]
        ) -> ToolResult:
            # Decoding and tracing preparation may block. Linearize admission at
            # the handler boundary rather than before those untrusted delays.
            # Running handlers still require cooperative effect gates or R3
            # isolated workers; this permit does not lock the entire execution.
            with self._epochs.permit(
                grant.authority.workspace_binding_id, grant.epoch, phase="start"
            ):
                if invocation.deadline.timestamp() <= time.time():
                    raise PermissionError("Invocation authorization expired")
            return binding.handler(invocation, decoded)

        result = invoke_binding(
            replace(binding, handler=admitted_handler), arguments, context=context
        )
        with self._epochs.permit(
            grant.authority.workspace_binding_id, grant.epoch, phase="release"
        ):
            if context.deadline.timestamp() <= time.time():
                raise PermissionError("Invocation authorization expired")
            return result
