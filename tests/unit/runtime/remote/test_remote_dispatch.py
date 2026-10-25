"""Remote read dispatch must derive identity exclusively from server authority."""

import importlib
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from capicove.contracts import InvocationContext, ToolResult, ToolSpec, ToolStatus
from capicove.contracts.remote import AuthorityKey, Principal, PrincipalKey
from capicove.mcp.bindings import McpToolBinding
from capicove.runtime.remote_epochs import EpochStore


def dispatcher(tmp_path: Path, *, allowed: bool = True) -> tuple[Any, Principal, list[str]]:
    assert importlib.util.find_spec("capicove.runtime.remote_dispatch") is not None
    module = importlib.import_module("capicove.runtime.remote_dispatch")
    key = PrincipalKey("https://issuer.test", "tenant", "app", "human", "alice")
    principal = Principal(key, frozenset({"read"}), time.time() + 120, "audience")
    authority = AuthorityKey(key, "workspace", "grant", "audience")
    calls: list[str] = []
    spec = ToolSpec(
        "capicove.test.echo",
        1,
        "Identity fixture",
        {"type": "object", "additionalProperties": False},
        {"type": "object", "properties": {"actor": {"type": "string"}}, "required": ["actor"]},
        required_permissions=frozenset({"read"}),
    )

    def handler(context: InvocationContext, arguments: Mapping[str, object]) -> ToolResult:
        calls.append(context.actor_id)
        return ToolResult(
            context.request_id,
            context.trace_id,
            spec.tool_id,
            1,
            ToolStatus.SUCCEEDED,
            result={"actor": context.actor_id},
        )

    def startup(spec: ToolSpec) -> InvocationContext:
        raise AssertionError("Startup context must not supply remote authority")

    binding = McpToolBinding("capicove_test_echo", spec, handler, startup)
    tiers = frozenset({"read"})
    grant = module.WorkspaceGrant(authority, "environment", tiers, tiers, tiers)
    return (
        module.RemoteDispatcher(
            grant,
            (binding,),
            deployment=tiers,
            resources=lambda authority, spec, args: tiers if allowed else frozenset(),
            epochs=EpochStore(tmp_path / "epochs.sqlite"),
        ),
        principal,
        calls,
    )


def test_authorized_read_uses_namespaced_actor(tmp_path: Path) -> None:
    dispatch, principal, calls = dispatcher(tmp_path)
    result = dispatch.invoke(principal, "capicove_test_echo", {})
    assert result["status"] == "succeeded"
    assert len(calls) == 1 and calls[0].startswith("remote:")


def test_resource_denial_prevents_handler(tmp_path: Path) -> None:
    dispatch, principal, calls = dispatcher(tmp_path, allowed=False)
    with pytest.raises(PermissionError):
        dispatch.invoke(principal, "capicove_test_echo", {})
    assert calls == []


def test_other_subject_cannot_use_grant(tmp_path: Path) -> None:
    dispatch, principal, calls = dispatcher(tmp_path)
    other = Principal(
        PrincipalKey("https://issuer.test", "tenant", "app", "human", "bob"),
        principal.token_permissions,
        principal.expires_at,
        principal.audience,
    )
    with pytest.raises(PermissionError):
        dispatch.invoke(other, "capicove_test_echo", {})
    assert calls == []


def test_token_for_another_resource_cannot_use_grant(tmp_path: Path) -> None:
    from dataclasses import replace

    dispatch, principal, calls = dispatcher(tmp_path)
    other_audience = replace(principal, audience="https://other.test/mcp")
    with pytest.raises(PermissionError):
        dispatch.invoke(other_audience, "capicove_test_echo", {})
    assert calls == []


def test_hidden_tool_denial_does_not_enter_resource_policy(tmp_path: Path) -> None:
    from dataclasses import replace

    dispatch, principal, calls = dispatcher(tmp_path)
    resource_calls: list[str] = []

    def resources(
        authority: AuthorityKey, spec: ToolSpec, arguments: Mapping[str, object]
    ) -> frozenset[str]:
        resource_calls.append(spec.tool_id)
        raise AssertionError("Hidden tools must not reach resource inspection")

    dispatch._resources = resources
    with pytest.raises(PermissionError):
        dispatch.invoke(replace(principal, token_permissions=frozenset()), "capicove_test_echo", {})
    assert resource_calls == []
    assert calls == []


def test_resource_lookup_crossing_expiry_never_starts_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dispatch, principal, calls = dispatcher(tmp_path)
    clock = [principal.expires_at - 1]
    monkeypatch.setattr("capicove.runtime.remote_dispatch.time.time", lambda: clock[0])

    def delayed_resources(*args: object) -> frozenset[str]:
        clock[0] = principal.expires_at + 1
        return frozenset({"read"})

    dispatch._resources = delayed_resources
    with pytest.raises(PermissionError):
        dispatch.invoke(principal, "capicove_test_echo", {})
    assert calls == []


def test_decoding_crossing_expiry_or_revocation_never_starts_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace
    from types import MappingProxyType

    dispatch, principal, calls = dispatcher(tmp_path)
    clock = [principal.expires_at - 1]
    monkeypatch.setattr("capicove.runtime.remote_dispatch.time.time", lambda: clock[0])

    def delayed_decode(arguments: Mapping[str, object]) -> dict[str, object]:
        clock[0] = principal.expires_at + 1
        dispatch._epochs.revoke("workspace")
        return dict(arguments)

    binding = dispatch._bindings["capicove_test_echo"]
    dispatch._bindings = MappingProxyType(
        {binding.name: replace(binding, decode_arguments=delayed_decode)}
    )
    with pytest.raises(PermissionError):
        dispatch.invoke(principal, binding.name, {})
    assert calls == []
