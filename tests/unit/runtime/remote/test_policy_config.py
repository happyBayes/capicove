"""Server enrollment must determine grant ownership before runtime composition."""

import importlib
import json
import os
from pathlib import Path
from typing import Any

import pytest


def document() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "environment_id": "production",
        "public_url": "https://mcp.test/mcp",
        "deployment_permissions": ["read"],
        "identity": {
            "schema_version": 1,
            "issuer": "https://issuer.test",
            "audience": "https://mcp.test/mcp",
            "applications": [
                {"client_id": "client", "tenant_id": "tenant", "application_id": "app"}
            ],
        },
        "grants": [
            {
                "client_id": "client",
                "subject_kind": "human",
                "subject": "alice",
                "workspace_binding_id": "workspace",
                "grant_id": "grant",
                "epoch": 0,
                "application_permissions": ["read"],
                "user_permissions": ["read"],
                "workspace_permissions": ["read"],
            }
        ],
    }


def decode(value: dict[str, Any]) -> Any:
    module = importlib.import_module("capicove.runtime.remote_config")
    assert hasattr(module, "decode_remote_policy"), "Remote policy enrollment decoder is missing"
    return module.decode_remote_policy(json.dumps(value))


def test_policy_grant_uses_only_enrolled_tenant_and_application() -> None:
    config = decode(document())
    grant = config.grants[0]
    assert grant.authority.principal_key.tenant_id == "tenant"
    assert grant.authority.principal_key.application_id == "app"
    assert grant.authority.audience == config.public_url
    assert grant.environment_id == "production"
    assert config.deployment_permissions == frozenset({"read"})


@pytest.mark.parametrize(
    "mutation",
    [
        "tenant_spoof",
        "unknown_client",
        "duplicate_grant",
        "audience",
        "inline_secret",
        "wrong_path",
        "permissions",
        "service_subject",
    ],
)
def test_invalid_server_grant_configuration_fails_closed(mutation: str) -> None:
    value = document()
    grant = value["grants"][0]
    if mutation == "tenant_spoof":
        grant["tenant_id"] = "other"
    elif mutation == "unknown_client":
        grant["client_id"] = "unknown"
    elif mutation == "duplicate_grant":
        value["grants"].append(dict(grant))
    elif mutation == "audience":
        value["public_url"] = "https://other.test/mcp"
    elif mutation == "inline_secret":
        # Synthetic value for rejecting inline credentials.
        value["secret"] = "not-allowed"  # pragma: allowlist secret
    elif mutation == "wrong_path":
        value["public_url"] = "https://mcp.test/other"
    elif mutation == "permissions":
        grant["user_permissions"] = ["read\nwrite"]
    elif mutation == "service_subject":
        grant["subject_kind"] = "service"
    with pytest.raises(ValueError):
        decode(value)


@pytest.mark.skipif(os.name != "posix", reason="POSIX private remote policy loader")
def test_private_policy_file_loads_and_rejects_symlink_or_shared_mode(tmp_path: Path) -> None:
    module = importlib.import_module("capicove.runtime.remote_config")
    assert hasattr(module, "load_remote_policy"), "Private remote policy loader is missing"
    path = tmp_path / "remote.json"
    path.write_text(json.dumps(document()))
    path.chmod(0o600)
    assert module.load_remote_policy(path).environment_id == "production"
    path.chmod(0o644)
    with pytest.raises(ValueError):
        module.load_remote_policy(path)
    path.chmod(0o600)
    alias = tmp_path / "alias.json"
    alias.symlink_to(path)
    with pytest.raises(ValueError):
        module.load_remote_policy(alias)
    directory_alias = tmp_path / "linked"
    directory_alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        module.load_remote_policy(directory_alias / "remote.json")


@pytest.mark.skipif(__import__("sys").platform != "darwin", reason="macOS extended ACL regression")
@pytest.mark.parametrize("target_kind", ["file", "directory"])
def test_extended_acl_cannot_make_private_policy_writable(tmp_path: Path, target_kind: str) -> None:
    import subprocess

    from capicove.runtime.remote_config import load_remote_policy

    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    path = directory / "remote.json"
    path.write_text(json.dumps(document()))
    path.chmod(0o600)
    target = path if target_kind == "file" else directory
    subprocess.run(["/bin/chmod", "+a", "everyone allow read,write", str(target)], check=True)
    try:
        with pytest.raises(ValueError):
            load_remote_policy(path)
    finally:
        subprocess.run(["/bin/chmod", "-N", str(target)], check=True)
