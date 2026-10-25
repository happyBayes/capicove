"""RFC-0016 amendment 1: operator model Tools in the workspace host."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from capicove.mcp.bindings import invoke_binding
from capicove.runtime.mcp_config import WorkspaceHostConfiguration, load_mcp_host_config
from capicove.runtime.mcp_host import compose_mcp_host


pytestmark = pytest.mark.skipif(os.name != "posix", reason="model host is POSIX-only")
WEIGHTS = b"synthetic weights" * 64
CONFIG = b'{"layers": 2}'


@dataclass(frozen=True)
class FileSpec:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repository: str
    revision: str
    size_bytes: int
    files: tuple[FileSpec, ...]


@dataclass(frozen=True)
class Plan:
    profile_id: str
    profile_version: int
    destination: Path
    models: tuple[ModelSpec, ...]
    total_size_bytes: int
    source_hosts: tuple[str, ...]
    manifest_sha256: str


def _file(path: str, payload: bytes) -> FileSpec:
    return FileSpec(path, len(payload), hashlib.sha256(payload).hexdigest())


@dataclass
class FakeInstaller:
    """Plan a two-file profile and write it on install, recording every call."""

    calls: list[tuple[str, Path, bool]] = field(default_factory=list)
    payload: bytes = WEIGHTS

    def model_plan(self, profile_id: str, destination: Path) -> Plan:
        if profile_id != "small":
            raise ValueError("unknown profile")
        files = (_file("weights.bin", WEIGHTS), _file("nested/config.json", CONFIG))
        model = ModelSpec(
            "detector", "example-org/detector", "a" * 40, len(WEIGHTS) + len(CONFIG), files
        )
        return Plan(
            "small", 1, destination, (model,), model.size_bytes, ("huggingface.co",), "m" * 64
        )

    def install_models(self, profile_id: str, destination: Path, *, confirmed: bool) -> Plan:
        self.calls.append((profile_id, destination, confirmed))
        (destination / "detector" / "nested").mkdir(parents=True, exist_ok=True)
        (destination / "detector" / "weights.bin").write_bytes(self.payload)
        (destination / "detector" / "nested" / "config.json").write_bytes(CONFIG)
        return self.model_plan(profile_id, destination)


@pytest.fixture
def installer(monkeypatch: pytest.MonkeyPatch) -> FakeInstaller:
    import capicove.runtime.model_mcp as model_mcp

    fake = FakeInstaller()

    def load(name: str) -> object:
        if name != "fake-parser":
            raise LookupError(name)
        return fake

    monkeypatch.setattr(model_mcp, "PLUGIN_LOADER", load)
    return fake


@dataclass(frozen=True)
class ModelSpace:
    root: Path
    config: Path

    def write(
        self,
        *,
        profile: str = "read",
        allowlist: list[dict[str, object]] | None = None,
        network: bool = False,
        tools: tuple[str, ...] = ("capicove.model.plan", "capicove.model.verify"),
    ) -> None:
        permissions = ["model:read"] + (["model:install"] if profile == "admin" else [])
        body = {
            "schema_version": 2,
            "server_name": "capicove-local",
            "workspace_id": "workspace",
            "actor_id": "owner",
            "environment_id": "local",
            "workspace_root": str(self.root),
            "profile": profile,
            "permissions": permissions,
            "enabled_tools": list(tools),
            "trace_store": None,
            "capabilities": {
                "model": {
                    "schema_version": 1,
                    "destinations": {"models": str(self.root / "models")},
                    "install_allowlist": allowlist or [],
                }
            },
            "network": {
                "enabled": network,
                "source_hosts": ["huggingface.co"] if network else [],
            },
        }
        self.config.write_text(json.dumps(body))
        self.config.chmod(0o600)

    def load(self) -> WorkspaceHostConfiguration:
        config = load_mcp_host_config(self.config)
        assert isinstance(config, WorkspaceHostConfiguration)
        return config

    def call(self, tool: str, arguments: Mapping[str, object]) -> dict[str, Any]:
        with compose_mcp_host(self.load()) as composed:
            bound = {item.spec.tool_id: item for item in composed}
            return invoke_binding(bound[f"capicove.model.{tool}"], arguments)


@pytest.fixture
def space(tmp_path: Path) -> ModelSpace:
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    result = ModelSpace(root, root / "host.json")
    result.write()
    return result


SELECT: dict[str, object] = {"plugin": "fake-parser", "profile": "small", "destination": "models"}


def admin(space: ModelSpace, digest: str, **entry: object) -> None:
    allowed: dict[str, object] = {
        "plan_digest": digest,
        "plugin": "fake-parser",
        "profile": "small",
        "destination": "models",
        "source_hosts": ["huggingface.co"],
    }
    allowed.update(entry)
    space.write(
        profile="admin",
        allowlist=[allowed],
        network=True,
        tools=("capicove.model.install", "capicove.model.plan", "capicove.model.verify"),
    )


def test_plan_is_offline_deterministic_and_path_free(
    space: ModelSpace, installer: FakeInstaller
) -> None:
    first = space.call("plan", SELECT)
    second = space.call("plan", SELECT)

    assert first["status"] == "succeeded", first
    plan = first["result"]
    assert plan["plan_digest"] == second["result"]["plan_digest"]
    assert plan["destination"] == "models"
    assert plan["source_hosts"] == ["huggingface.co"]
    assert plan["install_allowlisted"] is False
    assert str(space.root) not in json.dumps(first)
    assert installer.calls == []
    assert not (space.root / "models").exists()


def test_plan_digest_is_bound_to_the_workspace(space: ModelSpace, installer: FakeInstaller) -> None:
    digest = space.call("plan", SELECT)["result"]["plan_digest"]
    body = json.loads(space.config.read_text())
    body["workspace_id"] = "another"
    space.config.write_text(json.dumps(body))
    assert space.call("plan", SELECT)["result"]["plan_digest"] != digest


def test_plan_rejects_unknown_plugin_profile_and_destination(
    space: ModelSpace, installer: FakeInstaller
) -> None:
    assert space.call("plan", {**SELECT, "plugin": "other"})["error"]["code"] == "unavailable"
    assert space.call("plan", {**SELECT, "profile": "large"})["error"]["code"] == "invalid_input"
    assert space.call("plan", {**SELECT, "destination": "x"})["error"]["code"] == "invalid_input"


def test_verify_reports_missing_mismatched_and_verified_files(
    space: ModelSpace, installer: FakeInstaller
) -> None:
    missing = space.call("verify", SELECT)["result"]
    assert (missing["missing_files"], missing["complete"]) == (2, False)

    (space.root / "models").mkdir(mode=0o700)
    target = space.root / "models" / "fake-parser" / "small"
    installer.install_models("small", target, confirmed=True)
    assert space.call("verify", SELECT)["result"]["verified_files"] == 2

    (target / "detector" / "weights.bin").write_bytes(b"tampered")
    mismatched = space.call("verify", SELECT)["result"]
    assert (mismatched["mismatched_files"], mismatched["verified_files"]) == (1, 1)
    (target / "detector" / "weights.bin").unlink()
    (target / "detector" / "weights.bin").symlink_to(target / "detector" / "nested" / "config.json")
    assert space.call("verify", SELECT)["result"]["mismatched_files"] == 1


def test_install_needs_admin_network_profile(space: ModelSpace, installer: FakeInstaller) -> None:
    space.write(tools=("capicove.model.install",))
    with pytest.raises(ValueError):
        with compose_mcp_host(space.load()):
            pass
    space.write(profile="admin", tools=("capicove.model.install",), network=False)
    with pytest.raises(ValueError, match="network"):
        with compose_mcp_host(space.load()):
            pass


def test_install_requires_the_exact_allowlisted_digest(
    space: ModelSpace, installer: FakeInstaller
) -> None:
    digest = space.call("plan", SELECT)["result"]["plan_digest"]
    admin(space, "0" * 64)

    denied = space.call("install", {"plan_digest": digest, "destination": "models"})
    assert denied["error"]["code"] == "policy_rejected"
    stale = space.call("install", {"plan_digest": "0" * 64, "destination": "models"})
    assert stale["error"]["message"].startswith("plan_changed")
    assert installer.calls == []

    admin(space, digest)
    assert space.call("plan", SELECT)["result"]["install_allowlisted"] is True
    installed = space.call("install", {"plan_digest": digest, "destination": "models"})
    assert installed["status"] == "succeeded", installed
    assert installed["result"]["complete"] is True
    assert installer.calls == [("small", space.root / "models" / "fake-parser" / "small", True)]
    assert (space.root / "models").stat().st_mode & 0o077 == 0


def test_install_reports_a_corrupt_transfer_as_unqualified(
    space: ModelSpace, installer: FakeInstaller
) -> None:
    digest = space.call("plan", SELECT)["result"]["plan_digest"]
    admin(space, digest)
    installer.payload = b"truncated"

    result = space.call("install", {"plan_digest": digest, "destination": "models"})

    assert result["error"]["code"] == "unavailable"


def test_allowlist_hosts_must_be_network_authorized(
    space: ModelSpace, installer: FakeInstaller
) -> None:
    admin(space, "0" * 64, source_hosts=["example.com"])
    with pytest.raises(ValueError, match="source_hosts"):
        with compose_mcp_host(space.load()):
            pass


def test_models_cli_runtime_config_matches_mcp(
    space: ModelSpace, installer: FakeInstaller, capsys: pytest.CaptureFixture[str]
) -> None:
    from capicove.cli.app import main

    flags = ["fake-parser", "--profile", "small", "--destination", "models"]
    assert main(["models", "plan", "--runtime-config", str(space.config), *flags]) == 0
    cli = json.loads(capsys.readouterr().out)
    assert cli["result"] == space.call("plan", SELECT)["result"]
    # An incomplete installation is a successful verification with missing files.
    assert main(["models", "verify", "--runtime-config", str(space.config), *flags]) == 0
    assert json.loads(capsys.readouterr().out)["result"]["missing_files"] == 2

    admin(space, cli["result"]["plan_digest"])
    install = ["models", "install", "--runtime-config", str(space.config)]
    assert main([*install, "--plan-digest", cli["result"]["plan_digest"], "--yes"]) == 2
    assert (
        main([*install, "--plan-digest", cli["result"]["plan_digest"], "--destination", "models"])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["result"]["complete"] is True
