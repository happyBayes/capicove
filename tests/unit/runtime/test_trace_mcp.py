"""RFC-0016 amendment 1: operator trace Tools in the workspace host."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from capicove.contracts import PrivacyClass
from capicove.mcp.bindings import invoke_binding
from capicove.runtime.mcp_config import WorkspaceHostConfiguration, load_mcp_host_config
from capicove.runtime.mcp_host import compose_mcp_host
from capicove.traces import EpisodeCompleted, EpisodeStatus, TraceRecordDraft
from capicove.traces._adapters.sqlite import SqliteTraceStore
from tests.integration.traces.test_operator_read import NOW, _start


pytestmark = pytest.mark.skipif(os.name != "posix", reason="trace host is POSIX-only")
TOOLS = ["capicove.trace.export", "capicove.trace.inspect", "capicove.trace.validate"]


@dataclass(frozen=True)
class TraceSpace:
    root: Path
    store: Path
    config: Path

    def write(self, profile: str = "write", trace: Mapping[str, object] | None = None) -> None:
        permissions = {
            "read": ["trace:read"],
            "write": ["trace:export", "trace:read"],
            "admin": ["trace:copy_media", "trace:export", "trace:export_restricted", "trace:read"],
        }[profile]
        capability: dict[str, object] = {
            "schema_version": 1,
            "stores": {"local": str(self.store)},
            "export_roots": {"exports": str(self.root / "exports")},
            "media_root": None,
        }
        capability.update(trace or {})
        tools = TOOLS if profile != "read" else TOOLS[1:]
        body = {
            "schema_version": 2,
            "server_name": "capicove-local",
            "workspace_id": "workspace-one",
            "actor_id": "owner",
            "environment_id": "local",
            "workspace_root": str(self.root),
            "profile": profile,
            "permissions": permissions,
            "enabled_tools": tools,
            "trace_store": None,
            "capabilities": {"trace": capability},
            "network": {"enabled": False, "source_hosts": []},
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
            return invoke_binding(bound[f"capicove.trace.{tool}"], arguments)


@pytest.fixture
def space(tmp_path: Path) -> TraceSpace:
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    traces = tmp_path / "traces"
    traces.mkdir(mode=0o700)
    store = SqliteTraceStore(traces / "traces.sqlite3")
    for episode, privacy in (("done", PrivacyClass.INTERNAL), ("secret", PrivacyClass.RESTRICTED)):
        _start(store, episode, privacy=privacy)
        store.append(
            TraceRecordDraft(
                episode_id=episode,
                trace_id=f"trace-{episode}",
                occurred_at=NOW + timedelta(seconds=1),
                payload=EpisodeCompleted(EpisodeStatus.SUCCEEDED),
                privacy_class=privacy,
            ),
            idempotency_key="complete",
        )
    _start(store, "open", privacy=PrivacyClass.INTERNAL)
    result = TraceSpace(root, traces / "traces.sqlite3", root / "host.json")
    result.write()
    return result


def test_catalog_and_read_profile(space: TraceSpace) -> None:
    with compose_mcp_host(space.load(), catalog_only=True) as composed:
        assert [item.name for item in composed] == [
            "capicove_trace_export",
            "capicove_trace_inspect",
            "capicove_trace_validate",
        ]
    space.write(profile="read")
    with compose_mcp_host(space.load()) as composed:
        assert len(composed) == 2
    assert not (space.root / "exports").exists()


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"stores": {"local": "/nonexistent/traces.sqlite3"}}, "existing SQLite"),
        ({"export_roots": {"exports": "/tmp/outside"}}, "strictly inside"),
        ({"media_root": "relative"}, "absolute"),
        ({"stores": {}}, "1 to 16"),
    ],
)
def test_trace_config_is_closed(
    space: TraceSpace, override: dict[str, object], message: str
) -> None:
    space.write(trace=override)
    with pytest.raises(ValueError, match=message):
        with compose_mcp_host(space.load()):
            pass


def test_inspect_store_returns_counts_only(space: TraceSpace) -> None:
    result = space.call("inspect", {"store": "local"})

    assert result["result"] == {
        "kind": "store",
        "store": "local",
        "episode_count": 2,
        "incomplete_episode_count": 1,
        "omitted_restricted_episode_count": 1,
    }
    assert "private full message" not in json.dumps(result)
    assert space.call("inspect", {"store": "other"})["error"]["code"] == "invalid_input"


def test_export_is_idempotent_validated_and_content_free(space: TraceSpace) -> None:
    arguments: dict[str, object] = {
        "store": "local",
        "output_root": "exports",
        "idempotency_key": "nightly",
    }
    first = space.call("export", arguments)
    again = space.call("export", arguments)

    assert first["status"] == "succeeded", first
    package = first["result"]
    assert again["result"] == package
    assert package["included_episode_count"] == 1
    assert package["omitted_restricted_episode_count"] == 1
    assert str(space.root) not in json.dumps(first)
    assert (space.root / "exports").stat().st_mode & 0o077 == 0

    validated = space.call("validate", {"export_id": package["export_id"]})
    assert validated["result"]["valid"] is True
    assert validated["result"]["kind"] == "export"
    inspected = space.call("inspect", {"export_id": package["export_id"]})
    assert inspected["result"]["records_checked"] == validated["result"]["records_checked"]

    conflict = space.call("export", {**arguments, "include_incomplete": True})
    assert conflict["error"]["message"].startswith("idempotency_conflict")
    missing = space.call("validate", {"export_id": "export:" + "0" * 32})
    assert missing["error"]["code"] == "invalid_input"


def test_restricted_and_copy_exports_need_admin_authority(space: TraceSpace) -> None:
    base: dict[str, object] = {"store": "local", "output_root": "exports", "idempotency_key": "k"}
    restricted = space.call("export", {**base, "include_restricted": True})
    copied = space.call("export", {**base, "media_mode": "copy"})

    assert restricted["error"]["code"] == "permission_denied"
    assert copied["error"]["code"] == "permission_denied"
    assert not (space.root / "exports").exists()

    space.write(profile="admin")
    allowed = space.call("export", {**base, "include_restricted": True})
    assert allowed["result"]["included_episode_count"] == 2
    still_denied = space.call("export", {**base, "idempotency_key": "c", "media_mode": "copy"})
    assert still_denied["error"]["code"] == "permission_denied"


def test_export_filters_are_validated(space: TraceSpace) -> None:
    result = space.call(
        "export",
        {
            "store": "local",
            "output_root": "exports",
            "idempotency_key": "f",
            "started_after": "2026-10-01T00:00:00Z",
            "completed_before": "2026-09-01T00:00:00Z",
        },
    )
    assert result["error"]["code"] == "invalid_input"


def test_traces_cli_runtime_config_matches_mcp(
    space: TraceSpace, capsys: pytest.CaptureFixture[str]
) -> None:
    from capicove.cli.app import main

    assert (
        main(["traces", "inspect", "--runtime-config", str(space.config), "--store", "local"]) == 0
    )
    cli = json.loads(capsys.readouterr().out)
    assert cli["result"] == space.call("inspect", {"store": "local"})["result"]

    assert (
        main(
            [
                *("traces", "export", "--runtime-config", str(space.config)),
                *("--store-alias", "local", "--output-root", "exports"),
                *("--idempotency-key", "cli"),
            ]
        )
        == 0
    )
    exported = json.loads(capsys.readouterr().out)["result"]
    assert (
        main(["traces", "validate", "--runtime-config", str(space.config), exported["export_id"]])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["result"]["valid"] is True
    assert main(["traces", "inspect", "--runtime-config", str(space.config)]) == 2
