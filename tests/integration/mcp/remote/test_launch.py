"""Remote MCP launch configuration, readiness, catalog and served ASGI app (R2.04)."""

from __future__ import annotations

import importlib
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from tests.unit.runtime.remote.test_jwt_verifier import signing_key, token
from tests.unit.runtime.test_local_retrieval import retrieval_config_body


__all__ = ["signing_key"]
pytestmark = pytest.mark.skipif(os.name != "posix", reason="private POSIX policy loading")


def _private(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload))
    path.chmod(0o600)
    return path


def _setup(tmp_path: Path, **launch_overrides: Any) -> Path:
    root = (tmp_path / "workspace").resolve()
    notes, index = root / "notes", root / "index"
    notes.mkdir(parents=True)
    root.chmod(0o700)
    (notes / "a.md").write_text("Remote readiness token quokka.\n", encoding="utf-8")
    host = retrieval_config_body(root, notes, index)
    host["workspace_id"] = "workspace-1"
    host["environment_id"] = "production"
    workspace = _private(tmp_path / "host.json", host)
    policy = _private(
        tmp_path / "policy.json",
        {
            "schema_version": 1,
            "environment_id": "production",
            "public_url": "https://mcp.test/mcp",
            "identity": {
                "schema_version": 1,
                "issuer": "https://issuer.test",
                "audience": "https://mcp.test/mcp",
                "applications": [
                    {"client_id": "client", "tenant_id": "tenant", "application_id": "app"}
                ],
            },
            "deployment_permissions": ["evidence:read", "evidence:search", "remote:readiness"],
            "grants": [
                {
                    "client_id": "client",
                    "subject_kind": "human",
                    "subject": "alice",
                    "workspace_binding_id": "workspace-1",
                    "grant_id": "grant-1",
                    "epoch": 0,
                    "application_permissions": [
                        "evidence:read",
                        "evidence:search",
                        "remote:readiness",
                    ],
                    "user_permissions": ["evidence:read", "evidence:search", "remote:readiness"],
                    "workspace_permissions": [
                        "evidence:read",
                        "evidence:search",
                        "remote:readiness",
                    ],
                }
            ],
        },
    )
    launch: dict[str, Any] = {
        "schema_version": 1,
        "policy_path": str(policy),
        "workspace_config": str(workspace),
        "epoch_store": str(root / "remote" / "epochs.sqlite3"),
        "listen": {"host": "127.0.0.1", "port": 8443},
        "verifier": {"kind": "jwks", "url": "https://issuer.test/jwks", "subject_kind": "human"},
        "json_response": True,
        "publish_resource_metadata": True,
    }
    launch.update(launch_overrides)
    return _private(tmp_path / "launch.json", launch)


def test_launch_configuration_is_closed_private_and_loopback_only(tmp_path: Path) -> None:
    from capicove.runtime import remote_launch

    good = remote_launch.load_launch(_setup(tmp_path))
    assert good.listen_host == "127.0.0.1" and good.verifier.kind == "jwks"
    overrides: tuple[dict[str, Any], ...] = (
        {"listen": {"host": "0.0.0.0", "port": 8443}},
        {"listen": {"host": "127.0.0.1", "port": 0}},
        {
            "verifier": {
                "kind": "jwks",
                "url": "http://issuer.test/jwks",
                "subject_kind": "human",
            }
        },
        {"verifier": {"kind": "shared-secret", "secret": "x"}},
        {"policy_path": "relative/policy.json"},
        {"extra": True},
    )
    for index, override in enumerate(overrides):
        target = tmp_path / f"case-{index}"
        target.mkdir()
        with pytest.raises(ValueError):
            remote_launch.load_launch(_setup(target, **override))
    (tmp_path / "public").mkdir()
    public = _setup(tmp_path / "public")
    public.chmod(0o644)
    with pytest.raises(ValueError):
        remote_launch.load_launch(public)


def test_remote_check_reports_a_sanitized_catalog_without_network(tmp_path: Path) -> None:
    from capicove.runtime import remote_launch

    report: Any = remote_launch.remote_check(remote_launch.load_launch(_setup(tmp_path)))
    assert report["status"] == "ready"
    assert [tool["name"] for tool in report["catalog"]] == [
        "capicove_get_evidence",
        "capicove_remote_readiness",
        "capicove_search",
    ]
    assert report["grants"] == 1 and report["verifier"] == "jwks"
    serialized = json.dumps(report)
    for private in ("client", "alice", "tenant", str(tmp_path), "issuer.test/jwks"):
        assert private not in serialized
    assert len(report["catalog_sha256"]) == 64


def test_remote_check_rejects_foreign_workspace_grants(tmp_path: Path) -> None:
    from capicove.runtime import remote_launch

    launch = _setup(tmp_path)
    policy_path = Path(json.loads(launch.read_text())["policy_path"])
    policy = json.loads(policy_path.read_text())
    policy["grants"][0]["workspace_binding_id"] = "other-workspace"
    _private(policy_path, policy)
    with pytest.raises(ValueError, match="workspace"):
        remote_launch.remote_check(remote_launch.load_launch(launch))


@pytest.mark.anyio
async def test_served_app_verifies_real_tokens_and_serves_readiness(
    tmp_path: Path, signing_key: Any
) -> None:
    import httpx

    from capicove.runtime import remote_launch

    jwt = importlib.import_module("jwt")
    public = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(signing_key.public_key()))
    public.update(kid="key-1", alg="RS256", use="sig")
    jwks_requests: list[str] = []

    def identity(request: httpx.Request) -> httpx.Response:
        jwks_requests.append(str(request.url))
        return httpx.Response(200, json={"keys": [public]})

    config = remote_launch.load_launch(_setup(tmp_path))
    app = remote_launch.compose_remote_app(config, identity_transport=httpx.MockTransport(identity))
    credential = token(
        signing_key,
        scope="evidence:read evidence:search remote:readiness",
        exp=int(time.time()) + 120,
    )
    headers = {
        "authorization": f"Bearer {credential}",
        "accept": "application/json, text/event-stream",
    }
    async with (
        app.lifespan_context(),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="https://mcp.test", headers=headers
        ) as client,
    ):
        metadata = await client.get("/.well-known/oauth-protected-resource/mcp")
        assert metadata.status_code == 200
        assert metadata.json()["resource"] == "https://mcp.test/mcp"
        initialize = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "launch-test", "version": "1"},
                },
            },
        )
        assert initialize.status_code == 200, initialize.text
        session = {
            "mcp-session-id": initialize.headers["mcp-session-id"],
            "mcp-protocol-version": "2025-11-25",
        }
        notified = await client.post(
            "/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=session
        )
        assert notified.status_code == 202
        listed = await client.post(
            "/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, headers=session
        )
        names = [tool["name"] for tool in listed.json()["result"]["tools"]]
        assert names == ["capicove_get_evidence", "capicove_remote_readiness", "capicove_search"]
        called = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "capicove_remote_readiness", "arguments": {}},
            },
            headers=session,
        )
        result = called.json()["result"]
        assert result["isError"] is False
        readiness = result["structuredContent"]["result"]
        assert readiness["status"] == "ready" and readiness["protocol_version"] == "2025-11-25"
        assert "client" not in json.dumps(readiness)
        unauthenticated = await client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 4, "method": "tools/list"},
            headers={"authorization": "Bearer invalid.token.value", **session},
        )
        assert unauthenticated.status_code == 401
    assert jwks_requests == ["https://issuer.test/jwks"]


def test_cli_remote_check_and_catalog_print_sanitized_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from capicove.cli.app import main

    launch = _setup(tmp_path)
    assert main(["mcp", "remote-check", "--config", str(launch)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ready" and "client" not in json.dumps(report)
    assert main(["mcp", "remote-catalog", "--config", str(launch)]) == 0
    catalog = json.loads(capsys.readouterr().out)
    assert [item["name"] for item in catalog["catalog"]][1] == "capicove_remote_readiness"
    assert main(["mcp", "remote-check", "--config", "relative.json"]) == 2
    launch.chmod(0o644)
    assert main(["mcp", "remote-check", "--config", str(launch)]) == 2
    assert "Remote MCP check failed" in capsys.readouterr().err
