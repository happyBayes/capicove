"""Remote artifact and operation Tools through the composed authenticated endpoint."""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from tests.integration.mcp.remote.test_launch import _private, _setup
from tests.unit.runtime.remote.test_jwt_verifier import signing_key, token


__all__ = ["signing_key"]
pytestmark = pytest.mark.skipif(os.name != "posix", reason="private POSIX policy loading")
CONTROL = ["artifact:write", "artifact:read", "artifact:delete", "operation:submit"]
READ = ["evidence:read", "evidence:search", "remote:readiness"]


def _configure(tmp_path: Path, *, control: bool = True) -> Path:
    launch = _setup(tmp_path, **({"control_store": str(tmp_path / "control")} if control else {}))
    policy_path = tmp_path / "policy.json"
    policy = json.loads(policy_path.read_text())
    policy["deployment_permissions"] = READ + CONTROL
    alice = policy["grants"][0]
    for tier in ("application_permissions", "user_permissions", "workspace_permissions"):
        alice[tier] = READ + CONTROL
    bob = {**alice, "subject": "bob", "grant_id": "grant-2"}
    policy["grants"] = [alice, bob]
    _private(policy_path, policy)
    return launch


class Session:
    def __init__(self, client: Any, credential: str) -> None:
        self.client = client
        self.headers = {
            "authorization": f"Bearer {credential}",
            "accept": "application/json, text/event-stream",
        }
        self.sequence = 0

    async def open(self) -> None:
        response = await self.client.post(
            "/mcp",
            headers=self.headers,
            json={
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "control-test", "version": "1"},
                },
            },
        )
        assert response.status_code == 200, response.text
        self.headers.update(
            {
                "mcp-session-id": response.headers["mcp-session-id"],
                "mcp-protocol-version": "2025-11-25",
            }
        )
        notified = await self.client.post(
            "/mcp",
            headers=self.headers,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        assert notified.status_code == 202

    async def tools(self) -> list[str]:
        self.sequence += 1
        response = await self.client.post(
            "/mcp",
            headers=self.headers,
            json={"jsonrpc": "2.0", "id": self.sequence, "method": "tools/list"},
        )
        return sorted(tool["name"] for tool in response.json()["result"]["tools"])

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.sequence += 1
        response = await self.client.post(
            "/mcp",
            headers=self.headers,
            json={
                "jsonrpc": "2.0",
                "id": self.sequence,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
        )
        assert response.status_code == 200, response.text
        result = response.json()["result"]
        return dict(result["structuredContent"])


def _credential(key: Any, subject: str, scopes: list[str]) -> str:
    return token(key, sub=subject, scope=" ".join(scopes), exp=int(time.time()) + 120)


@pytest.mark.anyio
async def test_upload_commit_read_and_owner_isolation_over_remote_mcp(
    tmp_path: Path, signing_key: Any
) -> None:
    import httpx

    from capicove.runtime import remote_launch
    from capicove.runtime.remote_epochs import EpochStore

    jwt = importlib.import_module("jwt")
    public = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(signing_key.public_key()))
    public.update(kid="key-1", alg="RS256", use="sig")
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"keys": [public]}))
    config = remote_launch.load_launch(_configure(tmp_path))
    app = remote_launch.compose_remote_app(config, identity_transport=transport)
    data = os.urandom(300_000)
    async with (
        app.lifespan_context(),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="https://mcp.test"
        ) as client,
    ):
        alice = Session(client, _credential(signing_key, "alice", READ + CONTROL))
        await alice.open()
        names = await alice.tools()
        assert "capicove_artifact_begin_upload" in names and "capicove_operation_cancel" in names
        begun = await alice.call(
            "capicove_artifact_begin_upload",
            {
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "media_type": "application/octet-stream",
                "idempotency_key": "remote-1",
            },
        )
        upload_id = begun["result"]["upload_id"]
        for index, start in enumerate(range(0, len(data), 262_144)):
            written = await alice.call(
                "capicove_artifact_write_chunk",
                {
                    "upload_id": upload_id,
                    "index": index,
                    "data_base64": base64.b64encode(data[start : start + 262_144]).decode(),
                },
            )
            assert written["status"] == "succeeded", written
        committed = await alice.call("capicove_artifact_commit_upload", {"upload_id": upload_id})
        reference = committed["result"]
        assert reference["sha256"] == hashlib.sha256(data).hexdigest()
        assert str(tmp_path) not in json.dumps(committed)
        chunks = []
        for offset in range(0, len(data), 262_144):
            read = await alice.call(
                "capicove_artifact_read_chunk",
                {"artifact_id": reference["artifact_id"], "offset": offset, "length": 262_144},
            )
            chunks.append(base64.b64decode(read["result"]["data_base64"]))
        assert b"".join(chunks) == data

        bob = Session(client, _credential(signing_key, "bob", READ + CONTROL))
        await bob.open()
        foreign = await bob.call(
            "capicove_artifact_stat", {"artifact_id": reference["artifact_id"]}
        )
        assert foreign["status"] == "rejected"
        assert foreign["error"]["message"].endswith("not_found.")

        reader = Session(client, _credential(signing_key, "alice", READ))
        await reader.open()
        assert not any(name.startswith("capicove_artifact") for name in await reader.tools())

        EpochStore(config.epoch_store).revoke("workspace-1")
        revoked = await client.post(
            "/mcp",
            headers=alice.headers,
            json={
                "jsonrpc": "2.0",
                "id": 99,
                "method": "tools/call",
                "params": {
                    "name": "capicove_artifact_stat",
                    "arguments": {"artifact_id": reference["artifact_id"]},
                },
            },
        )
        # A revoked grant is refused at session admission, before any store call.
        assert revoked.status_code == 403


def test_without_control_store_the_endpoint_stays_read_only(tmp_path: Path) -> None:
    from capicove.runtime import remote_launch

    report: Any = remote_launch.remote_check(
        remote_launch.load_launch(_configure(tmp_path, control=False))
    )
    assert report["control_plane"] is False
    assert not any(entry["name"].startswith("capicove_artifact") for entry in report["catalog"])
    with pytest.raises(ValueError):
        remote_launch.decode_launch(
            json.dumps(
                {**json.loads((tmp_path / "launch.json").read_text()), "control_store": "rel"}
            )
        )
