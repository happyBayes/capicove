"""R6.03-shaped load measurement of the read-only preview endpoint (opt-in).

The server runs in its own process on a real loopback socket; 20 SDK clients from
10 enrolled applications (two users each) issue 10,000 catalog and readiness
calls. This measures the preview profile only: it is not full-capability R6.03
qualification (no mutation, upload, job or provider workloads, no soak).
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import platform
import socket
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


APPLICATIONS = 10
USERS = 2
TOTAL_CALLS = 10_000
PERMISSIONS = ["evidence:read", "evidence:search", "remote:readiness"]


def _private(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload))
    path.chmod(0o600)
    return path


def _configure(tmp_path: Path) -> Path:
    from tests.unit.runtime.test_local_retrieval import retrieval_config_body

    root = (tmp_path / "workspace").resolve()
    notes, index = root / "notes", root / "index"
    notes.mkdir(parents=True)
    root.chmod(0o700)
    (notes / "a.md").write_text("Load qualification note.\n", encoding="utf-8")
    host = retrieval_config_body(root, notes, index)
    host["workspace_id"] = "workspace-1"
    host["environment_id"] = "production"
    workspace = _private(tmp_path / "host.json", host)
    grants = [
        {
            "client_id": f"client-{app}",
            "subject_kind": "human",
            "subject": f"user-{app}-{user}",
            "workspace_binding_id": "workspace-1",
            "grant_id": f"grant-{app}-{user}",
            "epoch": 0,
            "application_permissions": PERMISSIONS,
            "user_permissions": PERMISSIONS,
            "workspace_permissions": PERMISSIONS,
        }
        for app in range(APPLICATIONS)
        for user in range(USERS)
    ]
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
                    {
                        "client_id": f"client-{app}",
                        "tenant_id": "tenant",
                        "application_id": f"app-{app}",
                    }
                    for app in range(APPLICATIONS)
                ],
            },
            "deployment_permissions": PERMISSIONS,
            "grants": grants,
        },
    )
    return _private(
        tmp_path / "launch.json",
        {
            "schema_version": 1,
            "policy_path": str(policy),
            "workspace_config": str(workspace),
            "epoch_store": str(root / "remote" / "epochs.sqlite3"),
            "listen": {"host": "127.0.0.1", "port": 8443},
            "verifier": {
                "kind": "jwks",
                "url": "https://issuer.test/jwks",
                "subject_kind": "human",
            },
            "json_response": True,
            "publish_resource_metadata": False,
        },
    )


def _serve(launch: str, jwk: dict[str, Any], port_file: str) -> None:
    import httpx
    import uvicorn

    from capicove.runtime import remote_launch

    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"keys": [jwk]}))
    app = remote_launch.compose_remote_app(
        remote_launch.load_launch(Path(launch)), identity_transport=transport
    )
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    Path(port_file).write_text(str(listener.getsockname()[1]))
    uvicorn.Server(uvicorn.Config(app, lifespan="on", log_level="error")).run(sockets=[listener])


class _FreshToken:
    """Mint a fresh short-lived token per request, as a real client refreshes."""

    def __init__(self, mint: Callable[[], str]) -> None:
        self._mint = mint

    def __call__(self, request: Any) -> Any:
        request.headers["authorization"] = f"Bearer {self._mint()}"
        return request


async def _client(
    port: int, mint: Callable[[], str], calls: int, samples: dict[str, list[float]]
) -> int:
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    errors = 0
    async with httpx.AsyncClient(
        trust_env=False, headers={"host": "mcp.test"}, auth=_FreshToken(mint), timeout=60
    ) as client:
        endpoint = f"http://127.0.0.1:{port}/mcp"
        async with streamable_http_client(endpoint, http_client=client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                for index in range(calls):
                    started = time.perf_counter()
                    if index % 2 == 0:
                        listed = await session.list_tools()
                        kind = "catalog"
                        failed = len(listed.tools) != 3
                    else:
                        result = await session.call_tool("capicove_remote_readiness", {})
                        kind = "status"
                        failed = result.isError is not False
                    samples[kind].append(time.perf_counter() - started)
                    errors += failed
    return errors


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


@pytest.mark.qualification
def test_preview_endpoint_load_profile(tmp_path: Path, signing_key: Any) -> None:
    import importlib

    jwt = importlib.import_module("jwt")
    from tests.unit.runtime.remote.test_jwt_verifier import token

    public = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(signing_key.public_key()))
    public.update(kid="key-1", alg="RS256", use="sig")
    launch = _configure(tmp_path)
    port_file = tmp_path / "port"
    process = multiprocessing.get_context("spawn").Process(
        target=_serve, args=(str(launch), public, str(port_file)), daemon=True
    )
    process.start()
    try:
        deadline = time.monotonic() + 60
        while not port_file.exists() or not port_file.read_text():
            assert process.is_alive() and time.monotonic() < deadline
            time.sleep(0.05)
        port = int(port_file.read_text())
        while True:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                assert time.monotonic() < deadline
                time.sleep(0.05)
        clients = APPLICATIONS * USERS
        per_client = TOTAL_CALLS // clients

        def minter(app: int, user: int) -> Callable[[], str]:
            return lambda: token(
                signing_key,
                sub=f"user-{app}-{user}",
                client_id=f"client-{app}",
                scope=" ".join(PERMISSIONS),
                exp=int(time.time()) + 120,
            )

        credentials = [minter(app, user) for app in range(APPLICATIONS) for user in range(USERS)]
        samples: dict[str, list[float]] = {"catalog": [], "status": []}

        async def run() -> list[int]:
            return list(
                await asyncio.gather(
                    *(_client(port, credential, per_client, samples) for credential in credentials)
                )
            )

        load_before = os.getloadavg()
        started = time.perf_counter()
        errors = asyncio.run(run())
        elapsed = time.perf_counter() - started
        total = len(samples["catalog"]) + len(samples["status"])
        report: dict[str, Any] = {
            "profile": "read-only preview: catalog + readiness",
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "load_average_before": load_before,
            "load_average_after": os.getloadavg(),
            "applications": APPLICATIONS,
            "concurrent_clients": clients,
            "calls": total,
            "errors": sum(errors),
            "throughput_per_second": total / elapsed,
            "latency_seconds": {
                kind: {
                    "p50": statistics.median(values),
                    "p95": _percentile(values, 0.95),
                    "p99": _percentile(values, 0.99),
                    "count": len(values),
                }
                for kind, values in samples.items()
            },
            "includes": (
                "client SDK processing and per-request RS256 token minting, JSON-RPC, HTTP"
                " and gateway on one host"
            ),
            "not_measured": ["gateway overhead in isolation", "soak", "overload", "mutations"],
        }
        destination = os.environ.get("CAPICOVE_REMOTE_QUALIFICATION_REPORT")
        if destination:
            Path(destination).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, sort_keys=True))
        assert total == TOTAL_CALLS and report["errors"] == 0
        assert report["latency_seconds"]["catalog"]["p95"] < 0.3, report
    finally:
        process.terminate()
        process.join(10)
