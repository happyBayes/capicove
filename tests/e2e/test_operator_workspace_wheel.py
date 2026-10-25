"""RFC-0016 amendment 1 acceptance: operator jobs and traces from a clean wheel.

Covers the JOB-001 journey: an MCP client plans and enqueues a job and
disconnects; with no worker the job stays queued and status explains why; the
operator-started worker runs it and CLI status reports success. A metadata-only
trace inspection (part of JOB-003) runs through the same configuration.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from tests.e2e.test_retrieval_wheel import install_wheel, run
from tests.e2e.wheel_cache import WheelCache
from tests.unit.runtime.test_batch_mcp import FINGERPRINT, PNG


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(os.name != "posix", reason="operator host is POSIX-only"),
]
SELECTORS = {
    "input_root": "images",
    "operation": "image_size_filter",
    "parameters": {"max_size": 4096},
    "provider_fingerprint": FINGERPRINT,
}


def config_body(workspace: Path, images: Path, traces: Path) -> dict[str, object]:
    return {
        "schema_version": 2,
        "server_name": "capicove-local",
        "workspace_id": "workspace",
        "actor_id": "owner",
        "environment_id": "local",
        "workspace_root": str(workspace),
        "profile": "write",
        "permissions": ["batch:read", "batch:write", "trace:export", "trace:read"],
        "enabled_tools": [
            "capicove.batch.enqueue",
            "capicove.batch.plan",
            "capicove.batch.status",
            "capicove.trace.inspect",
        ],
        "trace_store": None,
        "capabilities": {
            "batch": {
                "schema_version": 1,
                "input_roots": {"images": str(images)},
                "output_roots": {"results": str(workspace / "results")},
                "registry_root": str(workspace / "jobs"),
                "operations": ["image_size_filter"],
                "provider_fingerprints": {"image_size_filter": FINGERPRINT},
                "max_items": 100,
                "worker": {"concurrency": 2, "lease_seconds": 30, "poll_seconds": 0.5},
            },
            "trace": {
                "schema_version": 1,
                "stores": {"local": str(traces)},
                "export_roots": {"exports": str(workspace / "exports")},
                "media_root": None,
            },
        },
        "network": {"enabled": False, "source_hosts": []},
    }


async def enqueue_over_mcp(python: str, config: Path) -> dict[str, Any]:
    parameters = StdioServerParameters(
        command=python, args=["-m", "capicove", "mcp", "serve", "--config", str(config)]
    )
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(
            reader, writer, read_timeout_seconds=timedelta(seconds=60)
        ) as session:
            await session.initialize()
            listing = await session.list_tools()
            assert sorted(tool.name for tool in listing.tools) == [
                "capicove_batch_enqueue",
                "capicove_batch_plan",
                "capicove_batch_status",
                "capicove_trace_inspect",
            ]
            planned = await session.call_tool("capicove_batch_plan", SELECTORS)
            assert planned.structuredContent is not None
            plan_id = planned.structuredContent["result"]["plan_id"]
            queued = await session.call_tool(
                "capicove_batch_enqueue",
                {
                    **SELECTORS,
                    "output_root": "results",
                    "plan_id": plan_id,
                    "idempotency_key": "wheel",
                },
            )
            assert queued.structuredContent is not None
            return dict(queued.structuredContent)


@pytest.mark.slow
def test_operator_jobs_and_traces_from_installed_wheel(
    tmp_path: Path, wheel_cache: WheelCache
) -> None:
    python, capicove = install_wheel(wheel_cache)
    operator = tmp_path / "operator"
    operator.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    workspace.chmod(0o700)
    images = tmp_path / "images"
    images.mkdir()
    for name in ("a.png", "b.png"):
        (images / name).write_bytes(PNG + name.encode())
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir(mode=0o700)
    traces = trace_dir / "traces.sqlite3"
    subprocess.run(
        [
            python,
            "-c",
            "import sys; from pathlib import Path; "
            "from capicove.traces._adapters.sqlite import SqliteTraceStore; "
            "SqliteTraceStore(Path(sys.argv[1]))",
            str(traces),
        ],
        check=True,
        timeout=60,
    )
    config = workspace / "host.json"
    config.write_text(json.dumps(config_body(workspace, images, traces)))
    config.chmod(0o600)

    queued = asyncio.run(enqueue_over_mcp(python, config))
    job = queued["result"]["job"]
    assert job["state"] == "queued"

    runtime = ("--runtime-config", str(config))
    waiting = run(capicove, operator, "jobs", "status", job["job_id"], *runtime)
    assert waiting["result"]["job"]["state"] == "queued"  # type: ignore[index]
    assert "capicove jobs worker" in waiting["result"]["job"]["diagnostic"]  # type: ignore[index]

    worker = subprocess.run(
        [capicove, "jobs", "worker", "--config", str(config), "--once"],
        cwd=operator,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert worker.returncode == 0, worker.stderr
    assert '"job_finished"' in worker.stderr

    done = run(capicove, operator, "jobs", "status", job["job_id"], *runtime)
    assert done["result"]["job"]["state"] == "succeeded"  # type: ignore[index]
    assert [item["state"] for item in done["result"]["items"]] == ["succeeded"] * 2  # type: ignore[index]

    inspected = run(capicove, operator, "traces", "inspect", *runtime, "--store", "local")
    assert inspected["result"]["episode_count"] == 0  # type: ignore[index]
    assert {path.name for path in images.iterdir()} == {"a.png", "b.png"}
