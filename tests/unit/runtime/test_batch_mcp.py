"""RFC-0016 amendment 1: operator batch jobs in the version-2 workspace host."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event, Thread
from time import monotonic, sleep
from typing import Any

import pytest

from capicove.mcp.bindings import McpToolBinding, invoke_binding
from capicove.runtime.mcp_config import WorkspaceHostConfiguration, load_mcp_host_config
from capicove.runtime.mcp_host import compose_mcp_host


pytestmark = pytest.mark.skipif(os.name != "posix", reason="batch host is POSIX-only")
FINGERPRINT = "f" * 64
ALL_TOOLS = [
    "capicove.batch.cancel",
    "capicove.batch.enqueue",
    "capicove.batch.plan",
    "capicove.batch.status",
]
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"  # pragma: allowlist secret
    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"  # pragma: allowlist secret
)


@dataclass(frozen=True)
class BatchSpace:
    root: Path
    images: Path
    config: Path

    def batch(self, **overrides: object) -> dict[str, object]:
        body: dict[str, object] = {
            "schema_version": 1,
            "input_roots": {"images": str(self.images)},
            "output_roots": {"results": str(self.root / "results")},
            "registry_root": str(self.root / "jobs"),
            "operations": ["image_size_filter"],
            "provider_fingerprints": {"image_size_filter": FINGERPRINT},
            "max_items": 100,
            "worker": {"concurrency": 2, "lease_seconds": 30, "poll_seconds": 0.5},
        }
        body.update(overrides)
        return body

    def body(
        self, batch: dict[str, object] | None = None, **overrides: object
    ) -> dict[str, object]:
        body: dict[str, object] = {
            "schema_version": 2,
            "server_name": "capicove-local",
            "workspace_id": "workspace",
            "actor_id": "owner",
            "environment_id": "local",
            "workspace_root": str(self.root),
            "profile": "write",
            "permissions": ["batch:read", "batch:write"],
            "enabled_tools": ALL_TOOLS,
            "trace_store": None,
            "capabilities": {"batch": batch or self.batch()},
            "network": {"enabled": False, "source_hosts": []},
        }
        body.update(overrides)
        return body

    def write(self, batch: dict[str, object] | None = None, **overrides: object) -> Path:
        self.config.write_text(json.dumps(self.body(batch, **overrides)))
        self.config.chmod(0o600)
        return self.config

    def load(self) -> WorkspaceHostConfiguration:
        config = load_mcp_host_config(self.config)
        assert isinstance(config, WorkspaceHostConfiguration)
        return config


@pytest.fixture
def space(tmp_path: Path) -> BatchSpace:
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    images = tmp_path / "images"
    images.mkdir()
    for name in ("a.png", "b.png", "c.png"):
        (images / name).write_bytes(PNG + name.encode())
    result = BatchSpace(root, images, root / "host.json")
    result.write()
    return result


@pytest.fixture
def bindings(space: BatchSpace) -> Iterator[dict[str, McpToolBinding]]:
    with compose_mcp_host(space.load()) as composed:
        yield {item.spec.tool_id: item for item in composed}


def invoke(binding: McpToolBinding, arguments: Mapping[str, object]) -> dict[str, Any]:
    return invoke_binding(binding, arguments)


def call(
    bindings: dict[str, McpToolBinding], tool: str, arguments: Mapping[str, object]
) -> dict[str, Any]:
    return invoke(bindings[f"capicove.batch.{tool}"], arguments)


def selectors(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "input_root": "images",
        "operation": "image_size_filter",
        "parameters": {"max_size": 4096},
        "provider_fingerprint": FINGERPRINT,
    }
    body.update(overrides)
    return body


def enqueue(
    bindings: dict[str, McpToolBinding], key: str = "first", **overrides: object
) -> dict[str, Any]:
    planned = call(bindings, "plan", selectors(**overrides))
    assert planned["status"] == "succeeded", planned
    return call(
        bindings,
        "enqueue",
        {
            **selectors(**overrides),
            "output_root": "results",
            "plan_id": planned["result"]["plan_id"],
            "idempotency_key": key,
        },
    )


def run_worker_until(space: BatchSpace, done: Any, timeout: float = 30.0) -> list[str]:
    """Run the operator worker in a thread until ``done()`` holds."""
    from capicove.runtime.batch_mcp import run_worker, worker_runtime

    runtime = worker_runtime(space.load())
    runtime.settings = replace(runtime.settings, poll_seconds=0.05)
    stop = Event()
    events: list[str] = []
    thread = Thread(target=run_worker, args=(runtime, stop), kwargs={"report": events.append})
    thread.start()
    try:
        deadline = monotonic() + timeout
        while not done() and monotonic() < deadline:
            sleep(0.02)
    finally:
        stop.set()
        thread.join(timeout=30)
    assert not thread.is_alive()
    return events


# Configuration and catalog


def test_catalog_lists_selected_tools_without_creating_storage(space: BatchSpace) -> None:
    with compose_mcp_host(space.load(), catalog_only=True) as composed:
        assert [item.name for item in composed] == [
            "capicove_batch_cancel",
            "capicove_batch_enqueue",
            "capicove_batch_plan",
            "capicove_batch_status",
        ]
    assert not (space.root / "jobs").exists()
    assert not (space.root / "results").exists()


def test_read_profile_cannot_select_writing_tools(space: BatchSpace) -> None:
    space.write(profile="read", permissions=["batch:read"])
    with pytest.raises(ValueError, match=r"do not authorize|cannot enable a writing Tool"):
        with compose_mcp_host(space.load()):
            pass
    space.write(
        profile="read",
        permissions=["batch:read"],
        enabled_tools=["capicove.batch.plan", "capicove.batch.status"],
    )
    with compose_mcp_host(space.load()) as composed:
        assert len(composed) == 2


def test_missing_permission_fails_startup(space: BatchSpace) -> None:
    space.write(permissions=["batch:read"])
    with pytest.raises(ValueError, match="permissions do not authorize"):
        with compose_mcp_host(space.load()):
            pass


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"extra": 1}, "exactly the documented fields"),
        ({"schema_version": 2}, "version is unavailable"),
        ({"operations": ["image_size_filter", "rm_rf"]}, "allowlist"),
        ({"operations": []}, "allowlist"),
        ({"provider_fingerprints": {}}, "provider_fingerprints"),
        ({"provider_fingerprints": {"image_size_filter": "F" * 64}}, "provider_fingerprints"),
        ({"max_items": 0}, "max_items"),
        ({"max_items": 100_001}, "max_items"),
        ({"worker": {"concurrency": 9, "lease_seconds": 30, "poll_seconds": 1}}, "concurrency"),
        ({"worker": {"concurrency": 1, "lease_seconds": 29, "poll_seconds": 1}}, "lease"),
        ({"worker": {"concurrency": 1, "lease_seconds": 30, "poll_seconds": 0.1}}, "poll"),
        ({"worker": {"concurrency": 1, "lease_seconds": 30}}, "worker requires"),
        ({"input_roots": {}}, "1 to 16"),
        ({"input_roots": {"Bad": "/tmp"}}, "lowercase"),
        ({"output_roots": {"results": "/tmp/elsewhere"}}, "strictly inside workspace_root"),
        ({"registry_root": "relative"}, "absolute"),
        (
            {"operations": ["video_split"], "provider_fingerprints": {"video_split": "f" * 64}},
            "capabilities.video",
        ),
    ],
)
def test_batch_config_is_closed(
    space: BatchSpace, override: dict[str, object], message: str
) -> None:
    space.write(space.batch(**override))
    with pytest.raises(ValueError, match=message):
        with compose_mcp_host(space.load()):
            pass


def test_output_and_registry_roots_must_be_disjoint(space: BatchSpace) -> None:
    space.write(space.batch(registry_root=str(space.root / "results" / "jobs")))
    with pytest.raises(ValueError, match="disjoint"):
        with compose_mcp_host(space.load()):
            pass


def test_permissions_ceilings_follow_the_amendment(space: BatchSpace) -> None:
    space.write(profile="read", permissions=["batch:write"], enabled_tools=["capicove.batch.plan"])
    with pytest.raises(ValueError, match="profile ceiling"):
        space.load()


# Plan and enqueue


def test_plan_is_deterministic_and_has_no_side_effects(
    space: BatchSpace, bindings: dict[str, McpToolBinding]
) -> None:
    first = call(bindings, "plan", selectors())
    second = call(bindings, "plan", selectors())

    assert first["status"] == "succeeded"
    result = first["result"]
    assert result["plan_id"] == second["result"]["plan_id"]
    assert result["plan_id"].startswith("plan:")
    assert result["item_count"] == 3
    assert [item["relative_path"] for item in result["items"]] == ["a.png", "b.png", "c.png"]
    assert result["operation_fingerprint"]["provider_fingerprint"] == FINGERPRINT
    assert result["truncated"] is False
    assert not (space.root / "jobs").exists()
    assert str(space.images) not in json.dumps(first)


@pytest.mark.parametrize(
    ("override", "code"),
    [
        ({"input_root": "elsewhere"}, "invalid_input"),
        ({"operation": "image_shape_filter"}, "policy_rejected"),
        ({"provider_fingerprint": "0" * 64}, "policy_rejected"),
        ({"parameters": {"unknown": 1}}, "invalid_input"),
        ({"output_root": "missing"}, "invalid_input"),
    ],
)
def test_plan_rejects_selectors_outside_startup_authority(
    bindings: dict[str, McpToolBinding], override: dict[str, object], code: str
) -> None:
    result = call(bindings, "plan", selectors(**override))
    assert result["status"] == "failed"
    assert result["error"]["code"] == code


def test_plan_rejects_paths_and_unknown_fields_before_handlers(
    bindings: dict[str, McpToolBinding],
) -> None:
    from capicove.mcp.bindings import InvalidToolArguments

    for arguments in (
        selectors(input_root="/etc"),
        {**selectors(), "path": "/etc"},
        selectors(operation="shell"),
    ):
        with pytest.raises(InvalidToolArguments):
            call(bindings, "plan", arguments)


def test_plan_enforces_max_items(space: BatchSpace) -> None:
    space.write(space.batch(max_items=2))
    with compose_mcp_host(space.load()) as composed:
        result = invoke(
            {item.spec.tool_id: item for item in composed}["capicove.batch.plan"], selectors()
        )
    assert result["error"]["code"] == "policy_rejected"
    assert "max_items" in result["error"]["message"]


def test_enqueue_without_worker_stays_queued_and_explains(
    space: BatchSpace, bindings: dict[str, McpToolBinding]
) -> None:
    queued = enqueue(bindings)

    assert queued["status"] == "succeeded", queued
    job = queued["result"]["job"]
    assert job["job_id"].startswith("job:")
    assert job["state"] == "queued"
    assert job["operation"] == "image_size_filter"
    assert "capicove jobs worker" in job["diagnostic"]
    status = call(bindings, "status", {"job_id": job["job_id"]})
    assert status["result"]["job"] == job
    assert [item["state"] for item in status["result"]["items"]] == ["pending"] * 3
    assert (space.root / "jobs").stat().st_mode & 0o077 == 0
    assert not (space.root / "jobs" / "batch-jobs").exists()


def test_same_key_returns_same_job_and_conflicting_key_is_rejected(
    space: BatchSpace, bindings: dict[str, McpToolBinding]
) -> None:
    first = enqueue(bindings)
    again = enqueue(bindings)
    assert again["result"]["job"]["job_id"] == first["result"]["job"]["job_id"]

    (space.images / "d.png").write_bytes(PNG + b"d")
    conflict = enqueue(bindings)
    assert conflict["status"] == "failed"
    assert conflict["error"]["message"].startswith("idempotency_conflict")


def test_enqueue_rejects_a_changed_plan(
    space: BatchSpace, bindings: dict[str, McpToolBinding]
) -> None:
    planned = call(bindings, "plan", selectors())
    (space.images / "a.png").write_bytes(PNG + b"changed")
    result = call(
        bindings,
        "enqueue",
        {
            **selectors(),
            "output_root": "results",
            "plan_id": planned["result"]["plan_id"],
            "idempotency_key": "k",
        },
    )
    assert result["error"]["message"].startswith("plan_changed")
    assert not (space.root / "jobs" / "jobs.sqlite3").exists()


def test_status_is_scoped_and_paginated(
    space: BatchSpace, bindings: dict[str, McpToolBinding]
) -> None:
    missing = call(bindings, "status", {"job_id": "job:" + "0" * 24})
    assert missing["error"]["code"] == "invalid_input"
    job_id = enqueue(bindings)["result"]["job"]["job_id"]

    page = call(bindings, "status", {"job_id": job_id, "offset": 1, "limit": 1})
    assert [item["relative_path"] for item in page["result"]["items"]] == ["b.png"]
    assert page["result"]["next_offset"] == 2
    last = call(bindings, "status", {"job_id": job_id, "offset": 2, "limit": 1})
    assert last["result"]["next_offset"] is None

    space.write(workspace_id="another-workspace")
    with compose_mcp_host(space.load()) as composed:
        foreign = invoke(
            {item.spec.tool_id: item for item in composed}["capicove.batch.status"],
            {"job_id": job_id},
        )
    assert foreign["error"]["code"] == "invalid_input"


def test_cancel_is_revision_checked(bindings: dict[str, McpToolBinding]) -> None:
    job = enqueue(bindings)["result"]["job"]

    stale = call(bindings, "cancel", {"job_id": job["job_id"], "expected_revision": 9})
    assert stale["error"]["message"].startswith("revision_conflict")
    cancelled = call(bindings, "cancel", {"job_id": job["job_id"], "expected_revision": 1})
    assert cancelled["result"]["job"]["state"] == "cancelled"
    again = call(bindings, "cancel", {"job_id": job["job_id"], "expected_revision": 2})
    assert again["error"]["code"] == "invalid_input"


# Worker


def test_worker_runs_queued_job_after_client_session_ends(space: BatchSpace) -> None:
    with compose_mcp_host(space.load()) as composed:
        job_id = enqueue({item.spec.tool_id: item for item in composed})["result"]["job"]["job_id"]

    def state() -> str:
        with compose_mcp_host(space.load()) as reopened:
            found = invoke(
                {item.spec.tool_id: item for item in reopened}["capicove.batch.status"],
                {"job_id": job_id},
            )
        return str(found["result"]["job"]["state"])

    events = run_worker_until(space, lambda: state() == "succeeded")

    with compose_mcp_host(space.load()) as reopened:
        status = invoke(
            {item.spec.tool_id: item for item in reopened}["capicove.batch.status"],
            {"job_id": job_id},
        )["result"]
    assert status["job"]["state"] == "succeeded"
    assert status["job"]["completed_items"] == 3
    assert status["job"]["terminal_manifest_sha256"]
    assert status["job"]["diagnostic"] is None
    assert {item["state"] for item in status["items"]} == {"succeeded"}
    assert all(item["result"]["keep"] for item in status["items"])
    assert any('"job_finished"' in line for line in events)


def test_worker_skips_jobs_whose_operation_was_removed(space: BatchSpace) -> None:
    with compose_mcp_host(space.load()) as composed:
        bound = {item.spec.tool_id: item for item in composed}
        removed = enqueue(bound, key="removed")["result"]["job"]["job_id"]
    # A provider upgrade changes the fingerprint, so the older job no longer routes.
    space.write(space.batch(provider_fingerprints={"image_size_filter": "e" * 64}))
    with compose_mcp_host(space.load()) as composed:
        bound = {item.spec.tool_id: item for item in composed}
        kept = enqueue(
            bound,
            key="kept",
            provider_fingerprint="e" * 64,
        )["result"]["job"]["job_id"]

        def finished() -> bool:
            found = call(bound, "status", {"job_id": kept})
            return bool(found["result"]["job"]["state"] == "succeeded")

        events = run_worker_until(space, finished)
        assert call(bound, "status", {"job_id": removed})["result"]["job"]["state"] == "queued"
    skipped = [json.loads(line) for line in events if '"job_skipped"' in line]
    assert skipped == [{"event": "job_skipped", "job_id": removed, "reason": "operation_changed"}]


def test_worker_recovers_expired_lease_and_resume_requeues(space: BatchSpace) -> None:
    from capicove.batch._adapters.sqlite import SqliteBatchStore
    from capicove.batch.job_checkpoint_store import JobCheckpointStores
    from capicove.runtime.batch_mcp import resume_job, worker_runtime

    with compose_mcp_host(space.load()) as composed:
        bound = {item.spec.tool_id: item for item in composed}
        job_id = enqueue(bound)["result"]["job"]["job_id"]
    runtime = worker_runtime(space.load())
    registry = runtime.registry()
    assert registry is not None
    record = registry.status(workspace_id="workspace", job_id=job_id)
    service, _ = runtime.route(record)
    past = datetime.now(UTC) - timedelta(minutes=5)
    stores = JobCheckpointStores(SqliteBatchStore(space.root / "jobs" / "batch.sqlite3"))
    service.claim_and_prepare(
        job_id=job_id,
        worker_id="lost-worker",
        now=past,
        lease_until=past + timedelta(seconds=30),
        checkpoint_store=stores.for_job(job_id),
    )

    def interrupted() -> bool:
        return registry.status(workspace_id="workspace", job_id=job_id).state == "interrupted"

    run_worker_until(space, interrupted)
    with compose_mcp_host(space.load()) as composed:
        status = invoke(
            {item.spec.tool_id: item for item in composed}["capicove.batch.status"],
            {"job_id": job_id},
        )["result"]["job"]
    assert status["state"] == "interrupted"
    assert "capicove jobs resume" in status["diagnostic"]

    resumed = resume_job(runtime, job_id, status["revision"])
    assert resumed.state == "queued"
    run_worker_until(
        space,
        lambda: registry.status(workspace_id="workspace", job_id=job_id).state == "succeeded",
    )


def test_resume_rejects_changed_sources(space: BatchSpace) -> None:
    from capicove.batch._adapters.sqlite import SqliteBatchStore
    from capicove.batch.job_checkpoint_store import JobCheckpointStores
    from capicove.runtime.batch_mcp import resume_job, worker_runtime

    with compose_mcp_host(space.load()) as composed:
        job_id = enqueue({item.spec.tool_id: item for item in composed})["result"]["job"]["job_id"]
    runtime = worker_runtime(space.load())
    registry = runtime.registry()
    assert registry is not None
    service, _ = runtime.route(registry.status(workspace_id="workspace", job_id=job_id))
    past = datetime.now(UTC) - timedelta(minutes=5)
    stores = JobCheckpointStores(SqliteBatchStore(space.root / "jobs" / "batch.sqlite3"))
    service.claim_and_prepare(
        job_id=job_id,
        worker_id="lost-worker",
        now=past,
        lease_until=past + timedelta(seconds=30),
        checkpoint_store=stores.for_job(job_id),
    )
    service.recover_expired(
        job_id=job_id, now=datetime.now(UTC), checkpoint_store=stores.for_job(job_id)
    )
    (space.images / "b.png").write_bytes(PNG + b"tampered")
    record = registry.status(workspace_id="workspace", job_id=job_id)

    with pytest.raises(ValueError, match="source_changed"):
        resume_job(runtime, job_id, record.revision)


def test_worker_requires_write_permission(space: BatchSpace) -> None:
    from capicove.runtime.batch_mcp import worker_runtime

    space.write(profile="read", permissions=["batch:read"], enabled_tools=["capicove.batch.status"])
    with pytest.raises(ValueError, match="batch:write"):
        worker_runtime(space.load())


def test_tampered_operation_descriptor_is_not_routed(space: BatchSpace) -> None:
    with compose_mcp_host(space.load()) as composed:
        job_id = enqueue({item.spec.tool_id: item for item in composed})["result"]["job"]["job_id"]
    (descriptor,) = (space.root / "jobs" / "operations").iterdir()
    assert descriptor.stat().st_mode & 0o077 == 0
    body = json.loads(descriptor.read_text())
    body["parameters"]["max_size"] = 1
    descriptor.write_text(json.dumps(body))

    events = run_worker_until(space, lambda: False, timeout=0.5)

    assert {"event": "job_skipped", "job_id": job_id, "reason": "operation_changed"} in [
        json.loads(line) for line in events
    ]


# CLI


ERRORS: list[str] = []


def jobs_cli(
    capsys: pytest.CaptureFixture[str], space: BatchSpace, *argv: str
) -> tuple[int, dict[str, Any] | None]:
    from capicove.cli.app import main

    command, *rest = argv
    flag = "--config" if command in {"worker", "resume"} else "--runtime-config"
    status = main(["jobs", command, flag, str(space.config), *rest])
    captured = capsys.readouterr()
    ERRORS.append(captured.err)
    return status, json.loads(captured.out) if captured.out.strip() else None


def test_cli_matches_mcp_and_worker_once_drains_the_queue(
    space: BatchSpace, bindings: dict[str, McpToolBinding], capsys: pytest.CaptureFixture[str]
) -> None:
    selector_flags = (
        *("--input-root", "images", "--operation", "image_size_filter"),
        *("--parameters", '{"max_size": 4096}', "--provider-fingerprint", FINGERPRINT),
    )
    status, planned = jobs_cli(capsys, space, "plan", *selector_flags)
    assert status == 0 and planned is not None
    assert planned["result"] == call(bindings, "plan", selectors())["result"]

    status, queued = jobs_cli(
        capsys,
        space,
        "enqueue",
        *selector_flags,
        *("--output-root", "results", "--plan-id", planned["result"]["plan_id"]),
        *("--idempotency-key", "cli"),
    )
    assert status == 0 and queued is not None
    job_id = queued["result"]["job"]["job_id"]
    assert call(bindings, "status", {"job_id": job_id})["result"]["job"]["state"] == "queued"

    assert jobs_cli(capsys, space, "worker", "--once")[0] == 0
    status, observed = jobs_cli(capsys, space, "status", job_id, "--limit", "2")
    assert status == 0 and observed is not None
    assert observed["result"]["job"]["state"] == "succeeded"
    assert observed["result"]["next_offset"] == 2

    status, cancelled = jobs_cli(capsys, space, "cancel", job_id, "--expected-revision", "1")
    assert status == 1 and cancelled is not None
    assert cancelled["error"]["code"] == "invalid_input"


def test_cli_reports_startup_and_argument_failures(
    space: BatchSpace, capsys: pytest.CaptureFixture[str]
) -> None:
    status, _ = jobs_cli(
        capsys,
        space,
        "plan",
        *("--input-root", "images", "--operation", "image_size_filter"),
        *("--parameters", "[]", "--provider-fingerprint", FINGERPRINT),
    )
    assert status == 2
    assert "JSON object" in ERRORS[-1]

    assert jobs_cli(capsys, space, "resume", "job:bad", "--expected-revision", "1")[0] == 2
    assert jobs_cli(capsys, space, "resume", "job:" + "0" * 24, "--expected-revision", "1")[0] == 1
    assert "unavailable" in ERRORS[-1]

    space.write(space.batch(operations=["shell"]))
    assert jobs_cli(capsys, space, "worker", "--once")[0] == 2
    assert "Worker startup failed" in ERRORS[-1]


def test_worker_process_stops_cleanly_on_sigterm(space: BatchSpace) -> None:
    import signal
    import subprocess
    import sys

    process = subprocess.Popen(
        [sys.executable, "-m", "capicove", "jobs", "worker", "--config", str(space.config)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stderr is not None
        started = process.stderr.readline()
        assert '"worker_started"' in started
        process.send_signal(signal.SIGTERM)
        _, remaining = process.communicate(timeout=30)
    finally:
        if process.poll() is None:
            process.kill()
    assert process.returncode == 0
    assert '"worker_stopped"' in remaining
    assert not (space.root / "jobs").exists()


def test_setup_doctor_classifies_operator_readiness(
    space: BatchSpace, capsys: pytest.CaptureFixture[str]
) -> None:
    from capicove.cli.app import main

    assert main(["setup", "doctor", "--config", str(space.config)]) == 0
    assert json.loads(capsys.readouterr().out)["ready"] is True

    space.write(space.batch(output_roots={"results": "/tmp/outside-workspace"}))
    assert main(["setup", "doctor", "--config", str(space.config)]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["diagnostics"][0]["code"] == "unsafe_path"

    space.write(
        space.batch(
            operations=["video_filter"], provider_fingerprints={"video_filter": FINGERPRINT}
        )
    )
    assert main(["setup", "doctor", "--config", str(space.config)]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["diagnostics"][0]["code"] == "not_configured"


def test_http_worker_serves_batch_tools_from_the_same_configuration(space: BatchSpace) -> None:
    from capicove.api.workspace_server import WorkerProcess

    worker = WorkerProcess(space.config.read_bytes())
    worker.start()
    try:
        planned = worker.invoke("capicove.batch.plan", selectors(), timeout=30)
        queued = worker.invoke(
            "capicove.batch.enqueue",
            {
                **selectors(),
                "output_root": "results",
                "plan_id": planned["result"]["plan_id"],  # type: ignore[index]
                "idempotency_key": "http",
            },
            timeout=30,
        )
    finally:
        worker.terminate()

    job = queued["result"]["job"]  # type: ignore[index]
    assert job["state"] == "queued"
    with compose_mcp_host(space.load()) as composed:
        bound = {item.spec.tool_id: item for item in composed}
        assert call(bound, "status", {"job_id": job["job_id"]})["result"]["job"] == job


def test_conflicting_operation_descriptor_fails_enqueue_cleanly(
    space: BatchSpace, bindings: dict[str, McpToolBinding]
) -> None:
    enqueue(bindings)
    (descriptor,) = (space.root / "jobs" / "operations").iterdir()
    descriptor.write_text('{"tampered": true}')

    result = enqueue(bindings, key="second")

    assert result["status"] == "failed"
    assert result["error"]["code"] == "unavailable"
