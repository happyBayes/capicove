"""Operator batch jobs: configured Tool calls, the local worker, and resume.

`plan`, `enqueue`, `status`, and `cancel` invoke the same host bindings as MCP
(RFC-0016 amendment 1). `worker` and `resume` are operator-only commands that
read the same configuration; no MCP client can start them.
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import sys
from pathlib import Path
from threading import Event
from types import FrameType

from capicove.batch.compose import OPERATIONS


_TOOLS = {
    "plan": "capicove.batch.plan",
    "enqueue": "capicove.batch.enqueue",
    "status": "capicove.batch.status",
    "cancel": "capicove.batch.cancel",
}


def add_parser(parent: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register `capicove jobs`."""
    jobs = parent.add_parser("jobs", help="Plan, queue, observe, and run operator batch jobs.")
    commands = jobs.add_subparsers(dest="jobs_command")
    for name, help_text in (
        ("plan", "Plan one allowlisted operation over a configured input root."),
        ("enqueue", "Durably queue a plan for the local worker."),
    ):
        parser = commands.add_parser(name, help=help_text)
        _runtime_config(parser)
        parser.add_argument("--input-root", required=True, help="Configured input-root alias.")
        parser.add_argument("--operation", required=True, choices=OPERATIONS)
        parser.add_argument(
            "--parameters", default="{}", help="Operation parameters as a JSON object."
        )
        parser.add_argument(
            "--provider-fingerprint", required=True, help="Configured provider SHA-256."
        )
        parser.add_argument(
            "--output-root",
            required=name == "enqueue",
            help="Configured output-root alias (required for enqueue and video_split).",
        )
        if name == "enqueue":
            parser.add_argument("--plan-id", required=True)
            parser.add_argument("--idempotency-key", required=True)
    status = commands.add_parser("status", help="Read one job and a page of its items.")
    _runtime_config(status)
    status.add_argument("job_id")
    status.add_argument("--offset", type=int, default=0)
    status.add_argument("--limit", type=int, default=100)
    cancel = commands.add_parser("cancel", help="Request cooperative cancellation.")
    _runtime_config(cancel)
    cancel.add_argument("job_id")
    cancel.add_argument("--expected-revision", type=int, required=True)
    worker = commands.add_parser(
        "worker", help="Run queued jobs until stopped (operator-started local process)."
    )
    _config(worker)
    worker.add_argument(
        "--once", action="store_true", help="Exit when no queued job or expired lease remains."
    )
    resume = commands.add_parser("resume", help="Requeue one interrupted job after verification.")
    _config(resume)
    resume.add_argument("job_id")
    resume.add_argument("--expected-revision", type=int, required=True)


def _runtime_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runtime-config",
        type=Path,
        required=True,
        help="Version-2 workspace host configuration selecting the batch Tool.",
    )


def _config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Version-2 workspace host configuration with capabilities.batch.",
    )


def run(arguments: argparse.Namespace) -> int:
    """Dispatch one `capicove jobs` subcommand."""
    command = arguments.jobs_command
    if command in _TOOLS:
        return _run_tool(arguments)
    if command == "worker":
        return _run_worker(arguments)
    if command == "resume":
        return _run_resume(arguments)
    print(
        "choose a jobs subcommand: plan, enqueue, status, cancel, worker, resume", file=sys.stderr
    )
    return 2


def _run_tool(arguments: argparse.Namespace) -> int:
    from capicove.cli.configured import invoke_configured

    command = arguments.jobs_command
    request: dict[str, object]
    if command in {"plan", "enqueue"}:
        try:
            parameters = json.loads(arguments.parameters)
        except json.JSONDecodeError:
            parameters = None
        if not isinstance(parameters, dict):
            print("--parameters must be a JSON object", file=sys.stderr)
            return 2
        request = {
            "input_root": arguments.input_root,
            "operation": arguments.operation,
            "parameters": parameters,
            "provider_fingerprint": arguments.provider_fingerprint,
        }
        if arguments.output_root is not None:
            request["output_root"] = arguments.output_root
        if command == "enqueue":
            request["plan_id"] = arguments.plan_id
            request["idempotency_key"] = arguments.idempotency_key
    elif command == "status":
        request = {"job_id": arguments.job_id, "offset": arguments.offset, "limit": arguments.limit}
    else:
        request = {"job_id": arguments.job_id, "expected_revision": arguments.expected_revision}
    try:
        payload = invoke_configured(arguments.runtime_config, _TOOLS[command], request)
    except (OSError, ValueError, ImportError) as error:
        print(f"Runtime startup failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "succeeded" else 1


def _load_runtime(path: Path) -> object:
    """Validate the whole selected catalog as MCP startup does, then pin the worker."""
    from capicove.runtime.batch_mcp import worker_runtime
    from capicove.runtime.mcp_config import (
        WorkspaceHostConfiguration,
        parse_mcp_host_config,
        read_mcp_host_config_bytes,
    )
    from capicove.runtime.mcp_host import compose_mcp_host

    config = parse_mcp_host_config(read_mcp_host_config_bytes(path.absolute()))
    if not isinstance(config, WorkspaceHostConfiguration):
        raise ValueError("--config requires a version-2 workspace config")
    with compose_mcp_host(config, catalog_only=True):
        pass
    return worker_runtime(config)


def _run_worker(arguments: argparse.Namespace) -> int:
    from capicove.runtime.batch_mcp import BatchRuntime, run_worker

    try:
        runtime = _load_runtime(arguments.config)
    except (OSError, ValueError, ImportError) as error:
        print(f"Worker startup failed: {error}", file=sys.stderr)
        return 2
    assert isinstance(runtime, BatchRuntime)
    stop = Event()
    received: list[int] = []

    def request_stop(signum: int, frame: FrameType | None) -> None:
        # The first signal finishes the current job; a second one aborts it, and
        # its expired lease is recovered by the next worker.
        received.append(signum)
        if len(received) > 1:
            raise KeyboardInterrupt
        stop.set()

    previous = {
        number: signal.signal(number, request_stop) for number in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        run_worker(runtime, stop, until_idle=arguments.once)
    except KeyboardInterrupt:
        return 130
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    return 0


def _run_resume(arguments: argparse.Namespace) -> int:
    from capicove.batch.job_registry import JobTransitionError
    from capicove.runtime.batch_mcp import BatchRuntime, job_payload, resume_job

    try:
        runtime = _load_runtime(arguments.config)
    except (OSError, ValueError, ImportError) as error:
        print(f"Runtime startup failed: {error}", file=sys.stderr)
        return 2
    assert isinstance(runtime, BatchRuntime)
    if re.fullmatch(r"job:[0-9a-f]{24}", arguments.job_id) is None:
        print("job_id must be job:<24 lowercase hex>", file=sys.stderr)
        return 2
    try:
        record = resume_job(runtime, arguments.job_id, arguments.expected_revision)
    except KeyError:
        print("resume failed: the job is unavailable in this workspace", file=sys.stderr)
        return 1
    except (JobTransitionError, ValueError) as error:
        reason = str(error) if re.fullmatch(r"[a-z_ ]{1,64}", str(error)) else "rejected"
        print(f"resume failed: {reason}", file=sys.stderr)
        return 1
    print(json.dumps({"job": job_payload(runtime, record)}, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = ["add_parser", "run"]
