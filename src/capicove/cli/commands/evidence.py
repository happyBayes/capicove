"""Local CLI evidence translation; the reader still enforces resource access."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from capicove.cli.commands.search import encode_search_tool_result
from capicove.contracts import InvocationContext, ToolHandler, ToolStatus
from capicove.tools.evidence import SECTIONS


def add_parser(subcommands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the closed evidence read command."""
    parser = subcommands.add_parser("get-evidence", help="Read authorized evidence sections.")
    parser.add_argument("evidence_id")
    parser.add_argument("--section", choices=SECTIONS, action="append", default=[])
    from capicove.cli.commands.local_retrieval import add_runtime_config_argument

    add_runtime_config_argument(parser)


def run(
    arguments: argparse.Namespace,
    handler: ToolHandler,
    *,
    context: InvocationContext | None = None,
) -> int:
    """Invoke one reader; a host-composed context carries startup-owned identity."""
    identity = uuid4().hex
    context = context or InvocationContext(
        request_id=f"cli-request-{identity}",
        trace_id=f"cli-trace-{identity}",
        workspace_id="local",
        actor_id="capicove-cli",
        environment_id="local",
        deadline=datetime.now(UTC) + timedelta(seconds=30),
        permissions=frozenset({"evidence:read"}),
    )
    result = handler(
        context,
        {"evidence_id": arguments.evidence_id, "sections": arguments.section or ["metadata"]},
    )
    print(json.dumps(encode_search_tool_result(result), sort_keys=True))
    return 0 if result.status is ToolStatus.SUCCEEDED else 1
