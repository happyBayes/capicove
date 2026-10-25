"""Operator CLI adapter for the canonical local maintenance handlers."""

from __future__ import annotations

import argparse
import getpass
import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from capicove.contracts import InvocationContext, ToolStatus
from capicove.tools.wiki_maintenance import (
    MaintenanceToolHandler,
    encode_result,
    maintenance_operations,
    maintenance_specs,
)
from capicove.wiki.maintenance.service import MaintenanceService
from capicove.wiki.maintenance.store import read_authorized


def add_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register explicit operations; v3 migration remains separately gated."""
    maintain = commands.add_parser(
        "maintain", help="Operate an explicitly prepared Wiki v3 workspace."
    )
    maintain.add_argument("operation", choices=maintenance_operations())
    maintain.add_argument("workspace", type=Path)
    maintain.add_argument(
        "--request", type=Path, help="Closed JSON operation fields, excluding workspace_id."
    )
    maintain.add_argument("--source-root", action="append", type=Path, default=[])
    ask = commands.add_parser("ask", help="Return original Wiki evidence without generation.")
    ask.add_argument("workspace", type=Path)
    ask.add_argument("question")
    ask.add_argument("--include-history", action="store_true")


def run(arguments: argparse.Namespace) -> int:
    """Bind operator identity and reuse the exact canonical Tool implementation."""
    try:
        operation = "ask" if arguments.wiki_command == "ask" else arguments.operation
        workspace = arguments.workspace.resolve(strict=True)
        service = MaintenanceService(
            workspace, source_roots=tuple([workspace, *getattr(arguments, "source_root", [])])
        )
        if operation == "ask" and arguments.wiki_command == "ask":
            request = {"question": arguments.question, "include_history": arguments.include_history}
        elif arguments.request is None:
            request = {}
        else:
            path = arguments.request.absolute()

            def closed(pairs: list[tuple[str, object]]) -> dict[str, object]:
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError("duplicate request field")
                    result[key] = value
                return result

            request = json.loads(
                read_authorized(path, (path.parent.resolve(),), 1048576), object_pairs_hook=closed
            )
            if not isinstance(request, dict) or "workspace_id" in request:
                raise ValueError("invalid operator request")
        permissions = frozenset(
            permission
            for spec in maintenance_specs(enable_mutations=True)
            for permission in spec.required_permissions
        )
        context = InvocationContext(
            str(uuid.uuid4()),
            str(uuid.uuid4()),
            "local",
            "cli:" + getpass.getuser(),
            "local-cli",
            datetime.now(UTC) + timedelta(seconds=120),
            permissions,
        )
        result = MaintenanceToolHandler(operation, {"local": service}, enable_mutations=True)(
            context, {"workspace_id": "local", **request}
        )
        print(json.dumps(encode_result(result), ensure_ascii=False, sort_keys=True))
        return 0 if result.status == ToolStatus.SUCCEEDED else 2
    except (OSError, ValueError, TypeError, RecursionError):
        print(
            json.dumps(
                {
                    "error": {
                        "code": "wiki_invalid_request",
                        "message": "Invalid maintenance request or workspace.",
                    }
                }
            )
        )
        return 2
