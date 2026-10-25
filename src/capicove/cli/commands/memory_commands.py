"""Local memory commands translate bounded JSON requests into canonical Tools."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from capicove.contracts import InvocationContext, ToolStatus
from capicove.runtime.memory_config import ontology, startup_error, validate_config
from capicove.traces import TraceRecorder


_COMMANDS = {
    "remember": "remember_episode",
    "assert-fact": "assert_fact",
    "recall": "recall",
    "get-episode": "get_episode",
    "history": "get_fact_history",
    "retry": "retry_extraction",
    "archive": "archive_memory",
    "erase": "erase_scope",
    "doctor": "doctor",
    "work": "work_extractions",
    "resolve-entity": "resolve_entity",
}


def add_parser(subcommands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subcommands.add_parser(
        "memory", help="Use local temporal memory with explicit runtime configuration."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Host configuration JSON; never passed to a Tool.",
    )
    commands = parser.add_subparsers(dest="memory_command", required=True)
    for name in _COMMANDS:
        child = commands.add_parser(name)
        child.add_argument(
            "--request",
            type=Path,
            required=True,
            help="Closed canonical request JSON, including scope.",
        )


def run(arguments: argparse.Namespace, *, recorder: TraceRecorder | None = None) -> int:
    from capicove.runtime.memory_runtime import build_memory_runtime, close_memory_runtime
    from capicove.tools.memory_tools import (
        MEMORY_SPECS,
        encode_memory_tool_result,
        invoke_memory_tool,
    )

    phase = "config"
    try:
        if (
            arguments.config.stat().st_size > 65536
            or arguments.request.stat().st_size > 2 * 1024 * 1024
        ):
            raise ValueError("oversized input")
        config = validate_config(json.loads(arguments.config.read_text()))
        version, predicates = ontology(config)
        from capicove.runtime.memory_providers import configured_providers

        providers = configured_providers(config)
        handlers = build_memory_runtime(
            Path(config["root"]).expanduser(),
            workspace=config["workspace"],
            scopes=frozenset(config["scopes"]),
            ontology_version=version,
            predicates=predicates,
            backend=config.get("backend", "sqlite"),
            extractor=providers.extractor,
            dense=providers.dense,
        )
        try:
            operation = _COMMANDS[arguments.memory_command]
            phase = "request"
            request = json.loads(arguments.request.read_text())
            if not isinstance(request, dict):
                raise ValueError("invalid request")
            spec = MEMORY_SPECS[operation]
            identity = uuid4().hex
            context = InvocationContext(
                "cli-" + identity,
                "cli-trace-" + identity,
                config["workspace"],
                "capicove-cli",
                "local",
                datetime.now(UTC) + timedelta(seconds=30),
                spec.required_permissions,
            )
            result = invoke_memory_tool(
                recorder=recorder,
                spec=spec,
                handler=handlers[operation],
                context=context,
                arguments=request,
            )
            print(json.dumps(encode_memory_tool_result(result), sort_keys=True, ensure_ascii=False))
            return 0 if result.status is ToolStatus.SUCCEEDED else 1
        finally:
            close_memory_runtime(handlers)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps(startup_error(exc, phase), sort_keys=True))
        return 1
