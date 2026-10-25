"""One CLI for native knowledge configuration, ingestion, and cited answers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def add_parser(parent: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the native knowledge workflow without importing model runtimes."""
    parser = parent.add_parser("kb", help="Manage a native evidence-backed knowledge base.")
    commands = parser.add_subparsers(dest="kb_command", required=True)
    init = commands.add_parser("init", help="Create a private offline knowledge workspace.")
    init.add_argument("workspace", type=Path)
    init.add_argument("--source-root", action="append", default=[], metavar="ALIAS=PATH")
    for name in ("models", "generation", "audio"):
        init.add_argument("--" + name, type=Path)
    init.add_argument("--graph", action="store_true")
    init.add_argument("--allow-network", action="store_true")
    for name in ("status", "doctor", "rebuild", "update", "forget", "search", "ask", "evidence"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name in {"update", "forget"}:
            command.add_argument("--root", required=True)
            command.add_argument("path")
        if name in {"search", "ask"}:
            command.add_argument("question")
        if name == "evidence":
            command.add_argument("evidence_id")


def initialize(arguments: argparse.Namespace) -> dict[str, Any]:
    """Create only a new private directory; never replace existing configuration."""
    from capicove.runtime.knowledge_config import KnowledgeConfiguration
    from capicove.runtime.media_handoff import open_private_directory

    root = arguments.workspace.absolute()
    if root.exists() or root.is_symlink():
        raise ValueError("knowledge_workspace_already_exists")
    root.mkdir(mode=0o700)
    descriptor = open_private_directory(root, create=False)
    try:
        os.mkdir("inputs", 0o700, dir_fd=descriptor)
        roots = {"inputs": str(root / "inputs")}
        for selection in arguments.source_root:
            alias, separator, path = selection.partition("=")
            if not separator or alias in roots:
                raise ValueError("invalid_source_root_selection")
            roots[alias] = path
        config = KnowledgeConfiguration.from_mapping(
            {
                "schema_version": 1,
                "workspace_id": "knowledge",
                "storage_root": str(root / "store"),
                "source_roots": roots,
                **{
                    name: str(getattr(arguments, name).absolute())
                    if getattr(arguments, name)
                    else None
                    for name in ("models", "generation", "audio")
                },
                "graph": arguments.graph,
                "network_enabled": arguments.allow_network,
            }
        )
        file = os.open(
            "knowledge.json",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=descriptor,
        )
        with os.fdopen(file, "w") as stream:
            json.dump(config.to_mapping(), stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        from capicove.runtime.knowledge_host import host_configuration
        from capicove.runtime.native_knowledge import KnowledgeBase

        with KnowledgeBase(config):
            pass
        host = host_configuration(config, root / "knowledge.json")
        host_file = os.open(
            "host.json",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=descriptor,
        )
        with os.fdopen(host_file, "w") as stream:
            json.dump(host, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(descriptor)
        return {
            "status": "succeeded",
            "config": str(root / "knowledge.json"),
            "host_config": str(root / "host.json"),
        }
    finally:
        os.close(descriptor)


def run(arguments: argparse.Namespace) -> int:
    """Render bounded JSON outcomes; never print model exceptions or credentials."""
    from capicove.runtime.native_knowledge import KnowledgeBase, KnowledgeConfiguration

    try:
        if arguments.kb_command == "init":
            result = initialize(arguments)
        else:
            config = KnowledgeConfiguration.load(arguments.config.absolute())
            with KnowledgeBase(config) as kb:
                command = arguments.kb_command
                if command in {"status", "doctor"}:
                    result = kb.status()
                    if command == "doctor":
                        from capicove.runtime.knowledge_doctor import doctor

                        result["readiness"] = doctor(config)
                elif command == "rebuild":
                    result = kb.rebuild()
                elif command in {"update", "forget"}:
                    result = getattr(kb, command)(arguments.root, arguments.path)
                elif command in {"search", "ask"}:
                    result = getattr(kb, command)(arguments.question)
                else:
                    result = kb.evidence(arguments.evidence_id)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 1 if result.get("status") in {"failed", "partial"} else 0
    except Exception as error:
        code = "knowledge_operation_failed"
        # Only host-defined simple diagnostic identifiers are safe CLI errors.
        message = str(error)
        if (
            isinstance(error, ValueError)
            and message.isascii()
            and message.replace("_", "").isalnum()
            and len(message) <= 100
        ):
            code = message
        print(json.dumps({"status": "failed", "code": code}))
        return 1
