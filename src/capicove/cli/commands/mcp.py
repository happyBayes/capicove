"""Operator entry points for a configured local MCP stdio process."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def add_parser(parent: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register MCP commands without importing any optional SDK."""
    parser = parent.add_parser(
        "mcp", help="Serve explicitly configured Tools over local MCP stdio."
    )
    commands = parser.add_subparsers(dest="mcp_command", required=True)
    for name, text in (
        ("remote-check", "Validate a remote endpoint launch without listening or network calls."),
        ("remote-catalog", "Print the remote endpoint's sanitized read-Tool catalog."),
        ("serve-http", "Serve the remote endpoint on its loopback listener."),
    ):
        remote = commands.add_parser(name, help=text)
        remote.add_argument(
            "--config", required=True, type=Path, help="Absolute private remote launch JSON."
        )
    for name in ("serve", "catalog"):
        command = commands.add_parser(name, help=f"{name.capitalize()} configured MCP Tools.")
        command.add_argument(
            "--config",
            required=True,
            type=Path,
            help="Versioned host JSON file; serve requires an absolute path.",
        )


def _run_remote(arguments: argparse.Namespace) -> int:
    """Validate, list or serve the authenticated remote endpoint."""
    from capicove.runtime import remote_launch

    try:
        if not arguments.config.is_absolute():
            raise ValueError("remote launch requires an absolute --config path")
        launch = remote_launch.load_launch(arguments.config)
        if arguments.mcp_command == "serve-http":
            remote_launch.remote_check(launch)
            remote_launch.serve(launch)
            return 0
        report = remote_launch.remote_check(launch)
    except KeyboardInterrupt:
        return 130
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        print(f"Remote MCP check failed: {error}", file=sys.stderr)
        return 2
    payload = report if arguments.mcp_command == "remote-check" else {"catalog": report["catalog"]}
    print(json.dumps(payload, sort_keys=True))
    return 0


def run(arguments: argparse.Namespace) -> int:
    """Keep startup diagnostics on stderr and reserve stdout for JSON or MCP."""
    if arguments.mcp_command in {"remote-check", "remote-catalog", "serve-http"}:
        return _run_remote(arguments)
    from capicove.mcp.bindings import catalog_payload as build_catalog_payload
    from capicove.mcp.server import create_bound_mcp_server
    from capicove.mcp.stdio import protocol_stdout, run_stdio
    from capicove.runtime.mcp_config import load_mcp_host_config
    from capicove.runtime.mcp_host import compose_mcp_host

    try:
        if arguments.trace_store is not None:
            raise ValueError(
                "configure trace_store in the MCP host JSON, not the global CLI option"
            )
        config_path: Path = arguments.config
        if not config_path.is_absolute():
            if arguments.mcp_command == "serve":
                # MCP clients may launch the server from any working directory.
                raise ValueError(
                    f"mcp serve requires an absolute --config path; use {config_path.absolute()}"
                )
            config_path = config_path.absolute()
        config = load_mcp_host_config(config_path)
        with protocol_stdout() as protocol:
            if arguments.mcp_command == "serve" and importlib.util.find_spec("mcp") is None:
                raise RuntimeError("install capicove[mcp] to create an MCP server")
            try:
                catalog_payload: dict[str, object] | None = None
                with compose_mcp_host(
                    config, catalog_only=arguments.mcp_command == "catalog"
                ) as bindings:
                    if arguments.mcp_command == "serve":
                        server = create_bound_mcp_server(bindings, server_name=config.server_name)
                        run_stdio(server, stdout=protocol)
                    else:
                        payload = build_catalog_payload(config, bindings)
                        catalog_payload = payload
                if catalog_payload is not None:
                    print(json.dumps(catalog_payload, sort_keys=True), file=protocol)
            except Exception:
                print(
                    "MCP startup failed: selected local Tools could not start; "
                    "run capicove setup doctor with this config",
                    file=sys.stderr,
                )
                return 2
    except (ValueError, RuntimeError) as error:
        print(f"MCP startup failed: {error}", file=sys.stderr)
        return 2
    except OSError:
        print("MCP startup failed: configured local resources could not be opened", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    return 0
