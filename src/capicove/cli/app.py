"""Argument parsing and dispatch for the Capicove CLI."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from capicove.__about__ import __version__
from capicove.cli.commands.audio_transcribe import add_parsers as add_audio_parsers
from capicove.cli.commands.audio_transcribe import run_capabilities as run_audio_capabilities
from capicove.cli.commands.audio_transcribe import run_transcribe as run_audio_transcribe
from capicove.cli.commands.batch import add_parser as add_batch_parser
from capicove.cli.commands.batch import run as run_batch
from capicove.cli.commands.book import add_parser as add_book_parser
from capicove.cli.commands.book import run as run_book
from capicove.cli.commands.code_search import add_parser as add_code_parser
from capicove.cli.commands.code_search import run as run_code
from capicove.cli.commands.doctor import run as run_doctor
from capicove.cli.commands.evidence import add_parser as add_evidence_parser
from capicove.cli.commands.evidence import run as run_evidence
from capicove.cli.commands.evolution import add_parser as add_evolution_parser
from capicove.cli.commands.evolution import run as run_evolution
from capicove.cli.commands.http import add_parser as add_http_parser
from capicove.cli.commands.http import run as run_http
from capicove.cli.commands.image import add_parser as add_image_parser
from capicove.cli.commands.image import run as run_image
from capicove.cli.commands.jobs import add_parser as add_jobs_parser
from capicove.cli.commands.jobs import run as run_jobs
from capicove.cli.commands.knowledge_base import add_parser as add_kb_parser
from capicove.cli.commands.knowledge_base import run as run_kb
from capicove.cli.commands.local_retrieval import add_parsers as add_local_retrieval_parsers
from capicove.cli.commands.local_retrieval import run_index, run_source
from capicove.cli.commands.mcp import add_parser as add_mcp_parser
from capicove.cli.commands.mcp import run as run_mcp
from capicove.cli.commands.memory_commands import add_parser as add_memory_parser
from capicove.cli.commands.memory_commands import run as run_memory
from capicove.cli.commands.models import add_parser as add_models_parser
from capicove.cli.commands.models import run as run_models
from capicove.cli.commands.search import add_parser as add_search_parser
from capicove.cli.commands.search import run as run_search
from capicove.cli.commands.setup import add_parser as add_setup_parser
from capicove.cli.commands.setup import run as run_setup
from capicove.cli.commands.traces import add_parser as add_traces_parser
from capicove.cli.commands.traces import run_export as run_traces_export
from capicove.cli.commands.traces import run_inspect as run_traces_inspect
from capicove.cli.commands.traces import run_validate as run_traces_validate
from capicove.cli.commands.unified_retrieval import add_parser as add_unified_retrieval_parser
from capicove.cli.commands.unified_retrieval import run as run_unified_retrieval
from capicove.cli.commands.video_deduplicate import add_parser as add_video_deduplicate_parser
from capicove.cli.commands.video_deduplicate import run as run_video_deduplicate
from capicove.cli.commands.video_filter import add_parser as add_video_filter_parser
from capicove.cli.commands.video_filter import run as run_video_filter
from capicove.cli.commands.video_split import add_parser as add_video_split_parser
from capicove.cli.commands.video_split import run as run_video_split
from capicove.cli.commands.wiki import add_parser as add_wiki_parser
from capicove.cli.commands.wiki import run as run_wiki
from capicove.contracts import ToolHandler
from capicove.traces import TraceRecorder, TraceSanitizer, TraceStoreError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="capicove",
        description="Local-first multimodal evidence, search, knowledge, and memory infrastructure.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show the installed Capicove version and exit.",
    )
    parser.add_argument(
        "--trace-store",
        type=Path,
        help="Opt in to recording canonical Tool operations in a local SQLite file.",
    )
    subcommands = parser.add_subparsers(dest="command")
    subcommands.add_parser(
        "doctor",
        help="Report basic installation and runtime diagnostics.",
    )
    audio = subcommands.add_parser("audio", help="Process local audio evidence.")
    audio_subcommands = audio.add_subparsers(dest="audio_command")
    add_audio_parsers(audio_subcommands)
    video = subcommands.add_parser("video", help="Process video evidence.")
    video_subcommands = video.add_subparsers(dest="video_command")
    add_video_deduplicate_parser(video_subcommands)
    add_video_filter_parser(video_subcommands)
    add_video_split_parser(video_subcommands)
    add_image_parser(subcommands)
    add_models_parser(subcommands)
    add_search_parser(subcommands)
    add_evidence_parser(subcommands)
    add_local_retrieval_parsers(subcommands)
    add_unified_retrieval_parser(subcommands)
    add_code_parser(subcommands)
    add_batch_parser(subcommands)
    add_jobs_parser(subcommands)
    add_book_parser(subcommands)
    add_traces_parser(subcommands)
    add_wiki_parser(subcommands)
    add_memory_parser(subcommands)
    add_mcp_parser(subcommands)
    add_http_parser(subcommands)
    add_setup_parser(subcommands)
    add_kb_parser(subcommands)
    add_evolution_parser(subcommands)
    return parser


# Conventional shell status for a process stopped by SIGPIPE (128 + 13).
_BROKEN_PIPE_EXIT_CODE = 141


def main(
    argv: Sequence[str] | None = None,
    *,
    search_handler: ToolHandler | None = None,
    evidence_handler: ToolHandler | None = None,
) -> int:
    """Run the Capicove command-line interface."""
    try:
        status = _dispatch(argv, search_handler=search_handler, evidence_handler=evidence_handler)
        sys.stdout.flush()
    except BrokenPipeError:
        # A downstream reader such as `head` closed stdout; stop quietly like Unix tools.
        # Point stdout at devnull so the interpreter's final flush cannot fail again.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        os.close(devnull)
        return _BROKEN_PIPE_EXIT_CODE
    return status


def _dispatch(
    argv: Sequence[str] | None,
    *,
    search_handler: ToolHandler | None,
    evidence_handler: ToolHandler | None,
) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)

    if arguments.version:
        print(f"capicove {__version__}")
        return 0

    if arguments.command == "mcp":
        return run_mcp(arguments)

    if arguments.command == "evolution":
        return run_evolution(arguments)

    if arguments.command == "kb":
        return run_kb(arguments)

    if arguments.command == "setup":
        return run_setup(arguments)

    if arguments.command == "http":
        return run_http(arguments)

    if arguments.command == "jobs":
        return run_jobs(arguments)

    try:
        recorder = _build_trace_recorder(arguments)
    except TraceStoreError as error:
        # Store messages are stable and path-free by contract, so they are safe to show.
        parser.error(f"trace store could not be initialized: {error}")

    if arguments.command == "memory":
        return run_memory(arguments, recorder=recorder)

    if arguments.command == "doctor":
        return run_doctor()

    if arguments.command == "wiki":
        return run_wiki(arguments)

    if arguments.command == "book":
        return run_book(arguments)

    if arguments.command == "models":
        return run_models(arguments)

    if arguments.command == "source":
        return run_source(arguments)

    if arguments.command == "retrieval":
        return run_unified_retrieval(arguments)

    if arguments.command == "code":
        return run_code(arguments)

    if arguments.command == "index":
        return run_index(arguments)

    if arguments.command == "search":
        if arguments.runtime_config is not None:
            return _run_configured(arguments, "capicove.search", run_search)
        if search_handler is None:
            parser.error("text search runtime is not configured; pass --runtime-config")
        return run_search(arguments, search_handler)

    if arguments.command == "get-evidence":
        if arguments.runtime_config is not None:
            return _run_configured(arguments, "capicove.get_evidence", run_evidence)
        if evidence_handler is None:
            parser.error("evidence runtime is not configured; pass --runtime-config")
        return run_evidence(arguments, evidence_handler)

    if arguments.command == "audio" and arguments.audio_command == "transcribe":
        if recorder is None:
            return run_audio_transcribe(arguments)
        return run_audio_transcribe(arguments, recorder=recorder)

    if arguments.command == "audio" and arguments.audio_command == "import":
        from capicove.cli.commands.audio_transcribe import run_import

        return run_import(arguments)

    if arguments.command == "audio" and arguments.audio_command == "clip":
        from capicove.cli.commands.audio_clip import run as run_audio_clip

        return run_audio_clip(arguments, recorder=recorder)

    if arguments.command == "audio" and arguments.audio_command == "capabilities":
        return run_audio_capabilities(arguments)

    if arguments.command == "video" and arguments.video_command == "split":
        if recorder is None:
            return run_video_split(arguments)
        return run_video_split(arguments, recorder=recorder)

    if arguments.command == "video" and arguments.video_command == "filter":
        if recorder is None:
            return run_video_filter(arguments)
        return run_video_filter(arguments, recorder=recorder)

    if arguments.command == "video" and arguments.video_command == "deduplicate":
        return run_video_deduplicate(arguments)

    if arguments.command == "image" and arguments.image_command == "run":
        if recorder is None:
            return run_image(arguments)
        return run_image(arguments, recorder=recorder)

    if arguments.command == "image" and arguments.image_command in {"search", "serve"}:
        from capicove.cli.commands.image_search import run as run_image_search

        if arguments.trace_store is not None:
            parser.error("image search/serve do not yet support --trace-store")
        return run_image_search(arguments)

    if arguments.command == "batch" and arguments.batch_command == "run":
        try:
            if recorder is None:
                return run_batch(arguments)
            return run_batch(arguments, recorder=recorder)
        except ValueError as error:
            parser.error(str(error))

    if arguments.command == "traces":
        try:
            if arguments.traces_command == "export":
                return run_traces_export(arguments)
            if arguments.traces_command == "validate":
                return run_traces_validate(arguments)
            if arguments.traces_command == "inspect":
                return run_traces_inspect(arguments)
        except ValueError as error:
            parser.error(str(error))

    parser.print_help()
    return 0


def _run_configured(
    arguments: argparse.Namespace,
    tool_id: str,
    runner: Callable[..., int],
) -> int:
    """Run a read command over the same host-composed binding that MCP serves."""
    from capicove.cli.commands.local_retrieval import configured_tool

    try:
        with configured_tool(arguments.runtime_config, tool_id) as (handler, context):
            return runner(arguments, handler, context=context)
    except (OSError, ValueError, ImportError) as error:
        print(f"Runtime startup failed: {error}", file=sys.stderr)
        return 2


def _build_trace_recorder(arguments: argparse.Namespace) -> TraceRecorder | None:
    path = arguments.trace_store
    traced_command = arguments.command == "memory" or (
        (
            arguments.command == "audio"
            and getattr(arguments, "audio_command", None) in {"transcribe", "clip"}
        )
        or (
            arguments.command == "video"
            and getattr(arguments, "video_command", None) in {"filter", "split"}
        )
        or (arguments.command == "image" and getattr(arguments, "image_command", None) == "run")
        or (arguments.command == "batch" and getattr(arguments, "batch_command", None) == "run")
    )
    if path is None or not traced_command:
        return None
    from capicove.traces._adapters.sqlite import SqliteTraceStore

    return TraceRecorder(
        SqliteTraceStore(path.expanduser().resolve()),
        sanitizer=TraceSanitizer(),
        clock=lambda: datetime.now(UTC),
    )
