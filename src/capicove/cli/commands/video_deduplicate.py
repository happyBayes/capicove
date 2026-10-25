"""CLI for deterministic byte-exact video deduplication."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from capicove.processing.video.deduplication import (
    VideoDeduplicationDependencyError,
    deduplicate_video_files,
    delete_duplicate_files,
    move_duplicate_files,
    write_deduplication_manifest,
)


def add_parser(parent: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register `capicove video deduplicate` arguments."""
    parser = parent.add_parser(
        "deduplicate",
        help="Find byte-identical video files quickly.",
        description="Report exact duplicate videos without modifying files by default.",
    )
    parser.add_argument("inputs", nargs="+", help="Video files or directories to scan.")
    parser.add_argument("--mode", choices=("file",), default="file")
    parser.add_argument("--print-unique", action="store_true")
    parser.add_argument("--print-duplicates", action="store_true")
    parser.add_argument("--write-manifest", type=Path)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--move-duplicates", type=Path)
    actions.add_argument("--delete-duplicates", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--runtime-config",
        type=Path,
        help="Group explicit files with the host's read-only Tool (RFC-0021).",
    )


def run(arguments: argparse.Namespace) -> int:
    """Scan inputs, render decisions, and apply only explicit safe actions."""
    if arguments.runtime_config is not None:
        return _run_configured(arguments)
    if arguments.delete_duplicates and not arguments.force:
        print("--delete-duplicates requires --force", file=sys.stderr)
        return 2
    if arguments.force and not arguments.delete_duplicates:
        print("--force is valid only with --delete-duplicates", file=sys.stderr)
        return 2
    try:
        result = deduplicate_video_files(tuple(Path(value) for value in arguments.inputs))
        action = None
        if arguments.move_duplicates is not None:
            action = move_duplicate_files(result, arguments.move_duplicates)
        elif arguments.delete_duplicates:
            action = delete_duplicate_files(result, force=True)
        if arguments.write_manifest is not None:
            write_deduplication_manifest(arguments.write_manifest, result, action)
    except (OSError, ValueError, VideoDeduplicationDependencyError) as exc:
        print(f"video deduplication failed: {exc}", file=sys.stderr)
        return 1

    if arguments.print_unique:
        for path in result.unique_files:
            print(path)
    if arguments.print_duplicates:
        for group in result.duplicate_groups:
            print(
                json.dumps(
                    {
                        "group_id": group.group_id,
                        "keeper": str(group.keeper),
                        "eliminated": [str(path) for path in group.eliminated],
                        "size_bytes": group.size_bytes,
                        "xxh3_128": group.xxh3_128,
                        "sha256": group.sha256,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
    if not arguments.print_unique and not arguments.print_duplicates:
        print(f"discovered: {result.stats.discovered_files}")
        print(f"unique: {result.stats.unique_files}")
        print(f"duplicates: {result.stats.duplicate_files}")

    failures = list(result.failures)
    if action is not None:
        failures.extend(action.failures)
    for failure in failures:
        print(f"{failure.code}: {failure.path}: {failure.message}", file=sys.stderr)
    return 1 if failures else 0


def _run_configured(arguments: argparse.Namespace) -> int:
    """Group explicit files through the host Tool; scanning and actions stay legacy-only."""
    from capicove.cli.configured import invoke_configured

    legacy = [
        flag
        for flag, used in (
            ("--print-unique", arguments.print_unique),
            ("--print-duplicates", arguments.print_duplicates),
            ("--write-manifest", arguments.write_manifest is not None),
            ("--move-duplicates", arguments.move_duplicates is not None),
            ("--delete-duplicates", arguments.delete_duplicates),
            ("--force", arguments.force),
        )
        if used
    ]
    if legacy:
        print(
            f"{', '.join(legacy)} cannot be combined with --runtime-config; "
            "the host Tool only groups explicit files",
            file=sys.stderr,
        )
        return 2
    paths = [str(Path(value).absolute()) for value in arguments.inputs]
    try:
        payload = invoke_configured(
            arguments.runtime_config, "capicove.video.deduplicate", {"paths": paths}
        )
    except (OSError, ValueError, ImportError) as error:
        print(f"Runtime startup failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "succeeded" else 1
