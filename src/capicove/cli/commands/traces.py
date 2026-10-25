"""CLI translation for portable Agent operation trace exports."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from capicove.contracts import ToolStatus
from capicove.export import (
    MediaMode,
    TraceExporter,
    TraceExportError,
    TraceExportExistsError,
    TraceExportOptions,
    TraceValidationReport,
    WorkspaceExportMediaResolver,
    inspect_trace_export,
    validate_trace_export,
)
from capicove.traces import (
    EpisodeStatus,
    TraceModality,
    TraceQuery,
    TraceReader,
    TraceStoreError,
)


_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
_EnumT = TypeVar("_EnumT")


def add_parser(parent: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register `capicove traces export|validate|inspect`."""
    traces = parent.add_parser("traces", help="Export and validate Agent operation traces.")
    commands = traces.add_subparsers(dest="traces_command", required=True)

    export = commands.add_parser("export", help="Export selected complete trace Episodes.")
    _runtime_config(export)
    export.add_argument("--store", type=Path, help="Local trace SQLite file.")
    export.add_argument("--output", type=Path, help="New export directory.")
    export.add_argument("--store-alias", help="Configured trace store (--runtime-config only).")
    export.add_argument("--output-root", help="Configured export root (--runtime-config only).")
    export.add_argument("--idempotency-key", help="Export retry key (--runtime-config only).")
    export.add_argument("--episode-id", action="append", default=[])
    export.add_argument("--started-after", type=_utc_timestamp)
    export.add_argument("--completed-before", type=_utc_timestamp)
    export.add_argument(
        "--modality",
        action="append",
        choices=tuple(item.value for item in TraceModality),
        default=[],
    )
    export.add_argument("--tool-id", action="append", default=[])
    export.add_argument(
        "--tool-status",
        action="append",
        choices=tuple(item.value for item in ToolStatus),
        default=[],
    )
    export.add_argument(
        "--episode-status",
        action="append",
        choices=tuple(item.value for item in EpisodeStatus),
        default=[],
    )
    export.add_argument("--include-incomplete", action="store_true")
    export.add_argument("--include-restricted", action="store_true")
    export.add_argument(
        "--media-mode", choices=tuple(item.value for item in MediaMode), default="reference"
    )
    export.add_argument("--media-root", type=Path)
    export.add_argument("--max-records-per-shard", type=_positive_int, default=100_000)
    export.add_argument("--max-bytes-per-shard", type=_positive_int, default=536_870_912)
    export.add_argument("--max-record-bytes", type=_positive_int, default=16_777_216)
    export.add_argument("--format", choices=("human", "json"), default="human")

    validate = commands.add_parser("validate", help="Validate an exported trace package.")
    _runtime_config(validate)
    validate.add_argument(
        "directory", type=Path, help="Export directory, or an export ID with --runtime-config."
    )
    validate.add_argument("--no-verify-media", action="store_false", dest="verify_media")
    validate.add_argument("--max-record-bytes", type=_positive_int, default=16_777_216)
    validate.add_argument("--output", choices=("human", "json"), default="human")

    inspect = commands.add_parser("inspect", help="Inspect one trace export manifest.")
    _runtime_config(inspect)
    inspect.add_argument(
        "directory",
        type=Path,
        nargs="?",
        help="Export directory, or an export ID with --runtime-config.",
    )
    inspect.add_argument("--store", help="Configured trace store alias (--runtime-config only).")
    inspect.add_argument("--verify-media", action="store_true")
    inspect.add_argument("--output", choices=("human", "json"), default="human")


def _runtime_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runtime-config",
        type=Path,
        help="Run the configured capicove.trace Tool from a version-2 workspace config.",
    )


def run_export(arguments: argparse.Namespace, *, reader: TraceReader | None = None) -> int:
    """Translate CLI filters and produce one new trace export package."""
    if arguments.runtime_config is not None:
        return _configured_export(arguments)
    if arguments.store is None or arguments.output is None:
        raise ValueError("traces export requires --store and --output")
    if any(
        value is not None
        for value in (arguments.store_alias, arguments.output_root, arguments.idempotency_key)
    ):
        raise ValueError(
            "--store-alias, --output-root, and --idempotency-key need --runtime-config"
        )
    mode = MediaMode(arguments.media_mode)
    media_root = arguments.media_root
    if mode is MediaMode.COPY:
        if media_root is None:
            raise ValueError("copy media mode requires --media-root")
        if not media_root.is_absolute():
            raise ValueError("--media-root must be absolute")
    elif media_root is not None:
        raise ValueError("--media-root is only valid with copy media mode")

    query = TraceQuery(
        episode_ids=_optional_strings(arguments.episode_id),
        started_after=arguments.started_after,
        completed_before=arguments.completed_before,
        modalities=_optional_enums(arguments.modality, TraceModality),
        tool_ids=_optional_strings(arguments.tool_id),
        tool_statuses=_optional_enums(arguments.tool_status, ToolStatus),
        episode_statuses=_optional_enums(arguments.episode_status, EpisodeStatus),
        include_incomplete=arguments.include_incomplete,
        include_restricted=arguments.include_restricted,
    )
    options = TraceExportOptions(
        query=query,
        media_mode=mode,
        max_records_per_shard=arguments.max_records_per_shard,
        max_bytes_per_shard=arguments.max_bytes_per_shard,
        max_record_bytes=arguments.max_record_bytes,
    )
    output = arguments.output.resolve()
    try:
        if reader is None:
            from capicove.traces._adapters.sqlite import SqliteTraceStore

            reader = SqliteTraceStore(arguments.store.resolve())
        resolver = (
            WorkspaceExportMediaResolver(media_root)
            if mode is MediaMode.COPY and media_root is not None
            else None
        )
        exporter = TraceExporter(
            reader,
            clock=lambda: datetime.now(UTC),
            export_id_factory=lambda: str(uuid.uuid4()),
            media_resolver=resolver,
        )
        manifest = exporter.export(output, options=options)
    except TraceExportExistsError as error:
        raise ValueError("trace export output already exists") from error
    except TraceStoreError as error:
        raise ValueError("trace export store could not be opened") from error
    except TraceExportError as error:
        raise ValueError("trace export failed") from error
    payload: dict[str, object] = {
        "export_id": manifest.export_id,
        "episodes": manifest.included_episode_count,
        "records": manifest.record_count,
        "shards": len(manifest.shards),
        "media": manifest.media_count,
        "output": str(output),
    }
    _render(payload, arguments.format)
    return 0


def run_validate(arguments: argparse.Namespace) -> int:
    """Validate one local trace export and render its bounded report."""
    if arguments.runtime_config is not None:
        return _configured(
            arguments,
            "capicove.trace.validate",
            {"export_id": str(arguments.directory), "verify_media": arguments.verify_media},
        )
    report = validate_trace_export(
        arguments.directory.absolute(),
        verify_media=arguments.verify_media,
        max_record_bytes=arguments.max_record_bytes,
    )
    _render_validation(report, arguments.output)
    return 0 if report.valid else 1


def run_inspect(arguments: argparse.Namespace) -> int:
    """Render a trace manifest together with an optional media integrity check."""
    if arguments.runtime_config is not None:
        if (arguments.store is None) == (arguments.directory is None) or arguments.verify_media:
            print(
                "--runtime-config inspect takes exactly one of --store ALIAS or an export ID",
                file=sys.stderr,
            )
            return 2
        request: dict[str, object] = (
            {"store": arguments.store}
            if arguments.store is not None
            else {"export_id": str(arguments.directory)}
        )
        return _configured(arguments, "capicove.trace.inspect", request)
    if arguments.directory is None or arguments.store is not None:
        raise ValueError("traces inspect requires an export directory")
    directory = arguments.directory.absolute()
    manifest, report = inspect_trace_export(directory, verify_media=arguments.verify_media)
    payload: dict[str, object] = {
        "export_id": manifest.export_id,
        "created_at": _encode_timestamp(manifest.created_at),
        "format_version": manifest.format_version,
        "schema_version": manifest.schema_version,
        "record_count": manifest.record_count,
        "episode_count": manifest.included_episode_count,
        "shard_count": len(manifest.shards),
        "media_count": manifest.media_count,
        "missing_media_count": manifest.missing_media_count,
        "media_mode": manifest.options.media_mode.value,
        "valid": report.valid,
    }
    _render(payload, arguments.output)
    return 0 if report.valid else 1


def _configured_export(arguments: argparse.Namespace) -> int:
    legacy = [
        flag
        for flag, value in (
            ("--store", arguments.store),
            ("--output", arguments.output),
            ("--media-root", arguments.media_root),
        )
        if value is not None
    ]
    if legacy:
        print(f"{', '.join(legacy)} cannot be combined with --runtime-config", file=sys.stderr)
        return 2
    if None in (arguments.store_alias, arguments.output_root, arguments.idempotency_key):
        print(
            "--runtime-config export requires --store-alias, --output-root, and --idempotency-key",
            file=sys.stderr,
        )
        return 2
    request: dict[str, object] = {
        "store": arguments.store_alias,
        "output_root": arguments.output_root,
        "idempotency_key": arguments.idempotency_key,
        "media_mode": arguments.media_mode,
        "include_incomplete": arguments.include_incomplete,
        "include_restricted": arguments.include_restricted,
    }
    for name in ("episode_id", "modality", "tool_id", "tool_status", "episode_status"):
        values = getattr(arguments, name)
        if values:
            key = "modalities" if name == "modality" else f"{name}s"
            request[key] = sorted(set(values))
    for name in ("started_after", "completed_before"):
        value = getattr(arguments, name)
        if value is not None:
            request[name] = _encode_timestamp(value)
    return _configured(arguments, "capicove.trace.export", request)


def _configured(arguments: argparse.Namespace, tool_id: str, request: dict[str, object]) -> int:
    from capicove.cli.configured import invoke_configured

    try:
        payload = invoke_configured(arguments.runtime_config, tool_id, request)
    except (OSError, ValueError, ImportError) as error:
        print(f"Runtime startup failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "succeeded" else 1


def _render_validation(report: TraceValidationReport, output: str) -> None:
    payload: dict[str, object] = {
        "valid": report.valid,
        "records_checked": report.records_checked,
        "episodes_checked": report.episodes_checked,
        "media_checked": report.media_checked,
        "issues": [
            {
                "severity": issue.severity.value,
                "code": issue.code,
                "message": issue.message,
                "shard": issue.shard.as_posix() if issue.shard is not None else None,
                "line": issue.line,
                "episode_id": issue.episode_id,
                "record_id": issue.record_id,
            }
            for issue in report.issues
        ],
    }
    _render(payload, output)


def _render(payload: Mapping[str, object], output: str) -> None:
    if output == "json":
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        elif isinstance(value, bool):
            rendered = str(value).lower()
        else:
            rendered = str(value)
        print(f"{key}: {rendered}")


def _utc_timestamp(value: str) -> datetime:
    try:
        return datetime.strptime(value, _TIMESTAMP_FORMAT).replace(tzinfo=UTC)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "timestamp must use YYYY-MM-DDTHH:MM:SS.ffffffZ"
        ) from error


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _optional_strings(values: list[str]) -> frozenset[str] | None:
    return frozenset(values) if values else None


def _optional_enums(
    values: list[str], enum_type: Callable[[str], _EnumT]
) -> frozenset[_EnumT] | None:
    return frozenset(enum_type(value) for value in values) if values else None


def _encode_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime(_TIMESTAMP_FORMAT)
