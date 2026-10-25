"""CLI translation for the canonical video split Tool."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import tempfile
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import BinaryIO

from capicove.contracts import InvocationContext, PrivacyClass, ToolHandler, ToolResult, ToolStatus
from capicove.tools.video_split import VIDEO_SPLIT_SPEC, build_video_split_tool_handler
from capicove.traces import TraceRecorder


def add_parser(parent: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register `capicove video split` arguments."""
    parser = parent.add_parser(
        "split",
        help="Split a video into deterministic segments.",
        description="Split a video by duration, keyframes, or detected scenes.",
    )
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--input", help="Input path, or '-' to read bytes from stdin.")
    sources.add_argument("--artifact-ref", help="Immutable Capicove Artifact reference.")
    parser.add_argument("--artifact-sha256", help="Verified Artifact SHA-256 digest.")
    parser.add_argument(
        "--strategy",
        choices=("duration", "keyframe", "scene"),
        required=True,
    )
    parser.add_argument("--split-duration", type=float, default=60.0)
    parser.add_argument("--overlap-duration", type=float, default=0.0)
    parser.add_argument("--min-last-split-duration", type=float, default=0.0)
    parser.add_argument(
        "--min-keyframe-segment-duration",
        type=float,
        default=10.0,
        help="Merge adjacent GOPs until each keyframe-aligned segment is this long.",
    )
    parser.add_argument(
        "--detector", choices=("content", "adaptive", "threshold"), default="content"
    )
    parser.add_argument(
        "--scene-option",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Pass one allowlisted detector option; repeat for multiple options.",
    )
    parser.add_argument(
        "--output-format",
        choices=("path", "bytes"),
        help="Segment output form; defaults to path with --output-directory, else bytes.",
    )
    parser.add_argument("--output-directory", help="Directory for path output segments.")
    parser.add_argument("--backend", choices=("auto", "ffmpeg", "av"), default="auto")
    parser.add_argument(
        "--write-mode",
        choices=("accurate", "copy"),
        default="accurate",
        help="Use accurate transcoding by default, or opt into faster stream copying.",
    )
    parser.add_argument("--output", choices=("human", "json"), default="human")
    parser.add_argument(
        "--runtime-config",
        type=Path,
        help="Use the version-2 workspace host's video authority (RFC-0021).",
    )


def run(
    arguments: argparse.Namespace,
    *,
    handler: ToolHandler | None = None,
    recorder: TraceRecorder | None = None,
    stdin: BinaryIO | None = None,
) -> int:
    """Translate CLI values, invoke the Tool, and render its canonical result."""
    if getattr(arguments, "runtime_config", None) is not None:
        return _run_configured(arguments, stdin or sys.stdin.buffer)
    if arguments.output_format is None:
        arguments.output_format = "path" if arguments.output_directory is not None else "bytes"
    if arguments.output_format == "bytes" and arguments.output_directory is not None:
        print("--output-directory cannot be used with --output-format bytes", file=sys.stderr)
        return 2
    canonical = _canonical_arguments(arguments, stdin or sys.stdin.buffer)
    if handler is not None:
        return _invoke_and_render(arguments, canonical, handler, recorder=recorder)

    read_roots = _read_roots(arguments)
    if arguments.output_format == "path":
        if arguments.output_directory is None:
            print("--output-directory is required for path output", file=sys.stderr)
            return 2
        output_root = Path(arguments.output_directory).resolve()
        composed = build_video_split_tool_handler(
            "cli-local",
            read_roots,
            (output_root,),
            output_root,
        )
        return _invoke_and_render(arguments, canonical, composed, recorder=recorder)

    with tempfile.TemporaryDirectory(prefix="capicove-video-split-") as temporary:
        output_root = Path(temporary).resolve()
        composed = build_video_split_tool_handler(
            "cli-local",
            read_roots,
            (output_root,),
            output_root,
        )
        return _invoke_and_render(arguments, canonical, composed, recorder=recorder)


def _run_configured(arguments: argparse.Namespace, stdin: BinaryIO) -> int:
    from capicove.cli.configured import invoke_configured

    conflicts = [
        flag
        for flag, value in (
            ("--output-directory", arguments.output_directory),
            ("--artifact-ref", arguments.artifact_ref),
            ("--input -", "-" if arguments.input == "-" else None),
        )
        if value is not None
    ]
    if conflicts:
        print(
            f"{', '.join(conflicts)} cannot be combined with --runtime-config; "
            "the host config owns input roots and output_root",
            file=sys.stderr,
        )
        return 2
    if arguments.output_format is None:
        arguments.output_format = "path"
    try:
        payload = invoke_configured(
            arguments.runtime_config,
            VIDEO_SPLIT_SPEC.tool_id,
            _canonical_arguments(arguments, stdin),
        )
    except (OSError, ValueError, ImportError) as error:
        print(f"Runtime startup failed: {error}", file=sys.stderr)
        return 2
    if arguments.output == "json":
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        _render_human(payload)
    return 0 if payload["status"] == "succeeded" else 1


def _invoke_and_render(
    arguments: argparse.Namespace,
    canonical: Mapping[str, object],
    handler: ToolHandler,
    *,
    recorder: TraceRecorder | None,
) -> int:
    request_id = str(uuid.uuid4())
    context = InvocationContext(
        request_id=request_id,
        trace_id=str(uuid.uuid4()),
        workspace_id="cli-local",
        actor_id="cli-user",
        environment_id="cli-local",
        deadline=datetime.now(UTC) + timedelta(seconds=300),
        permissions=frozenset({"video:read", "artifact:write", "subprocess:execute"}),
        privacy_class=PrivacyClass.CONFIDENTIAL,
    )
    from capicove.cli.tracing import invoke_cli_tool

    result = invoke_cli_tool(
        recorder=recorder,
        spec=VIDEO_SPLIT_SPEC,
        handler=handler,
        context=context,
        arguments=canonical,
    )
    payload = _result_payload(result)
    if arguments.output == "json":
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        _render_human(payload)
    return 0 if result.status is ToolStatus.SUCCEEDED else 1


def _canonical_arguments(arguments: argparse.Namespace, stdin: BinaryIO) -> dict[str, object]:
    if arguments.input == "-":
        source: dict[str, object] = {
            "type": "bytes",
            "base64": base64.b64encode(stdin.read()).decode("ascii"),
        }
    elif arguments.input is not None:
        source = {"type": "path", "path": arguments.input}
    else:
        if not arguments.artifact_sha256:
            raise ValueError("--artifact-sha256 is required with --artifact-ref")
        source = {
            "type": "artifact",
            "ref": arguments.artifact_ref,
            "sha256": arguments.artifact_sha256,
        }
    config: dict[str, object]
    if arguments.strategy == "duration":
        config = {
            "split_duration": arguments.split_duration,
            "overlap_duration": arguments.overlap_duration,
            "min_last_split_duration": arguments.min_last_split_duration,
        }
    elif arguments.strategy == "keyframe":
        config = {"min_segment_duration": arguments.min_keyframe_segment_duration}
    else:
        config = {
            "detector": arguments.detector,
            "options": _scene_options(arguments.scene_option),
        }
    canonical: dict[str, object] = {
        "source": source,
        "strategy": arguments.strategy,
        "config": config,
        "output_format": arguments.output_format,
        "backend": arguments.backend,
        "write_mode": arguments.write_mode,
    }
    if arguments.output_directory is not None:
        canonical["output_directory"] = arguments.output_directory
    return canonical


def _scene_options(values: list[str]) -> dict[str, str | int | float | bool]:
    options: dict[str, str | int | float | bool] = {}
    for item in values:
        name, separator, raw_value = item.partition("=")
        if not separator or not name:
            raise ValueError("scene options use NAME=VALUE")
        options[name] = _parse_scalar(raw_value)
    return options


def _parse_scalar(value: str) -> str | int | float | bool:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _read_roots(arguments: argparse.Namespace) -> tuple[Path, ...]:
    if arguments.input not in (None, "-"):
        return (Path(arguments.input).resolve().parent,)
    return ()


def _result_payload(result: ToolResult) -> dict[str, object]:
    payload: dict[str, object] = {
        "request_id": result.request_id,
        "trace_id": result.trace_id,
        "tool_id": result.tool_id,
        "tool_version": result.tool_version,
        "status": result.status.value,
        "result": dict(result.result) if result.result is not None else None,
        "evidence_refs": list(result.evidence_refs),
        "error": None,
    }
    if result.error is not None:
        payload["error"] = {
            "code": result.error.code.value,
            "message": result.error.message,
            "retryable": result.error.retryable,
        }
    return payload


def _render_human(payload: Mapping[str, object]) -> None:
    result = payload.get("result")
    if not isinstance(result, Mapping):
        error = payload.get("error")
        print(f"video split failed: {error}")
        return
    segments = result.get("segments", [])
    warnings = result.get("warnings", [])
    print(f"status: {payload['status']}")
    print(f"segments: {len(segments) if isinstance(segments, list) else 0}")
    if isinstance(segments, list):
        for segment in segments:
            if not isinstance(segment, Mapping):
                continue
            start = segment.get("start_seconds")
            end = segment.get("end_seconds")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                continue
            destination = segment.get("path") or segment.get("output_id") or "inline-bytes"
            print(
                f"segment {segment.get('index', '?')}: "
                f"{float(start):.3f}-{float(end):.3f}s -> {destination}"
            )
    if isinstance(warnings, list):
        for warning in warnings:
            if isinstance(warning, Mapping):
                print(f"warning: {warning.get('code', 'unknown')}")
