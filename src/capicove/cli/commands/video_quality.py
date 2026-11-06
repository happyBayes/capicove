"""CLI translation for the canonical offline video quality Tool."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from capicove.contracts import InvocationContext, ToolHandler
from capicove.mcp.bindings import encode_result
from capicove.tools.video_quality import VIDEO_QUALITY_SPEC, build_video_quality_tool_handler
from capicove.traces import TraceRecorder


def add_parser(parent: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register inspection options without running a media provider."""
    from capicove.processing.video.quality import CHECKS

    parser = parent.add_parser("quality", help="Inspect offline video media quality with FFmpeg.")
    parser.add_argument("--input", required=True, help="Explicit local MP4/MOV or MKV/WebM file.")
    parser.add_argument("--checks", nargs="+", choices=CHECKS)
    parser.add_argument("--video-stream-index", type=int)
    parser.add_argument("--audio-stream-index", type=int)
    parser.add_argument("--detector-options", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", choices=("human", "json"), default="human")
    parser.add_argument(
        "--runtime-config", type=Path, help="Use a version-2 workspace host configuration."
    )


def _read_object(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        data = stream.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise ValueError("quality configuration exceeds its byte limit")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("quality configuration must contain an object")
    return dict(value)


def run(
    arguments: argparse.Namespace,
    *,
    handler: ToolHandler | None = None,
    recorder: TraceRecorder | None = None,
) -> int:
    """Invoke the canonical handler and preserve policy versus execution outcomes."""
    path = Path(arguments.input).absolute()
    canonical: dict[str, object] = {"path": str(path)}
    try:
        for key in ("checks", "video_stream_index", "audio_stream_index"):
            if (value := getattr(arguments, key)) is not None:
                canonical[key] = value
        for key in ("detector_options", "policy"):
            if (value := getattr(arguments, key)) is not None:
                canonical[key] = _read_object(value)
    except (OSError, ValueError, UnicodeError):
        print("Quality configuration must be a bounded readable JSON object.", file=sys.stderr)
        return 2
    if arguments.runtime_config is not None:
        from capicove.cli.configured import invoke_configured

        try:
            payload = invoke_configured(
                arguments.runtime_config, VIDEO_QUALITY_SPEC.tool_id, canonical
            )
        except (ValueError, OSError, ImportError) as error:
            print(f"Video quality host configuration rejected: {error}", file=sys.stderr)
            return 2
    else:
        composed = handler or build_video_quality_tool_handler("cli-local", (path.parent,))
        context = InvocationContext(
            str(uuid.uuid4()),
            str(uuid.uuid4()),
            "cli-local",
            "cli-user",
            "cli-local",
            datetime.now(UTC) + timedelta(seconds=VIDEO_QUALITY_SPEC.timeout_seconds),
            VIDEO_QUALITY_SPEC.required_permissions,
            VIDEO_QUALITY_SPEC.privacy_class,
        )
        from capicove.cli.tracing import invoke_cli_tool

        result = invoke_cli_tool(
            recorder=recorder,
            spec=VIDEO_QUALITY_SPEC,
            handler=composed,
            context=context,
            arguments=canonical,
        )
        payload = encode_result(result)
    if arguments.output == "json":
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
    else:
        report = payload.get("result")
        print(f"status: {payload['status']}")
        if isinstance(report, dict):
            print(f"verdict: {report['verdict']} (selected checks only; no numerical grade)")
            for check in report["checks"]:
                print(f"{check['check_id']}: {check['status']} / {check['verdict']}")
                for name, measured in check["metrics"].items():
                    print(
                        f"  {name}: {measured['value']} {measured['unit'] or ''} {measured['reason'] or ''}".rstrip()
                    )
                for event in check["events"]:
                    print(
                        f"  review interval: {event['start_seconds']:.3f}-{event['end_seconds']:.3f} s"
                    )
                for finding in check["findings"]:
                    print(f"  {finding['severity']}: {finding['code']} — {finding['message']}")
                if check["unavailable_reason"]:
                    print(f"  unavailable: {check['unavailable_reason']}")
        if payload["error"]:
            print(f"error: {payload['error']}")
    if payload["status"] != "succeeded":
        return 2
    report = payload.get("result")
    return 1 if isinstance(report, dict) and report.get("verdict") == "fail" else 0
