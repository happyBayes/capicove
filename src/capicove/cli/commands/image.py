"""CLI translation for canonical image Tools."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from capicove.contracts import InvocationContext, PrivacyClass, ToolHandler, ToolResult, ToolStatus
from capicove.traces import TraceRecorder


_IMAGE_TOOL_NAMES = (
    "image_remove_background_mapper",
    "imgdiff_difference_area_generator_mapper",
    "imgdiff_difference_caption_generator_mapper",
    "image_aesthetics_filter",
    "image_aspect_ratio_filter",
    "image_nsfw_filter",
    "image_shape_filter",
    "image_size_filter",
    "image_subplot_filter",
    "image_text_matching_filter",
    "image_text_similarity_filter",
    "image_watermark_filter",
    "image_pair_similarity_filter",
    "image_phash",
    "image_lsh_search",
    "image_perceptual_deduplicator",
    "ray_image_perceptual_deduplicator",
    "image_ocr",
)
# Tools the CLI composes without a configured provider; keep in sync with
# ``_build_default_handler``.
_DEFAULT_IMAGE_TOOL_NAMES = (
    "image_aspect_ratio_filter",
    "image_shape_filter",
    "image_size_filter",
    "image_subplot_filter",
    "image_phash",
    "image_lsh_search",
)
_PROVIDER_IMAGE_TOOL_NAMES = tuple(
    name for name in _IMAGE_TOOL_NAMES if name not in _DEFAULT_IMAGE_TOOL_NAMES
)


def add_parser(parent: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register `capicove image run`."""
    image = parent.add_parser("image", help="Process local image evidence.")
    commands = image.add_subparsers(dest="image_command")
    from capicove.cli.commands.image_search import add_parsers

    add_parsers(commands)
    run_parser = commands.add_parser("run", help="Invoke one canonical image Tool.")
    run_parser.add_argument(
        "--tool",
        choices=_IMAGE_TOOL_NAMES,
        required=True,
        metavar="TOOL",
        help=(
            f"Ready without configuration: {', '.join(_DEFAULT_IMAGE_TOOL_NAMES)}. "
            f"Other choices ({', '.join(_PROVIDER_IMAGE_TOOL_NAMES)}) "
            "need a provider configured through the Python API or MCP."
        ),
    )
    run_parser.add_argument("--input", action="append", default=[])
    run_parser.add_argument(
        "--arguments",
        type=Path,
        help="Path to a UTF-8 JSON file holding canonical Tool arguments.",
    )
    run_parser.add_argument("--output", choices=("human", "json"), default="human")


def run(
    arguments: argparse.Namespace,
    *,
    handler: ToolHandler | None = None,
    recorder: TraceRecorder | None = None,
) -> int:
    """Invoke one image Tool and render its canonical result."""
    from capicove.tools.image import IMAGE_TOOL_SPECS_BY_NAME

    if not arguments.tool:
        raise ValueError("--tool is required")
    try:
        canonical = _read_arguments(arguments)
    except (OSError, UnicodeError, json.JSONDecodeError):
        print("--arguments must name a readable UTF-8 JSON file", file=sys.stderr)
        return 2
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    if handler is None and arguments.tool not in _DEFAULT_IMAGE_TOOL_NAMES:
        print(
            f"{arguments.tool} requires an explicitly configured local provider; "
            "use the Python API or MCP",
            file=sys.stderr,
        )
        return 2
    composed = handler or _build_default_handler(
        arguments.tool,
        arguments.input,
        canonical,
    )
    spec = IMAGE_TOOL_SPECS_BY_NAME[arguments.tool]
    context = InvocationContext(
        request_id=str(uuid.uuid4()),
        trace_id=str(uuid.uuid4()),
        workspace_id="cli-local",
        actor_id="cli-user",
        environment_id="cli-local",
        deadline=datetime.now(UTC) + timedelta(seconds=spec.timeout_seconds),
        permissions=spec.required_permissions,
        privacy_class=PrivacyClass.CONFIDENTIAL,
    )
    from capicove.cli.tracing import invoke_cli_tool

    result = invoke_cli_tool(
        recorder=recorder,
        spec=spec,
        handler=composed,
        context=context,
        arguments=canonical,
    )
    payload = _encode(result)
    if arguments.output == "json":
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(f"status: {payload['status']}")
        print(json.dumps(payload["result"], ensure_ascii=False, sort_keys=True))
    return 0 if result.status is ToolStatus.SUCCEEDED else 1


def _read_arguments(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.arguments is not None:
        value = json.loads(arguments.arguments.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("image arguments JSON must contain an object")
        return value
    if not arguments.input:
        raise ValueError("at least one --input or --arguments is required")
    return {"sources": [{"path": path} for path in arguments.input]}


def _build_default_handler(
    tool: str,
    inputs: list[str],
    canonical: Mapping[str, object],
) -> ToolHandler:
    from capicove.processing.image import (
        ImageAspectRatioFilter,
        ImageLshSearch,
        ImagePhash,
        ImageShapeFilter,
        ImageSizeFilter,
        ImageSubplotFilter,
    )
    from capicove.tools.image import build_image_tool_handler

    roots = tuple(
        dict.fromkeys(
            Path(path).resolve().parent for path in (*inputs, *_canonical_image_paths(canonical))
        )
    )
    capabilities: dict[str, object] = {
        "image_aspect_ratio_filter": ImageAspectRatioFilter(),
        "image_shape_filter": ImageShapeFilter(),
        "image_size_filter": ImageSizeFilter(),
        "image_subplot_filter": ImageSubplotFilter(),
        "image_phash": ImagePhash(),
        "image_lsh_search": ImageLshSearch(),
    }
    if tool not in capabilities:
        raise ValueError(
            f"{tool} requires an explicitly configured local provider; use the Python API or MCP"
        )
    return build_image_tool_handler(tool, capabilities[tool], read_roots=roots)


def _canonical_image_paths(arguments: Mapping[str, object]) -> tuple[str, ...]:
    paths: list[str] = []
    raw_sources: list[object] = []
    sources = arguments.get("sources")
    if isinstance(sources, Sequence) and not isinstance(sources, (str, bytes, bytearray)):
        raw_sources.extend(sources)
    records = arguments.get("records")
    if isinstance(records, Sequence) and not isinstance(records, (str, bytes, bytearray)):
        for raw_record in records:
            if not isinstance(raw_record, Mapping):
                continue
            images = raw_record.get("images")
            if isinstance(images, Sequence) and not isinstance(
                images,
                (str, bytes, bytearray),
            ):
                raw_sources.extend(images)
    for raw_source in raw_sources:
        if isinstance(raw_source, Mapping) and isinstance(raw_source.get("path"), str):
            paths.append(raw_source["path"])
    return tuple(paths)


def _encode(result: ToolResult) -> dict[str, object]:
    return {
        "request_id": result.request_id,
        "trace_id": result.trace_id,
        "tool_id": result.tool_id,
        "tool_version": result.tool_version,
        "status": result.status.value,
        "result": dict(result.result) if result.result is not None else None,
        "evidence_refs": list(result.evidence_refs),
        "error": (
            {"code": result.error.code.value, "message": result.error.message}
            if result.error is not None
            else None
        ),
    }
