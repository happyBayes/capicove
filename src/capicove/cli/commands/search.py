"""CLI projection helpers for canonical text search."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from capicove.contracts import InvocationContext, ToolHandler, ToolResult, ToolStatus


def add_parser(subcommands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the provider-neutral text search command."""
    parser = subcommands.add_parser("search", help="Search configured text or video evidence.")
    parser.add_argument("query", nargs="?", help="Text query to search for.")
    parser.add_argument("--image", help="Authorized local query image; resolved by the runtime.")
    parser.add_argument(
        "--target", action="append", choices=("keyframe", "video_segment", "text_chunk"), default=[]
    )
    parser.add_argument("--language", help="Language hint for a text query.")
    parser.add_argument("--start-ms", type=int, help="Start of a temporal filter in milliseconds.")
    parser.add_argument("--end-ms", type=int, help="End of a temporal filter in milliseconds.")
    parser.add_argument("--corpus-ref", action="append", default=[])
    parser.add_argument("--artifact-ref", action="append", default=[])
    parser.add_argument("--media-type", action="append", default=[])
    parser.add_argument(
        "--text-origin", action="append", choices=("document", "ocr", "asr"), default=[]
    )
    parser.add_argument("--detail-level", choices=("compact", "full"), default="compact")
    parser.add_argument("--cursor", help="Opaque continuation cursor.")
    parser.add_argument("--top-k", type=int, default=20, help="Maximum number of hits.")
    parser.add_argument(
        "--profile",
        default=None,
        help="Configured retrieval profile reference.",
    )
    from capicove.cli.commands.local_retrieval import add_runtime_config_argument

    add_runtime_config_argument(parser)


def run(
    arguments: argparse.Namespace,
    handler: ToolHandler,
    *,
    context: InvocationContext | None = None,
) -> int:
    """Execute text search through the canonical Tool boundary.

    A host-composed ``context`` carries startup-owned identity and permissions;
    without one, an injected local handler receives the legacy CLI identity.
    """
    now = datetime.now(UTC)
    invocation_id = uuid4().hex
    try:
        payload = arguments_to_search(
            arguments.query,
            top_k=arguments.top_k,
            profile=arguments.profile,
            image=arguments.image,
            targets=tuple(arguments.target),
            language=arguments.language,
            time_range=(arguments.start_ms, arguments.end_ms),
            corpus_refs=tuple(arguments.corpus_ref),
            artifact_refs=tuple(arguments.artifact_ref),
            media_types=tuple(arguments.media_type),
            text_origins=tuple(arguments.text_origin),
        )
        if arguments.detail_level != "compact":
            payload["detail_level"] = arguments.detail_level
        if arguments.cursor is not None:
            payload["cursor"] = arguments.cursor
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    result = handler(
        context
        or InvocationContext(
            request_id=f"cli-request-{invocation_id}",
            trace_id=f"cli-trace-{invocation_id}",
            workspace_id="local",
            actor_id="capicove-cli",
            environment_id="local",
            deadline=now + timedelta(seconds=30),
            permissions=frozenset({"evidence:search"}),
        ),
        payload,
    )
    print(json.dumps(encode_search_tool_result(result), sort_keys=True))
    return 0 if result.status is ToolStatus.SUCCEEDED else 1


def arguments_to_search(
    query: str | None,
    *,
    top_k: int,
    profile: str | None,
    image: str | None = None,
    targets: tuple[str, ...] = (),
    language: str | None = None,
    time_range: tuple[int | None, int | None] = (None, None),
    corpus_refs: tuple[str, ...] = (),
    artifact_refs: tuple[str, ...] = (),
    media_types: tuple[str, ...] = (),
    text_origins: tuple[str, ...] = (),
) -> dict[str, object]:
    """Map CLI values to the canonical Tool argument shape."""
    if (query is None) == (image is None):
        raise ValueError("provide exactly one text query or --image")
    if image is not None and not targets:
        raise ValueError("--image requires at least one --target")
    if image is not None and language is not None:
        raise ValueError("language is not supported for image queries")
    if language is not None and not language.strip():
        raise ValueError("language must not be empty")
    if (time_range[0] is None) != (time_range[1] is None):
        raise ValueError("--start-ms and --end-ms must be provided together")
    value: object = query
    if targets:
        if any(target not in {"keyframe", "video_segment", "text_chunk"} for target in targets):
            raise ValueError("unsupported search target")
        if "text_chunk" in targets and set(targets) != {"text_chunk"}:
            raise ValueError("text_chunk cannot be combined with video targets")
        if image is not None and "text_chunk" in targets:
            raise ValueError("text_chunk requires a text query")
        value = (
            {"type": "text", "text": query, **({"language": language} if language else {})}
            if image is None
            else {"type": "image", "source": {"type": "local_path", "path": image}}
        )
    result: dict[str, object] = {
        "query": value,
        "top_k": top_k,
        "retrieval_profile_ref": profile
        or ("video-local-exact" if targets and set(targets) != {"text_chunk"} else "local-default"),
    }
    if targets:
        result["targets"] = list(dict.fromkeys(targets))
    elif language is not None:
        result["language"] = language
    if time_range[0] is not None and time_range[1] is not None:
        result["time_range"] = {"start_ms": time_range[0], "end_ms": time_range[1]}
    for field, values in (
        ("corpus_refs", corpus_refs),
        ("artifact_refs", artifact_refs),
        ("media_types", media_types),
        ("text_origins", text_origins),
    ):
        if values:
            result[field] = list(dict.fromkeys(values))
    return result


def encode_search_tool_result(result: ToolResult) -> dict[str, object]:
    """Encode one Tool result for JSON CLI output."""
    error = result.error
    return {
        "request_id": result.request_id,
        "trace_id": result.trace_id,
        "tool_id": result.tool_id,
        "tool_version": result.tool_version,
        "status": result.status.value,
        "result": None if result.result is None else dict(result.result),
        "evidence_refs": list(result.evidence_refs),
        "error": None
        if error is None
        else {"code": error.code.value, "message": error.message, "retryable": error.retryable},
    }


__all__ = ["add_parser", "arguments_to_search", "encode_search_tool_result", "run"]
