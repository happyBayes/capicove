"""CLI translation for the canonical unified video-filter Tool."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections.abc import Mapping
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from pathlib import Path

from capicove.contracts import InvocationContext, PrivacyClass, ToolHandler, ToolResult, ToolStatus
from capicove.tools.video_filter import VIDEO_FILTER_SPEC, build_video_filter_tool_handler
from capicove.traces import TraceRecorder


def add_parser(parent: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register `capicove video filter` arguments."""
    parser = parent.add_parser(
        "filter",
        help="Evaluate one or more videos with unified local filter criteria.",
        description="Apply metadata, visual, model, tag, and text criteria locally.",
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input video path; repeat to evaluate multiple videos.",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Text paired by position with each input; required for similarity criteria.",
    )
    parser.add_argument(
        "--criteria",
        required=True,
        help="UTF-8 JSON file containing a non-empty criterion array.",
    )
    parser.add_argument("--aggregation", choices=("all", "any"), default="all")
    parser.add_argument("--output", choices=("human", "json"), default="human")
    parser.add_argument("--ocr-provider", choices=("paddleocr", "none"), default="paddleocr")
    parser.add_argument(
        "--model-config", type=Path, help="Offline model factory JSON/TOML configuration."
    )
    parser.add_argument(
        "--model-binding", default="video.ocr", help="Configured image OCR binding."
    )
    parser.add_argument(
        "--ocr-model-root", type=Path, help="Verified local Paddle book-balanced model root."
    )
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
) -> int:
    """Translate CLI values, invoke the Tool, and render its canonical result."""
    try:
        canonical = _canonical_arguments(arguments)
    except (OSError, UnicodeError, json.JSONDecodeError):
        print("criteria must be a readable UTF-8 JSON file", file=sys.stderr)
        return 2
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    if getattr(arguments, "runtime_config", None) is not None:
        if (
            getattr(arguments, "ocr_model_root", None) is not None
            or getattr(arguments, "ocr_provider", "paddleocr") != "paddleocr"
            or getattr(arguments, "model_config", None) is not None
        ):
            print(
                "OCR flags apply to standalone video filtering; host mode uses its configured authority",
                file=sys.stderr,
            )
            return 2
        from capicove.cli.configured import invoke_configured

        try:
            payload = invoke_configured(
                arguments.runtime_config, VIDEO_FILTER_SPEC.tool_id, canonical
            )
        except (OSError, ValueError, ImportError) as error:
            print(f"Runtime startup failed: {error}", file=sys.stderr)
            return 2
        if arguments.output == "json":
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        else:
            _render_human(payload)
        return 0 if payload["status"] == "succeeded" else 1
    from capicove.processing.image.ocr_configuration import ImageOcrConfiguration

    with ExitStack() as stack:
        factory = None
        if getattr(arguments, "model_config", None) is not None:
            from capicove.model import ModelCapability, ModelFactory, load_configuration

            try:
                if (
                    getattr(arguments, "ocr_model_root", None) is not None
                    or getattr(arguments, "ocr_provider", "paddleocr") != "paddleocr"
                ):
                    raise ValueError("model configuration owns OCR provider and model root")
                factory = stack.enter_context(
                    ModelFactory(load_configuration(arguments.model_config))
                )
                factory.profile(arguments.model_binding, ModelCapability.IMAGE_OCR)
            except (OSError, ValueError, ImportError):
                print("invalid local model configuration", file=sys.stderr)
                return 2
        composed = handler or build_video_filter_tool_handler(
            "cli-local",
            _read_roots(arguments.input, arguments.criteria),
            ocr_configuration=ImageOcrConfiguration(
                provider=getattr(arguments, "ocr_provider", "paddleocr"),
                model_root=getattr(arguments, "ocr_model_root", None),
            ),
            model_factory=factory,
            ocr_binding=getattr(arguments, "model_binding", "video.ocr"),
        )
        return _invoke_local(arguments, canonical, composed, recorder)


def _invoke_local(
    arguments: argparse.Namespace,
    canonical: Mapping[str, object],
    composed: ToolHandler,
    recorder: TraceRecorder | None,
) -> int:
    """Invoke while the startup-selected model factory remains alive."""
    request_id = str(uuid.uuid4())
    context = InvocationContext(
        request_id=request_id,
        trace_id=str(uuid.uuid4()),
        workspace_id="cli-local",
        actor_id="cli-user",
        environment_id="cli-local",
        deadline=datetime.now(UTC) + timedelta(seconds=300),
        permissions=frozenset({"video:read", "subprocess:execute"}),
        privacy_class=PrivacyClass.CONFIDENTIAL,
    )
    from capicove.cli.tracing import invoke_cli_tool

    result = invoke_cli_tool(
        recorder=recorder,
        spec=VIDEO_FILTER_SPEC,
        handler=composed,
        context=context,
        arguments=canonical,
    )
    payload = _result_payload(result)
    if arguments.output == "json":
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        _render_human(payload)
    return 0 if result.status is ToolStatus.SUCCEEDED else 1


def _canonical_arguments(arguments: argparse.Namespace) -> dict[str, object]:
    inputs = list(arguments.input)
    texts = list(arguments.text)
    if texts and len(texts) != len(inputs):
        raise ValueError("--text must be omitted or repeated once per --input")
    criteria_value = json.loads(Path(arguments.criteria).read_text(encoding="utf-8"))
    if not isinstance(criteria_value, list) or not criteria_value:
        raise ValueError("criteria JSON must contain a non-empty array")
    if any(not isinstance(value, dict) for value in criteria_value):
        raise ValueError("every criteria JSON entry must be an object")
    sources: list[dict[str, object]] = []
    for index, path in enumerate(inputs, start=1):
        source: dict[str, object] = {
            "item_id": f"video-{index}",
            "source": {"type": "path", "path": path},
        }
        if texts:
            source["text"] = texts[index - 1]
        sources.append(source)
    return {
        "sources": sources,
        "criteria": criteria_value,
        "aggregation": arguments.aggregation,
    }


def _read_roots(inputs: list[str], criteria_path: str) -> tuple[Path, ...]:
    roots = [Path(value).resolve().parent for value in inputs]
    roots.append(Path(criteria_path).resolve().parent)
    return tuple(dict.fromkeys(roots))


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
        print(f"video filter failed: {payload.get('error')}")
        return
    keep = result.get("keep")
    keep_text = "unknown" if keep is None else str(keep).lower()
    print(f"status: {payload['status']}")
    print(f"keep: {keep_text}")
    criteria = result.get("criteria", [])
    if isinstance(criteria, list):
        for criterion in criteria:
            if isinstance(criterion, Mapping):
                print(
                    f"criterion {criterion.get('criterion_id', '?')}: "
                    f"{criterion.get('passed', 'incomplete')}"
                )
    failures = result.get("failures", [])
    if isinstance(failures, list):
        for failure in failures:
            if isinstance(failure, Mapping):
                print(f"failure: {failure.get('code', 'unknown')}")
