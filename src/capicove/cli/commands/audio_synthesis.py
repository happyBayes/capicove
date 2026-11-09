"""Thin CLI translation for shared synthesis Tools."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from capicove.cli.tracing import invoke_cli_tool
from capicove.contracts import InvocationContext
from capicove.mcp.bindings import encode_result
from capicove.runtime.speech import SpeechRuntime
from capicove.traces import TraceRecorder


def add_parsers(parent: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register short local speech generation and reference conditioning."""
    doctor = parent.add_parser(
        "speech-capabilities", help="Verify local speech assets without loading a model."
    )
    doctor.add_argument("--runtime-config", type=Path, required=True)
    doctor.add_argument("--signing-key-file", type=Path)
    for name in ("synthesize", "clone"):
        parser = parent.add_parser(
            name, help="Generate a local WAV using configured speech assets."
        )
        parser.add_argument("--runtime-config", type=Path, required=True)
        parser.add_argument("--signing-key-file", type=Path)
        parser.add_argument("--text", required=True)
        parser.add_argument("--language", default="zh")
        parser.add_argument("--voice")
        if name == "clone":
            parser.add_argument("--reference", type=Path, required=True)
            parser.add_argument("--authorization", required=True)
            parser.add_argument("--reference-text")


def run(arguments: argparse.Namespace, *, recorder: TraceRecorder | None = None) -> int:
    """Use the same handler and result envelope as MCP."""
    runtime = None
    try:
        key = getattr(arguments, "signing_key_file", None)
        runtime = SpeechRuntime(
            arguments.runtime_config.absolute(),
            signing_key_file=key.absolute() if key is not None else None,
        )
        if arguments.audio_command == "speech-capabilities":
            print(json.dumps(runtime.capabilities(), ensure_ascii=False))
            return 0
        handler = runtime.clone if arguments.audio_command == "clone" else runtime.synthesize
        values: dict[str, object] = {"text": arguments.text, "language": arguments.language}
        for name in ("voice", "reference", "authorization", "reference_text"):
            value = getattr(arguments, name, None)
            if value is not None:
                values[name] = str(value.absolute()) if isinstance(value, Path) else value
        context = InvocationContext(
            str(uuid.uuid4()),
            str(uuid.uuid4()),
            "cli-local",
            "cli-user",
            "cli-local",
            datetime.now(UTC) + timedelta(seconds=240),
            handler.spec.required_permissions,
            handler.spec.privacy_class,
        )
        result = invoke_cli_tool(
            recorder=recorder, spec=handler.spec, handler=handler, context=context, arguments=values
        )
        print(json.dumps(encode_result(result), ensure_ascii=False))
        return 0 if result.error is None else 1
    except (ValueError, OSError):
        print("Speech configuration or local assets unavailable.", file=sys.stderr)
        return 2
    finally:
        if runtime is not None:
            runtime.close()
