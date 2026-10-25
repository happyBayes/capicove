"""CLI adapters for the canonical clipping Tool and explicit backend discovery."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from capicove.contracts import InvocationContext
from capicove.processing.audio import AudioClippingError, AudioClippingService, AudioProcessor
from capicove.processing.audio._adapters.audio_clip import FfmpegAudioClipBackend
from capicove.processing.audio._adapters.ffmpeg import AUDIO_DEMUXERS, FfmpegAudioValidator
from capicove.processing.audio._adapters.ffprobe import FfprobeAudioStreamInspector
from capicove.tools.audio_clip import AUDIO_CLIP_SPEC, AudioClipToolHandler
from capicove.traces import TraceRecorder


def _timeout(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 < parsed <= 300:
        raise argparse.ArgumentTypeError("timeout must be finite and between 0 and 300 seconds")
    return parsed


def add_parser(parent: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register one local extraction command with optional workspace authority."""
    parser = parent.add_parser("clip", help="Extract a sample-accurate range to a new PCM16 WAV.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--sample-rate-hz", type=int)
    parser.add_argument("--channels", type=int, choices=(1, 2))
    parser.add_argument("--profile", choices=("pcm16-wav",), default="pcm16-wav")
    parser.add_argument("--store")
    parser.add_argument("--ffmpeg")
    parser.add_argument("--ffprobe")
    parser.add_argument("--timeout", type=_timeout)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--output", choices=("human", "json"), default="human")


def run(arguments: argparse.Namespace, *, recorder: TraceRecorder | None = None) -> int:
    """Use the canonical handler for both standalone and configured execution."""
    from capicove.cli.tracing import invoke_cli_tool
    from capicove.mcp.bindings import encode_result

    source = Path(arguments.input).expanduser().absolute()
    destination = Path(arguments.destination).expanduser().absolute()
    tool_arguments: dict[str, object] = {
        "source": {"type": "path", "path": str(source)},
        "destination": {"type": "path", "path": str(destination)},
        "start_seconds": arguments.start,
        "end_seconds": arguments.end,
        "profile": arguments.profile,
    }
    for name in ("sample_rate_hz", "channels"):
        if getattr(arguments, name) is not None:
            tool_arguments[name] = getattr(arguments, name)
    if arguments.runtime_config is not None:
        if any(
            getattr(arguments, name) is not None
            for name in ("store", "ffmpeg", "ffprobe", "timeout")
        ):
            print(
                "--runtime-config owns the evidence store, executables, and timeout",
                file=sys.stderr,
            )
            return 2
        from capicove.cli.configured import invoke_configured

        try:
            payload = invoke_configured(
                arguments.runtime_config, AUDIO_CLIP_SPEC.tool_id, tool_arguments
            )
        except (ValueError, OSError, ImportError) as error:
            print(f"Runtime startup failed: {error}", file=sys.stderr)
            return 2
    else:
        timeout = arguments.timeout or 300
        ffmpeg, ffprobe = arguments.ffmpeg or "ffmpeg", arguments.ffprobe or "ffprobe"
        service = AudioClippingService(
            AudioProcessor(
                FfprobeAudioStreamInspector(ffprobe, timeout, restrict_inputs=True),
                FfmpegAudioValidator(ffmpeg, timeout, restrict_inputs=True),
            ),
            FfmpegAudioClipBackend(ffmpeg, ffprobe),
        )
        handler = AudioClipToolHandler(
            service,
            read_roots=(source.parent,),
            output_root=destination.parent,
            store_root=Path(arguments.store or ".capicove/audio-evidence").expanduser().absolute(),
        )
        context = InvocationContext(
            str(uuid.uuid4()),
            str(uuid.uuid4()),
            "cli-local",
            "cli-user",
            "cli-local",
            datetime.now(UTC) + timedelta(seconds=timeout),
            AUDIO_CLIP_SPEC.required_permissions,
            AUDIO_CLIP_SPEC.privacy_class,
        )
        payload = encode_result(
            invoke_cli_tool(
                recorder=recorder,
                spec=AUDIO_CLIP_SPEC,
                handler=handler,
                context=context,
                arguments=tool_arguments,
            )
        )
    if arguments.output == "json":
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    elif payload["status"] == "succeeded":
        result = payload["result"]
        assert isinstance(result, dict)
        artifact = result["artifact"]
        output = result["output"]
        assert isinstance(artifact, dict) and isinstance(output, dict)
        print(f"Created {artifact['destination_name']} ({output['duration_seconds']:.6f} seconds)")
    else:
        print(json.dumps(payload["error"]))
    return 0 if payload["status"] == "succeeded" else 1


def clip_capabilities(ffmpeg: str, ffprobe: str) -> dict[str, object]:
    """Explicitly probe the backend; this never runs during host catalog discovery."""
    reasons: list[str] = []
    for name, executable in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)):
        if shutil.which(executable) is None:
            reasons.append(f"{name}_unavailable")
    version = "unavailable"
    backend = FfmpegAudioClipBackend(ffmpeg, ffprobe)
    primitives = {
        "directory_fd": all(
            function in os.supports_dir_fd
            for function in (os.open, os.mkdir, os.stat, os.link, os.unlink)
        ),
        "nofollow": hasattr(os, "O_NOFOLLOW"),
        "hard_link": hasattr(os, "link"),
        "pread": hasattr(os, "pread"),
        "fstatvfs": hasattr(os, "fstatvfs"),
        "directory_fsync": hasattr(os, "fsync"),
        "descriptor_inheritance": os.name == "posix",
    }
    if not reasons:
        try:
            version = backend.preflight(datetime.now(UTC) + timedelta(seconds=15))
        except AudioClippingError as error:
            reasons.append(error.reason or "filesystem_profile_unavailable")
    return {
        "component": AUDIO_CLIP_SPEC.tool_id,
        "version": 1,
        "available": not reasons,
        "unavailable_reasons": reasons,
        "backend": "ffmpeg",
        "backend_version": version,
        "ffprobe_version": backend.ffprobe_version,
        "fd_protocols": backend.fd_protocols,
        "platform": {
            "python_minimum": "3.11",
            "supported_os": ["linux", "darwin"],
            "current_os": sys.platform,
        },
        "filesystem_primitives": primitives,
        "input_formats": [suffix.removeprefix(".") for suffix in AUDIO_DEMUXERS],
        "output_profile": "pcm16-wav",
        "sample_rate_hz": {"minimum": 8000, "maximum": 96000},
        "channels": [1, 2],
        "audio_stream_policy": "exactly_one",
        "overwrite": False,
        "network_access": False,
        "execution": "synchronous",
        "max_timeout_seconds": 300,
        "cancellation": "not_supported",
        "availability_scope": "host_backend_only",
    }
