"""Generic VLM registration inspection, JSON inference and HTTP serving."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import sys
from pathlib import Path

from capicove.model import ModelCapability, ModelFactory, load_configuration
from capicove.model.http_profile import HttpModelProfile
from capicove.model.vlm import MAX_REQUEST_BYTES, parse_chat_request, parse_json
from capicove.model.vlm_service import model_info, registered_vlms


def add_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Keep parser registration independent of web and model SDK dependencies."""
    root = commands.add_parser("vlm", help="Use registered OpenAI-compatible vision models.")
    operations = root.add_subparsers(dest="vlm_command", required=True)
    for name in ("list", "generate", "count-tokens", "serve"):
        parser = operations.add_parser(name)
        parser.add_argument(
            "--config", type=Path, required=True, help="Version-2 JSON/TOML model configuration."
        )
        if name in {"generate", "count-tokens"}:
            parser.add_argument("--binding", required=True)
            parser.add_argument(
                "--input", required=True, help="Chat Completions JSON file, or - for stdin."
            )
        elif name == "serve":
            parser.add_argument("--host", default="127.0.0.1")
            parser.add_argument("--port", type=int, default=8769)
            parser.add_argument(
                "--api-key-env",
                help="CAPICOVE_ environment variable for incoming bearer authentication.",
            )


def serve(factory: ModelFactory, *, host: str, port: int, api_key: str | None) -> None:
    """Serve all bound VLM profiles without importing any GPU model runtime."""
    import uvicorn

    from capicove.api.model_server import create_app

    uvicorn.run(
        create_app(
            factory.configuration,
            models=tuple(registered_vlms(factory.configuration)),
            api_key=api_key,
        ),
        host=host,
        port=port,
        workers=1,
        access_log=False,
    )


def run(arguments: argparse.Namespace) -> int:
    """Use explicit configuration and keep input, media and credentials out of errors."""
    try:
        if getattr(arguments, "trace_store", None) is not None:
            raise ValueError("VLM commands do not support trace storage")
        configuration = load_configuration(arguments.config)
        catalog = registered_vlms(configuration)
        if arguments.vlm_command == "list":
            print(
                json.dumps(
                    {"object": "list", "data": [model_info(p) for _, p in catalog.values()]},
                    ensure_ascii=False,
                )
            )
            return 0
        with ModelFactory(configuration) as factory:
            if arguments.vlm_command in {"generate", "count-tokens"}:
                if arguments.input == "-":
                    payload = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
                else:
                    with Path(arguments.input).open("rb") as stream:
                        payload = stream.read(MAX_REQUEST_BYTES + 1)
                raw = parse_json(payload)
                profile = factory.profile(arguments.binding, ModelCapability.MULTIMODAL_GENERATION)
                assert isinstance(profile, HttpModelProfile)
                if raw.get("model", profile.name) not in {profile.name, profile.model}:
                    raise ValueError("input model does not match selected binding")
                request = parse_chat_request(raw)
                with factory.vlm(arguments.binding) as provider:
                    result = (
                        provider.count_tokens(request).to_dict()
                        if arguments.vlm_command == "count-tokens"
                        else provider.generate(request).to_openai()
                    )
                    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
                return 0
            loopback = (
                arguments.host == "localhost" or ipaddress.ip_address(arguments.host).is_loopback
            )
            if type(arguments.port) is not int or not 0 <= arguments.port <= 65535:
                raise ValueError("invalid listen port")
            key = None
            if arguments.api_key_env is not None:
                if not re.fullmatch(r"CAPICOVE_[A-Z0-9_]{1,100}", arguments.api_key_env):
                    raise ValueError("invalid incoming key environment reference")
                key = os.environ.get(arguments.api_key_env, "")
                if not key or len(key) > 8192 or any(not 33 <= ord(c) <= 126 for c in key):
                    raise ValueError("incoming key environment value is missing or invalid")
            if not loopback and key is None:
                print("VLM non-loopback serving requires --api-key-env.", file=sys.stderr)
                return 2
            for _, profile in catalog.values():
                if profile.api_key_env is not None and not os.environ.get(profile.api_key_env):
                    raise ValueError("upstream key environment value is missing")
            serve(factory, host=arguments.host, port=arguments.port, api_key=key)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, RecursionError, TypeError):
        print(
            "VLM command failed: check configuration, request, endpoint, key environment and capicove[vlm,api] dependencies.",
            file=sys.stderr,
        )
        return 2
