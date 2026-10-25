"""Start an explicitly selected local embedding server without eager SDK imports."""

from __future__ import annotations

import argparse
import importlib.util
import ipaddress
import os
import re
import sys
from pathlib import Path


def serve(
    *,
    model_directory: Path,
    manifest_sha256: str,
    device: str,
    batch_size: int,
    max_tokens: int,
    host: str,
    port: int,
    api_key: str | None,
) -> None:
    """Warm one verified model, then serve it until normal process shutdown."""
    import uvicorn

    from capicove.api.bge_m3 import build_encoder, create_app

    encoder = build_encoder(
        model_directory,
        expected_model_manifest_sha256=manifest_sha256,
        device=device,
        batch_size=batch_size,
        max_tokens=max_tokens,
    )
    try:
        uvicorn.run(
            create_app(encoder, api_key=api_key), host=host, port=port, workers=1, access_log=False
        )
    finally:
        encoder.close()


def run(arguments: argparse.Namespace) -> int:
    """Validate startup policy before importing optional model or web runtimes."""
    try:
        loopback = arguments.host == "localhost" or ipaddress.ip_address(arguments.host).is_loopback
        if not 0 <= arguments.port <= 65535:
            raise ValueError("invalid port")
        if not 1 <= arguments.max_tokens <= 8192 or not 1 <= arguments.batch_size <= 128:
            raise ValueError("invalid token or batch limit")
        if not re.fullmatch(r"[0-9a-f]{64}", arguments.model_manifest_sha256):
            raise ValueError("a trusted model manifest SHA-256 is required")
        if (
            not arguments.model_directory.is_absolute()
            or arguments.model_directory.resolve() != arguments.model_directory
        ):
            raise ValueError("model directory must be an absolute unlinked path")
        if not re.fullmatch(r"cpu|mps|cuda(?::[0-9]+)?", arguments.device):
            raise ValueError("an explicit supported model device is required")
        api_key = None
        if arguments.api_key_env is not None:
            if not re.fullmatch(r"CAPICOVE_[A-Z0-9_]{1,100}", arguments.api_key_env):
                raise ValueError("API key environment reference must start with CAPICOVE_")
            api_key = os.environ.get(arguments.api_key_env, "")
            if not api_key or len(api_key) > 8192 or any(not 33 <= ord(c) <= 126 for c in api_key):
                raise ValueError("API key environment value is missing or invalid")
        if not loopback and api_key is None:
            raise ValueError("non-loopback serving requires --api-key-env")
        if getattr(arguments, "trace_store", None) is not None:
            raise ValueError("model serving does not support --trace-store")
        if any(
            importlib.util.find_spec(name) is None
            for name in ("uvicorn", "fastapi", "sentence_transformers", "torch")
        ):
            raise ValueError("install capicove[api,text-embedding] to serve BGE-M3")
        serve(
            model_directory=arguments.model_directory,
            manifest_sha256=arguments.model_manifest_sha256,
            device=arguments.device,
            batch_size=arguments.batch_size,
            max_tokens=arguments.max_tokens,
            host=arguments.host,
            port=arguments.port,
            api_key=api_key,
        )
        return 0
    except (ValueError, OSError, ImportError):
        print(
            "BGE-M3 startup failed: check local model, trusted digest, device, limits, "
            "API key and capicove[api,text-embedding] dependencies.",
            file=sys.stderr,
        )
        return 2
