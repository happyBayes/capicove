"""Explicit local reranker CLI and HTTP server composition."""

from __future__ import annotations

import argparse
import importlib.util
import ipaddress
import json
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from capicove.model.reranker_assets import MANIFEST_SHA256, MODEL_ID


if TYPE_CHECKING:
    from capicove.retrieval.bge_reranker import BgeReranker


def add_rerank_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register a JSON file or standard-input reranking command."""
    parser = commands.add_parser(
        "rerank", help="Rerank text candidates with an installed local BGE reranker."
    )
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--model-manifest-sha256")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--input", required=True, help="JSON request file, or - for stdin.")


def build_ranker(
    *, model_directory: Path, device: str, batch_size: int, max_tokens: int
) -> BgeReranker:
    """Verify local assets, load once and complete a real warmup forward pass."""
    from capicove.retrieval.bge_reranker import BgeReranker

    ranker = BgeReranker(
        model_directory, device=device, batch_size=batch_size, max_tokens=max_tokens
    )
    try:
        ranker.rerank("warmup", ("warmup",))
    except BaseException:
        ranker.close()
        raise
    return ranker


def serve(
    *,
    model_directory: Path,
    device: str,
    batch_size: int,
    max_tokens: int,
    host: str,
    port: int,
    api_key: str | None,
) -> None:
    """Keep the warm model resident until listener shutdown."""
    import uvicorn

    from capicove.api.reranker import create_app

    ranker = build_ranker(
        model_directory=model_directory, device=device, batch_size=batch_size, max_tokens=max_tokens
    )
    try:
        uvicorn.run(
            create_app(ranker, api_key=api_key), host=host, port=port, workers=1, access_log=False
        )
    finally:
        ranker.close()


def run(arguments: argparse.Namespace) -> int:
    """Validate policy and requests before importing optional model runtimes."""
    try:
        root = arguments.model_directory
        if not root.is_absolute() or root.resolve() != root or not root.is_dir():
            raise ValueError("model directory must be an existing absolute unlinked path")
        if arguments.model_manifest_sha256 not in (None, MANIFEST_SHA256):
            raise ValueError("reranker trusted manifest mismatch")
        if not re.fullmatch(r"cpu|mps|cuda(?::[0-9]+)?", arguments.device):
            raise ValueError("explicit supported model device required")
        if not 1 <= arguments.batch_size <= 128 or not 1 <= arguments.max_tokens <= 8192:
            raise ValueError("invalid batch or token limit")
        if getattr(arguments, "trace_store", None) is not None:
            raise ValueError("reranker commands do not support --trace-store")
        options = {
            "model_directory": root,
            "device": arguments.device,
            "batch_size": arguments.batch_size,
            "max_tokens": arguments.max_tokens,
        }
        if arguments.models_command == "rerank":
            from capicove.retrieval.reranking import (
                MAX_REQUEST_BYTES,
                parse_rerank_json,
                parse_rerank_request,
                render_rerank_result,
            )

            if arguments.input == "-":
                payload = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
            else:
                with Path(arguments.input).open("rb") as stream:
                    payload = stream.read(MAX_REQUEST_BYTES + 1)
            value = parse_rerank_json(payload)
            if value.get("model", MODEL_ID) not in {MODEL_ID, "bge-reranker-v2-m3"}:
                raise ValueError("model not served")
            query, documents, top_n, include = parse_rerank_request(value)
            ranker = build_ranker(**options)
            try:
                result = render_rerank_result(
                    ranker.rerank(query, documents, top_n=top_n),
                    documents,
                    return_documents=include,
                )
                print(json.dumps(result, ensure_ascii=False, allow_nan=False))
            finally:
                ranker.close()
            return 0
        loopback = arguments.host == "localhost" or ipaddress.ip_address(arguments.host).is_loopback
        if not 0 <= arguments.port <= 65535:
            raise ValueError("invalid port")
        key = None
        if arguments.api_key_env is not None:
            if not re.fullmatch(r"CAPICOVE_[A-Z0-9_]{1,100}", arguments.api_key_env):
                raise ValueError("API key environment reference must start with CAPICOVE_")
            key = os.environ.get(arguments.api_key_env, "")
            if not key or len(key) > 8192 or any(not 33 <= ord(c) <= 126 for c in key):
                raise ValueError("API key environment value is missing or invalid")
        if not loopback and key is None:
            raise ValueError("non-loopback serving requires --api-key-env")
        if any(
            importlib.util.find_spec(n) is None
            for n in ("torch", "transformers", "fastapi", "uvicorn")
        ):
            raise ValueError("install capicove[model-serving] to serve the reranker")
        serve(**options, host=arguments.host, port=arguments.port, api_key=key)
        return 0
    except (ValueError, OSError, ImportError, RuntimeError, RecursionError, TypeError):
        print(
            "Reranker command failed: check request, local pinned assets, device, limits, API key and capicove[model-serving] dependencies.",
            file=sys.stderr,
        )
        return 2
