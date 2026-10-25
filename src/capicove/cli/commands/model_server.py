"""Unified configured model catalog and HTTP startup."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import sys

from capicove.model import load_configuration
from capicove.model.serving import catalog, select_models


def run(arguments: argparse.Namespace) -> int:
    """Validate selection and listen policy before loading optional HTTP runtimes."""
    try:
        if getattr(arguments, "trace_store", None) is not None:
            raise ValueError("use workspace Tools for tracing")
        configuration = load_configuration(arguments.config)
        if arguments.models_command == "list":
            print(
                json.dumps({"object": "list", "data": catalog(configuration)}, ensure_ascii=False)
            )
            return 0
        if arguments.provider is not None or arguments.model_directory is not None:
            raise ValueError("configuration cannot be combined with legacy model arguments")
        if bool(arguments.model) == bool(arguments.all):
            raise ValueError("select --model or --all")
        selected = select_models(configuration, tuple(arguments.model) if arguments.model else None)
        loopback = arguments.host == "localhost" or ipaddress.ip_address(arguments.host).is_loopback
        port = arguments.port if arguments.port is not None else 8769
        if not 0 <= port <= 65535:
            raise ValueError("invalid port")
        key = None
        if arguments.api_key_env is not None:
            if not re.fullmatch(r"CAPICOVE_[A-Z0-9_]{1,100}", arguments.api_key_env):
                raise ValueError("invalid key environment reference")
            key = os.environ.get(arguments.api_key_env, "")
            if not key or len(key) > 8192 or any(not 33 <= ord(c) <= 126 for c in key):
                raise ValueError("invalid incoming API key")
        if not loopback and key is None:
            raise ValueError("non-loopback serving requires --api-key-env")
        import uvicorn

        from capicove.api.model_server import create_app
        from capicove.api.ocr import OcrHttpSettings

        settings = OcrHttpSettings(
            max_body_bytes=arguments.max_body_bytes,
            max_pixels=arguments.max_pixels,
            max_document_bytes=arguments.max_document_bytes,
            max_document_pages=arguments.max_document_pages,
            max_document_page_points=arguments.max_document_page_points,
            document_temp_directory=arguments.document_temp_directory,
        )

        uvicorn.run(
            create_app(selected, api_key=key, ocr_settings=settings),
            host=arguments.host,
            port=port,
            workers=1,
            access_log=False,
        )
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, TypeError):
        print(
            "Model service failed: check configuration, explicit selection, capacity, local assets, dependencies and listen authentication.",
            file=sys.stderr,
        )
        return 2
