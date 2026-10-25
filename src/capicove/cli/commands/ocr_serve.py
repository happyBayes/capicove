"""Compose a portable local PaddleOCR HTTP listener without eager SDK imports."""

from __future__ import annotations

import argparse
import importlib.util
import ipaddress
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast


if TYPE_CHECKING:
    from capicove.api.ocr import OcrProvider
    from capicove.documents import DocumentParserBinding


class _DocumentProfile(Protocol):
    """Read only the profile identity from the public installation plan."""

    profile_id: str
    profile_version: int


def build_provider(model_directory: Path, *, device: str, enable_mkldnn: bool) -> OcrProvider:
    """Verify installed assets and warm the public provider without acquiring models."""
    from PIL import Image

    from capicove.processing.image.ocr_configuration import ImageOcrConfiguration

    provider = cast(
        "OcrProvider",
        ImageOcrConfiguration(
            model_root=model_directory, device=device, enable_mkldnn=enable_mkldnn
        ).create_provider(),
    )
    try:
        provider.recognize(Image.new("RGB", (64, 64), "white"), ("chi_sim", "eng"))
        return provider
    except Exception:
        provider.close()
        raise


def build_document_parser(
    model_directory: Path, *, device: str, enable_mkldnn: bool
) -> DocumentParserBinding:
    """Select a public parser plugin and its installed profile without downloads."""
    from capicove.documents import DocumentParserBinding
    from capicove.plugins import DocumentParserModelInstaller, load_document_parser_plugin

    plugin = load_document_parser_plugin("paddleocr")
    if not plugin.availability().available:
        raise RuntimeError("Paddle document parser dependencies are unavailable")
    if not isinstance(plugin, DocumentParserModelInstaller):
        raise TypeError("Paddle document parser must declare its model profile")
    plan = cast("_DocumentProfile", plugin.model_plan("book-balanced", model_directory))
    return DocumentParserBinding(
        plugin.create(
            {"model_root": str(model_directory), "device": device, "enable_mkldnn": enable_mkldnn}
        ),
        plan.profile_id,
        plan.profile_version,
    )


def serve(
    *,
    model_directory: Path,
    device: str,
    enable_mkldnn: bool,
    host: str,
    port: int,
    api_key: str | None,
    max_body_bytes: int,
    max_pixels: int,
    max_document_bytes: int = 50 * 1024 * 1024,
    max_document_pages: int = 10,
    max_document_page_points: int = 2000,
    document_temp_directory: Path | None = None,
) -> None:
    """Warm native code before Uvicorn installs its process signal handlers."""
    import uvicorn

    from capicove.api.ocr import OcrHttpSettings, OcrRuntime, create_app

    settings = OcrHttpSettings(
        max_body_bytes=max_body_bytes,
        max_pixels=max_pixels,
        max_document_bytes=max_document_bytes,
        max_document_pages=max_document_pages,
        max_document_page_points=max_document_page_points,
        document_temp_directory=document_temp_directory,
    )
    runtime = OcrRuntime(
        lambda: build_provider(model_directory, device=device, enable_mkldnn=enable_mkldnn)
    )
    try:
        # Native initialization precedes Uvicorn signal installation. Both the
        # lifespan and startup-failure path ask the same owner to release once.
        runtime.start()
        document_parser = build_document_parser(
            model_directory, device=device, enable_mkldnn=enable_mkldnn
        )
        app = create_app(
            runtime, api_key=api_key, document_parser=document_parser, settings=settings
        )
        uvicorn.run(app, host=host, port=port, workers=1, access_log=False)
    finally:
        runtime.close()


def run(arguments: argparse.Namespace) -> int:
    """Validate operator-owned configuration before loading optional runtimes."""
    try:
        loopback = arguments.host == "localhost" or ipaddress.ip_address(arguments.host).is_loopback
        if not 0 <= arguments.port <= 65535:
            raise ValueError("port must be within 0-65535")
        root = arguments.model_directory
        if not root.is_absolute() or root.resolve() != root or not root.is_dir():
            raise ValueError("model directory must be an existing absolute unlinked directory")
        if arguments.device != "cpu":
            raise ValueError("the installed PaddleOCR adapter currently supports only CPU")
        if getattr(arguments, "trace_store", None) is not None:
            raise ValueError("configure tracing through workspace Tools instead")
        api_key = None
        if arguments.api_key_env is not None:
            if not re.fullmatch(r"CAPICOVE_[A-Z0-9_]{1,100}", arguments.api_key_env):
                raise ValueError("API key environment reference must start with CAPICOVE_")
            api_key = os.environ.get(arguments.api_key_env, "")
            if (
                not api_key
                or len(api_key) > 8192
                or any(not 33 <= c <= 126 for c in map(ord, api_key))
            ):
                raise ValueError("API key environment value is missing or invalid")
        if not loopback and api_key is None:
            raise ValueError("non-loopback serving requires --api-key-env")
        if any(importlib.util.find_spec(name) is None for name in ("fastapi", "uvicorn")):
            raise ValueError("install capicove[api] to serve OCR")
        if importlib.util.find_spec("capicove_processor_paddleocr") is None:
            raise ValueError("install capicove-processor-paddleocr and a compatible Paddle runtime")
        from capicove.api.ocr import OcrHttpSettings

        settings = OcrHttpSettings(
            max_body_bytes=arguments.max_body_bytes,
            max_pixels=arguments.max_pixels,
            max_document_bytes=arguments.max_document_bytes,
            max_document_pages=arguments.max_document_pages,
            max_document_page_points=arguments.max_document_page_points,
            document_temp_directory=arguments.document_temp_directory,
        )
        serve(
            model_directory=root,
            device=arguments.device,
            enable_mkldnn=arguments.enable_mkldnn,
            host=arguments.host,
            port=arguments.port,
            api_key=api_key,
            max_body_bytes=settings.max_body_bytes,
            max_pixels=settings.max_pixels,
            max_document_bytes=settings.max_document_bytes,
            max_document_pages=settings.max_document_pages,
            max_document_page_points=settings.max_document_page_points,
            document_temp_directory=settings.document_temp_directory,
        )
        return 0
    except (ValueError, OSError, ImportError, LookupError, RuntimeError, TypeError) as error:
        print(f"OCR startup failed: {error}", file=sys.stderr)
        return 2
