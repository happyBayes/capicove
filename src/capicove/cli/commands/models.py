"""Explicit model installation commands for capability-neutral plugins."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

from capicove.model.installers import ModelInstaller, load_model_installer


class _ModelSpec(Protocol):
    """Describe the model fields rendered by this command."""

    key: str
    repository: str
    revision: str
    size_bytes: int
    files: tuple[_ModelFileSpec, ...]


class _ModelFileSpec(Protocol):
    """Describe one immutable model file rendered by this command."""

    path: str
    size_bytes: int
    sha256: str


class _InstallationPlan(Protocol):
    """Describe the installation plan fields rendered by this command."""

    profile_id: str
    profile_version: int
    supported_languages: tuple[str, ...]
    inference_engine: tuple[tuple[str, str], ...]
    pipeline_configuration: tuple[tuple[str, str | int | bool], ...]
    destination: Path
    models: tuple[_ModelSpec, ...]
    total_size_bytes: int
    source_hosts: tuple[str, ...]
    manifest_sha256: str


def add_parser(parent: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register `capicove models` commands."""
    models = parent.add_parser("models", help="Manage optional local model profiles.")
    commands = models.add_subparsers(dest="models_command", required=True)
    serve = commands.add_parser("serve", help="Serve an installed local model over HTTP.")
    serve.add_argument("provider", nargs="?", choices=("bge-m3", "paddleocr", "bge-reranker-v2-m3"))
    serve.add_argument("--model-directory", type=Path)
    serve.add_argument("--config", type=Path)
    selection = serve.add_mutually_exclusive_group()
    selection.add_argument("--model", action="append", help="Registered name; may be repeated.")
    selection.add_argument("--all", action="store_true", help="Serve all explicitly bound models.")
    listing = commands.add_parser("list", help="List configured models and HTTP endpoints.")
    listing.add_argument("--config", type=Path, required=True)
    serve.add_argument("--model-manifest-sha256", help="Required for BGE-M3.")
    serve.add_argument("--device", default="cpu")
    serve.add_argument("--batch-size", type=int, default=8)
    serve.add_argument("--max-tokens", type=int, default=512)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, help="Default: 8768 for reranking, 8766 otherwise.")
    serve.add_argument(
        "--api-key-env", help="CAPICOVE_ environment variable containing a bearer key."
    )
    serve.add_argument(
        "--max-body-bytes",
        type=int,
        default=10 * 1024 * 1024,
        help="OCR encoded image byte limit (default: 10 MiB).",
    )
    serve.add_argument(
        "--max-pixels",
        type=int,
        default=20_000_000,
        help="OCR decoded image pixel limit (default: 20000000).",
    )
    serve.add_argument("--enable-mkldnn", action="store_true", help="Enable Paddle CPU MKL-DNN.")
    serve.add_argument(
        "--max-document-bytes",
        type=int,
        default=50 * 1024 * 1024,
        help="PDF upload byte limit (default: 50 MiB).",
    )
    serve.add_argument(
        "--max-document-pages",
        type=int,
        default=10,
        help="Maximum selected PDF pages per request (default: 10).",
    )
    serve.add_argument(
        "--max-document-page-points",
        type=int,
        default=2000,
        help="Maximum raw or UserUnit-scaled PDF page side (default: 2000).",
    )
    serve.add_argument(
        "--document-temp-directory",
        type=Path,
        help="Existing private-upload parent; default: OS temporary directory.",
    )
    from capicove.cli.commands.reranker import add_rerank_parser
    from capicove.cli.commands.vlm import add_parser as add_vlm_parser

    add_rerank_parser(commands)
    add_vlm_parser(commands)
    install = commands.add_parser(
        "install", help="Plan or explicitly install one local model profile."
    )
    _runtime_config(install)
    install.add_argument("plugin", nargs="?")
    install.add_argument("--profile")
    install.add_argument(
        "--destination", help="Model directory, or a destination alias with --runtime-config."
    )
    install.add_argument("--plan-digest", help="Allowlisted plan digest (--runtime-config only).")
    confirmation = install.add_mutually_exclusive_group()
    confirmation.add_argument("--dry-run", action="store_true")
    confirmation.add_argument("--yes", action="store_true")
    for name, help_text in (
        ("plan", "Show the pinned installation plan for a configured destination."),
        ("verify", "Hash an installed profile against its plan without downloading."),
    ):
        parser = commands.add_parser(name, help=help_text)
        _runtime_config(parser, required=True)
        parser.add_argument("plugin")
        parser.add_argument("--profile", required=True)
        parser.add_argument("--destination", required=True, help="Configured destination alias.")


def _runtime_config(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    parser.add_argument(
        "--runtime-config",
        type=Path,
        required=required,
        help="Run the configured capicove.model Tool from a version-2 workspace config.",
    )


def _run_configured(arguments: argparse.Namespace) -> int:
    from capicove.cli.configured import invoke_configured

    command = arguments.models_command
    if command == "install":
        if (
            arguments.plan_digest is None
            or arguments.destination is None
            or arguments.plugin is not None
            or arguments.profile is not None
            or arguments.dry_run
            or arguments.yes
        ):
            print(
                "--runtime-config install takes only --plan-digest and --destination; "
                "the configured allowlist is the approval",
                file=sys.stderr,
            )
            return 2
        request: dict[str, object] = {
            "plan_digest": arguments.plan_digest,
            "destination": arguments.destination,
        }
    else:
        request = {
            "plugin": arguments.plugin,
            "profile": arguments.profile,
            "destination": arguments.destination,
        }
    try:
        payload = invoke_configured(arguments.runtime_config, f"capicove.model.{command}", request)
    except (OSError, ValueError, ImportError) as error:
        print(f"Runtime startup failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "succeeded" else 1


def run(
    arguments: argparse.Namespace,
    *,
    plugin_loader: Callable[[str], object] = load_model_installer,
    interactive: bool | None = None,
) -> int:
    """Plan or perform one explicitly authorized model installation."""
    if arguments.models_command == "list" or (
        arguments.models_command == "serve" and getattr(arguments, "config", None) is not None
    ):
        from capicove.cli.commands.model_server import run as run_server

        return run_server(arguments)
    if arguments.models_command == "serve" and (
        arguments.provider is None
        or arguments.model_directory is None
        or arguments.model
        or arguments.all
    ):
        print(
            "serve requires --config and --model/--all, or a legacy provider and --model-directory",
            file=sys.stderr,
        )
        return 2
    if arguments.models_command == "vlm":
        from capicove.cli.commands.vlm import run as run_vlm

        return run_vlm(arguments)
    if arguments.models_command == "serve" and arguments.port is None:
        arguments.port = 8768 if arguments.provider == "bge-reranker-v2-m3" else 8766
    if arguments.models_command == "rerank" or (
        arguments.models_command == "serve" and arguments.provider == "bge-reranker-v2-m3"
    ):
        from capicove.cli.commands.reranker import run as run_reranker

        return run_reranker(arguments)
    if arguments.models_command == "serve" and arguments.provider == "paddleocr":
        from capicove.cli.commands.ocr_serve import run as run_ocr_serve

        return run_ocr_serve(arguments)
    if arguments.models_command == "serve":
        if arguments.model_manifest_sha256 is None:
            print("BGE-M3 startup failed: --model-manifest-sha256 is required", file=sys.stderr)
            return 2
        from capicove.cli.commands.model_serve import run as run_serve

        return run_serve(arguments)
    if getattr(arguments, "runtime_config", None) is not None:
        return _run_configured(arguments)
    if arguments.plugin is None or arguments.profile is None:
        _emit_error("model_arguments_required", "install requires a plugin and --profile")
        return 2
    if getattr(arguments, "plan_digest", None) is not None:
        _emit_error("model_arguments_invalid", "--plan-digest requires --runtime-config")
        return 2
    try:
        plugin = plugin_loader(arguments.plugin)
    except LookupError:
        message = "model installer plugin is unavailable"
        if arguments.plugin == "paddleocr":
            message = (
                f"{message}; install capicove-processor-paddleocr first "
                "(repository checkout: uv pip install -e ./plugins/processor-paddleocr)"
            )
        _emit_error("model_plugin_unavailable", message)
        return 2
    except Exception:
        _emit_error("model_plugin_unavailable", "model installer plugin could not be loaded")
        return 2
    if not isinstance(plugin, ModelInstaller):
        _emit_error(
            "model_installer_unavailable",
            f"plugin {arguments.plugin!r} does not install models",
        )
        return 2
    try:
        destination = (
            Path(arguments.destination)
            if arguments.destination is not None
            else _default_destination(arguments.plugin, arguments.profile)
        )
        plan = cast(
            _InstallationPlan,
            plugin.model_plan(arguments.profile, destination),
        )
        rendered = _render_plan(plan)
        if arguments.dry_run:
            _emit({"operation": "model_install_plan", **rendered})
            return 0
        is_interactive = sys.stdin.isatty() if interactive is None else interactive
        confirmed = bool(arguments.yes)
        if not confirmed and is_interactive:
            _emit({"operation": "model_install_plan", **rendered})
            confirmed = input("Download these models? [y/N] ").strip().lower() in {"y", "yes"}
        if not confirmed:
            _emit(
                {
                    "error": {
                        "code": "model_confirmation_required",
                        "message": "use --yes to authorize this network download",
                    },
                    "plan": rendered,
                }
            )
            return 2
        installed = cast(
            _InstallationPlan,
            plugin.install_models(
                arguments.profile,
                destination,
                confirmed=True,
            ),
        )
        _emit({"operation": "model_install_complete", **_render_plan(installed)})
        return 0
    except Exception:
        _emit_error(
            "model_install_failed",
            "Model plan or installation failed; check the local plugin, manifest, and destination.",
        )
        return 2


def _render_plan(plan: _InstallationPlan) -> dict[str, object]:
    return {
        "destination": str(plan.destination),
        "manifest_sha256": plan.manifest_sha256,
        "models": [
            {
                "key": model.key,
                "repository": model.repository,
                "revision": model.revision,
                "size_bytes": model.size_bytes,
                "files": [
                    {
                        "path": item.path,
                        "size_bytes": item.size_bytes,
                        "sha256": item.sha256,
                    }
                    for item in model.files
                ],
            }
            for model in plan.models
        ],
        "profile_id": plan.profile_id,
        "profile_version": plan.profile_version,
        "supported_languages": list(plan.supported_languages),
        "inference_engine": dict(plan.inference_engine),
        "pipeline_configuration": dict(plan.pipeline_configuration),
        "source_hosts": list(plan.source_hosts),
        "total_size_bytes": plan.total_size_bytes,
    }


def _emit_error(code: str, message: str) -> None:
    _emit({"error": {"code": code, "message": message}})


def _default_destination(plugin: str, profile: str) -> Path:
    configured = os.environ.get("CAPICOVE_MODEL_HOME")
    root = (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".local/share/capicove/models"
    )
    return (root / plugin / profile).resolve()


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


__all__ = ["add_parser", "run"]
