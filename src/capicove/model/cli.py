"""Command line interface for explicit image-model acquisition."""

from __future__ import annotations

import argparse
import os
import platform
import shlex
import subprocess  # nosec B404
import sys
from collections.abc import Sequence
from pathlib import Path

from capicove.model.downloader import (
    DownloadOptions,
    download_models,
    render_download_commands,
)
from capicove.model.manifest import load_manifest, load_packaged_manifest
from capicove.model.selection import resolve_model_home, select_models


# subprocess is used only to classify failures from the isolated hf CLI adapter.


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="capicove-download-image-models",
        description="Explicitly download immutable image-model revisions for local Capicove use.",
    )
    parser.add_argument("--profile", choices=("default",))
    parser.add_argument("--model", action="append", default=[], metavar="KEY")
    parser.add_argument("--include-restricted", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--token-env", metavar="NAME")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a selection and optionally download it through the official hf CLI."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.max_workers <= 0:
            raise ValueError("--max-workers must be positive")
        manifest = (
            load_manifest(arguments.manifest)
            if arguments.manifest is not None
            else load_packaged_manifest()
        )
        models = select_models(
            manifest,
            default_profile=arguments.profile == "default",
            keys=tuple(arguments.model),
            include_restricted=arguments.include_restricted,
        )
        destination = resolve_model_home(
            arguments.destination,
            os.environ,
            platform.system(),
            Path.home(),
        )
    except (OSError, ValueError) as error:
        parser.print_usage(sys.stderr)
        print(f"error: {error}", file=sys.stderr)
        return 2
    if arguments.dry_run:
        for model in models:
            print(
                f"model={model.key} repository={model.repository} "
                f"revision={model.revision} license={model.license} "
                f"restricted={str(model.restricted).lower()}"
            )
        for command in render_download_commands(
            models,
            destination,
            max_workers=arguments.max_workers,
        ):
            print(shlex.join(command))
        return 0
    try:
        installed = download_models(
            models,
            manifest_sha256=manifest.source_sha256,
            options=DownloadOptions(
                destination=destination,
                max_workers=arguments.max_workers,
                force=arguments.force,
                token_environment=arguments.token_env,
            ),
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for installed_model in installed:
        print(
            f"status={installed_model.status} model={installed_model.key} "
            f"path={installed_model.path} revision={installed_model.revision} "
            f"model_manifest_sha256={installed_model.model_manifest_sha256}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
