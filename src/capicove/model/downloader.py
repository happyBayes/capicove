"""Explicit Hugging Face image-model download orchestration."""

from __future__ import annotations

import os
import re
import shutil
import subprocess  # nosec B404
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from capicove.model.manifest import ImageModelSpec
from capicove.model.receipt import (
    build_receipt,
    read_receipt,
    receipt_matches,
    verify_receipt_files,
    write_receipt,
)


# subprocess invokes a resolved hf executable with a fixed argv structure and no shell.


_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_ENVIRONMENT_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


@dataclass(frozen=True, slots=True)
class DownloadOptions:
    """Configure one explicit model download invocation."""

    destination: Path
    max_workers: int = 4
    force: bool = False
    token_environment: str | None = None


@dataclass(frozen=True, slots=True)
class InstalledModel:
    """Report one completed or already available model directory."""

    key: str
    path: Path
    repository: str
    revision: str
    model_manifest_sha256: str
    status: Literal["installed", "available"]


def render_download_commands(
    models: tuple[ImageModelSpec, ...],
    destination: Path,
    *,
    max_workers: int,
    executable: str = "hf",
) -> tuple[tuple[str, ...], ...]:
    """Render auditable pinned commands without filesystem or network access."""
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    commands: list[tuple[str, ...]] = []
    for model in models:
        staging = destination / f".capicove-staging-{model.key}"
        commands.extend(_commands_for_stage(model, staging, max_workers, executable))
    return tuple(commands)


def download_models(
    models: tuple[ImageModelSpec, ...],
    *,
    manifest_sha256: str,
    options: DownloadOptions,
    environ: Mapping[str, str] | None = None,
) -> tuple[InstalledModel, ...]:
    """Download, verify, and atomically install locked model revisions."""
    if options.max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if _DIGEST_PATTERN.fullmatch(manifest_sha256) is None:
        raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")
    executable = shutil.which("hf")
    if executable is None:
        raise RuntimeError(
            "Hugging Face hf CLI is unavailable. Install it with: "
            "curl -LsSf https://hf.co/cli/install.sh | bash"
        )
    environment = _subprocess_environment(environ, options.token_environment)
    destination = options.destination.expanduser().resolve()
    existing = tuple(
        _existing_result(model, destination, manifest_sha256, options.force) for model in models
    )
    if all(item is not None for item in existing):
        return tuple(item for item in existing if item is not None)
    destination.mkdir(parents=True, exist_ok=True)
    hf_version = _hf_version(executable, environment)
    results: list[InstalledModel] = []
    for model, available in zip(models, existing, strict=True):
        if available is not None:
            results.append(available)
            continue
        results.append(
            _download_one(
                model,
                manifest_sha256=manifest_sha256,
                options=options,
                destination=destination,
                executable=executable,
                environment=environment,
                hf_version=hf_version,
            )
        )
    return tuple(results)


def _existing_result(
    model: ImageModelSpec,
    destination: Path,
    manifest_sha256: str,
    force: bool,
) -> InstalledModel | None:
    target = destination / model.key
    if not target.exists():
        return None
    if not target.is_dir():
        raise ValueError(f"model destination is not a directory: {target}")
    receipt = read_receipt(target)
    verify_receipt_files(target, receipt)
    if receipt_matches(receipt, model, manifest_sha256=manifest_sha256) and not force:
        return InstalledModel(
            key=model.key,
            path=target,
            repository=model.repository,
            revision=model.revision,
            model_manifest_sha256=_model_manifest_digest(receipt),
            status="available",
        )
    if not force:
        raise ValueError(f"existing model receipt does not match the lock entry: {target}")
    if receipt.get("model_key") != model.key or receipt.get("repository") != model.repository:
        raise ValueError("--force cannot replace a different model key or repository")
    return None


def _download_one(
    model: ImageModelSpec,
    *,
    manifest_sha256: str,
    options: DownloadOptions,
    destination: Path,
    executable: str,
    environment: Mapping[str, str],
    hf_version: str,
) -> InstalledModel:
    staging = Path(tempfile.mkdtemp(prefix=f".capicove-staging-{model.key}-", dir=destination))
    target = destination / model.key
    try:
        download_command, verify_command = _commands_for_stage(
            model, staging, options.max_workers, executable
        )
        # The argv is generated from validated immutable lock entries.
        subprocess.run(  # nosec B603
            download_command,
            check=True,
            env=environment,
        )
        _remove_hf_local_metadata(staging)
        subprocess.run(  # nosec B603
            verify_command,
            check=True,
            env=environment,
        )
        receipt = build_receipt(
            model,
            staging,
            manifest_sha256=manifest_sha256,
            hf_version=hf_version,
        )
        write_receipt(staging, receipt)
        _finalize(staging, target, model, force=options.force)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return InstalledModel(
        key=model.key,
        path=target,
        repository=model.repository,
        revision=model.revision,
        model_manifest_sha256=_model_manifest_digest(receipt),
        status="installed",
    )


def _remove_hf_local_metadata(staging: Path) -> None:
    """Remove reproducible metadata created by ``hf download --local-dir``."""
    cache_root = staging / ".cache"
    metadata = cache_root / "huggingface"
    if metadata.exists():
        if metadata.is_symlink() or not metadata.is_dir():
            raise ValueError("Hugging Face local metadata path must be a directory")
        shutil.rmtree(metadata)
    if cache_root.is_dir() and not any(cache_root.iterdir()):
        cache_root.rmdir()


def _model_manifest_digest(receipt: Mapping[str, object]) -> str:
    value = receipt.get("model_manifest_sha256")
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ValueError("model receipt has an invalid file-manifest digest")
    return value


def _finalize(staging: Path, target: Path, model: ImageModelSpec, *, force: bool) -> None:
    if not target.exists():
        staging.replace(target)
        return
    if not force:
        raise ValueError(f"model destination appeared during download: {target}")
    receipt = read_receipt(target)
    if receipt.get("model_key") != model.key or receipt.get("repository") != model.repository:
        raise ValueError("--force cannot replace a different model key or repository")
    backup = target.parent / f".capicove-backup-{model.key}-{uuid.uuid4().hex}"
    target.replace(backup)
    try:
        staging.replace(target)
    except Exception:
        backup.replace(target)
        raise
    shutil.rmtree(backup)


def _commands_for_stage(
    model: ImageModelSpec,
    staging: Path,
    max_workers: int,
    executable: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (
        (
            executable,
            "download",
            model.repository,
            "--revision",
            model.revision,
            "--local-dir",
            str(staging),
            "--max-workers",
            str(max_workers),
        ),
        (
            executable,
            "cache",
            "verify",
            model.repository,
            "--revision",
            model.revision,
            "--local-dir",
            str(staging),
            "--fail-on-missing-files",
            "--fail-on-extra-files",
        ),
    )


def _subprocess_environment(
    environ: Mapping[str, str] | None,
    token_environment: str | None,
) -> dict[str, str]:
    environment = dict(os.environ if environ is None else environ)
    if token_environment is None:
        return environment
    if _ENVIRONMENT_NAME_PATTERN.fullmatch(token_environment) is None:
        raise ValueError("token environment name is invalid")
    token = environment.get(token_environment)
    if not token:
        raise ValueError(f"token environment variable is unavailable: {token_environment}")
    environment["HF_TOKEN"] = token
    return environment


def _hf_version(executable: str, environment: Mapping[str, str]) -> str:
    # The executable is resolved by shutil.which and invoked without a shell.
    completed = subprocess.run(  # nosec B603
        (executable, "version"),
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return completed.stdout.strip() or "unknown"
