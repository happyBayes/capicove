"""Explicit immutable model acquisition for the book-balanced profile."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess  # nosec B404
import tempfile
import tomllib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path, PurePosixPath


@dataclass(frozen=True, slots=True)
class PaddleModelFileSpec:
    """Bind one immutable repository-relative model file to its bytes."""

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class PaddleModelSpec:
    """Describe one pinned official PaddlePaddle model repository."""

    key: str
    repository: str
    revision: str
    size_bytes: int
    model_parameter: str
    files: tuple[PaddleModelFileSpec, ...]


@dataclass(frozen=True, slots=True)
class PaddleModelManifest:
    """Describe one immutable parser model profile."""

    profile_id: str
    profile_version: int
    supported_languages: tuple[str, ...]
    inference_engine: tuple[tuple[str, str], ...]
    pipeline_configuration: tuple[tuple[str, str | int | bool], ...]
    models: tuple[PaddleModelSpec, ...]
    source_sha256: str


@dataclass(frozen=True, slots=True)
class ModelInstallationPlan:
    """Report network and storage effects before model acquisition."""

    profile_id: str
    profile_version: int
    supported_languages: tuple[str, ...]
    inference_engine: tuple[tuple[str, str], ...]
    pipeline_configuration: tuple[tuple[str, str | int | bool], ...]
    destination: Path
    models: tuple[PaddleModelSpec, ...]
    total_size_bytes: int
    source_hosts: tuple[str, ...]
    manifest_sha256: str


def load_book_balanced_manifest() -> PaddleModelManifest:
    """Load the packaged pinned book-balanced model manifest."""
    payload = (
        resources.files("capicove_processor_paddleocr")
        .joinpath("book_balanced_models.toml")
        .read_bytes()
    )
    raw = tomllib.loads(payload.decode("utf-8"))
    if raw.get("manifest_version") != 1 or raw.get("profile_id") != "book-balanced":
        raise ValueError("unsupported PaddleOCR model manifest")
    values = raw.get("models")
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("book-balanced requires exactly three models")
    models: list[PaddleModelSpec] = []
    model_keys: set[str] = set()
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("model entry must be a table")
        key = _text(value, "key")
        if key in model_keys or PurePosixPath(key).name != key or "\\" in key:
            raise ValueError("model key must be a unique path segment")
        model_keys.add(key)
        raw_files = value.get("files")
        if not isinstance(raw_files, list) or not raw_files:
            raise ValueError("model entry requires immutable file identities")
        files: list[PaddleModelFileSpec] = []
        file_paths: set[str] = set()
        for raw_file in raw_files:
            if not isinstance(raw_file, dict):
                raise ValueError("model file entry must be a table")
            path = _relative_file_path(raw_file)
            if path in file_paths:
                raise ValueError("model file paths must be unique")
            file_paths.add(path)
            files.append(
                PaddleModelFileSpec(
                    path,
                    _positive_int(raw_file, "size_bytes"),
                    _sha256(raw_file, "sha256"),
                )
            )
        size_bytes = _positive_int(value, "size_bytes")
        if size_bytes != sum(item.size_bytes for item in files):
            raise ValueError("model size must equal its immutable file sizes")
        models.append(
            PaddleModelSpec(
                key=key,
                repository=_text(value, "repository"),
                revision=_revision(value),
                size_bytes=size_bytes,
                model_parameter=_text(value, "model_parameter"),
                files=tuple(files),
            )
        )
    return PaddleModelManifest(
        profile_id="book-balanced",
        profile_version=_positive_int(raw, "profile_version"),
        supported_languages=_string_array(raw, "supported_languages"),
        inference_engine=_string_table(raw, "inference_engine"),
        pipeline_configuration=_configuration_table(raw, "pipeline_configuration"),
        models=tuple(models),
        source_sha256=hashlib.sha256(payload).hexdigest(),
    )


def plan_model_installation(
    manifest: PaddleModelManifest, destination: Path
) -> ModelInstallationPlan:
    """Create a side-effect-free installation plan."""
    return ModelInstallationPlan(
        profile_id=manifest.profile_id,
        profile_version=manifest.profile_version,
        supported_languages=manifest.supported_languages,
        inference_engine=manifest.inference_engine,
        pipeline_configuration=manifest.pipeline_configuration,
        destination=destination.expanduser().resolve(),
        models=manifest.models,
        total_size_bytes=sum(item.size_bytes for item in manifest.models),
        source_hosts=("huggingface.co",),
        manifest_sha256=manifest.source_sha256,
    )


def install_models(
    manifest: PaddleModelManifest,
    destination: Path,
    *,
    confirmed: bool,
) -> ModelInstallationPlan:
    """Download, verify, and atomically install the pinned profile."""
    if not confirmed:
        raise PermissionError("explicit model download confirmation is required")
    if destination.expanduser().is_symlink():
        raise ValueError("model root must not be a symbolic link")
    plan = plan_model_installation(manifest, destination)
    plan.destination.mkdir(parents=True, exist_ok=True)
    executable: str | None = None
    for model in plan.models:
        target = plan.destination / model.key
        if target.is_symlink():
            raise ValueError("model directory must not be a symbolic link")
        if target.is_dir():
            _verify_model_files(model, target)
            continue
        if target.exists():
            raise ValueError("model path must be a directory")
        if executable is None:
            executable = shutil.which("hf")
            if executable is None:
                raise RuntimeError("Hugging Face hf CLI is unavailable")
        staging = Path(tempfile.mkdtemp(prefix=f".capicove-{model.key}-", dir=plan.destination))
        try:
            subprocess.run(  # nosec B603
                (
                    executable,
                    "download",
                    model.repository,
                    "--revision",
                    model.revision,
                    "--local-dir",
                    str(staging),
                ),
                check=True,
                env=dict(os.environ),
            )
            shutil.rmtree(staging / ".cache", ignore_errors=True)
            _verify_model(executable, model, staging)
            _verify_model_files(model, staging)
            staging.replace(target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    _write_receipt(plan.destination / "book-balanced.receipt.json", _receipt(manifest))
    return plan


def verify_installed_models(manifest: PaddleModelManifest, destination: Path) -> None:
    """Fail closed unless the installed receipt and every model byte are pinned."""
    requested = destination.expanduser()
    if requested.is_symlink():
        raise ValueError("model root must not be a symbolic link")
    root = requested.resolve()
    receipt_path = root / "book-balanced.receipt.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("verified model installation receipt is unavailable")
    if receipt_path.stat().st_size > 1_000_000:
        raise ValueError("verified model installation receipt is invalid")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("verified model installation receipt is invalid") from error
    if receipt != _receipt(manifest):
        raise ValueError("verified model installation receipt does not match the profile")
    for model in manifest.models:
        _verify_model_files(model, root / model.key)


def _verify_model(executable: str, model: PaddleModelSpec, directory: Path) -> None:
    """Verify existing or staged bytes against the requested immutable revision."""
    subprocess.run(  # nosec B603
        (
            executable,
            "cache",
            "verify",
            model.repository,
            "--revision",
            model.revision,
            "--local-dir",
            str(directory),
            "--fail-on-missing-files",
            "--fail-on-extra-files",
        ),
        check=True,
        env=dict(os.environ),
    )


def _verify_model_files(model: PaddleModelSpec, directory: Path) -> None:
    """Verify a closed local tree against packaged file sizes and digests."""
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"verified model directory is unavailable: {model.key}")
    actual: dict[str, Path] = {}
    pending = [directory]
    while pending:
        current = pending.pop()
        for child in current.iterdir():
            if child.is_symlink():
                raise ValueError(f"model tree contains a symbolic link: {model.key}")
            if child.is_dir():
                pending.append(child)
            elif child.is_file():
                relative = child.relative_to(directory).as_posix()
                actual[relative] = child
            else:
                raise ValueError(f"model tree contains an unsupported entry: {model.key}")
    expected = {item.path: item for item in model.files}
    if actual.keys() != expected.keys():
        raise ValueError(f"model file set does not match the manifest: {model.key}")
    for path, specification in expected.items():
        local = actual[path]
        if local.stat().st_size != specification.size_bytes:
            raise ValueError(f"model file size does not match the manifest: {model.key}/{path}")
        with local.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != specification.sha256:
            raise ValueError(f"model file digest does not match the manifest: {model.key}/{path}")


def _receipt(manifest: PaddleModelManifest) -> dict[str, object]:
    return {
        "schema_version": 2,
        "profile_id": manifest.profile_id,
        "profile_version": manifest.profile_version,
        "supported_languages": list(manifest.supported_languages),
        "inference_engine": dict(manifest.inference_engine),
        "pipeline_configuration": dict(manifest.pipeline_configuration),
        "manifest_sha256": manifest.source_sha256,
        "models": [
            {
                "key": model.key,
                "repository": model.repository,
                "revision": model.revision,
                "files": [
                    {
                        "path": item.path,
                        "size_bytes": item.size_bytes,
                        "sha256": item.sha256,
                    }
                    for item in model.files
                ],
            }
            for model in manifest.models
        ],
    }


def _write_receipt(path: Path, receipt: dict[str, object]) -> None:
    if path.is_symlink():
        raise ValueError("model installation receipt must not be a symbolic link")
    payload = json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    try:
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _text(value: dict[str, object], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return item


def _positive_int(value: dict[str, object], name: str) -> int:
    item = value.get(name)
    if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return item


def _string_array(value: dict[str, object], name: str) -> tuple[str, ...]:
    raw = value.get(name)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{name} must be a nonempty string array")
    items = tuple(raw)
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise ValueError(f"{name} must be a nonempty string array")
    if len(set(items)) != len(items) or items != tuple(sorted(items)):
        raise ValueError(f"{name} must be unique and sorted")
    return items


def _string_table(value: dict[str, object], name: str) -> tuple[tuple[str, str], ...]:
    raw = value.get(name)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{name} must be a nonempty string table")
    items: list[tuple[str, str]] = []
    for key, item in raw.items():
        if (
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(item, str)
            or not item.strip()
        ):
            raise ValueError(f"{name} must be a nonempty string table")
        items.append((key, item))
    return tuple(sorted(items))


def _configuration_table(
    value: dict[str, object], name: str
) -> tuple[tuple[str, str | int | bool], ...]:
    raw = value.get(name)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{name} must be a nonempty scalar table")
    items: list[tuple[str, str | int | bool]] = []
    for key, item in raw.items():
        if (
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(item, (str, int, bool))
            or (isinstance(item, str) and not item.strip())
        ):
            raise ValueError(f"{name} must be a nonempty scalar table")
        items.append((key, item))
    return tuple(sorted(items))


def _revision(value: dict[str, object]) -> str:
    revision = _text(value, "revision")
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("revision must be a lowercase Git SHA")
    return revision


def _relative_file_path(value: dict[str, object]) -> str:
    path = _text(value, "path")
    parsed = PurePosixPath(path)
    if (
        parsed.is_absolute()
        or parsed.as_posix() != path
        or any(part in {"", ".", ".."} for part in parsed.parts)
        or "\\" in path
    ):
        raise ValueError("model file path must be canonical and relative")
    return path


def _sha256(value: dict[str, object], name: str) -> str:
    digest = _text(value, name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return digest


__all__ = [
    "ModelInstallationPlan",
    "PaddleModelFileSpec",
    "PaddleModelManifest",
    "PaddleModelSpec",
    "install_models",
    "load_book_balanced_manifest",
    "plan_model_installation",
    "verify_installed_models",
]
