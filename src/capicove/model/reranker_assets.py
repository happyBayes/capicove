"""Pinned public BGE reranker installation; inference never downloads assets."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import stat
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from capicove.model.local_assets import LocalModelAssets


MODEL_ID = "BAAI/bge-reranker-v2-m3"
MODEL_KEY = "bge-reranker-v2-m3"
# Public upstream commit and file digests below are immutable asset identities.
REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"  # pragma: allowlist secret


@dataclass(frozen=True)
class RerankerFile:
    """Pin one upstream safetensors or fast-tokenizer file."""

    path: str
    size_bytes: int
    sha256: str


FILES = (
    RerankerFile(
        "config.json",
        795,
        "13dcd6c31d9fec9d1d8e158702072f62d7fa7d312a64b9fe057bec9a08cfe41a",  # pragma: allowlist secret
    ),
    RerankerFile(
        "model.safetensors",
        2271071852,
        "d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286",  # pragma: allowlist secret
    ),
    RerankerFile(
        "special_tokens_map.json",
        964,
        "8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835",  # pragma: allowlist secret
    ),
    RerankerFile(
        "tokenizer.json",
        17098273,
        "69564b696052886ed0ac63fa393e928384e0f8caada38c1f4864a9bfbf379c15",  # pragma: allowlist secret
    ),
    RerankerFile(
        "tokenizer_config.json",
        1173,
        "7e4c1cc848840aeccdd763458c18dd525eb0f795c992e00ebe9c28554e7db2d4",  # pragma: allowlist secret
    ),
)
_PAIRS = tuple((f.path, f.sha256) for f in FILES)
MANIFEST_SHA256 = hashlib.sha256(
    json.dumps(_PAIRS, ensure_ascii=False, separators=(",", ":")).encode()
).hexdigest()


def reranker_assets(root: Path) -> LocalModelAssets:
    """Read standard profile roots and legacy flat installations offline."""
    nested = root / MODEL_KEY
    return LocalModelAssets(nested if nested.is_dir() else root, _PAIRS, MANIFEST_SHA256)


def _check_receipt(path: Path, *, parent_fd: int | None = None) -> None:
    """Reject linked or nonregular receipts before acquisition or publication."""
    try:
        metadata = os.stat(
            path.name if parent_fd is not None else path, dir_fd=parent_fd, follow_symlinks=False
        )
    except FileNotFoundError:
        return
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError("model receipt must be a regular unlinked file")


def _open_directory(path: Path) -> int:
    """Pin a directory by walking components without following symlinks."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(path.anchor, flags)
    try:
        for component in path.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _write_receipt(destination: Path) -> None:
    """Publish complete provenance atomically without truncating existing files."""
    path = destination.with_name(destination.name + ".receipt.json")
    receipt = {
        "repository": MODEL_ID,
        "repository_type": "model",
        "revision": REVISION,
        "upstream": MODEL_ID,
        "upstream_revision": REVISION,
        "license": "apache-2.0",
        "format": "safetensors",
        "transformation": "none; upstream float32 weights and fast tokenizer",
        "runtime": "capicove[model-serving]; transformers>=4.50,<6; torch>=2.2,<3",
        "asset_manifest_sha256": MANIFEST_SHA256,
        "asset_files": _PAIRS,
    }
    parent = _open_directory(path.parent)
    temporary = ".capicove-receipt-" + uuid.uuid4().hex
    created = False
    try:
        _check_receipt(path, parent_fd=parent)
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent
        )
        created = True
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _check_receipt(path, parent_fd=parent)
        os.replace(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent)
        os.fsync(parent)
    finally:
        try:
            if created:
                try:
                    os.unlink(temporary, dir_fd=parent)
                except FileNotFoundError:
                    pass
        finally:
            os.close(parent)


@dataclass(frozen=True)
class RerankerModel:
    """Expose the existing model installer inventory shape."""

    key: str = MODEL_KEY
    repository: str = MODEL_ID
    revision: str = REVISION
    files: tuple[RerankerFile, ...] = FILES
    size_bytes: int = sum(f.size_bytes for f in FILES)


@dataclass(frozen=True)
class RerankerInstallPlan:
    """Separate a network-free installation plan from explicit acquisition."""

    destination: Path
    profile_id: str = "default"
    profile_version: int = 1
    supported_languages: tuple[str, ...] = ("multilingual",)
    inference_engine: tuple[tuple[str, str], ...] = (
        ("engine", "transformers"),
        ("format", "safetensors"),
    )
    pipeline_configuration: tuple[tuple[str, str | int | bool], ...] = (
        ("license", "apache-2.0"),
        ("max_tokens", 512),
        ("trust_remote_code", False),
    )
    models: tuple[RerankerModel, ...] = (RerankerModel(),)
    total_size_bytes: int = sum(f.size_bytes for f in FILES)
    source_hosts: tuple[str, ...] = ("huggingface.co", "*.xethub.hf.co")
    manifest_sha256: str = MANIFEST_SHA256


class BgeRerankerInstaller:
    """Install only the reviewed public model allowlist after confirmation."""

    def model_plan(self, profile_id: str, destination: Path) -> RerankerInstallPlan:
        if profile_id != "default":
            raise ValueError("unknown reranker installation profile")
        if not destination.is_absolute() or destination.resolve() != destination:
            raise ValueError("model destination must be an absolute unlinked path")
        size = sum(file.size_bytes for file in FILES)
        return RerankerInstallPlan(
            destination,
            models=(RerankerModel(files=FILES, size_bytes=size),),
            total_size_bytes=size,
            manifest_sha256=MANIFEST_SHA256,
        )

    def install_models(
        self, profile_id: str, destination: Path, *, confirmed: bool
    ) -> RerankerInstallPlan:
        plan = self.model_plan(profile_id, destination)
        if confirmed is not True:
            raise ValueError("explicit model download confirmation required")
        receipt = destination.with_name(destination.name + ".receipt.json")
        _check_receipt(receipt)
        target = destination / MODEL_KEY
        legacy: dict[Path, os.stat_result] = {}
        if destination.exists():
            if not destination.is_dir():
                raise ValueError("model destination must be a directory")
            if target.exists():
                reranker_assets(destination).verify()
                _write_receipt(destination)
                return plan
            if any(destination.iterdir()):
                # Migrate only a complete, verified flat inventory. Keep it
                # intact until the canonical copy and receipt are published.
                reranker_assets(destination).verify()
                legacy = {
                    destination / file.path: (destination / file.path).stat() for file in FILES
                }
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".capicove-reranker-", dir=destination.parent
        ) as name:
            stage = Path(name)
            # The Hub cache remains separate from the closed runtime inventory.
            for file in FILES:
                if legacy:
                    source: str | Path = destination / file.path
                else:
                    hub = importlib.import_module("huggingface_hub")
                    source = hub.hf_hub_download(
                        MODEL_ID, file.path, revision=REVISION, token=False
                    )
                shutil.copyfile(source, stage / file.path)
            reranker_assets(stage).verify()
            destination.mkdir(mode=0o700, exist_ok=True)
            if destination.resolve() != destination:
                raise ValueError("model destination changed during installation")
            if target.exists():
                reranker_assets(destination).verify()
                _write_receipt(destination)
            else:
                if not legacy and any(destination.iterdir()):
                    raise ValueError("model destination changed during installation")
                # A failed receipt write cannot publish a completed model. A
                # retry can replace an orphan receipt or repair a missing one.
                _write_receipt(destination)
                stage.rename(target)
        if legacy:
            for path, before in legacy.items():
                after = path.stat(follow_symlinks=False)
                identity = (
                    "st_dev",
                    "st_ino",
                    "st_mode",
                    "st_nlink",
                    "st_size",
                    "st_mtime_ns",
                    "st_ctime_ns",
                )
                if all(getattr(before, field) == getattr(after, field) for field in identity):
                    path.unlink()
        return plan
