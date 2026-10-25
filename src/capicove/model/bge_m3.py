"""Explicit pinned BGE-M3 acquisition and offline safetensors conversion."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import subprocess  # nosec B404
import tempfile
import tomllib
from importlib import resources
from pathlib import Path
from typing import Any


def load_download_lock() -> dict[str, Any]:
    """Return the packaged official dense-only immutable download inventory."""
    return tomllib.loads(
        resources.files("capicove.model").joinpath("bge_m3.lock.toml").read_text(encoding="utf-8")
    )


def verify_upstream(root: Path, lock: dict[str, Any]) -> None:
    """Verify all allowlisted bytes before a restricted state-dict conversion."""
    for name, identity in lock["files"].items():
        path = root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError("unsafe BGE-M3 upstream file")
        if path.stat().st_size != identity["size_bytes"]:
            raise ValueError("BGE-M3 upstream size mismatch")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != identity["sha256"]:
                raise ValueError("BGE-M3 upstream digest mismatch")


def provision_bge_m3(destination: Path, *, confirmed: bool) -> dict[str, Any]:
    """Download only the locked dense inventory and atomically publish a bundle.

    Acquisition is operator-only. Runtime never accepts pickle weights. Conversion
    uses verified official input, weights_only=True, mmap=True and CPU loading.
    """
    if not confirmed:
        raise PermissionError("explicit model download confirmation is required")
    if (
        not destination.is_absolute()
        or destination.is_symlink()
        or destination.resolve() != destination
    ):
        raise ValueError("absolute unlinked destination is required")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "bge-m3"
    receipt_path = destination / "bge-m3.receipt.json"
    if target.exists() or target.is_symlink() or receipt_path.exists() or receipt_path.is_symlink():
        raise ValueError("BGE-M3 destination already exists")
    executable = shutil.which("hf")
    if executable is None:
        raise RuntimeError("Hugging Face hf CLI is required")
    lock = load_download_lock()
    with tempfile.TemporaryDirectory(prefix=".capicove-bge-m3-", dir=destination) as temporary:
        staging = Path(temporary)
        upstream, bundle = staging / "upstream", staging / "bundle"
        bundle.mkdir(mode=0o700)
        subprocess.run(  # nosec B603
            [
                executable,
                "download",
                lock["repository"],
                *sorted(lock["files"]),
                "--revision",
                lock["revision"],
                "--local-dir",
                str(upstream),
                "--max-workers",
                "2",
            ],
            check=True,
        )
        verify_upstream(upstream, lock)
        safetensors = importlib.import_module("safetensors")
        torch = importlib.import_module("torch")
        save_file = importlib.import_module("safetensors.torch").save_file

        weights = torch.load(
            upstream / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True
        )
        if (
            not isinstance(weights, dict)
            or not weights
            or any(
                not isinstance(key, str) or not isinstance(value, torch.Tensor)
                for key, value in weights.items()
            )
        ):
            raise ValueError("BGE-M3 upstream is not a tensor state dictionary")
        save_file(weights, str(bundle / "model.safetensors"), metadata={"format": "pt"})
        del weights
        for name in lock["files"]:
            if name != "pytorch_model.bin":
                path = bundle / name
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(upstream / name, path)
        files = []
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                with path.open("rb") as stream:
                    files.append(
                        (
                            path.relative_to(bundle).as_posix(),
                            hashlib.file_digest(stream, "sha256").hexdigest(),
                        )
                    )
        digest = hashlib.sha256(
            json.dumps(files, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        receipt = {
            "schema_version": 1,
            "repository_type": "model",
            "upstream": lock,
            "format": "safetensors",
            "scope": "dense-cls-only",
            "asset_files": files,
            "asset_manifest_sha256": digest,
            "conversion": {
                "torch": torch.__version__,
                "safetensors": safetensors.__version__,
                "weights_only": True,
                "mmap": True,
                "device": "cpu",
                "quantization": None,
            },
        }
        from capicove.model.local_assets import LocalModelAssets

        LocalModelAssets(bundle, tuple(files), digest).verify()
        os.rename(bundle, target)
        with receipt_path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
        return receipt
