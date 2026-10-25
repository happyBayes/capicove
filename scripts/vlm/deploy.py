#!/usr/bin/env python3
"""Explicit pinned model acquisition and offline vLLM launch from verified assets."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any


MANIFEST = "capicove-assets.json"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(root: Path) -> dict[str, str]:
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts[0] == ".cache" or str(relative) == MANIFEST:
            continue
        if path.is_symlink():
            raise ValueError("model directory must not contain symlinks")
        if path.is_file():
            files[relative.as_posix()] = digest(path)
    return files


def verify_assets(root: Path, manifest: dict[str, Any]) -> None:
    """Reject changed, missing, extra or escaping files before starting a runtime."""
    expected = manifest.get("files")
    if manifest.get("schema_version") != 1 or not isinstance(expected, dict) or not expected:
        raise ValueError("invalid model inventory")
    for name, checksum in expected.items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not re.fullmatch(r"[0-9a-f]{64}", checksum)
        ):
            raise ValueError("invalid model inventory path or hash")
    actual = inventory(root)
    if set(actual) != set(expected):
        raise ValueError("model inventory differs from receipt")
    if actual != expected:
        raise ValueError("model file hash differs from receipt")


def load_config(path: Path) -> dict[str, Any]:
    config = tomllib.loads(path.read_text())
    required = {
        "repository",
        "revision",
        "license",
        "vllm_version",
        "served_model_name",
        "arguments",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("invalid deployment configuration fields")
    for field in required - {"arguments"}:
        if not isinstance(config[field], str) or not config[field].strip():
            raise ValueError("deployment identities must be nonempty strings")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", config["repository"]):
        raise ValueError("invalid model repository")
    if not re.fullmatch(r"[0-9a-f]{40}", config["revision"]):
        raise ValueError("model revision must be an immutable commit")
    args = config["arguments"]
    if not isinstance(args, list) or any(not isinstance(a, str) for a in args):
        raise ValueError("runtime arguments must be strings")
    forbidden = {"--api-key", "--model", "--served-model-name", "--revision", "--trust-remote-code"}
    if any(a.split("=", 1)[0] in forbidden for a in args):
        raise ValueError("runtime arguments override managed model or credential policy")
    return config


def serve_command(root: Path, config: dict[str, Any]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(root),
        "--served-model-name",
        config["served_model_name"],
        *config["arguments"],
    ]


def download(root: Path, config: dict[str, Any], allow_network: bool) -> None:
    if not allow_network:
        raise ValueError("download requires --allow-network")
    project = Path(__file__).resolve().parents[2]
    if root == project or project in root.parents:
        raise ValueError("model weights must be stored outside the source repository")
    if (root / MANIFEST).exists():
        previous = json.loads((root / MANIFEST).read_text())
        if any(
            previous.get(k) != config[k]
            for k in ("repository", "revision", "license", "vllm_version")
        ):
            raise ValueError("existing receipt differs from requested deployment")
        verify_assets(root, previous)
        print("Verified existing pinned model assets.")
        return
    hf = shutil.which("hf")
    if hf is None:
        raise ValueError("install the Hugging Face hf CLI in the acquisition environment")
    common = [config["repository"], "--revision", config["revision"], "--local-dir", str(root)]
    subprocess.run([hf, "download", *common], check=True)
    subprocess.run(
        [hf, "cache", "verify", *common, "--fail-on-missing-files", "--fail-on-extra-files"],
        check=True,
    )
    files = inventory(root)
    if "config.json" not in files or not any(name.endswith(".safetensors") for name in files):
        raise ValueError("download is missing configuration or safetensors weights")
    manifest = {
        "schema_version": 1,
        "repository": config["repository"],
        "revision": config["revision"],
        "upstream_repository": config["repository"],
        "upstream_revision": config["revision"],
        "license": config["license"],
        "format": "safetensors",
        "conversion": "none",
        "vllm_version": config["vllm_version"],
        "files": files,
    }
    staging = root / (MANIFEST + ".partial")
    staging.write_text(json.dumps(manifest, indent=2) + "\n")
    staging.replace(root / MANIFEST)
    print(
        json.dumps({"model_root": str(root), "revision": config["revision"], "files": len(files)})
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download", "verify", "serve"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("qwen35-9b.toml"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="Verify assets and print launch arguments."
    )
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        root = args.root.resolve()
        if args.action == "download":
            download(root, config, args.allow_network)
            return 0
        manifest = json.loads((root / MANIFEST).read_text())
        for field in ("repository", "revision", "license", "vllm_version"):
            if manifest.get(field) != config[field]:
                raise ValueError("deployment configuration differs from asset receipt")
        verify_assets(root, manifest)
        if args.action == "verify":
            print("Verified model inventory and SHA-256 hashes.")
            return 0
        command = serve_command(root, config)
        if args.dry_run:
            print(json.dumps(command))
            return 0
        if importlib.metadata.version("vllm") != config["vllm_version"]:
            raise ValueError("vLLM runtime version differs from pinned deployment configuration")
        key = os.environ.get("CAPICOVE_VLM_API_KEY", "")
        if not key or len(key) > 8192 or any(not 33 <= ord(c) <= 126 for c in key):
            raise ValueError("set CAPICOVE_VLM_API_KEY before serving")
        environment = {
            **os.environ,
            "VLLM_API_KEY": key,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "VLLM_NO_USAGE_STATS": "1",
            "DO_NOT_TRACK": "1",
        }
        os.execvpe(command[0], command, environment)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.CalledProcessError,
        importlib.metadata.PackageNotFoundError,
    ):
        print(
            "VLM deployment failed: check configuration, verified assets, runtime version and credential environment.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
