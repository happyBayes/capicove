"""Verify existing local assets and compile local Vision helpers; never download."""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure(model_paths, output, tesseract):
    if platform.system() != "Darwin":
        raise RuntimeError("The experimental Vision route requires macOS")
    assets = tomllib.loads((ROOT / "assets.lock.toml").read_text())
    for kind, model in assets["models"].items():
        directory = Path(model_paths[kind]).resolve(strict=True)
        for name, expected in model["files"].items():
            if digest(directory / name) != expected["sha256"]:
                raise ValueError(f"Model asset mismatch: {kind}/{name}")
        model["directory"] = str(directory)
    executable = Path(tesseract).resolve(strict=True)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sdk = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True, timeout=30).strip()
    binaries = {"tesseract": {"path": str(executable), "sha256": digest(executable)}}
    for kind, source in {"vision": "recognize.swift", "numeric": "numeric_boxes.swift"}.items():
        target = output / kind
        subprocess.run(
            ["xcrun", "swiftc", "-sdk", sdk, str(ROOT / source), "-o", str(target)],
            check=True,
            timeout=120,
        )
        binaries[kind] = {"path": str(target), "sha256": digest(target)}
    (output / "assets.json").write_text(json.dumps(assets, indent=2) + "\n")
    config = {
        "assets": str(output / "assets.json"),
        "binaries": binaries,
        "repository": str(ROOT.parents[3]),
        "cache_directory": str(output / "runtime-cache"),
    }
    (output / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    return output / "configuration.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for kind in ("en", "el", "server"):
        parser.add_argument(f"--{kind}-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tesseract", default=shutil.which("tesseract"))
    args = parser.parse_args()
    if not args.tesseract:
        parser.error("An existing Tesseract executable is required")
    print(
        configure(
            {kind: getattr(args, f"{kind}_model") for kind in ("en", "el", "server")},
            args.output,
            args.tesseract,
        )
    )


if __name__ == "__main__":
    main()
