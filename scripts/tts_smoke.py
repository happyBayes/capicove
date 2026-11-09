"""Run a bounded, local-file Qwen TTS experiment, outside the product API.

Provision models and the executable separately. Linux and Windows default to CPU;
macOS uses Metal. This script does not download, install, or expose a server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
import wave
from pathlib import Path


ASSETS = {
    "Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf": "ac7931aeb2e7aad1a6ed6602d353a5679c9d096b18ce8204ac730a8408d572e1",  # pragma: allowlist secret -- Public model SHA-256, not a credential.
    "mmproj-Qwen3-TTS-12Hz-1.7B-Base-bf16.gguf": "b9503e95e44705739cf82c15ce909b040b96a31ad79d02cf4dc5e1484399265d",  # pragma: allowlist secret -- Public model SHA-256, not a credential.
}
REVISION = "ca27d74bc954b73dadab5b71ca265d87fc861a7c"  # pragma: allowlist secret -- Public model revision, not a credential.


def build_command(root, system, text, reference, output):
    """Build argv for an operator-owned local smoke directory."""
    if system not in {"windows", "linux", "darwin"} or not text.strip() or len(text) > 200:
        raise ValueError("unsupported platform or text outside 1-200 characters")
    profiles = Path(__file__).parents[1] / "docs/development/tts-local-profiles.json"
    profile = json.loads(profiles.read_text(encoding="utf-8"))["platforms"][system]
    names = list(ASSETS)
    command = [
        str(root / profile["executable"]),
        "-m",
        str(root / "models" / names[0]),
        "--mmproj",
        str(root / "models" / names[1]),
        "--tts-lang",
        "zh",
        "-p",
        text,
        "-t",
        "2",
        "-ngl",
        str(profile["gpu_layers"]),
        "-n",
        "250",
        "--seed",
        "42",
        "--output",
        str(output),
    ]
    if reference is not None:
        command += ["--tts-speaker-file", str(reference)]
    command += ["--offline"]
    if system != "darwin":
        command += ["--no-mmproj-offload"]
    return command


def digest(path):
    """Hash a file without reading the complete weight into memory."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    """Verify provisioned assets, invoke one process, and retain a smoke receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    reference = args.reference.resolve(strict=True) if args.reference else None
    if output.exists() or output.with_suffix(".json").exists():
        parser.error("output or receipt already exists")
    for name, expected in ASSETS.items():
        if digest(root / "models" / name) != expected:
            parser.error(f"asset integrity failure: {name}")
    command = build_command(root, platform.system().lower(), args.text, reference, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    with output.with_suffix(".log").open("w") as log:
        subprocess.run(command, check=True, timeout=240, env=env, stdout=log, stderr=log)
    elapsed = time.monotonic() - started
    with wave.open(str(output), "rb") as audio:
        duration = audio.getnframes() / audio.getframerate()
        if duration <= 0:
            raise ValueError("empty generated audio")
        sample_rate = audio.getframerate()
    receipt = {
        "accepted": False,
        "scope": "file-generation smoke; completeness and voice similarity unreviewed",
        "model_revision": REVISION,
        "assets": ASSETS,
        "binary_sha256": digest(Path(command[0])),
        "reference_sha256": digest(reference) if reference else None,
        "text_sha256": hashlib.sha256(args.text.encode()).hexdigest(),
        "output_sha256": digest(output),
        "duration_seconds": duration,
        "wall_seconds": elapsed,
        "rtf_including_load": elapsed / duration,
        "sample_rate": sample_rate,
        "system": platform.system(),
    }
    output.with_suffix(".json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
