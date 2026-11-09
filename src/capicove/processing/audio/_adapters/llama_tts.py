"""Explicit local-file llama.cpp speech adapter with bounded child ownership."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import tempfile
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from capicove.processing.audio.synthesis import (
    SpeechFeatures,
    SpeechRequest,
    SynthesisError,
    file_digest,
)


class LlamaSpeechProvider:
    """Own no resident model; every call launches one local executable."""

    features = SpeechFeatures(("zh", "en", "de", "it", "pt", "es", "ja", "ko", "fr", "ru"), True)

    def __init__(
        self,
        executable: Path,
        model: Path,
        mmproj: Path,
        *,
        hashes: Mapping[str, str],
        device: str,
        threads: int,
        max_frames: int,
        backend_version: str,
    ) -> None:
        self.executable, self.model, self.mmproj = executable, model, mmproj
        self.hashes = dict(hashes)
        self.device, self.threads, self.max_frames = device, threads, max_frames
        self.model_id = model.name
        self.lineage: Mapping[str, object] = {
            "provider": "llama-cpp-tts",
            "backend_version": backend_version,
            "hashes": dict(hashes),
            "device": device,
            "threads": threads,
            "max_frames": max_frames,
            "seed": 42,
            "conditioning": "speaker_features",
        }

    def close(self) -> None:
        """No idle processes or GPU contexts are retained."""

    def verify_assets(self) -> None:
        """Check exact local files without loading a model or starting a process."""
        for name, path in (
            ("executable", self.executable),
            ("model", self.model),
            ("mmproj", self.mmproj),
        ):
            if file_digest(path) != self.hashes[name]:
                raise SynthesisError("asset_integrity")

    def synthesize(self, request: SpeechRequest, output: Path, *, deadline: datetime) -> None:
        """Launch a local child; kill and reap it on deadline or interruption."""
        self.verify_assets()
        if datetime.now(UTC) >= deadline:
            raise SynthesisError("timeout")
        command = [
            str(self.executable),
            "-m",
            str(self.model),
            "--mmproj",
            str(self.mmproj),
            "--tts-lang",
            request.language,
            "-p",
            request.text,
            "-t",
            str(self.threads),
            "-ngl",
            "0" if self.device == "cpu" else "99",
            "-n",
            str(self.max_frames),
            "--seed",
            "42",
            "--offline",
            "--verbosity",
            "3",
            "--output",
            str(output),
        ]
        if self.device == "cpu":
            command += ["--no-mmproj-offload"]
        if request.reference:
            command += ["--tts-speaker-file", str(request.reference)]
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("LLAMA_", "HF_", "HUGGING_FACE_"))
        }
        env.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
        with tempfile.TemporaryFile() as log:
            process = subprocess.Popen(
                command,
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                env=env,
                start_new_session=os.name != "nt",
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                if os.name == "nt"
                else 0,
            )
            try:
                while process.poll() is None:
                    if datetime.now(UTC) >= deadline:
                        raise SynthesisError("timeout")
                    if os.fstat(log.fileno()).st_size > 1024 * 1024 or (
                        output.exists() and output.stat().st_size > 16 * 1024 * 1024
                    ):
                        raise SynthesisError("output_invalid")
                    time.sleep(0.05)
                if process.returncode:
                    raise SynthesisError("generation_failed")
                log.seek(0)
                diagnostic = log.read(1024 * 1024).decode(errors="replace")
                match = re.search(r"generated (\d+) frames,", diagnostic)
                if not match or int(match[1]) >= self.max_frames:
                    raise SynthesisError("completion_unverified")
            finally:
                if process.poll() is None:
                    if os.name == "nt":
                        subprocess.run(
                            [
                                str(Path(os.environ["SystemRoot"]) / "System32/taskkill.exe"),
                                "/PID",
                                str(process.pid),
                                "/T",
                                "/F",
                            ],
                            check=False,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
