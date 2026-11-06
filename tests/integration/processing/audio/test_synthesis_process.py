"""Exercise owned native processes with authored scripts, never real models."""

import hashlib
import os
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from capicove.processing.audio._adapters.llama_tts import LlamaSpeechProvider
from capicove.processing.audio.synthesis import SpeechRequest, SynthesisError


pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX executable fixture")


def provider(root: Path, body: str) -> LlamaSpeechProvider:
    executable = root / "authored-tts"
    executable.write_text("#!" + sys.executable + "\n" + body)
    executable.chmod(0o700)
    model = root / "authored-model.gguf"
    mmproj = root / "authored-mmproj.gguf"
    model.write_bytes(b"authored placeholder, not weights")
    mmproj.write_bytes(b"authored placeholder, not weights")
    return LlamaSpeechProvider(
        executable,
        model,
        mmproj,
        hashes={
            key: hashlib.sha256(path.read_bytes()).hexdigest()
            for key, path in (("executable", executable), ("model", model), ("mmproj", mmproj))
        },
        device="cpu",
        threads=2,
        max_frames=10,
        backend_version="authored-process-fixture",
    )


def deadline() -> datetime:
    return datetime.now(UTC) + timedelta(seconds=5)


def test_native_argv_is_offline_and_treats_text_as_literal(tmp_path: Path) -> None:
    output = tmp_path / "out.wav"
    marker = tmp_path / "must-not-exist"
    adapter = provider(
        tmp_path,
        """import pathlib, sys
assert "--offline" in sys.argv
assert "--no-mmproj-offload" in sys.argv
assert "-hf" not in sys.argv
assert sys.argv[sys.argv.index("-t") + 1] == "2"
pathlib.Path(sys.argv[sys.argv.index("--output") + 1]).write_bytes(b"authored artifact")
print("generated 8 frames, authored fixture")
""",
    )
    adapter.synthesize(SpeechRequest("$(touch " + str(marker) + ")"), output, deadline=deadline())
    assert output.read_bytes() == b"authored artifact"
    assert not marker.exists()


@pytest.mark.parametrize("diagnostic", ["generated 10 frames, capped", "no completion evidence"])
def test_capped_or_unverified_generation_cannot_claim_completion(
    tmp_path: Path,
    diagnostic: str,
) -> None:
    adapter = provider(tmp_path, "print(" + repr(diagnostic) + ")\n")
    with pytest.raises(SynthesisError, match="completion_unverified"):
        adapter.synthesize(SpeechRequest("Hello"), tmp_path / "out.wav", deadline=deadline())


def test_native_nonzero_exit_is_a_failure(tmp_path: Path) -> None:
    adapter = provider(tmp_path, "import sys\nsys.exit(7)\n")
    with pytest.raises(SynthesisError, match="generation_failed"):
        adapter.synthesize(SpeechRequest("Hello"), tmp_path / "out.wav", deadline=deadline())


def test_expired_deadline_never_starts_native_process(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-exist"
    adapter = provider(
        tmp_path, "import pathlib\npathlib.Path(" + repr(str(marker)) + ").touch()\n"
    )
    with pytest.raises(SynthesisError, match="timeout"):
        adapter.synthesize(
            SpeechRequest("Hello"),
            tmp_path / "out.wav",
            deadline=datetime.now(UTC) - timedelta(seconds=1),
        )
    assert not marker.exists()


def test_timeout_kills_and_reaps_owned_process(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    adapter = provider(
        tmp_path,
        "import os, pathlib, time\npathlib.Path("
        + repr(str(pid_file))
        + ").write_text(str(os.getpid()))\ntime.sleep(30)\n",
    )
    with pytest.raises(SynthesisError, match="timeout"):
        adapter.synthesize(
            SpeechRequest("Hello"),
            tmp_path / "out.wav",
            deadline=datetime.now(UTC) + timedelta(seconds=1),
        )
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_asset_change_prevents_process_launch(tmp_path: Path) -> None:
    adapter = provider(tmp_path, "raise AssertionError('must not execute')\n")
    adapter.model.write_bytes(b"changed placeholder")
    with pytest.raises(SynthesisError, match="asset_integrity"):
        adapter.synthesize(SpeechRequest("Hello"), tmp_path / "out.wav", deadline=deadline())


def test_cancellation_reaps_owned_native_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = provider(tmp_path, "import time\ntime.sleep(30)\n")
    original = subprocess.Popen
    children: list[subprocess.Popen[bytes]] = []

    def capture(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        process = original(*args, **kwargs)
        children.append(process)
        return process

    def cancel(seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "Popen", capture)
    monkeypatch.setattr(time, "sleep", cancel)
    with pytest.raises(KeyboardInterrupt):
        adapter.synthesize(SpeechRequest("Hello"), tmp_path / "out.wav", deadline=deadline())
    assert len(children) == 1
    assert children[0].returncode is not None
