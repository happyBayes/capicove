"""Exercise speech onboarding through the normal CLI without real weights."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from capicove.cli.app import main


def test_plan_is_read_only_and_identifies_exact_downloads(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "speech"
    assert main(["audio", "setup-speech", "--root", str(root), "--plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["model_repository"] == "ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF"
    # This is the immutable public model revision.
    revision = "ca27d74bc954b73dadab5b71ca265d87fc861a7c"  # pragma: allowlist secret
    assert plan["model_revision"] == revision
    assert {asset["name"] for asset in plan["models"]} == {
        "Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf",
        "mmproj-Qwen3-TTS-12Hz-1.7B-Base-bf16.gguf",
    }
    assert not root.exists()


def test_download_requires_confirmation_before_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "speech"
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert main(["audio", "setup-speech", "--root", str(root)]) == 2
    assert "--yes" in capsys.readouterr().err
    assert not root.exists()


def test_installer_requires_network_opt_in_before_staging(tmp_path: Path) -> None:
    from capicove.runtime.speech_setup import install_speech

    root = tmp_path / "speech"
    with pytest.raises(ValueError, match="confirm downloads"):
        install_speech(root, None, None, download_authorized=False)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX process fixture")
def test_setup_reuses_install_and_authorizes_reference_for_offline_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import hashlib
    import sys
    import wave

    from capicove.runtime import speech_setup

    assets = tmp_path / "local-assets"
    assets.mkdir()
    fixtures = []
    for original in speech_setup.MODEL_FILES:
        payload = original.name.encode()
        (assets / original.name).write_bytes(payload)
        fixtures.append(
            speech_setup.Asset(original.name, len(payload), hashlib.sha256(payload).hexdigest())
        )
    monkeypatch.setattr(speech_setup, "MODEL_FILES", tuple(fixtures))
    executable = tmp_path / "llama-tts"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import sys, wave\n"
        "if '--help' in sys.argv:\n"
        "    print('--offline --mmproj --tts-speaker-file'); sys.exit(0)\n"
        "assert '--offline' in sys.argv\n"
        "with wave.open(sys.argv[sys.argv.index('--output') + 1], 'wb') as wav:\n"
        "    wav.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))\n"
        "    wav.writeframes(b'\\x01\\x00' * 2400)\n"
        "print('generated 12 frames, fixture')\n"
    )
    executable.chmod(0o700)
    root = tmp_path / "speech"
    command = ["audio", "setup-speech", "--root", str(root)]
    assert (
        main(
            [
                *command,
                "--assets-dir",
                str(assets),
                "--llama-tts",
                str(executable),
                "--test-text",
                "Hello",
            ]
        )
        == 0
    )
    first = json.loads(capsys.readouterr().out)
    assert Path(first["trial"]["path"]).is_file()
    key = (root / "signing.key").read_bytes()
    reference = tmp_path / "reference.wav"
    with wave.open(str(reference), "wb") as wav:
        wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\x01\x00" * 72000)
    assert main([*command, "--reference", str(reference)]) == 2
    capsys.readouterr()
    assert (
        main(
            [
                *command,
                "--reference",
                str(reference),
                "--authorize-reference",
                "--test-text",
                "Hello",
            ]
        )
        == 0
    )
    second = json.loads(capsys.readouterr().out)
    assert (root / "signing.key").read_bytes() == key
    assert (
        second["trial"]["lineage"]["reference_sha256"]
        == hashlib.sha256(reference.read_bytes()).hexdigest()
    )
    assert Path(second["reference"]).read_bytes() == reference.read_bytes()
    assert (
        main(
            [
                "audio",
                "speech-capabilities",
                "--runtime-config",
                second["runtime_config"],
                "--signing-key-file",
                second["signing_key_file"],
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["reference_audio"] is True


@pytest.mark.parametrize("corrupt", [False, True])
def test_confirmed_download_verifies_both_files_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    corrupt: bool,
) -> None:
    import hashlib
    import io

    from capicove.runtime import speech_setup

    assets = tuple(
        speech_setup.Asset(
            asset.name, len(asset.name), hashlib.sha256(asset.name.encode()).hexdigest()
        )
        for asset in speech_setup.MODEL_FILES
    )
    monkeypatch.setattr(speech_setup, "MODEL_FILES", assets)
    calls = []

    class Opener:
        def open(self, url: str, *, timeout: int) -> io.BytesIO:
            calls.append(url)
            name = url.rsplit("/", 1)[1]
            content = name.encode()
            if corrupt and name == assets[1].name:
                content = b"!" + content[1:]
            return io.BytesIO(content)

    monkeypatch.setattr(speech_setup, "build_opener", lambda *_: Opener())
    monkeypatch.setattr(speech_setup, "_check_backend", lambda _: None)
    executable = tmp_path / "llama-tts"
    executable.write_bytes(b"authored binary placeholder")
    executable.chmod(0o700)
    root = tmp_path / "speech"
    code = main(
        [
            "audio",
            "setup-speech",
            "--root",
            str(root),
            "--llama-tts",
            str(executable),
            "--yes",
        ]
    )
    captured = capsys.readouterr()
    assert calls == [
        f"https://huggingface.co/{speech_setup.MODEL_REPOSITORY}/resolve/"
        f"{speech_setup.MODEL_REVISION}/{asset.name}"
        for asset in assets
    ]
    if corrupt:
        assert code == 2
        assert "SHA-256 verification" in captured.err
        assert not root.exists()
        assert not list(tmp_path.glob(".speech.staging-*"))
        assert not (tmp_path / ".speech.setup-lock").exists()
    else:
        assert code == 0
        assert json.loads(captured.out)["capabilities"]["available"] is True
        assert (
            json.loads((root / "provisioning.json").read_text())["models"][1]["sha256"]
            == assets[1].sha256
        )


def test_source_script_delegates_to_cli(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[3] / "scripts" / "setup_speech.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--root", str(tmp_path / "speech"), "--plan"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout)["backend_download"] is True
    assert not (tmp_path / "speech").exists()
