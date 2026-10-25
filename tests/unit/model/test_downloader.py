"""Tests for verified Hugging Face image-model downloads."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path

import pytest

from capicove.model.downloader import (
    DownloadOptions,
    download_models,
    render_download_commands,
)
from capicove.model.manifest import ImageModelSpec, load_packaged_manifest
from capicove.model.receipt import RECEIPT_NAME


def _model(key: str) -> ImageModelSpec:
    return next(model for model in load_packaged_manifest().models if model.key == key)


def test_dry_run_renders_pinned_hf_commands_without_writes(tmp_path: Path) -> None:
    destination = tmp_path / "models"

    commands = render_download_commands(
        (_model("clip-vit-b32"),),
        destination,
        max_workers=4,
    )

    assert commands[0][0:5] == (
        "hf",
        "download",
        "openai/clip-vit-base-patch32",
        "--revision",
        "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",  # pragma: allowlist secret
    )
    assert commands[0][-2:] == ("--max-workers", "4")
    assert commands[1][0:6] == (
        "hf",
        "cache",
        "verify",
        "openai/clip-vit-base-patch32",
        "--revision",
        "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",  # pragma: allowlist secret
    )
    assert not destination.exists()


def _fake_hf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    executable = tmp_path / "bin" / "hf"
    executable.parent.mkdir()
    log = tmp_path / "hf.log"
    script = f"""#!{sys.executable}
import json
import os
import pathlib
import sys

args = sys.argv[1:]
log = pathlib.Path(os.environ["FAKE_HF_LOG"])
with log.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({{"args": args, "token_present": bool(os.environ.get("HF_TOKEN"))}}) + "\\n")
if args == ["version"]:
    print("hf 1.2.3")
elif args[0] == "download":
    local_dir = pathlib.Path(args[args.index("--local-dir") + 1])
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "config.json").write_text('{{"model":"locked"}}', encoding="utf-8")
    metadata = local_dir / ".cache" / "huggingface" / "download"
    metadata.mkdir(parents=True)
    (metadata / "config.json.metadata").write_text("ephemeral", encoding="utf-8")
elif args[:2] == ["cache", "verify"]:
    local_dir = pathlib.Path(args[args.index("--local-dir") + 1])
    if not (local_dir / "config.json").is_file():
        raise SystemExit(7)
    if (local_dir / ".cache" / "huggingface").exists():
        raise SystemExit(9)
else:
    raise SystemExit(8)
"""
    executable.write_text(script, encoding="utf-8")
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("FAKE_HF_LOG", str(log))
    monkeypatch.setenv("PATH", f"{executable.parent}{os.pathsep}{os.environ.get('PATH', '')}")
    return log


def test_download_verifies_writes_receipt_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = _fake_hf(tmp_path, monkeypatch)
    manifest = load_packaged_manifest()
    model = _model("clip-vit-b32")
    destination = tmp_path / "models"
    monkeypatch.setenv("PRIVATE_HF_TOKEN", "do-not-record-this-token")
    options = DownloadOptions(
        destination=destination,
        max_workers=3,
        token_environment="PRIVATE_HF_TOKEN",
    )

    installed = download_models(
        (model,),
        manifest_sha256=manifest.source_sha256,
        options=options,
    )

    assert installed[0].status == "installed"
    assert installed[0].path == destination / "clip-vit-b32"
    receipt_path = installed[0].path / RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_file_hash = hashlib.sha256(b'{"model":"locked"}').hexdigest()
    expected_revision = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"  # pragma: allowlist secret
    assert receipt["repository"] == "openai/clip-vit-base-patch32"
    assert receipt["revision"] == expected_revision
    assert receipt["hf_version"] == "hf 1.2.3"
    assert receipt["files"] == [{"path": "config.json", "sha256": expected_file_hash, "size": 18}]
    assert not (installed[0].path / ".cache").exists()
    assert installed[0].model_manifest_sha256 == receipt["model_manifest_sha256"]
    assert "do-not-record-this-token" not in receipt_path.read_text(encoding="utf-8")
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert calls[1]["args"][0:5] == [
        "download",
        "openai/clip-vit-base-patch32",
        "--revision",
        "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",  # pragma: allowlist secret
        "--local-dir",
    ]
    assert calls[1]["token_present"] is True
    assert calls[2]["args"][0:3] == ["cache", "verify", "openai/clip-vit-base-patch32"]

    available = download_models(
        (model,),
        manifest_sha256=manifest.source_sha256,
        options=options,
    )

    assert available[0].status == "available"
    assert available[0].model_manifest_sha256 == receipt["model_manifest_sha256"]
    assert len(log.read_text(encoding="utf-8").splitlines()) == len(calls)


def test_download_rejects_untrusted_existing_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_hf(tmp_path, monkeypatch)
    target = tmp_path / "models" / "clip-vit-b32"
    target.mkdir(parents=True)
    (target / "untrusted.bin").write_bytes(b"untrusted")

    with pytest.raises(ValueError, match="valid Capicove model receipt"):
        download_models(
            (_model("clip-vit-b32"),),
            manifest_sha256=load_packaged_manifest().source_sha256,
            options=DownloadOptions(destination=tmp_path / "models"),
        )

    assert (target / "untrusted.bin").read_bytes() == b"untrusted"


def test_download_rejects_content_that_no_longer_matches_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_hf(tmp_path, monkeypatch)
    manifest = load_packaged_manifest()
    destination = tmp_path / "models"
    options = DownloadOptions(destination=destination)
    download_models(
        (_model("clip-vit-b32"),),
        manifest_sha256=manifest.source_sha256,
        options=options,
    )
    (destination / "clip-vit-b32" / "config.json").write_text(
        '{"model":"tampered"}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="contents do not match its receipt"):
        download_models(
            (_model("clip-vit-b32"),),
            manifest_sha256=manifest.source_sha256,
            options=options,
        )


def test_missing_hf_fails_before_destination_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    destination = tmp_path / "models"

    with pytest.raises(RuntimeError, match="Install it with"):
        download_models(
            (_model("clip-vit-b32"),),
            manifest_sha256=load_packaged_manifest().source_sha256,
            options=DownloadOptions(destination=destination),
        )

    assert not destination.exists()
