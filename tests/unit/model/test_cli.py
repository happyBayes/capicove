"""Tests for the explicit image-model download command."""

from __future__ import annotations

from pathlib import Path

import pytest

from capicove.model.cli import main
from capicove.model.downloader import InstalledModel


def test_cli_dry_run_prints_locked_revision_without_hf(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "models"
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))

    result = main(
        [
            "--model",
            "clip-vit-b32",
            "--destination",
            str(destination),
            "--dry-run",
        ]
    )

    output = capsys.readouterr()
    assert result == 0
    assert "openai/clip-vit-base-patch32" in output.out
    assert "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268" in output.out  # pragma: allowlist secret
    assert "license=mit" in output.out
    assert output.err == ""
    assert not destination.exists()


def test_cli_requires_a_selection(capsys: pytest.CaptureFixture[str]) -> None:
    result = main([])

    output = capsys.readouterr()
    assert result == 2
    assert "select --profile default or at least one --model" in output.err


def test_cli_default_profile_excludes_restricted_model(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main(
        [
            "--profile",
            "default",
            "--destination",
            str(tmp_path / "models"),
            "--dry-run",
        ]
    )

    output = capsys.readouterr()
    assert result == 0
    assert "nsfw-falconsai" in output.out
    assert "aesthetics-cafe" not in output.out


def test_cli_requires_explicit_restricted_authorization(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main(
        [
            "--model",
            "aesthetics-cafe",
            "--destination",
            str(tmp_path / "models"),
            "--dry-run",
        ]
    )

    output = capsys.readouterr()
    assert result == 2
    assert "--include-restricted" in output.err


def test_cli_prints_external_model_manifest_trust_anchor(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "models" / "clip-vit-b32"
    monkeypatch.setattr(
        "capicove.model.cli.download_models",
        lambda *args, **kwargs: (
            InstalledModel(
                key="clip-vit-b32",
                path=model_path,
                repository="openai/clip-vit-base-patch32",
                revision="a" * 40,
                model_manifest_sha256="b" * 64,
                status="installed",
            ),
        ),
    )

    result = main(
        [
            "--model",
            "clip-vit-b32",
            "--destination",
            str(tmp_path / "models"),
        ]
    )

    output = capsys.readouterr()
    assert result == 0
    assert f"model_manifest_sha256={'b' * 64}" in output.out
