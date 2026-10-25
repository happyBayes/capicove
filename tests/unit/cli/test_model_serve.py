"""Installed CLI entry and safe startup policy for the BGE-M3 HTTP service."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from capicove.cli.app import main


def test_help_is_available_without_importing_model_or_web_runtimes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from capicove.cli.app import main; import sys; "
            "main(['models','serve','bge-m3','--help'])",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--model-manifest-sha256" in result.stdout
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "from capicove.cli.app import _build_parser; import sys; "
            "_build_parser().parse_args(['models','serve','bge-m3','--model-directory','/tmp/m',"
            "'--model-manifest-sha256','abc']); "
            "assert not {'fastapi','uvicorn','torch','sentence_transformers'} & sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr


def test_serve_dispatches_selected_local_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from capicove.cli.commands import model_serve

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(model_serve, "serve", lambda **kwargs: calls.append(kwargs))
    assert (
        main(
            [
                "models",
                "serve",
                "bge-m3",
                "--model-directory",
                str(tmp_path),
                "--model-manifest-sha256",
                "a" * 64,
            ]
        )
        == 0
    )
    assert calls[0]["model_directory"] == tmp_path
    assert calls[0]["host"] == "127.0.0.1" and calls[0]["port"] == 8766


def test_remote_bind_requires_and_uses_a_startup_environment_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from capicove.cli.commands import model_serve

    calls: list[dict[str, Any]] = []
    monkeypatch.setenv("CAPICOVE_TEST_BGE_KEY", "test-only-key")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(model_serve, "serve", lambda **kwargs: calls.append(kwargs))
    assert (
        main(
            [
                "models",
                "serve",
                "bge-m3",
                "--model-directory",
                str(tmp_path),
                "--model-manifest-sha256",
                "a" * 64,
                "--host",
                "0.0.0.0",
                "--api-key-env",
                "CAPICOVE_TEST_BGE_KEY",
            ]
        )
        == 0
    )
    # The environment above contains a deliberately synthetic test credential.
    assert calls[0]["api_key"] == "test-only-key"  # pragma: allowlist secret


def test_missing_optional_dependencies_are_reported_without_loading_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert (
        main(
            [
                "models",
                "serve",
                "bge-m3",
                "--model-directory",
                str(tmp_path),
                "--model-manifest-sha256",
                "a" * 64,
            ]
        )
        == 2
    )
    assert "capicove[api,text-embedding]" in capsys.readouterr().err


def test_serve_closes_encoder_when_listener_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import uvicorn

    from capicove.api import bge_m3
    from capicove.cli.commands import model_serve
    from tests.unit.api.test_bge_m3_api import Encoder

    encoder = Encoder()
    monkeypatch.setattr(bge_m3, "build_encoder", lambda *args, **kwargs: encoder)

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("listener unavailable")

    monkeypatch.setattr(uvicorn, "run", fail)
    with pytest.raises(OSError):
        model_serve.serve(
            model_directory=tmp_path,
            manifest_sha256="a" * 64,
            device="cpu",
            batch_size=8,
            max_tokens=512,
            host="127.0.0.1",
            port=0,
            api_key=None,
        )
    assert encoder.closed


@pytest.mark.parametrize(
    "options",
    [
        ["--host", "0.0.0.0"],
        ["--port", "-1"],
        ["--api-key-env", "OPENAI_API_KEY"],
        ["--api-key-env", "CAPICOVE_MISSING_TEST_KEY"],
        ["--max-tokens", "8193"],
    ],
)
def test_invalid_startup_is_rejected_before_loading_model(
    options: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "models",
                "serve",
                "bge-m3",
                "--model-directory",
                str(tmp_path),
                "--model-manifest-sha256",
                "a" * 64,
                *options,
            ]
        )
        == 2
    )
    assert "startup failed" in capsys.readouterr().err.lower()
