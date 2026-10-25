"""Lightweight parser and generic configured VLM command composition."""

import json
from pathlib import Path
from typing import Any

import pytest

from capicove.cli.app import _build_parser
from capicove.cli.commands.models import run
from tests.unit.model.test_vlm import configuration


def test_vlm_models_lists_generic_registration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "models.json"
    path.write_text(json.dumps(configuration("another/served-vision")))
    args = _build_parser().parse_args(["models", "vlm", "list", "--config", str(path)])
    assert run(args) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["data"][0]["upstream_model"] == "another/served-vision"


def test_vlm_nonloopback_service_requires_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "models.json"
    path.write_text(json.dumps(configuration()))
    args = _build_parser().parse_args(
        ["models", "vlm", "serve", "--config", str(path), "--host", "0.0.0.0"]
    )
    assert run(args) == 2
    assert "non-loopback" in capsys.readouterr().err


def test_vlm_cli_generate_uses_factory_and_preserves_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from tests.unit.api.test_vlm_api import FakeVlm, factory

    path = tmp_path / "models.json"
    path.write_text(json.dumps(configuration()))
    payload = tmp_path / "request.json"
    payload.write_text(json.dumps({"messages": [{"role": "user", "content": "describe"}]}))

    def runtime(*args: Any, **kwargs: Any) -> Any:
        return factory(FakeVlm())

    monkeypatch.setattr("capicove.cli.commands.vlm.ModelFactory", runtime)
    args = _build_parser().parse_args(
        [
            "models",
            "vlm",
            "generate",
            "--config",
            str(path),
            "--binding",
            "assistant.vision",
            "--input",
            str(payload),
        ]
    )
    assert run(args) == 0
    assert json.loads(capsys.readouterr().out)["choices"][0]["message"]["content"] == "red square"
