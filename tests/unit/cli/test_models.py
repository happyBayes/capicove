from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
from pytest import CaptureFixture

from capicove.cli.commands.models import run


@dataclass(frozen=True)
class _Model:
    key: str
    repository: str
    revision: str
    size_bytes: int
    files: tuple[_ModelFile, ...]


@dataclass(frozen=True)
class _ModelFile:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class _Plan:
    profile_id: str
    profile_version: int
    supported_languages: tuple[str, ...]
    inference_engine: tuple[tuple[str, str], ...]
    pipeline_configuration: tuple[tuple[str, str | int | bool], ...]
    destination: Path
    models: tuple[_Model, ...]
    total_size_bytes: int
    source_hosts: tuple[str, ...]
    manifest_sha256: str


class _Plugin:
    def __init__(self, plan: _Plan) -> None:
        self.plan = plan
        self.install_calls: list[tuple[str, Path, bool]] = []

    def model_plan(self, profile_id: str, destination: Path) -> _Plan:
        assert profile_id == "book-balanced"
        assert destination == self.plan.destination
        return self.plan

    def install_models(self, profile_id: str, destination: Path, *, confirmed: bool) -> _Plan:
        self.install_calls.append((profile_id, destination, confirmed))
        return self.plan


def _arguments(tmp_path: Path, *, dry_run: bool = False, yes: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        models_command="install",
        plugin="paddleocr",
        profile="book-balanced",
        destination=tmp_path,
        dry_run=dry_run,
        yes=yes,
    )


def _plugin(tmp_path: Path) -> _Plugin:
    model = _Model(
        "layout",
        "PaddlePaddle/layout",
        "a" * 40,
        123,
        (_ModelFile("inference.json", 123, "c" * 64),),
    )
    return _Plugin(
        _Plan(
            "book-balanced",
            1,
            ("en", "ja", "zh"),
            (("name", "PP-StructureV3"), ("paddleocr", ">=3.3,<4")),
            (("device", "$requested_device"), ("text_det_limit_side_len", 736)),
            tmp_path,
            (model,),
            123,
            ("huggingface.co",),
            "b" * 64,
        )
    )


def test_models_install_dry_run_emits_immutable_plan(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    plugin = _plugin(tmp_path)

    result = run(
        _arguments(tmp_path, dry_run=True),
        plugin_loader=lambda name: plugin,
        interactive=False,
    )

    assert result == 0
    assert plugin.install_calls == []
    payload = json.loads(capsys.readouterr().out)
    assert payload["operation"] == "model_install_plan"
    assert payload["models"][0]["revision"] == "a" * 40
    assert payload["models"][0]["files"] == [
        {"path": "inference.json", "size_bytes": 123, "sha256": "c" * 64}
    ]
    assert payload["supported_languages"] == ["en", "ja", "zh"]
    assert payload["inference_engine"]["paddleocr"] == ">=3.3,<4"
    assert payload["pipeline_configuration"]["text_det_limit_side_len"] == 736
    assert payload["source_hosts"] == ["huggingface.co"]


def test_models_install_rejects_noninteractive_implicit_confirmation(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    plugin = _plugin(tmp_path)

    result = run(
        _arguments(tmp_path),
        plugin_loader=lambda name: plugin,
        interactive=False,
    )

    assert result == 2
    assert plugin.install_calls == []
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "model_confirmation_required"


def test_models_install_yes_performs_explicit_installation(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    plugin = _plugin(tmp_path)

    result = run(
        _arguments(tmp_path, yes=True),
        plugin_loader=lambda name: plugin,
        interactive=False,
    )

    assert result == 0
    assert plugin.install_calls == [("book-balanced", tmp_path, True)]
    assert json.loads(capsys.readouterr().out)["operation"] == "model_install_complete"


def test_models_install_reports_missing_plugin(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    def missing(name: str) -> object:
        raise LookupError(f"unknown document parser plugin: {name}")

    result = run(
        _arguments(tmp_path, dry_run=True),
        plugin_loader=missing,
        interactive=False,
    )

    assert result == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "model_plugin_unavailable"
    assert "uv pip install -e ./plugins/processor-paddleocr" in payload["error"]["message"]


def test_models_install_converts_downloader_failure_to_stable_error(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    class _FailingPlugin(_Plugin):
        def install_models(self, profile_id: str, destination: Path, *, confirmed: bool) -> _Plan:
            raise subprocess.CalledProcessError(1, ("hf", "download"))

    result = run(
        _arguments(tmp_path, yes=True),
        plugin_loader=lambda name: _FailingPlugin(_plugin(tmp_path).plan),
        interactive=False,
    )

    assert result == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "model_install_failed"
    assert "hf" not in payload["error"]["message"]
    assert "manifest" in payload["error"]["message"]


def test_models_install_never_echoes_downloader_credentials(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    class _FailingPlugin(_Plugin):
        def install_models(self, profile_id: str, destination: Path, *, confirmed: bool) -> _Plan:
            raise subprocess.CalledProcessError(
                1, ("hf", "download", "--token", "private-model-token")
            )

    result = run(
        _arguments(tmp_path, yes=True),
        plugin_loader=lambda name: _FailingPlugin(_plugin(tmp_path).plan),
        interactive=False,
    )

    captured = capsys.readouterr()
    assert result == 2
    assert json.loads(captured.out)["error"]["code"] == "model_install_failed"
    assert "private-model-token" not in captured.out + captured.err


@pytest.mark.parametrize("error_type", [LookupError, RuntimeError, KeyError, OSError])
def test_models_install_never_echoes_plugin_loader_credentials(
    tmp_path: Path, capsys: CaptureFixture[str], error_type: type[Exception]
) -> None:
    def failing_loader(_name: str) -> object:
        raise error_type("private-plugin-token")

    result = run(
        _arguments(tmp_path, dry_run=True),
        plugin_loader=failing_loader,
        interactive=False,
    )

    captured = capsys.readouterr()
    assert result == 2
    assert json.loads(captured.out)["error"]["code"] == "model_plugin_unavailable"
    assert "private-plugin-token" not in captured.out + captured.err


def test_models_install_never_echoes_unexpected_plugin_failure(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    class _FailingPlugin(_Plugin):
        def install_models(self, profile_id: str, destination: Path, *, confirmed: bool) -> _Plan:
            raise KeyError("private-model-token")

    result = run(
        _arguments(tmp_path, yes=True),
        plugin_loader=lambda name: _FailingPlugin(_plugin(tmp_path).plan),
        interactive=False,
    )

    captured = capsys.readouterr()
    assert result == 2
    assert json.loads(captured.out)["error"]["code"] == "model_install_failed"
    assert "private-model-token" not in captured.out + captured.err
