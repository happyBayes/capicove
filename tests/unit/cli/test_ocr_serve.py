"""Portable OCR listener selection and fail-closed startup."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from capicove.cli import main


def test_help_does_not_import_optional_runtimes() -> None:
    code = """import sys
from capicove.cli.app import _build_parser
_build_parser().parse_args(['models','serve','paddleocr','--model-directory','/tmp/models'])
assert not {'fastapi','uvicorn','paddleocr','paddle','torch'} & sys.modules.keys()
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_paddle_dispatches_portable_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from capicove.cli.commands import ocr_serve

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(ocr_serve, "serve", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setenv("CAPICOVE_TEST_OCR_KEY", "test-only-key")
    assert (
        main(
            [
                "models",
                "serve",
                "paddleocr",
                "--model-directory",
                str(tmp_path),
                "--host",
                "0.0.0.0",
                "--port",
                "8767",
                "--api-key-env",
                "CAPICOVE_TEST_OCR_KEY",
                "--max-pixels",
                "123456",
            ]
        )
        == 0
    )
    assert calls[0]["model_directory"] == tmp_path
    assert calls[0]["max_pixels"] == 123456
    # Synthetic credential used only by the test listener.
    assert calls[0]["api_key"] == "test-only-key"  # pragma: allowlist secret


@pytest.mark.parametrize(
    "options",
    [
        ["--host", "0.0.0.0"],
        ["--device", "cuda"],
        ["--max-pixels", "0"],
        ["--max-body-bytes", "0"],
        ["--api-key-env", "OTHER_KEY"],
        ["--port", "65536"],
    ],
)
def test_invalid_startup_does_not_load_models(
    options: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["models", "serve", "paddleocr", "--model-directory", str(tmp_path), *options]) == 2
    assert "startup failed" in capsys.readouterr().err.lower()


def test_provider_warms_once_and_closes_on_warmup_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from capicove.cli.commands.ocr_serve import build_provider
    from capicove.processing.image.ocr_configuration import ImageOcrConfiguration
    from tests.unit.api.test_ocr_api import Provider

    provider = Provider()
    monkeypatch.setattr(ImageOcrConfiguration, "create_provider", lambda self: provider)
    assert build_provider(tmp_path, device="cpu", enable_mkldnn=False) is provider
    assert provider.calls == 1
    assert not provider.closed
    provider.fail = True
    with pytest.raises(RuntimeError, match="private"):
        build_provider(tmp_path, device="cpu", enable_mkldnn=False)
    assert provider.closed


def test_missing_web_dependencies_report_actionable_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert main(["models", "serve", "paddleocr", "--model-directory", str(tmp_path)]) == 2
    assert "capicove[api]" in capsys.readouterr().err


def test_native_runtime_loads_before_listener_signal_handlers_and_closes_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import uvicorn

    from capicove.cli.commands import ocr_serve
    from tests.unit.api.test_ocr_api import Provider

    events: list[str] = []
    provider = Provider()

    def build(*args: Any, **kwargs: Any) -> Provider:
        events.append("provider")
        return provider

    def listen(*args: Any, **kwargs: Any) -> None:
        assert events == ["provider"]
        raise OSError("listener failed")

    monkeypatch.setattr(ocr_serve, "build_provider", build)
    monkeypatch.setattr(ocr_serve, "build_document_parser", lambda *args, **kwargs: None)
    monkeypatch.setattr(uvicorn, "run", listen)
    with pytest.raises(OSError, match="listener failed"):
        ocr_serve.serve(
            model_directory=tmp_path,
            device="cpu",
            enable_mkldnn=False,
            host="127.0.0.1",
            port=0,
            api_key=None,
            max_body_bytes=1024,
            max_pixels=100,
        )
    assert provider.closed


def test_document_limits_are_dispatched_and_invalid_limits_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from capicove.cli.commands import ocr_serve

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(ocr_serve, "serve", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    args = ["models", "serve", "paddleocr", "--model-directory", str(tmp_path)]
    assert (
        main(
            [
                *args,
                "--max-document-pages",
                "2",
                "--max-document-bytes",
                "5000",
                "--document-temp-directory",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert calls[0]["max_document_pages"] == 2
    assert calls[0]["max_document_bytes"] == 5000
    assert calls[0]["document_temp_directory"] == tmp_path
    assert main([*args, "--max-document-pages", "0"]) == 2
    assert main([*args, "--max-document-page-points", "0"]) == 2
    assert main([*args, "--max-document-bytes", "0"]) == 2


def test_document_parser_is_installed_through_public_plugin_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    from capicove.cli.commands.ocr_serve import build_document_parser
    from capicove.plugins import PluginAvailability
    from tests.unit.api.test_document_api import Parser

    parser = Parser()
    configurations: list[dict[str, object]] = []

    class Plugin:
        def availability(self) -> PluginAvailability:
            return PluginAvailability(True)

        def create(self, configuration: dict[str, object]) -> Parser:
            configurations.append(configuration)
            return parser

        def model_plan(self, profile_id: str, destination: Path) -> object:
            return SimpleNamespace(profile_id=profile_id, profile_version=17)

        def install_models(self, *args: Any, **kwargs: Any) -> object:
            raise AssertionError("serving must not install models")

    monkeypatch.setattr("capicove.plugins.load_document_parser_plugin", lambda name: Plugin())
    binding = build_document_parser(tmp_path, device="cpu", enable_mkldnn=False)
    assert binding.backend is parser
    assert binding.profile_version == 17
    assert configurations == [
        {"model_root": str(tmp_path), "device": "cpu", "enable_mkldnn": False}
    ]
