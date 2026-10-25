"""Capacity scripts reject invalid requests before generating large artifacts."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def load_script(name: str) -> ModuleType:
    path = Path(__file__).parents[3] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("script unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--count", "19"),
        ("--dimension", "1"),
        ("--queries", "0"),
        ("--threads", "0"),
        ("--ef-search", "4097"),
        ("--seed", "-1"),
    ],
)
def test_ann_invalid_parameters_create_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str, value: str
) -> None:
    output = tmp_path / "absent"
    monkeypatch.setattr("sys.argv", ["benchmark", "--output", str(output), flag, value])
    with pytest.raises(SystemExit) as error:
        load_script("benchmark_video_ann").main()
    assert error.value.code == 2
    assert not output.exists()


def test_reopen_invalid_budget_does_not_access_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark",
            "--root",
            str(tmp_path / "missing"),
            "--report",
            str(tmp_path / "report.json"),
            "--ef-search",
            "0",
        ],
    )
    with pytest.raises(SystemExit) as error:
        load_script("benchmark_video_ann_reopen").main()
    assert error.value.code == 2
    assert not (tmp_path / "report.json").exists()


def test_runtime_reopen_requires_existing_fixture_before_optional_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "resource", None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark",
            "--root",
            str(tmp_path / "missing"),
            "--report",
            str(tmp_path / "report.json"),
            "--model-pid",
            "123",
            "--text-pid",
            "456",
            "--model-manifest-sha256",
            "a" * 64,
        ],
    )
    with pytest.raises(SystemExit) as error:
        load_script("benchmark_video_runtime_reopen").main()
    assert error.value.code == 2
    assert not (tmp_path / "report.json").exists()
