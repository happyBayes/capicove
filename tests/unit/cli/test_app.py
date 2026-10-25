import subprocess
import sys
from pathlib import Path

from pytest import CaptureFixture, MonkeyPatch

from capicove.cli import main


def test_version_prints_capicove_version(capsys: CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.startswith("capicove ")


def test_doctor_reports_runtime_ready(capsys: CaptureFixture[str]) -> None:
    assert main(["doctor"]) == 0

    output = capsys.readouterr().out
    assert "Capicove diagnostics" in output
    assert "status: ok" in output
    assert "Model inference has not been checked" in output


def test_cli_module_runs_the_version_command() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "capicove.cli", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stdout.startswith("capicove ")
    assert completed.stderr == ""


def test_package_module_runs_the_version_command() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "capicove", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stdout.startswith("capicove ")
    assert completed.stderr == ""


def test_package_module_runs_doctor() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "capicove", "doctor"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Capicove diagnostics" in completed.stdout
    assert "status: ok" in completed.stdout
    assert completed.stderr == ""


def test_closed_stdout_stops_quietly_with_sigpipe_status(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    import capicove.cli.app as app

    def closed_reader(*_: object, **__: object) -> int:
        raise BrokenPipeError

    sink = (tmp_path / "stdout").open("w")
    monkeypatch.setattr(app, "_dispatch", closed_reader)
    monkeypatch.setattr(sys, "stdout", sink)
    try:
        assert app.main(["doctor"]) == 141
    finally:
        sink.close()
