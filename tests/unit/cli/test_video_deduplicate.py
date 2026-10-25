import argparse
import json
from pathlib import Path

from pytest import CaptureFixture, MonkeyPatch

from capicove.cli.app import _build_parser, main
from capicove.cli.commands.video_deduplicate import run


def _parse(arguments: list[str]) -> argparse.Namespace:
    return _build_parser().parse_args(["video", "deduplicate", *arguments])


def test_print_unique_is_pipeline_safe(capsys: CaptureFixture[str], tmp_path: Path) -> None:
    keeper = tmp_path / "a.mp4"
    duplicate = tmp_path / "b.mp4"
    keeper.write_bytes(b"same")
    duplicate.write_bytes(b"same")

    code = run(_parse([str(tmp_path), "--print-unique"]))

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.splitlines() == [str(keeper.resolve())]
    assert captured.err == ""


def test_print_duplicates_emits_keeper_and_eliminated_json(
    capsys: CaptureFixture[str], tmp_path: Path
) -> None:
    keeper = tmp_path / "a.mp4"
    duplicate = tmp_path / "b.mp4"
    keeper.write_bytes(b"same")
    duplicate.write_bytes(b"same")

    assert run(_parse([str(tmp_path), "--print-duplicates"])) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["keeper"] == str(keeper.resolve())
    assert payload["eliminated"] == [str(duplicate.resolve())]


def test_delete_without_force_returns_usage_error(
    capsys: CaptureFixture[str], tmp_path: Path
) -> None:
    assert run(_parse([str(tmp_path), "--delete-duplicates"])) == 2
    assert "requires --force" in capsys.readouterr().err


def test_app_dispatches_video_deduplicate(monkeypatch: MonkeyPatch) -> None:
    observed: list[argparse.Namespace] = []

    def fake_run(arguments: argparse.Namespace) -> int:
        observed.append(arguments)
        return 0

    monkeypatch.setattr("capicove.cli.app.run_video_deduplicate", fake_run)

    assert main(["video", "deduplicate", "clips"]) == 0
    assert observed[0].video_command == "deduplicate"
