"""Tests for `capicove traces export|validate|inspect` translation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pytest import CaptureFixture

from capicove.cli import main


def test_traces_export_validate_and_inspect_empty_local_store(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    store = tmp_path / "traces.sqlite3"
    output = tmp_path / "export"

    assert (
        main(
            [
                "traces",
                "export",
                "--store",
                str(store),
                "--output",
                str(output),
                "--format",
                "json",
            ]
        )
        == 0
    )
    export_payload = json.loads(capsys.readouterr().out)
    assert export_payload["records"] == 0
    assert export_payload["episodes"] == 0
    assert output.is_dir()

    assert main(["traces", "validate", str(output), "--output", "json"]) == 0
    validation_payload = json.loads(capsys.readouterr().out)
    assert validation_payload["valid"] is True

    assert main(["traces", "inspect", str(output), "--output", "json"]) == 0
    inspect_payload = json.loads(capsys.readouterr().out)
    assert inspect_payload["record_count"] == 0
    assert inspect_payload["export_id"] == export_payload["export_id"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX no-follow export directory handles")
def test_traces_validate_cli_rejects_symlinked_export_root(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    output = tmp_path / "export"
    assert (
        main(
            [
                "traces",
                "export",
                "--store",
                str(tmp_path / "traces.sqlite3"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    capsys.readouterr()
    alias = tmp_path / "alias"
    alias.symlink_to(output, target_is_directory=True)

    assert main(["traces", "validate", str(alias), "--output", "json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert not report["valid"]
    assert any(issue["code"] == "EXPORT_UNSAFE_PATH" for issue in report["issues"])

    with pytest.raises(SystemExit) as captured:
        main(["traces", "inspect", str(alias), "--output", "json"])
    assert captured.value.code == 2
    assert str(output) not in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments",
    [
        ["traces", "export"],
        ["traces", "export", "--store", "db", "--output", "out", "--media-mode", "copy"],
        [
            "traces",
            "export",
            "--store",
            "db",
            "--output",
            "out",
            "--media-root",
            "relative",
        ],
        [
            "traces",
            "export",
            "--store",
            "db",
            "--output",
            "out",
            "--started-after",
            "not-a-time",
        ],
        [
            "traces",
            "export",
            "--store",
            "db",
            "--output",
            "out",
            "--max-records-per-shard",
            "0",
        ],
    ],
)
def test_traces_usage_errors_exit_two(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as captured:
        main(arguments)
    assert captured.value.code == 2


def test_export_supports_all_episode_filters_and_rejects_existing_output(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    store = tmp_path / "traces.sqlite3"
    output = tmp_path / "export"
    arguments = [
        "traces",
        "export",
        "--store",
        str(store),
        "--output",
        str(output),
        "--episode-id",
        "episode-1",
        "--started-after",
        "2026-08-18T12:00:00.000000Z",
        "--completed-before",
        "2026-08-19T12:00:00.000000Z",
        "--modality",
        "image",
        "--tool-id",
        "capicove.image.ocr",
        "--tool-status",
        "succeeded",
        "--episode-status",
        "succeeded",
        "--include-incomplete",
        "--include-restricted",
        "--max-records-per-shard",
        "10",
        "--max-bytes-per-shard",
        "1024",
    ]

    assert main(arguments) == 0
    assert "export_id:" in capsys.readouterr().out
    with pytest.raises(SystemExit) as captured:
        main(arguments)
    assert captured.value.code == 2
    assert str(store) not in capsys.readouterr().err


def test_copy_mode_requires_absolute_media_root(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as captured:
        main(
            [
                "traces",
                "export",
                "--store",
                str(tmp_path / "store.sqlite3"),
                "--output",
                str(tmp_path / "export"),
                "--media-mode",
                "copy",
                "--media-root",
                "relative",
            ]
        )
    assert captured.value.code == 2


def test_trace_export_maps_filesystem_failure_to_path_safe_usage_error(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    store = tmp_path / "traces.sqlite3"
    output = tmp_path / "missing-parent" / "export"

    with pytest.raises(SystemExit) as captured:
        main(
            [
                "traces",
                "export",
                "--store",
                str(store),
                "--output",
                str(output),
            ]
        )

    assert captured.value.code == 2
    assert str(output) not in capsys.readouterr().err
