"""End-to-end tests for the unified directory batch CLI."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from pytest import CaptureFixture

from capicove.cli.app import main


def test_batch_cli_runs_real_image_tool_and_resumes_from_content_cache(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    (input_root / "frame.png").write_bytes(b"local-image-evidence")
    parameters = tmp_path / "parameters.json"
    parameters.write_text(
        json.dumps({"min_size": "1B", "max_size": "1MiB"}),
        encoding="utf-8",
    )
    arguments = [
        "batch",
        "run",
        "--input",
        str(input_root),
        "--operation",
        "image_size_filter",
        "--parameters",
        str(parameters),
        "--state",
        str(tmp_path / "batch.sqlite3"),
        "--provider-fingerprint",
        "d" * 64,
        "--output",
        "json",
    ]

    assert main(arguments) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["cache_hits"] == 0
    assert first["items"][0]["result"]["source"]["relative_path"] == "frame.png"
    assert first["items"][0]["result"]["tool"]["tool_id"] == ("capicove.image.filter.size")
    assert first["items"][0]["result"]["tool"]["result"]["keep"] is True

    assert main(arguments) == 0
    resumed = json.loads(capsys.readouterr().out)
    assert resumed["plan_id"] == first["plan_id"]
    assert resumed["cache_hits"] == 1
    assert resumed["items"][0]["attempts"] == 1


def test_batch_cli_trace_store_records_actual_tool_attempt(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    (input_root / "frame.png").write_bytes(b"local-image-evidence")
    parameters = tmp_path / "parameters.json"
    parameters.write_text("{}", encoding="utf-8")
    trace_store = tmp_path / "traces.sqlite3"

    assert (
        main(
            [
                "--trace-store",
                str(trace_store),
                "batch",
                "run",
                "--input",
                str(input_root),
                "--operation",
                "image_size_filter",
                "--parameters",
                str(parameters),
                "--state",
                str(tmp_path / "batch.sqlite3"),
                "--provider-fingerprint",
                "e" * 64,
            ]
        )
        == 0
    )

    with sqlite3.connect(trace_store) as connection:
        records = connection.execute(
            "SELECT record_type, record_json FROM trace_records ORDER BY sequence"
        ).fetchall()
    assert [record_type for record_type, _ in records] == [
        "episode.started",
        "tool.call",
        "tool.result",
        "episode.completed",
    ]
    tool_call = json.loads(records[1][1])
    assert tool_call["payload"]["attempt"] == 1
    assert tool_call["payload"]["tool_id"] == "capicove.image.filter.size"
