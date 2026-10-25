"""Public CLI validation failures remain controlled operator diagnostics."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("module", ["capicove", "capicove.cli"])
@pytest.mark.parametrize(
    "scenario",
    [
        "missing-image-tool",
        "empty-criteria",
        "object-criteria",
        "broken-json",
        "non-utf8",
        "image-inline-arguments",
        "image-broken-arguments",
        "image-array-arguments",
        "image-missing-sources",
    ],
)
def test_invalid_media_arguments_exit_without_traceback(
    tmp_path: Path, module: str, scenario: str
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"source-must-not-be-processed")
    if scenario == "missing-image-tool":
        arguments = ["image", "run", "--input", str(source)]
        expected = "--tool"
    elif scenario == "image-inline-arguments":
        arguments = ["image", "run", "--tool", "image_phash", "--arguments", '{"sources": []}']
        expected = "--arguments must name a readable UTF-8 JSON file"
    elif scenario.startswith("image-"):
        payload = tmp_path / "arguments.json"
        payload.write_bytes(
            {
                "image-broken-arguments": b"{",
                "image-array-arguments": b"[]",
                "image-missing-sources": b"",
            }[scenario]
        )
        arguments = ["image", "run", "--tool", "image_phash"]
        if scenario == "image-missing-sources":
            expected = "at least one --input or --arguments is required"
        else:
            arguments += ["--arguments", str(payload)]
            expected = (
                "image arguments JSON must contain an object"
                if scenario == "image-array-arguments"
                else "--arguments must name a readable UTF-8 JSON file"
            )
    else:
        criteria = tmp_path / "criteria.json"
        criteria.write_bytes(
            {
                "empty-criteria": b"[]",
                "object-criteria": b"{}",
                "broken-json": b"{",
                "non-utf8": b"\xff",
            }[scenario]
        )
        arguments = [
            "video",
            "filter",
            "--input",
            str(source),
            "--criteria",
            str(criteria),
            "--output",
            "json",
        ]
        expected = (
            "criteria JSON must contain a non-empty array"
            if scenario in {"empty-criteria", "object-criteria"}
            else "criteria must be a readable UTF-8 JSON file"
        )
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    completed = subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert expected in completed.stderr
    assert "Traceback" not in completed.stderr
    assert completed.stdout == ""
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pipe semantics")
def test_closed_stdout_pipe_exits_without_traceback(tmp_path: Path) -> None:
    read_end, write_end = os.pipe()
    os.close(read_end)
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "capicove", "doctor"],
            cwd=tmp_path,
            stdout=write_end,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    finally:
        os.close(write_end)

    assert completed.returncode == 141
    assert completed.stderr == ""
