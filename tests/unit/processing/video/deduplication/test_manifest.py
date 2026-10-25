import csv
import json
from pathlib import Path

import pytest

from capicove.processing.video.deduplication import (
    VideoDeduplicationResult,
    deduplicate_video_files,
    write_deduplication_manifest,
)


def _result(tmp_path: Path) -> VideoDeduplicationResult:
    keeper = tmp_path / "a.mp4"
    duplicate = tmp_path / "b.mp4"
    keeper.write_bytes(b"same")
    duplicate.write_bytes(b"same")
    return deduplicate_video_files((keeper, duplicate))


def test_json_marks_keeper_copy_and_eliminated_skip(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"

    write_deduplication_manifest(output, _result(tmp_path))

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert [item["upload_mode"] for item in payload["items"]] == ["copy", "skip"]
    assert payload["items"][1]["keeper_path"].endswith("a.mp4")


def test_csv_has_the_same_upload_decisions(tmp_path: Path) -> None:
    output = tmp_path / "manifest.csv"

    write_deduplication_manifest(output, _result(tmp_path))

    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["upload_mode"] for row in rows] == ["copy", "skip"]
    assert rows[0]["size_bytes"] == "4"


def test_unknown_manifest_extension_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"json or \.csv"):
        write_deduplication_manifest(tmp_path / "manifest.txt", _result(tmp_path))
