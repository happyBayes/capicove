import json
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image
from pytest import CaptureFixture

from capicove.cli.app import _build_parser, main
from capicove.cli.commands.image import run
from capicove.contracts import InvocationContext, ToolResult, ToolStatus


class FakeHandler:
    def __init__(self) -> None:
        self.calls: list[tuple[InvocationContext, Mapping[str, object]]] = []

    def __call__(self, context: InvocationContext, arguments: Mapping[str, object]) -> ToolResult:
        self.calls.append((context, arguments))
        return ToolResult(
            request_id="request-1",
            trace_id="trace-1",
            tool_id="capicove.image.filter.aspect-ratio",
            tool_version=1,
            status=ToolStatus.SUCCEEDED,
            result={"keep": True},
            evidence_refs=(),
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )


def test_importing_image_cli_does_not_load_image_backends() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import capicove.cli.commands.image; assert 'cv2' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_building_unrelated_cli_help_does_not_load_image_backends() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from capicove.cli.app import main; "
                "main(['video', 'filter', '--help']); assert 'cv2' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_image_cli_maps_multiple_inputs_to_canonical_sources(
    capsys: CaptureFixture[str],
) -> None:
    arguments = _build_parser().parse_args(
        [
            "image",
            "run",
            "--tool",
            "image_aspect_ratio_filter",
            "--input",
            "one.png",
            "--input",
            "two.png",
            "--output",
            "json",
        ]
    )
    handler = FakeHandler()

    assert run(arguments, handler=handler) == 0

    assert handler.calls[0][1] == {
        "sources": [
            {"path": "one.png"},
            {"path": "two.png"},
        ]
    }
    assert json.loads(capsys.readouterr().out)["tool_id"] == ("capicove.image.filter.aspect-ratio")


def test_image_cli_reads_canonical_arguments_file(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    arguments_file = tmp_path / "arguments.json"
    arguments_file.write_text(json.dumps({"sources": [{"path": "one.png"}]}), encoding="utf-8")
    arguments = _build_parser().parse_args(
        [
            "image",
            "run",
            "--tool",
            "image_aspect_ratio_filter",
            "--arguments",
            str(arguments_file),
            "--output",
            "json",
        ]
    )
    handler = FakeHandler()

    assert run(arguments, handler=handler) == 0
    assert handler.calls[0][1] == {"sources": [{"path": "one.png"}]}
    assert json.loads(capsys.readouterr().out)["status"] == "succeeded"


def test_image_cli_exposes_public_phash_and_lsh_tools() -> None:
    parser = _build_parser()

    phash = parser.parse_args(["image", "run", "--tool", "image_phash"])
    search = parser.parse_args(["image", "run", "--tool", "image_lsh_search"])

    assert phash.tool == "image_phash"
    assert search.tool == "image_lsh_search"


def test_image_lsh_search_cli_uses_default_local_capability(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    arguments_file = tmp_path / "lsh.json"
    arguments_file.write_text(
        json.dumps(
            {
                "query": {"hashes": ["0000000000000000"]},
                "candidates": [{"record_id": "match", "hashes": ["0000000000000001"]}],
                "hash_version": "phash-64-v1",
                "max_hamming_distance": 1,
            }
        ),
        encoding="utf-8",
    )
    arguments = _build_parser().parse_args(
        [
            "image",
            "run",
            "--tool",
            "image_lsh_search",
            "--arguments",
            str(arguments_file),
            "--output",
            "json",
        ]
    )

    assert run(arguments) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["matches"][0]["record_id"] == "match"


def test_image_phash_cli_authorizes_paths_from_canonical_records(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    arguments_file = tmp_path / "phash.json"
    arguments_file.write_text(
        json.dumps(
            {
                "records": [{"record_id": "image", "images": [{"path": str(image_path)}]}],
                "enable_lsh": True,
            }
        ),
        encoding="utf-8",
    )
    arguments = _build_parser().parse_args(
        [
            "image",
            "run",
            "--tool",
            "image_phash",
            "--arguments",
            str(arguments_file),
            "--output",
            "json",
        ]
    )

    assert run(arguments) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["records"][0]["record_id"] == "image"
    assert payload["result"]["lineage"]["candidate_strategy"] == "lsh"


def test_unreadable_arguments_file_is_a_usage_error(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    arguments = _build_parser().parse_args(
        ["image", "run", "--tool", "image_phash", "--arguments", str(tmp_path / "missing.json")]
    )

    assert run(arguments) == 2
    assert "--arguments must name a readable UTF-8 JSON file" in capsys.readouterr().err


def test_non_object_arguments_file_is_a_usage_error(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    payload = tmp_path / "arguments.json"
    payload.write_text("[]", encoding="utf-8")
    arguments = _build_parser().parse_args(
        ["image", "run", "--tool", "image_phash", "--arguments", str(payload)]
    )

    assert run(arguments) == 2
    assert "image arguments JSON must contain an object" in capsys.readouterr().err


_PROVIDER_TOOLS = (
    "image_remove_background_mapper",
    "imgdiff_difference_area_generator_mapper",
    "imgdiff_difference_caption_generator_mapper",
    "image_aesthetics_filter",
    "image_nsfw_filter",
    "image_text_matching_filter",
    "image_text_similarity_filter",
    "image_watermark_filter",
    "image_pair_similarity_filter",
    "image_perceptual_deduplicator",
    "ray_image_perceptual_deduplicator",
    "image_ocr",
)


@pytest.mark.parametrize("tool", _PROVIDER_TOOLS)
def test_provider_tool_without_default_composition_is_a_usage_error(
    tmp_path: Path, capsys: CaptureFixture[str], tool: str
) -> None:
    image = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(image)

    assert main(["image", "run", "--tool", tool, "--input", str(image)]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"{tool} requires an explicitly configured local provider" in captured.err
    assert "Traceback" not in captured.err
