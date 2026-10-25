"""RFC-0021: local video Tools in the version-2 workspace host."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from capicove.cli.app import main
from capicove.runtime.mcp_config import WorkspaceHostConfiguration, load_mcp_host_config
from capicove.runtime.mcp_host import compose_mcp_host


FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
pytestmark = [
    pytest.mark.skipif(os.name != "posix", reason="host media handoff is POSIX-only (ADR-0026)"),
    pytest.mark.skipif(FFMPEG is None or FFPROBE is None, reason="needs local FFmpeg"),
]
ALL_TOOLS = ["capicove.video.deduplicate", "capicove.video.filter", "capicove.video.split"]


def _render(target: Path, source: str) -> None:
    assert FFMPEG is not None
    subprocess.run(
        [
            *(FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", source),
            *("-c:v", "libx264", "-g", "10", "-pix_fmt", "yuv420p", str(target)),
        ],
        check=True,
        timeout=60,
    )


@pytest.fixture(scope="session")
def rendered(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("video-fixtures")
    _render(directory / "moving.mp4", "testsrc=duration=4:size=64x48:rate=10")
    _render(directory / "still.mp4", "color=c=gray:duration=4:size=64x48:rate=10")
    return directory


@dataclass(frozen=True)
class VideoSpace:
    root: Path
    clips: Path
    config: Path

    def body(self, **overrides: object) -> dict[str, object]:
        video: dict[str, object] = {
            "schema_version": 1,
            "read_roots": [str(self.clips)],
            "staging_root": str(self.root / "staging"),
            "output_root": str(self.root / "output"),
            "backend": "ffmpeg",
            "ffmpeg": FFMPEG,
            "ffprobe": FFPROBE,
            "max_input_bytes": 64 * 1024 * 1024,
        }
        for key in [key for key in overrides if key.startswith("video_")]:
            value = overrides.pop(key)
            name = key.removeprefix("video_")
            if value is None and name in {"output_root"}:
                video.pop(name)
            else:
                video[name] = value
        body: dict[str, object] = {
            "schema_version": 2,
            "server_name": "capicove-local",
            "workspace_id": "workspace",
            "actor_id": "owner",
            "environment_id": "local",
            "workspace_root": str(self.root),
            "profile": "write",
            "permissions": ["artifact:write", "subprocess:execute", "video:read"],
            "enabled_tools": ALL_TOOLS,
            "trace_store": None,
            "capabilities": {"video": video},
            "network": {"enabled": False, "source_hosts": []},
        }
        body.update(overrides)
        return body

    def write(self, **overrides: object) -> Path:
        self.config.write_text(json.dumps(self.body(**overrides)))
        self.config.chmod(0o600)
        return self.config

    def load(self) -> WorkspaceHostConfiguration:
        config = load_mcp_host_config(self.config)
        assert isinstance(config, WorkspaceHostConfiguration)
        return config


@pytest.fixture
def space(tmp_path: Path, rendered: Path) -> VideoSpace:
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    clips = tmp_path / "clips"
    clips.mkdir()
    for name in ("moving.mp4", "still.mp4"):
        shutil.copyfile(rendered / name, clips / name)
    shutil.copyfile(rendered / "moving.mp4", clips / "copy.mp4")
    result = VideoSpace(root, clips, root / "host.json")
    result.write()
    return result


@pytest.fixture
def bindings(space: VideoSpace) -> Iterator[dict[str, object]]:
    with compose_mcp_host(space.load()) as composed:
        yield {item.spec.tool_id: item for item in composed}


def call(
    bindings: dict[str, object], tool_id: str, arguments: dict[str, object]
) -> dict[str, object]:
    from capicove.mcp.bindings import invoke_binding

    return invoke_binding(bindings[tool_id], arguments)  # type: ignore[arg-type]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_no_staging_leak(payload: object, space: VideoSpace) -> None:
    assert str(space.root / "staging") not in json.dumps(payload)


# Configuration, authority, and catalog


def test_catalog_lists_the_selected_video_tools_without_side_effects(
    space: VideoSpace, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_process(*args: object, **kwargs: object) -> None:
        raise AssertionError("catalog must not start a subprocess")

    monkeypatch.setattr(subprocess, "Popen", no_process)

    assert main(["mcp", "catalog", "--config", str(space.config)]) == 0

    payload = json.loads(capsys.readouterr().out)
    tools = {item["tool_id"]: item for item in payload["tools"]}
    assert sorted(tools) == ALL_TOOLS
    assert tools["capicove.video.split"]["annotations"]["readOnlyHint"] is False
    assert tools["capicove.video.deduplicate"]["annotations"]["readOnlyHint"] is True
    assert not (space.root / "staging").exists() and not (space.root / "output").exists()


def test_read_profile_cannot_select_split(
    space: VideoSpace, capsys: pytest.CaptureFixture[str]
) -> None:
    space.write(profile="read", permissions=["subprocess:execute", "video:read"])
    assert main(["mcp", "catalog", "--config", str(space.config)]) == 2
    assert capsys.readouterr().out == ""

    space.write(
        profile="read",
        permissions=["subprocess:execute", "video:read"],
        enabled_tools=["capicove.video.deduplicate", "capicove.video.filter"],
    )
    assert main(["mcp", "catalog", "--config", str(space.config)]) == 0


def test_missing_permissions_fail_startup(
    space: VideoSpace, capsys: pytest.CaptureFixture[str]
) -> None:
    space.write(permissions=["artifact:write", "video:read"])

    assert main(["mcp", "catalog", "--config", str(space.config)]) == 2


@pytest.mark.parametrize(
    "override",
    [
        {"video_extra": 1},
        {"video_schema_version": 2},
        {"video_read_roots": ["relative/clips"]},
        {"video_staging_root": "/tmp/outside-staging"},
        {"video_ffmpeg": "ffmpeg"},
        {"video_ffprobe": "/nonexistent/ffprobe"},
        {"video_backend": "gpu"},
        {"video_max_input_bytes": 0},
        {"video_max_input_bytes": 64 * 1024**3 + 1},
        {"video_output_root": None},
    ],
)
def test_video_config_is_closed_and_selection_checked(
    space: VideoSpace, override: dict[str, object]
) -> None:
    space.write(**override)

    with pytest.raises((ValueError, ImportError)):
        with compose_mcp_host(space.load(), catalog_only=True):
            pass


def test_staging_inside_a_read_root_is_rejected(space: VideoSpace) -> None:
    media = space.root / "media"
    media.mkdir()
    space.write(video_read_roots=[str(media)], video_staging_root=str(media / "staging"))
    with pytest.raises(ValueError, match="disjoint"):
        with compose_mcp_host(space.load(), catalog_only=True):
            pass


def test_deduplicate_alone_needs_no_media_executables(space: VideoSpace) -> None:
    space.write(
        enabled_tools=["capicove.video.deduplicate"],
        permissions=["video:read"],
        video_ffmpeg=None,
        video_ffprobe=None,
        video_backend="av",
        video_output_root=None,
    )

    with compose_mcp_host(space.load(), catalog_only=True) as composed:
        assert [item.spec.tool_id for item in composed] == ["capicove.video.deduplicate"]


# Split


def test_split_writes_expected_segments_with_original_identity(
    space: VideoSpace, bindings: dict[str, object]
) -> None:
    source = space.clips / "moving.mp4"
    original = source.read_bytes()

    payload = call(
        bindings,
        "capicove.video.split",
        {
            "source": {"type": "path", "path": str(source)},
            "strategy": "duration",
            "config": {
                "split_duration": 2.0,
                "overlap_duration": 0.0,
                "min_last_split_duration": 0.0,
            },
            "output_format": "path",
            "backend": "auto",
        },
    )

    assert payload["status"] == "succeeded", payload
    result = payload["result"]
    assert isinstance(result, dict)
    assert result["original"]["path"] == str(source)
    assert result["original"]["sha256"] == hashlib.sha256(original).hexdigest()
    ranges = [(s["start_seconds"], s["end_seconds"]) for s in result["segments"]]
    assert [(round(a, 1), round(b, 1)) for a, b in ranges] == [(0.0, 2.0), (2.0, 4.0)]
    for segment in result["segments"]:
        assert Path(segment["path"]).parent == space.root / "output"
        assert Path(segment["path"]).is_file()
    assert source.read_bytes() == original
    assert_no_staging_leak(payload, space)
    assert list((space.root / "staging").iterdir()) == []


@pytest.mark.parametrize(
    "source",
    [
        {"type": "bytes", "base64": "AAAA"},
        {"type": "artifact", "ref": "artifact:x", "sha256": "a" * 64},
    ],
)
def test_split_rejects_non_path_sources_before_intake(
    space: VideoSpace, bindings: dict[str, object], source: dict[str, object]
) -> None:
    payload = call(
        bindings,
        "capicove.video.split",
        {
            "source": source,
            "strategy": "keyframe",
            "config": {"min_segment_duration": 1.0},
            "output_format": "path",
            "backend": "auto",
        },
    )

    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "invalid_input"  # type: ignore[index]
    assert not (space.root / "staging").exists()


def test_paths_outside_read_roots_and_symlinks_are_denied(
    space: VideoSpace, bindings: dict[str, object], tmp_path: Path
) -> None:
    outside = tmp_path / "outside.mp4"
    shutil.copyfile(space.clips / "moving.mp4", outside)
    (space.clips / "linked.mp4").symlink_to(outside)

    for path in (outside, space.clips / "linked.mp4", space.clips / "missing.mp4"):
        payload = call(
            bindings,
            "capicove.video.split",
            {
                "source": {"type": "path", "path": str(path)},
                "strategy": "keyframe",
                "config": {"min_segment_duration": 1.0},
                "output_format": "path",
                "backend": "auto",
            },
        )
        assert payload["status"] == "failed", path
        assert payload["error"]["code"] == "permission_denied"  # type: ignore[index]
    assert not (space.root / "output").exists() or list((space.root / "output").iterdir()) == []


def test_source_change_during_intake_rejects_without_output(
    space: VideoSpace, bindings: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    import capicove.runtime.media_handoff as handoff

    source = space.clips / "moving.mp4"
    original_read = handoff._read_chunk

    def changing(descriptor: int, size: int) -> bytes:
        chunk = original_read(descriptor, size)
        with source.open("ab") as stream:
            stream.write(b"changed")
        return chunk

    monkeypatch.setattr(handoff, "_read_chunk", changing)

    payload = call(
        bindings,
        "capicove.video.split",
        {
            "source": {"type": "path", "path": str(source)},
            "strategy": "keyframe",
            "config": {"min_segment_duration": 1.0},
            "output_format": "path",
            "backend": "auto",
        },
    )

    assert payload["status"] == "failed"
    assert not (space.root / "output").exists() or list((space.root / "output").iterdir()) == []
    assert list((space.root / "staging").iterdir()) == []


def test_backend_reads_the_verified_copy_after_intake(
    space: VideoSpace, bindings: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    import capicove.runtime.media_handoff as handoff

    source = space.clips / "moving.mp4"
    expected = sha256(source)
    original_stage = handoff.MediaIntake.stage

    def stage_then_replace(self: object, paths: object) -> object:
        manager = original_stage(self, paths)  # type: ignore[arg-type]

        class Replacing:
            def __enter__(self) -> object:
                staged = manager.__enter__()
                shutil.copyfile(space.clips / "still.mp4", source)
                return staged

            def __exit__(self, *details: Any) -> object:
                return manager.__exit__(*details)

        return Replacing()

    monkeypatch.setattr(handoff.MediaIntake, "stage", stage_then_replace)

    payload = call(
        bindings,
        "capicove.video.split",
        {
            "source": {"type": "path", "path": str(source)},
            "strategy": "duration",
            "config": {
                "split_duration": 2.0,
                "overlap_duration": 0.0,
                "min_last_split_duration": 0.0,
            },
            "output_format": "path",
            "backend": "auto",
        },
    )

    assert payload["status"] == "succeeded"
    assert payload["result"]["original"]["sha256"] == expected  # type: ignore[index]


def test_stale_staging_from_dead_processes_is_removed(
    space: VideoSpace, bindings: dict[str, object]
) -> None:
    staging = space.root / "staging"
    staging.mkdir(mode=0o700)
    stale = staging / "inv-999999999-0123456789abcdef"
    stale.mkdir(mode=0o700)
    (stale / "leftover.mp4").write_bytes(b"x")

    call(
        bindings,
        "capicove.video.deduplicate",
        {"paths": [str(space.clips / "moving.mp4")]},
    )
    call(
        bindings,
        "capicove.video.split",
        {
            "source": {"type": "path", "path": str(space.clips / "moving.mp4")},
            "strategy": "keyframe",
            "config": {"min_segment_duration": 1.0},
            "output_format": "path",
            "backend": "auto",
        },
    )

    assert not stale.exists()


# Filter


def test_filter_metadata_and_motion_decisions(
    space: VideoSpace, bindings: dict[str, object]
) -> None:
    payload = call(
        bindings,
        "capicove.video.filter",
        {
            "sources": [
                {
                    "item_id": "moving",
                    "source": {"type": "path", "path": str(space.clips / "moving.mp4")},
                },
                {
                    "item_id": "still",
                    "source": {"type": "path", "path": str(space.clips / "still.mp4")},
                },
            ],
            "criteria": [
                {
                    "type": "metadata",
                    "criterion_id": "long",
                    "min_duration_seconds": 3.0,
                    "videos_aggregation": "all",
                },
                {
                    "type": "motion",
                    "criterion_id": "moves",
                    "min_score": 0.05,
                    "videos_aggregation": "any",
                },
            ],
        },
    )

    assert payload["status"] == "succeeded", payload
    result = payload["result"]
    assert isinstance(result, dict)
    items = {item["item_id"]: item for item in result["items"]}
    assert abs(items["moving"]["metadata"]["duration_seconds"] - 4.0) < 0.15
    assert items["moving"]["metadata"]["encoded_width"] == 64
    assert items["moving"]["metadata"]["encoded_height"] == 48
    assert items["moving"]["source_sha256"] == sha256(space.clips / "moving.mp4")
    criteria = {criterion["criterion_id"]: criterion for criterion in result["criteria"]}
    assert criteria["long"]["passed"] is True
    motion = {video["item_id"]: video["passed"] for video in criteria["moves"]["video_results"]}
    assert motion == {"moving": True, "still": False}
    assert_no_staging_leak(payload, space)


@pytest.mark.parametrize("criterion", ["ocr_area", "tags", "aesthetics", "frame_text_similarity"])
def test_filter_rejects_unavailable_criteria_before_intake(
    space: VideoSpace, bindings: dict[str, object], criterion: str
) -> None:
    payload = call(
        bindings,
        "capicove.video.filter",
        {
            "sources": [{"source": {"type": "path", "path": str(space.clips / "moving.mp4")}}],
            "criteria": [{"type": criterion, "criterion_id": "x"}],
        },
    )

    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "invalid_input"  # type: ignore[index]
    assert not (space.root / "staging").exists()


# Deduplicate


def test_deduplicate_groups_exact_copies(space: VideoSpace, bindings: dict[str, object]) -> None:
    paths = [space.clips / name for name in ("still.mp4", "moving.mp4", "copy.mp4", "gone.mp4")]

    payload = call(bindings, "capicove.video.deduplicate", {"paths": [str(path) for path in paths]})

    assert payload["status"] == "succeeded", payload
    result = payload["result"]
    assert isinstance(result, dict)
    assert result["file_count"] == 4
    assert result["groups"] == [
        {
            "paths": sorted([str(space.clips / "copy.mp4"), str(space.clips / "moving.mp4")]),
            "sha256": sha256(space.clips / "moving.mp4"),
            "size_bytes": (space.clips / "moving.mp4").stat().st_size,
            "xxh3_128": result["groups"][0]["xxh3_128"],
        }
    ]
    assert result["failures"] == [{"path": str(space.clips / "gone.mp4"), "code": "unreadable"}]
    assert result["lineage"]["fast_hash"] == "xxh3_128"
    assert result["lineage"]["confirmation"] == "sha256"
    assert {path.name for path in space.clips.iterdir()} == {"still.mp4", "moving.mp4", "copy.mp4"}


@pytest.mark.parametrize("paths", [[], ["relative.mp4"], [f"/x/{i}.mp4" for i in range(1001)]])
def test_deduplicate_input_bounds(
    space: VideoSpace, bindings: dict[str, object], paths: list[str]
) -> None:
    with pytest.raises(ValueError):
        call(bindings, "capicove.video.deduplicate", {"paths": paths})


def test_deduplicate_denies_paths_outside_read_roots(
    space: VideoSpace, bindings: dict[str, object], tmp_path: Path
) -> None:
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"x")

    payload = call(bindings, "capicove.video.deduplicate", {"paths": [str(outside)]})

    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "permission_denied"  # type: ignore[index]


# CLI parity and authority


def test_configured_cli_matches_the_host_binding(
    space: VideoSpace, bindings: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    dedup_paths = [str(space.clips / "moving.mp4"), str(space.clips / "copy.mp4")]
    assert main(["video", "deduplicate", "--runtime-config", str(space.config), *dedup_paths]) == 0
    cli = json.loads(capsys.readouterr().out)
    host = call(bindings, "capicove.video.deduplicate", {"paths": dedup_paths})
    assert cli["result"] == host["result"]

    assert (
        main(
            [
                "video",
                "split",
                "--runtime-config",
                str(space.config),
                "--input",
                str(space.clips / "moving.mp4"),
                "--strategy",
                "duration",
                "--split-duration",
                "2",
                "--output",
                "json",
            ]
        )
        == 0
    )
    split = json.loads(capsys.readouterr().out)
    assert split["status"] == "succeeded"
    assert len(split["result"]["segments"]) == 2
    assert_no_staging_leak(split, space)


@pytest.mark.parametrize(
    "argv",
    [
        [
            "video",
            "split",
            "--input",
            "X",
            "--strategy",
            "keyframe",
            "--output-directory",
            "/tmp/x",
        ],
        ["video", "deduplicate", "X", "--print-unique"],
    ],
)
def test_configured_cli_rejects_conflicting_authority(
    space: VideoSpace, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    command = [part if part != "X" else str(space.clips / "moving.mp4") for part in argv]
    command[2:2] = ["--runtime-config", str(space.config)]

    assert main(command) == 2
    assert "--runtime-config" in capsys.readouterr().err


def test_configured_tracing_records_no_staging_locator(space: VideoSpace, tmp_path: Path) -> None:
    traces = space.root / "traces"
    traces.mkdir(mode=0o700)
    space.write(trace_store=str(traces / "traces.sqlite3"))

    with compose_mcp_host(space.load()) as composed:
        selected = {item.spec.tool_id: item for item in composed}
        call(
            selected,  # type: ignore[arg-type]
            "capicove.video.deduplicate",
            {"paths": [str(space.clips / "moving.mp4"), str(space.clips / "copy.mp4")]},
        )

    database = traces / "traces.sqlite3"
    assert database.is_file()
    import sqlite3

    with sqlite3.connect(database) as connection:
        tables = [
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ]
        dump = "".join(
            json.dumps([str(value) for value in row])
            for table in tables
            for row in connection.execute(f"SELECT * FROM {table}")
        )
    assert "capicove.video.deduplicate" in dump
    assert str(space.root / "staging") not in dump


def test_setup_doctor_classifies_video_readiness(
    space: VideoSpace, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["setup", "doctor", "--config", str(space.config)]) == 0
    assert json.loads(capsys.readouterr().out)["ready"] is True

    space.write(video_ffprobe="/nonexistent/ffprobe")
    assert main(["setup", "doctor", "--config", str(space.config)]) == 2
    statuses = {
        item["tool_id"]: item["status"] for item in json.loads(capsys.readouterr().out)["tools"]
    }
    assert statuses["capicove.video.filter"] == "missing_dependency"
    assert statuses["capicove.video.deduplicate"] == "ready"


# Review findings (2026-10-06)


def test_output_root_swap_cannot_redirect_writes(
    space: VideoSpace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from capicove.processing.video._adapters import ffmpeg as adapter

    external = tmp_path / "external"
    external.mkdir()
    output = space.root / "output"
    output.mkdir(mode=0o700)
    original_write = adapter.FfmpegClipWriter.write
    swapped: list[bool] = []

    def swap_then_write(self: object, *args: Any, **kwargs: Any) -> Any:
        if not swapped:
            output.rename(space.root / "output-moved")
            output.symlink_to(external)
            swapped.append(True)
        return original_write(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(adapter.FfmpegClipWriter, "write", swap_then_write)
    with compose_mcp_host(space.load()) as composed:
        split = {item.spec.tool_id: item for item in composed}
        call(
            split,  # type: ignore[arg-type]
            "capicove.video.split",
            {
                "source": {"type": "path", "path": str(space.clips / "moving.mp4")},
                "strategy": "duration",
                "config": {
                    "split_duration": 2.0,
                    "overlap_duration": 0.0,
                    "min_last_split_duration": 0.0,
                },
                "output_format": "path",
                "backend": "ffmpeg",
            },
        )

    assert swapped
    assert list(external.iterdir()) == []


def test_av_backend_runs_the_configured_ffmpeg_not_path(
    space: VideoSpace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    impostors = tmp_path / "impostors"
    impostors.mkdir()
    marker = tmp_path / "impostor-ran"
    for name in ("ffmpeg", "ffprobe"):
        script = impostors / name
        script.write_text(f"#!/bin/sh\ntouch {marker}\nexit 1\n")
        script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{impostors}:{os.environ['PATH']}")
    space.write(video_backend="av", enabled_tools=["capicove.video.split"])

    with compose_mcp_host(space.load()) as composed:
        split = {item.spec.tool_id: item for item in composed}
        payload = call(
            split,  # type: ignore[arg-type]
            "capicove.video.split",
            {
                "source": {"type": "path", "path": str(space.clips / "moving.mp4")},
                "strategy": "duration",
                "config": {
                    "split_duration": 2.0,
                    "overlap_duration": 0.0,
                    "min_last_split_duration": 0.0,
                },
                "output_format": "path",
                "backend": "av",
            },
        )

    assert payload["status"] == "succeeded", payload
    assert not marker.exists()


def test_av_split_without_configured_ffmpeg_fails_startup(space: VideoSpace) -> None:
    space.write(video_backend="av", video_ffmpeg=None, enabled_tools=["capicove.video.split"])

    with pytest.raises((ValueError, ImportError)):
        with compose_mcp_host(space.load(), catalog_only=True):
            pass


def test_replaced_ffprobe_is_never_run(space: VideoSpace, tmp_path: Path, rendered: Path) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    assert FFPROBE is not None
    shutil.copyfile(FFPROBE, tools / "ffprobe")
    (tools / "ffprobe").chmod(0o755)
    space.write(video_ffprobe=str(tools / "ffprobe"), enabled_tools=["capicove.video.filter"])
    marker = tmp_path / "replacement-ran"

    with compose_mcp_host(space.load()) as composed:
        replacement = tools / "ffprobe.new"
        replacement.write_text(f"#!/bin/sh\ntouch {marker}\nexit 1\n")
        replacement.chmod(0o755)
        os.replace(replacement, tools / "ffprobe")
        selected = {item.spec.tool_id: item for item in composed}
        payload = call(
            selected,  # type: ignore[arg-type]
            "capicove.video.filter",
            {
                "sources": [{"source": {"type": "path", "path": str(space.clips / "moving.mp4")}}],
                "criteria": [
                    {"type": "metadata", "criterion_id": "m", "min_duration_seconds": 1.0}
                ],
            },
        )

    assert payload["status"] == "failed"
    assert not marker.exists()
