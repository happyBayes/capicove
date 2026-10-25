import hashlib
import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from capicove.processing.video import (
    AcceptedVideoSource,
    TimeRange,
    VideoBackend,
    VideoWriteMode,
)
from capicove.processing.video._adapters.ffmpeg import (
    DefaultVideoBackendResolver,
    FfmpegClipWriter,
    FfprobeVideoInspector,
)
from capicove.processing.video._adapters.filesystem import (
    ExplicitPathAuthorizer,
    FilesystemOutputStore,
)
from capicove.processing.video.ports import (
    BackendUnavailableError,
    OutputStoreError,
    PathPolicyError,
    WrittenClip,
)


def _deadline() -> datetime:
    return datetime.now(UTC) + timedelta(minutes=1)


def test_path_authorizer_rejects_escape_and_unknown_policy(tmp_path: Path) -> None:
    read_root = tmp_path / "read"
    write_root = tmp_path / "write"
    read_root.mkdir()
    write_root.mkdir()
    authorizer = ExplicitPathAuthorizer({"policy-1": ((read_root,), (write_root,))})

    authorizer.authorize_read("policy-1", read_root / "video.mp4")
    authorizer.authorize_write("policy-1", write_root / "segment.mp4")
    with pytest.raises(PathPolicyError):
        authorizer.authorize_read("policy-1", tmp_path / "read-sibling" / "video.mp4")
    with pytest.raises(PathPolicyError):
        authorizer.authorize_write("missing-policy", write_root / "segment.mp4")


def test_path_authorizer_rejects_relative_roots() -> None:
    with pytest.raises(ValueError, match="absolute"):
        ExplicitPathAuthorizer({"policy-1": ((Path("read"),), (Path("write"),))})


def test_filesystem_store_finalizes_and_loads_verified_output(tmp_path: Path) -> None:
    store = FilesystemOutputStore(tmp_path)
    identity = "a" * 64
    pending = store.allocate(identity, ".mp4")
    payload = b"segment"
    pending.path.write_bytes(payload)

    stored = store.finalize(
        pending,
        WrittenClip(
            pending.path,
            hashlib.sha256(payload).hexdigest(),
            "video/mp4",
        ),
    )

    assert stored.path == tmp_path / f"{identity}.mp4"
    assert stored.path.read_bytes() == payload
    assert store.load_bytes(stored) == payload
    assert not pending.path.exists()


def test_filesystem_store_reuses_only_matching_content(tmp_path: Path) -> None:
    store = FilesystemOutputStore(tmp_path)
    identity = "b" * 64
    payload = b"same"
    existing = tmp_path / f"{identity}.mp4"
    existing.write_bytes(payload)
    pending = store.allocate(identity, ".mp4")
    pending.path.write_bytes(payload)

    stored = store.finalize(
        pending,
        WrittenClip(pending.path, hashlib.sha256(payload).hexdigest(), "video/mp4"),
    )

    assert stored.path == existing
    assert not pending.path.exists()


def test_filesystem_store_rejects_existing_content_collision(tmp_path: Path) -> None:
    store = FilesystemOutputStore(tmp_path)
    identity = "c" * 64
    (tmp_path / f"{identity}.mp4").write_bytes(b"existing")
    pending = store.allocate(identity, ".mp4")
    pending.path.write_bytes(b"different")

    with pytest.raises(OutputStoreError, match="collision"):
        store.finalize(
            pending,
            WrittenClip(
                pending.path,
                hashlib.sha256(b"different").hexdigest(),
                "video/mp4",
            ),
        )


def test_ffprobe_inspector_uses_argument_array_without_shell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "video.mp4"
    source_path.write_bytes(b"video")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((args, kwargs))
        payload = {
            "format": {"duration": "12.5"},
            "frames": [
                {"key_frame": 1, "best_effort_timestamp_time": "0.0"},
                {"key_frame": 0, "best_effort_timestamp_time": "1.0"},
                {"key_frame": 1, "best_effort_timestamp_time": "4.0"},
            ],
        }
        return subprocess.CompletedProcess(args, 0, json.dumps(payload), "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    inspector = FfprobeVideoInspector(executable="ffprobe", version="7.0")

    metadata = inspector.inspect(
        AcceptedVideoSource(
            source_id="source-1",
            sha256="d" * 64,
            mime_type="video/mp4",
            path=source_path,
        ),
        _deadline(),
    )

    assert metadata.duration_seconds == 12.5
    assert metadata.keyframe_timestamps == (0.0, 4.0)
    assert calls[0][0][0] == "ffprobe"
    assert calls[0][1].get("shell", False) is False
    assert calls[0][1]["start_new_session"] is True


def test_ffmpeg_writer_cleans_temporary_byte_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_input: Path | None = None

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal observed_input
        observed_input = Path(args[args.index("-i") + 1])
        destination = Path(args[-1])
        destination.write_bytes(b"segment")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    writer = FfmpegClipWriter(executable="ffmpeg")
    destination = FilesystemOutputStore(tmp_path).allocate("e" * 64, ".mp4")

    written = writer.write(
        AcceptedVideoSource(
            source_id="source-1",
            sha256="f" * 64,
            mime_type="video/mp4",
            data=b"video",
        ),
        TimeRange(1.0, 3.0),
        destination,
        _deadline(),
    )

    assert written.sha256 == hashlib.sha256(b"segment").hexdigest()
    assert observed_input is not None
    assert not observed_input.exists()


def test_ffmpeg_copy_mode_uses_stream_copy_without_reencoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        observed.extend(args)
        Path(args[-1]).write_bytes(b"segment")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"video")
    destination = FilesystemOutputStore(tmp_path / "outputs").allocate("f" * 64, ".mp4")

    FfmpegClipWriter(write_mode=VideoWriteMode.COPY).write(
        AcceptedVideoSource(
            source_id="source-1",
            sha256="a" * 64,
            mime_type="video/mp4",
            path=source_path,
        ),
        TimeRange(0.0, 1.0),
        destination,
        _deadline(),
    )

    assert observed[observed.index("-c") + 1] == "copy"
    assert "libx264" not in observed


def test_backend_resolver_reports_unavailable_explicit_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)

    with pytest.raises(BackendUnavailableError, match="ffmpeg"):
        DefaultVideoBackendResolver().resolve(VideoBackend.FFMPEG)
