import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from capicove.processing.video import AcceptedVideoSource, TimeRange
from capicove.processing.video._adapters.ffmpeg import FfmpegClipWriter, FfprobeVideoInspector
from capicove.processing.video._adapters.filesystem import FilesystemOutputStore


pytestmark = pytest.mark.integration


@pytest.fixture
def generated_video(tmp_path: Path) -> Path:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required for this integration test")
    output = tmp_path / "synthetic.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=16x16:d=2:r=10",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=2",
            "-c:v",
            "libx264",
            "-g",
            "10",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(output),
        ],
        check=True,
        capture_output=True,
        start_new_session=True,
    )
    return output


def test_ffmpeg_backend_probes_and_splits_generated_video(
    tmp_path: Path,
    generated_video: Path,
) -> None:
    deadline = datetime.now(UTC) + timedelta(minutes=1)
    source_hash = hashlib.sha256(generated_video.read_bytes()).hexdigest()
    source = AcceptedVideoSource(
        source_id=f"sha256:{source_hash}",
        sha256=source_hash,
        mime_type="video/mp4",
        path=generated_video,
    )
    inspector = FfprobeVideoInspector()
    store = FilesystemOutputStore(tmp_path / "outputs")
    pending = store.allocate("a" * 64, ".mp4")

    metadata = inspector.inspect(source, deadline)
    written = FfmpegClipWriter().write(source, TimeRange(0.0, 1.0), pending, deadline)
    stored = store.finalize(pending, written)

    assert metadata.duration_seconds == pytest.approx(2.0, abs=0.2)
    assert metadata.keyframe_timestamps[0] == pytest.approx(0.0)
    assert stored.path.stat().st_size > 0
    assert hashlib.sha256(stored.path.read_bytes()).hexdigest() == stored.sha256
    assert generated_video.exists()


def test_ffmpeg_writer_is_frame_accurate_and_preserves_audio(
    tmp_path: Path,
    generated_video: Path,
) -> None:
    deadline = datetime.now(UTC) + timedelta(minutes=1)
    source_hash = hashlib.sha256(generated_video.read_bytes()).hexdigest()
    source = AcceptedVideoSource(
        source_id=f"sha256:{source_hash}",
        sha256=source_hash,
        mime_type="video/mp4",
        path=generated_video,
    )
    pending = FilesystemOutputStore(tmp_path / "accurate").allocate("b" * 64, ".mp4")

    written = FfmpegClipWriter().write(
        source,
        TimeRange(0.5, 1.5),
        pending,
        deadline,
    )
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,start_time,duration",
            "-of",
            "json",
            str(written.path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = {item["codec_type"]: item for item in json.loads(completed.stdout)["streams"]}

    assert set(streams) == {"video", "audio"}
    assert float(streams["video"]["start_time"]) == pytest.approx(0.0, abs=0.05)
    assert float(streams["audio"]["start_time"]) == pytest.approx(0.0, abs=0.05)
    assert float(streams["video"]["duration"]) == pytest.approx(1.0, abs=0.05)
    assert float(streams["audio"]["duration"]) == pytest.approx(1.0, abs=0.05)


def test_local_dataset_video_can_be_probed() -> None:
    dataset = Path("tests/datasets/muscian_sound.mp4")
    if not dataset.exists():
        pytest.skip("local ignored test video is not present")
    payload_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()

    metadata = FfprobeVideoInspector().inspect(
        AcceptedVideoSource(
            source_id=f"sha256:{payload_hash}",
            sha256=payload_hash,
            mime_type="video/mp4",
            path=dataset.resolve(),
        ),
        datetime.now(UTC) + timedelta(minutes=1),
    )

    assert metadata.duration_seconds > 0
