import json
import subprocess
from datetime import UTC, datetime, timedelta

import pytest

from capicove.processing.video import AcceptedVideoSource
from capicove.processing.video.filtering._adapters.ffprobe_video_metadata import (
    FfprobeVideoMetadataProbe,
)
from capicove.processing.video.filtering.video_filter_ports import (
    BackendUnavailableFilterError,
    UnsupportedMediaFilterError,
)


def _source() -> AcceptedVideoSource:
    return AcceptedVideoSource(
        source_id="source-1",
        sha256="a" * 64,
        mime_type="video/mp4",
        data=b"video",
    )


def _deadline() -> datetime:
    return datetime.now(UTC) + timedelta(minutes=1)


def test_probe_normalizes_rotated_display_dimensions() -> None:
    payload = {
        "program_version": {"version": "7.1"},
        "streams": [
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "sample_aspect_ratio": "1:1",
                "display_aspect_ratio": "16:9",
                "tags": {"rotate": "90"},
            }
        ],
        "format": {"duration": "12.5"},
    }

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    metadata = FfprobeVideoMetadataProbe(runner=runner).probe(_source(), _deadline())

    assert (metadata.encoded_width, metadata.encoded_height) == (1920, 1080)
    assert (metadata.display_width, metadata.display_height) == (1080, 1920)
    assert metadata.rotation_degrees == 90
    assert metadata.duration_seconds == 12.5
    assert metadata.backend_version == "7.1"


def test_probe_uses_side_data_rotation_and_cleans_byte_source(tmp_path: object) -> None:
    del tmp_path
    observed_path = ""
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "width": 640,
                "height": 360,
                "side_data_list": [{"rotation": -90}],
            }
        ],
        "format": {"duration": "2.0"},
    }

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal observed_path
        del kwargs
        observed_path = command[-1]
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    FfprobeVideoMetadataProbe(runner=runner).probe(_source(), _deadline())

    from pathlib import Path

    assert observed_path
    assert not Path(observed_path).exists()


def test_probe_maps_missing_executable_to_unavailable() -> None:
    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del command, kwargs
        raise FileNotFoundError("ffprobe")

    with pytest.raises(BackendUnavailableFilterError, match="ffprobe"):
        FfprobeVideoMetadataProbe(runner=runner).probe(_source(), _deadline())


def test_probe_rejects_payload_without_video_stream() -> None:
    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        payload = {"streams": [{"codec_type": "audio"}], "format": {"duration": "1"}}
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    with pytest.raises(UnsupportedMediaFilterError, match="video stream"):
        FfprobeVideoMetadataProbe(runner=runner).probe(_source(), _deadline())
