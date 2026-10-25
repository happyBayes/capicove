import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from capicove.processing.video import VideoSource
from capicove.processing.video._adapters.filesystem import ExplicitPathAuthorizer
from capicove.processing.video.filtering._adapters.ffprobe_video_metadata import (
    FfprobeVideoMetadataProbe,
)
from capicove.processing.video.filtering._adapters.pyav_frame_sampler import (
    PyavVideoFrameSampler,
)
from capicove.processing.video.filtering.video_filter_contracts import (
    MetadataCriterion,
    VideoFilterItem,
    VideoFilterRequest,
)
from capicove.processing.video.filtering.video_filter_pipeline import VideoFilterPipeline


pytestmark = pytest.mark.integration


def test_generated_video_passes_metadata_ranges(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required")
    source = tmp_path / "metadata.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=160x90:d=1:r=5",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    request = VideoFilterRequest(
        request_id="request-1",
        trace_id="trace-1",
        attempt_id="attempt-1",
        operation_id="operation-1",
        execution_policy_ref="policy-1",
        items=(VideoFilterItem("item-1", VideoSource(path=source)),),
        criteria=(
            MetadataCriterion(
                "metadata",
                min_duration_seconds=0.9,
                max_duration_seconds=1.1,
                min_width=160,
                max_width=160,
                min_height=90,
                max_height=90,
                min_aspect_ratio=1.77,
                max_aspect_ratio=1.78,
            ),
        ),
        deadline=datetime.now(UTC) + timedelta(minutes=1),
    )
    pipeline = VideoFilterPipeline(
        metadata_probe=FfprobeVideoMetadataProbe(),
        frame_sampler=PyavVideoFrameSampler(),
        evaluators={},
        path_authorizer=ExplicitPathAuthorizer({"policy-1": ((tmp_path.resolve(),), ())}),
    )

    result = pipeline.evaluate(request)

    assert result.keep is True
    assert result.item_results[0].metadata is not None
    assert result.item_results[0].metadata.encoded_width == 160
    assert result.item_results[0].metadata.encoded_height == 90
