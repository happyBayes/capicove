"""Bounded server-owned policy tests."""

import pytest

from capicove.retrieval.video_profiles import VideoLanePolicy, VideoRetrievalProfile


def test_default_bounds() -> None:
    p = VideoRetrievalProfile.video_local_ann()
    assert p.visual_depth(20) == 200
    assert p.text_depth(20) == 160
    assert p.visual_depth(100) == 500
    assert p.text_depth(100) == 300
    for invalid in (0, 101, True):
        with pytest.raises(ValueError):
            p.visual_depth(invalid)


def test_invalid_profile_rejected() -> None:
    with pytest.raises(ValueError):
        VideoLanePolicy("ocr_lexical", float("nan"), True)
    with pytest.raises(ValueError):
        VideoRetrievalProfile("x", (), maximum_temporal_candidates=0)
