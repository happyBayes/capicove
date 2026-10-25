"""Opt-in real local Ray workers; no remote cluster or model downloads."""

from __future__ import annotations

import importlib
import os

import pytest

from capicove.processing.image.deduplication import (
    ImagePerceptualDeduplicator,
    ImageRecord,
    RayImagePerceptualDeduplicator,
)
from tests.unit.processing.image.authored_images import authored_source


pytestmark = pytest.mark.ray


def test_real_ray_workers_preserve_local_deduplication() -> None:
    if os.environ.get("CAPICOVE_TEST_REAL_RAY") != "1":
        pytest.skip("Set CAPICOVE_TEST_REAL_RAY=1 to start a disposable local Ray runtime")
    ray = importlib.import_module("ray")

    assert not ray.is_initialized()
    with pytest.raises(RuntimeError, match="explicitly initialized"):
        RayImagePerceptualDeduplicator().group(())
    ray.init(
        num_cpus=2,
        include_dashboard=False,
        log_to_driver=False,
        object_store_memory=80 * 1024 * 1024,
    )
    try:
        records = (
            ImageRecord("b", (authored_source("landscape"),)),
            ImageRecord("a", (authored_source("landscape"),)),
            ImageRecord("c", (authored_source("portrait"),)),
        )
        expected = ImagePerceptualDeduplicator().group(records)
        actual = RayImagePerceptualDeduplicator(timeout_seconds=60).group(records)
        assert actual == expected
    finally:
        ray.shutdown()
    assert not ray.is_initialized()
