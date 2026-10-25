"""Cursor identity is stable across hash seeds only with a shared signing key."""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import BoundedSemaphore

import pytest

from capicove.contracts import DetailLevel, SearchFilters, SearchRequest, TargetType, TextQuery
from capicove.retrieval.video_profiles import StaticVideoRetrievalProfileRegistry
from capicove.retrieval.video_search import VideoSearchService
from tests.unit.retrieval.test_video_search_text import (
    Catalog,
    Encoder,
    Generations,
    Visual,
)


def _page(cursor: str | None, change: str) -> str | None:
    with ThreadPoolExecutor(max_workers=4) as executor:
        service = VideoSearchService(
            workspace_ref="w",
            generation_sets=Generations(),
            profiles=StaticVideoRetrievalProfileRegistry(),
            catalog_resolver=lambda pin: Catalog(),
            visual_retriever=Visual(),
            query_encoder=Encoder(),
            executor=executor,
            concurrency=BoundedSemaphore(4),
            cursor_secret=(b"different-test-key" if change == "key" else b"shared-test-key") * 3,
        )
        request = SearchRequest(
            TextQuery("different" if change == "query" else "slide"),
            frozenset({TargetType.KEYFRAME, TargetType.VIDEO_SEGMENT}),
            top_k=1,
            detail_level=DetailLevel.FULL,
            retrieval_profile_ref="video-local-exact",
            filters=SearchFilters(
                artifact_refs=frozenset(f"artifact-{i}" for i in range(32)),
            ),
        )
        return service.search(replace(request, cursor=cursor)).next_cursor


@pytest.mark.slow
@pytest.mark.parametrize("change", ["none", "query", "key"])
def test_cursor_across_process_hash_seeds(change: str) -> None:
    root = Path(__file__).resolve().parents[3]
    command = [sys.executable, "-m", "tests.unit.retrieval.test_video_cursor_process"]
    first = subprocess.run(
        command,
        cwd=root,
        env=dict(os.environ, PYTHONHASHSEED="1"),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert first.returncode == 0, first.stderr
    token = first.stdout.strip()
    assert token and token != "None"
    second = subprocess.run(
        [*command, token, change],
        cwd=root,
        env=dict(os.environ, PYTHONHASHSEED="2"),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if change == "none":
        assert second.returncode == 0, second.stderr
    else:
        assert second.returncode != 0 and "stale_cursor" in second.stderr


if __name__ == "__main__":
    print(
        _page(
            sys.argv[1] if len(sys.argv) > 1 else None, sys.argv[2] if len(sys.argv) > 2 else "none"
        )
    )
