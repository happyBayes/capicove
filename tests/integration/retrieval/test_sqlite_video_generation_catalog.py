"""Integration tests for atomic video generation aliases and durable leases."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from capicove.contracts import TargetType
from capicove.retrieval._adapters.sqlite_video_generations import (
    SqliteVideoGenerationSetCatalog,
)
from capicove.retrieval.generations import PinnedTextIndexGeneration
from capicove.retrieval.video_models import (
    PinnedVideoGenerationSet,
    PinnedVisualIndexGeneration,
    VideoGenerationSetManifest,
    VideoGenerationSetRecord,
    VideoGenerationSetState,
    VideoGenerationTransitionError,
)
from tests.unit.retrieval.test_video_models import (
    text_manifest,
    visual_manifest,
    visual_space,
)


def _record(set_id: str) -> VideoGenerationSetRecord:
    keyframe = visual_manifest(TargetType.KEYFRAME, generation_id=f"{set_id}-keyframe")
    segment = visual_manifest(TargetType.VIDEO_SEGMENT, generation_id=f"{set_id}-segment")
    asr = replace(text_manifest("asr"), generation_id=f"{set_id}-asr")
    manifest = VideoGenerationSetManifest(
        set_id=set_id,
        workspace_ref="workspace-1",
        profile_ref="video-local-exact",
        corpus_refs=("corpus-a", "corpus-b"),
        unit_catalog_ref=f"/tmp/{set_id}-units.sqlite3",
        unit_manifest_sha256="c" * 64,
        keyframe_visual=keyframe,
        segment_visual=segment,
        asr_text=asr,
        ocr_text=None,
        fusion_profile_ref="video-weighted-rrf-v1",
    )
    pin = PinnedVideoGenerationSet(
        set_id=set_id,
        manifest_sha256=manifest.manifest_sha256,
        workspace_ref=manifest.workspace_ref,
        profile_ref=manifest.profile_ref,
        unit_catalog_ref=manifest.unit_catalog_ref,
        unit_manifest_sha256=manifest.unit_manifest_sha256,
        keyframe_visual=PinnedVisualIndexGeneration(
            keyframe.generation_id,
            keyframe.manifest_sha256,
            f"/tmp/{set_id}-keyframe",
            TargetType.KEYFRAME,
            visual_space(),
        ),
        segment_visual=PinnedVisualIndexGeneration(
            segment.generation_id,
            segment.manifest_sha256,
            f"/tmp/{set_id}-segment",
            TargetType.VIDEO_SEGMENT,
            visual_space(),
        ),
        asr_text=PinnedTextIndexGeneration(
            asr.generation_id, asr.manifest_sha256, f"/tmp/{set_id}-asr"
        ),
        ocr_text=None,
        fusion_profile_ref=manifest.fusion_profile_ref,
    )
    return VideoGenerationSetRecord(manifest, pin, VideoGenerationSetState.READY, 0)


def test_activation_is_atomic_and_old_set_stays_leased(tmp_path: Path) -> None:
    path = (tmp_path / "video-generations.sqlite3").resolve()
    catalog = SqliteVideoGenerationSetCatalog(path)
    first, second = _record("set-1"), _record("set-2")
    assert catalog.register(first) == first
    assert catalog.register(first) == first
    catalog.register(second)
    assert catalog.record_activation(first.pin).state is VideoGenerationSetState.ACTIVE

    lease_id, leased = catalog.acquire_active_lease(
        workspace_ref="workspace-1", profile_ref="video-local-exact"
    )
    assert leased.set_id == "set-1"
    assert catalog.record_activation(second.pin).state is VideoGenerationSetState.ACTIVE
    active = catalog.active(workspace_ref="workspace-1", profile_ref="video-local-exact")
    assert active is not None
    assert active.manifest.set_id == "set-2"
    assert catalog.get("set-1").state is VideoGenerationSetState.SUPERSEDED  # type: ignore[union-attr]
    assert catalog.lease_count("set-1") == 1
    assert catalog.can_evict("set-1") is False

    catalog.release_lease(lease_id)
    catalog.release_lease(lease_id)
    assert catalog.lease_count("set-1") == 0
    assert catalog.can_evict("set-1") is True


def test_reopen_rollback_history_and_stale_pin_rejection(tmp_path: Path) -> None:
    path = (tmp_path / "video-generations.sqlite3").resolve()
    catalog = SqliteVideoGenerationSetCatalog(path)
    first, second = _record("set-1"), _record("set-2")
    catalog.register(first)
    catalog.register(second)
    catalog.record_activation(first.pin)
    catalog.record_activation(second.pin)

    reopened = SqliteVideoGenerationSetCatalog(path)
    active = reopened.active(workspace_ref="workspace-1", profile_ref="video-local-exact")
    assert active is not None
    assert active.manifest.set_id == "set-2"
    rolled_back = reopened.record_activation(first.pin)
    assert rolled_back.state is VideoGenerationSetState.ACTIVE
    assert reopened.get("set-2").state is VideoGenerationSetState.SUPERSEDED  # type: ignore[union-attr]
    assert len(reopened.history("set-1")) == 4

    with pytest.raises(VideoGenerationTransitionError, match="pin"):
        reopened.record_activation(replace(first.pin, manifest_sha256="9" * 64))


def test_catalog_rejects_symlink_database_path(tmp_path: Path) -> None:
    real = (tmp_path / "real.sqlite3").resolve()
    real.touch()
    link = tmp_path / "linked.sqlite3"
    link.symlink_to(real)
    with pytest.raises(ValueError, match="symlink"):
        SqliteVideoGenerationSetCatalog(link)
