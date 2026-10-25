"""Tests for the complete video generation-set service."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from capicove.retrieval.generations import TextIndexValidationReport
from capicove.retrieval.ports import TextIndexGenerationLifecycle
from capicove.retrieval.video_generations import (
    VideoGenerationSetService,
    VideoTextMemberValidator,
    open_exact_video_runtime,
    open_video_runtime,
)
from capicove.retrieval.video_models import (
    PinnedVideoGenerationSet,
    VideoGenerationSetEvent,
    VideoGenerationSetRecord,
    VideoGenerationSetState,
    VideoGenerationTransitionError,
)
from tests.integration.retrieval.test_sqlite_video_generation_catalog import _record


class _Catalog:
    def __init__(self) -> None:
        self.records: dict[str, VideoGenerationSetRecord] = {}
        self.active_record: VideoGenerationSetRecord | None = None
        self.released: list[str] = []

    def register(self, record: VideoGenerationSetRecord) -> VideoGenerationSetRecord:
        self.records[record.manifest.set_id] = record
        return record

    def get(self, set_id: str) -> VideoGenerationSetRecord | None:
        return self.records.get(set_id)

    def record_activation(self, pin: PinnedVideoGenerationSet) -> VideoGenerationSetRecord:
        current = self.records[pin.set_id]
        updated = VideoGenerationSetRecord(
            current.manifest, current.pin, VideoGenerationSetState.ACTIVE, current.revision + 1
        )
        self.records[pin.set_id] = updated
        self.active_record = updated
        return updated

    def active(self, **kwargs: str) -> VideoGenerationSetRecord | None:
        return self.active_record

    def acquire_active_lease(self, **kwargs: str) -> tuple[str, PinnedVideoGenerationSet]:
        if self.active_record is None:
            raise VideoGenerationTransitionError("no active generation set")
        return "lease-1", self.active_record.pin

    def release_lease(self, lease_id: str) -> None:
        self.released.append(lease_id)

    def can_evict(self, set_id: str) -> bool:
        return True

    def history(self, set_id: str) -> tuple[VideoGenerationSetEvent, ...]:
        return ()

    def lease_count(self, set_id: str) -> int:
        return 0


def test_service_register_activate_resolve_and_release_lease() -> None:
    catalog = _Catalog()
    validated: list[str] = []
    service = VideoGenerationSetService(
        catalog, validator=lambda record: validated.append(record.manifest.set_id)
    )
    source = _record("set-1")
    assert service.register(source.manifest, source.pin).state is VideoGenerationSetState.READY
    assert service.activate("set-1").state is VideoGenerationSetState.ACTIVE
    assert validated == ["set-1"]
    assert (
        service.resolve_active(workspace_ref="workspace-1", profile_ref="video-local-exact").set_id
        == "set-1"
    )
    with service.lease_active(workspace_ref="workspace-1", profile_ref="video-local-exact") as pin:
        assert pin.set_id == "set-1"
    assert catalog.released == ["lease-1"]


def test_service_requires_ready_record_and_configured_validator() -> None:
    source = _record("set-1")
    catalog = _Catalog()
    VideoGenerationSetService(catalog).register(source.manifest, source.pin)
    with pytest.raises(VideoGenerationTransitionError, match="validator"):
        VideoGenerationSetService(catalog).activate("set-1")
    catalog.records["set-1"] = VideoGenerationSetRecord(
        source.manifest, source.pin, VideoGenerationSetState.CORRUPT, 1
    )
    with pytest.raises(VideoGenerationTransitionError, match="corrupt"):
        VideoGenerationSetService(catalog, validator=lambda record: None).activate("set-1")


def test_exact_runtime_cannot_activate_unverified_text_members(tmp_path: Path) -> None:
    runtime = open_exact_video_runtime(
        generation_database_path=tmp_path / "sets.sqlite3", index_root=tmp_path / "indexes"
    )
    record = _record("set-1")
    runtime.generation_sets.register(record.manifest, record.pin)
    try:
        with pytest.raises(VideoGenerationTransitionError, match="text member validator"):
            runtime.generation_sets.activate(record.manifest.set_id)
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "mutation",
    [
        {},
        {"manifest_sha256": "f" * 64},
        {"generation_id": "different-generation"},
        {"physical_index_ref": "different-index"},
        {"mapping_sha256": "0" * 64},
        {"expected_unit_count": 42, "observed_unit_count": 42},
        {"issue_codes": ("write_block_missing",)},
    ],
)
def test_text_member_validator_requires_every_physical_identity(
    mutation: dict[str, Any],
) -> None:
    record = _record("set-1")
    manifest, pin = record.manifest.asr_text, record.pin.asr_text
    report = TextIndexValidationReport(
        pin.generation_id,
        pin.manifest_sha256,
        pin.physical_index_ref,
        manifest.mapping_sha256,
        manifest.expected_unit_count,
        manifest.expected_unit_count,
        (),
    )
    lifecycle = Mock(spec=TextIndexGenerationLifecycle)
    lifecycle.validate.return_value = replace(report, **mutation)
    validator = VideoTextMemberValidator(asr=lifecycle)
    if mutation:
        with pytest.raises(VideoGenerationTransitionError, match="physical text"):
            validator(record)
    else:
        validator(record)
    lifecycle.validate.assert_called_once_with(manifest)
    lifecycle.activate.assert_not_called()


def test_runtime_accepts_injected_visual_backend_and_owns_cleanup(tmp_path: Path) -> None:
    backend = Mock()
    runtime = open_video_runtime(
        generation_database_path=tmp_path / "generations.sqlite3", visual_backend=backend
    )
    assert runtime.visual_backend is backend
    runtime.close()
    backend.close.assert_called_once()


def test_runtime_reuses_catalog_but_rejects_changed_immutable_file(tmp_path: Path) -> None:
    from capicove.retrieval._adapters.sqlite_video_units import (
        SqliteVideoUnitCatalogWriter,
        VideoUnitCatalogError,
    )
    from capicove.retrieval.video_models import searchable_units_manifest_sha256
    from tests.integration.retrieval.test_sqlite_video_unit_catalog import _units

    path = tmp_path / "units.sqlite3"
    units = _units()
    digest = searchable_units_manifest_sha256(units)
    SqliteVideoUnitCatalogWriter(path).publish(
        units=units, unit_manifest_sha256=digest, keyframe_to_segments={}, text_to_segments={}
    )
    pin = replace(_record("set-1").pin, unit_catalog_ref=str(path), unit_manifest_sha256=digest)
    runtime = open_video_runtime(
        generation_database_path=tmp_path / "sets.sqlite3", visual_backend=Mock()
    )
    reader = runtime.catalog_resolver(pin)
    assert runtime.catalog_resolver(pin) is reader
    path.chmod(0o600)
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(VideoUnitCatalogError, match="changed"):
        reader.get(units[0].unit_id)
    with pytest.raises(VideoUnitCatalogError, match="changed"):
        runtime.catalog_resolver(pin)
