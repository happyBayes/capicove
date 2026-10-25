"""Integration tests for the immutable SQLite video unit catalog."""

from __future__ import annotations

from pathlib import Path

import pytest

from capicove.contracts import (
    CharacterRange,
    ProcessingFingerprint,
    SearchableUnit,
    TargetType,
    TemporalRange,
    TextChunkLocator,
    TextOrigin,
)
from capicove.retrieval._adapters.sqlite_video_units import (
    SqliteVideoUnitCatalogReader,
    SqliteVideoUnitCatalogWriter,
)
from capicove.retrieval.video_models import searchable_units_manifest_sha256


def _unit(unit_id: str, target: TargetType) -> SearchableUnit:
    values: dict[str, object] = {
        "unit_id": unit_id,
        "unit_type": target,
        "source_hash": "a" * 64,
        "artifact_ref": "artifact-1",
        "evidence_ref": f"evidence:{unit_id}",
        "representation_refs": (f"representation:{unit_id}",),
        "processing_run_ref": "run-1",
        "processing_version": "video-v1",
        "processing_fingerprint": ProcessingFingerprint("b" * 64),
    }
    if target is TargetType.KEYFRAME:
        values.update(timestamp_ms=1_000, scene_range=TemporalRange(0, 2_000))
    elif target is TargetType.VIDEO_SEGMENT:
        values["time_range"] = TemporalRange(0, 2_000)
    else:
        locator = TextChunkLocator(
            TextOrigin.ASR,
            CharacterRange(0, 4),
            time_range=TemporalRange(500, 1_500),
        )
        values["text_locator"] = locator
    return SearchableUnit(**values)  # type: ignore[arg-type]


def _units() -> tuple[SearchableUnit, ...]:
    return (
        _unit("keyframe-1", TargetType.KEYFRAME),
        _unit("segment-1", TargetType.VIDEO_SEGMENT),
        _unit("asr-1", TargetType.TEXT_CHUNK),
    )


def test_catalog_round_trips_units_relations_and_reopens_read_only(tmp_path: Path) -> None:
    target = (tmp_path / "units.sqlite3").resolve()
    units = _units()
    writer = SqliteVideoUnitCatalogWriter(target)
    catalog_ref = writer.publish(
        units=units,
        unit_manifest_sha256=searchable_units_manifest_sha256(units),
        keyframe_to_segments={"keyframe-1": ("segment-1",)},
        text_to_segments={"asr-1": (("segment-1", 1_000),)},
    )

    reader = SqliteVideoUnitCatalogReader(Path(catalog_ref))
    assert reader.get("keyframe-1") == units[0]
    assert reader.get_many(("segment-1", "keyframe-1")) == (units[1], units[0])
    assert reader.segments_for_keyframe("keyframe-1") == ("segment-1",)
    assert reader.segments_for_keyframes(("keyframe-1",)) == {"keyframe-1": ("segment-1",)}
    assert reader.segments_for_text("asr-1") == (("segment-1", 1_000),)
    assert reader.segments_for_texts(("asr-1",)) == {"asr-1": (("segment-1", 1_000),)}
    assert reader.unit_manifest_sha256 == searchable_units_manifest_sha256(units)
    assert writer.publish(
        units=units,
        unit_manifest_sha256=searchable_units_manifest_sha256(units),
        keyframe_to_segments={"keyframe-1": ("segment-1",)},
        text_to_segments={"asr-1": (("segment-1", 1_000),)},
    ) == str(target)


def test_catalog_rejects_manifest_mismatch_and_invalid_relations(tmp_path: Path) -> None:
    units = _units()
    writer = SqliteVideoUnitCatalogWriter((tmp_path / "units.sqlite3").resolve())
    with pytest.raises(ValueError, match="manifest"):
        writer.publish(
            units=units,
            unit_manifest_sha256="f" * 64,
            keyframe_to_segments={},
            text_to_segments={},
        )
    with pytest.raises(ValueError, match="unknown"):
        writer.publish(
            units=units,
            unit_manifest_sha256=searchable_units_manifest_sha256(units),
            keyframe_to_segments={"missing": ("segment-1",)},
            text_to_segments={},
        )


def test_bulk_catalog_reads_bind_unit_ids_as_values(tmp_path: Path) -> None:
    target = (tmp_path / "quoted # catalog.sqlite3").resolve()
    units = _units()
    SqliteVideoUnitCatalogWriter(target).publish(
        units=units,
        unit_manifest_sha256=searchable_units_manifest_sha256(units),
        keyframe_to_segments={"keyframe-1": ("segment-1",)},
        text_to_segments={"asr-1": (("segment-1", 1000),)},
    )
    reader = SqliteVideoUnitCatalogReader(target)
    literal = "unknown' quotation; ordinary opaque identifier"
    assert reader.get_many((literal,)) == ()
    assert reader.segments_for_keyframes((literal,)) == {literal: ()}
    assert reader.segments_for_texts((literal,)) == {literal: ()}
    assert reader.get_many(("keyframe-1",)) == (units[0],)


def test_catalog_rejects_symlink_and_unbounded_reads(tmp_path: Path) -> None:
    real = (tmp_path / "real").resolve()
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        SqliteVideoUnitCatalogWriter(link / "units.sqlite3")

    target = (tmp_path / "units.sqlite3").resolve()
    units = _units()
    SqliteVideoUnitCatalogWriter(target).publish(
        units=units,
        unit_manifest_sha256=searchable_units_manifest_sha256(units),
        keyframe_to_segments={},
        text_to_segments={},
    )
    reader = SqliteVideoUnitCatalogReader(target)
    with pytest.raises(ValueError, match="1,000"):
        reader.get_many(tuple(f"unit-{index}" for index in range(1_001)))
