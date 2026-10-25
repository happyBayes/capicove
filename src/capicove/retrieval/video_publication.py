"""Publish prepared video units through existing immutable index lifecycles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from capicove.contracts import ProcessingFingerprint, TargetType, TextOrigin
from capicove.processing.video.retrieval.models import VideoRetrievalPreparationResult
from capicove.retrieval._adapters.sqlite_video_units import SqliteVideoUnitCatalogWriter
from capicove.retrieval.generations import TextIndexGenerationManifest
from capicove.retrieval.models import TextIndexEntry
from capicove.retrieval.ports import TextRetrieverBackend
from capicove.retrieval.video_generations import ExactVideoRuntimeComponents
from capicove.retrieval.video_models import (
    PinnedVideoGenerationSet,
    VideoGenerationSetManifest,
    VisualIndexEntry,
    VisualIndexGenerationManifest,
    visual_entries_content_sha256,
)


def publish_prepared_video(
    prepared: VideoRetrievalPreparationResult,
    *,
    runtime: ExactVideoRuntimeComponents,
    workspace_root: Path,
    workspace_ref: str,
    corpus_refs: tuple[str, ...],
    text_backend: TextRetrieverBackend,
    text_index_format: str,
    read_resource: Callable[[str], bytes],
) -> PinnedVideoGenerationSet:
    """Activate only after every physical member validates; retain the old alias on failure.

    Each build is a complete single-video snapshot, not an incremental corpus append.
    Failed builds may leave unreferenced immutable members for operator cleanup.
    """
    set_id = f"video-{uuid4().hex}"
    units = {unit.unit_id: unit for unit in prepared.searchable_units}
    catalog_path = workspace_root / "catalogs" / f"{set_id}.sqlite3"
    key_relations: dict[str, list[str]] = {}
    text_relations: dict[str, list[tuple[str, int]]] = {}
    for segment in prepared.segments:
        for key in segment.keyframe_unit_ids:
            key_relations.setdefault(key, []).append(segment.unit_id)
        for text in (*segment.asr_unit_ids, *segment.ocr_unit_ids):
            locator = units[text].text_locator
            if locator is None or locator.time_range is None:
                raise ValueError("video text requires temporal evidence")
            overlap = min(segment.time_range.end_ms, locator.time_range.end_ms) - max(
                segment.time_range.start_ms, locator.time_range.start_ms
            )
            if overlap > 0:
                text_relations.setdefault(text, []).append((segment.unit_id, overlap))
    SqliteVideoUnitCatalogWriter(catalog_path).publish(
        units=prepared.searchable_units,
        unit_manifest_sha256=prepared.unit_manifest_sha256,
        keyframe_to_segments={key: tuple(value) for key, value in key_relations.items()},
        text_to_segments={key: tuple(value) for key, value in text_relations.items()},
    )
    fingerprint = ProcessingFingerprint.from_manifest({"publisher": "video-local-exact-v1"})
    manifests = []
    pins = []
    for target, members, vectors in (
        (
            TargetType.KEYFRAME,
            tuple(k.unit_id for k in prepared.keyframes),
            prepared.keyframe_vectors,
        ),
        (
            TargetType.VIDEO_SEGMENT,
            tuple(s.unit_id for s in prepared.segments),
            prepared.segment_vectors,
        ),
    ):
        entries = tuple(
            VisualIndexEntry(units[member], vector)
            for member, vector in zip(members, vectors, strict=True)
        )
        manifest = VisualIndexGenerationManifest(
            f"{set_id}-{target.value}",
            workspace_ref,
            corpus_refs,
            target,
            prepared.unit_manifest_sha256,
            visual_entries_content_sha256(entries),
            prepared.embedding_space,
            runtime.visual_backend.provider_id,
            "numpy-exact-v1",
            fingerprint,
            len(entries),
        )
        runtime.visual_backend.create(manifest)
        runtime.visual_backend.add(index_generation_id=manifest.generation_id, entries=entries)
        pins.append(
            runtime.visual_backend.activate(manifest, runtime.visual_backend.validate(manifest))
        )
        manifests.append(manifest)
    text_entries = tuple(
        TextIndexEntry(
            unit,
            read_resource(unit.representation_refs[0]).decode("utf-8"),
            corpus_refs,
            "video/mp4",
        )
        for unit in prepared.searchable_units
        if unit.unit_type is TargetType.TEXT_CHUNK
    )
    lifecycle = text_backend.generation_lifecycle
    text_manifest = TextIndexGenerationManifest(
        f"{set_id}-text",
        workspace_ref,
        corpus_refs,
        prepared.unit_manifest_sha256,
        hashlib.sha256(
            json.dumps(
                [asdict(entry) for entry in text_entries], sort_keys=True, default=list
            ).encode()
        ).hexdigest(),
        lifecycle.mapping_sha256(()),
        (),
        text_backend.provider_id,
        text_index_format,
        fingerprint,
        len(text_entries),
    )
    lifecycle.create(text_manifest)
    text_backend.index_writer.add(
        index_generation_id=text_manifest.generation_id, entries=text_entries
    )
    report = lifecycle.validate(text_manifest)
    if not report.valid:
        raise ValueError("text generation failed validation")
    # Pin the validated physical member without mutating any unrelated text alias.
    from capicove.retrieval.generations import PinnedTextIndexGeneration

    text_pin = PinnedTextIndexGeneration(
        text_manifest.generation_id, text_manifest.manifest_sha256, report.physical_index_ref
    )
    has_ocr = any(
        entry.unit.text_locator and entry.unit.text_locator.origin is TextOrigin.OCR
        for entry in text_entries
    )
    manifest_set = VideoGenerationSetManifest(
        set_id,
        workspace_ref,
        "video-local-exact",
        corpus_refs,
        str(catalog_path),
        prepared.unit_manifest_sha256,
        manifests[0],
        manifests[1],
        text_manifest,
        text_manifest if has_ocr else None,
        "video-weighted-rrf-v1",
    )
    pin = PinnedVideoGenerationSet(
        set_id,
        manifest_set.manifest_sha256,
        workspace_ref,
        "video-local-exact",
        str(catalog_path),
        prepared.unit_manifest_sha256,
        pins[0],
        pins[1],
        text_pin,
        text_pin if has_ocr else None,
        "video-weighted-rrf-v1",
    )
    runtime.generation_sets.register(manifest_set, pin)
    runtime.generation_sets.activate(set_id)
    return pin
