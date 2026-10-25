"""Integration tests for explicit media copy mode during one-pass export."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from capicove.export import MediaMode, TraceExporter, TraceExportError, TraceExportOptions
from capicove.export.media import WorkspaceExportMediaResolver
from capicove.traces import (
    EpisodeStarted,
    MediaContent,
    MediaLocator,
    MediaLocatorKind,
    MediaReference,
    MessageObserved,
    MessageRole,
    TraceModality,
    TraceQuery,
    TraceRecord,
    TraceRecordType,
)
from capicove.traces.codec import canonical_record_bytes, trace_record_sha256

from .test_jsonl_export import Reader, Snapshot


SHA256 = "a" * 64
NOW = datetime(2026, 8, 18, tzinfo=UTC)


def _media_records(payload: bytes) -> tuple[TraceRecord, ...]:
    reference = MediaReference(
        "media-1",
        TraceModality.IMAGE,
        "image/png",
        hashlib.sha256(payload).hexdigest(),
        len(payload),
        locator=MediaLocator(MediaLocatorKind.WORKSPACE_RELATIVE, "inputs/image.png"),
    )
    started = TraceRecord(
        1,
        TraceRecordType.EPISODE_STARTED,
        1,
        "record-0",
        "episode-1",
        "trace-1",
        0,
        NOW,
        EpisodeStarted("Copy media.", "workspace", "actor", "local", 1, SHA256, {}),
    )
    message = TraceRecord(
        1,
        TraceRecordType.MESSAGE_OBSERVED,
        1,
        "record-1",
        "episode-1",
        "trace-1",
        1,
        NOW,
        MessageObserved(MessageRole.USER, (MediaContent("media-1"),)),
        modalities=(TraceModality.IMAGE,),
        media=(reference,),
        previous_record_sha256=trace_record_sha256(started),
    )
    return started, message


def test_copy_mode_bundles_verified_media_without_rewriting_trace_records(
    tmp_path: Path,
) -> None:
    payload = b"small png fixture bytes"
    workspace = tmp_path / "workspace"
    source = workspace / "inputs" / "image.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    records = _media_records(payload)
    reader = Reader(Snapshot(records, 1))
    output = tmp_path / "export"
    exporter = TraceExporter(
        reader,  # type: ignore[arg-type]
        clock=lambda: NOW,
        export_id_factory=lambda: "export-copy",
        media_resolver=WorkspaceExportMediaResolver(workspace),
    )

    manifest = exporter.export(
        output,
        options=TraceExportOptions(
            media_mode=MediaMode.COPY,
            query=TraceQuery(include_incomplete=True),
        ),
    )

    digest = hashlib.sha256(payload).hexdigest()
    assert (output / f"media/sha256/{digest[:2]}/{digest}.png").read_bytes() == payload
    assert manifest.media_count == 1
    assert manifest.media_size_bytes == len(payload)
    assert manifest.missing_media_count == 0
    lines = (output / "records-00000.jsonl").read_bytes().splitlines()
    assert lines == [canonical_record_bytes(record) for record in records]


def test_copy_mode_refuses_unresolved_media_and_reference_mode_never_resolves(
    tmp_path: Path,
) -> None:
    records = _media_records(b"missing")

    class Resolver:
        def __init__(self) -> None:
            self.calls = 0

        def resolve(self, reference: MediaReference, /) -> Path | None:
            del reference
            self.calls += 1
            return None

    resolver = Resolver()
    reference_output = tmp_path / "reference"
    TraceExporter(
        Reader(Snapshot(records, 1)),  # type: ignore[arg-type]
        clock=lambda: NOW,
        export_id_factory=lambda: "reference",
        media_resolver=resolver,
    ).export(
        reference_output,
        options=TraceExportOptions(query=TraceQuery(include_incomplete=True)),
    )
    assert resolver.calls == 0

    copy_output = tmp_path / "copy"
    with pytest.raises(TraceExportError, match="media"):
        TraceExporter(
            Reader(Snapshot(records, 1)),  # type: ignore[arg-type]
            clock=lambda: NOW,
            export_id_factory=lambda: "copy",
            media_resolver=resolver,
        ).export(
            copy_output,
            options=TraceExportOptions(
                media_mode=MediaMode.COPY,
                query=TraceQuery(include_incomplete=True),
            ),
        )
    assert not copy_output.exists()
