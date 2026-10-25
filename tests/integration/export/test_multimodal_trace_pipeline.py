"""End-to-end reconstruction of one cleanable multimodal Agent trajectory."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from capicove.batch import BatchState
from capicove.contracts import InvocationContext, ToolResult, ToolSpec, ToolStatus
from capicove.export import (
    MediaMode,
    TraceExporter,
    TraceExportOptions,
    WorkspaceExportMediaResolver,
    validate_trace_export,
)
from capicove.tools import AUDIO_TRANSCRIBE_SPEC, VIDEO_FILTER_SPEC, VIDEO_SPLIT_SPEC
from capicove.tools.image import IMAGE_TOOL_SPECS_BY_NAME
from capicove.tools.tracing import tool_catalog_sha256
from capicove.traces import (
    AgentLineage,
    ArtifactProduced,
    BatchItemObserved,
    EpisodeFeedback,
    EpisodeStatus,
    LineageAvailability,
    MediaContent,
    MediaLocator,
    MediaLocatorKind,
    MediaReference,
    MessageObserved,
    MessageRole,
    TextContent,
    ToolCallObserved,
    ToolResultObserved,
    TraceModality,
    TraceQuery,
    TraceReader,
    TraceRecorder,
    TraceSanitizer,
)
from capicove.traces._adapters.sqlite import SqliteTraceStore
from capicove.traces.codec import decode_trace_record


pytestmark = pytest.mark.integration
_NOW = datetime(2026, 8, 18, 12, tzinfo=UTC)


def test_mixed_modal_episode_round_trips_as_model_neutral_trajectory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    references = _write_media(workspace)
    store = SqliteTraceStore((tmp_path / "traces.sqlite3").resolve())
    specs = (
        IMAGE_TOOL_SPECS_BY_NAME["image_ocr"],
        VIDEO_SPLIT_SPEC,
        VIDEO_FILTER_SPEC,
        AUDIO_TRANSCRIBE_SPEC,
    )
    session = TraceRecorder(
        store,
        sanitizer=TraceSanitizer(),
        clock=lambda: _NOW,
    ).start_session(
        episode_id="episode-multimodal",
        trace_id="trace-multimodal",
        goal="Inspect visible image, video, and audio evidence.",
        workspace_id="workspace-1",
        actor_id="agent-user",
        environment_id="offline-local",
        environment_version=1,
        tool_catalog_sha256=tool_catalog_sha256(specs),
        active_asset_versions={"asset-bundle": 3},
        agent_lineage=AgentLineage(
            LineageAvailability.CAPTURED,
            agent_id="agent-1",
            agent_version="1.0",
            model_id="omni-model",
            model_revision="revision-1",
            generation_parameters_sha256="e" * 64,
            prompt_template_sha256="f" * 64,
        ),
    )
    session.record_message(
        message_id="user-1",
        role=MessageRole.USER,
        content=(
            TextContent("Compare these inputs: "),
            MediaContent("image-1"),
            TextContent(" then "),
            MediaContent("video-1"),
            TextContent(" and finally "),
            MediaContent("audio-1"),
        ),
        media=(references["image-1"], references["video-1"], references["audio-1"]),
    )

    _record_tool(
        session,
        specs[0],
        request_id="request-image",
        arguments={"source": {"type": "media_reference", "media_id": "image-1"}},
        media=(references["image-1"],),
        result={
            "words": [
                {
                    "text": "Capicove",
                    "region": {"x": 10, "y": 20, "width": 100, "height": 24},
                }
            ],
            "lineage": {"provider": "local-ocr", "model_sha256": "1" * 64},
        },
    )
    _record_tool(
        session,
        specs[1],
        request_id="request-video-split",
        arguments={
            "source": {"type": "media_reference", "media_id": "video-1"},
            "strategy": "duration",
            "config": {"split_duration": 2.0},
        },
        media=(references["video-1"],),
        result={
            "segments": [
                {
                    "index": 0,
                    "start_seconds": 0.0,
                    "end_seconds": 2.0,
                    "frame_indices": [0, 12, 24],
                    "media_id": "clip-1",
                }
            ]
        },
    )
    session.record_artifact(
        event_id="clip-1",
        artifact_ref="artifact:clip-1",
        media=references["clip-1"],
        derived_from=("video-1",),
    )
    _record_tool(
        session,
        specs[2],
        request_id="request-video-filter",
        arguments={
            "sources": [{"media_id": "clip-1", "start_seconds": 0.0, "end_seconds": 2.0}],
            "criteria": [{"type": "duration", "min_seconds": 1.0}],
            "aggregation": "all",
        },
        media=(references["clip-1"],),
        result={"keep": True, "frame_indices": [0, 12, 24], "score": 0.98},
    )
    _record_tool(
        session,
        specs[3],
        request_id="request-audio",
        arguments={"source": {"type": "media_reference", "media_id": "audio-1"}},
        media=(references["audio-1"],),
        result={
            "text": "hello world",
            "language": "en",
            "warnings": ["TIMESTAMP_CLAMPED"],
            "segments": [
                {"index": 0, "start_seconds": 0.0, "end_seconds": 0.8, "text": "hello"},
                {"index": 1, "start_seconds": 0.8, "end_seconds": 1.5, "text": "world"},
            ],
            "lineage": {"backend": "local", "model_sha256": "2" * 64},
        },
    )
    for event_id, state, attempts, cache_hit, detail in (
        ("retried", BatchState.SUCCEEDED, 2, False, "succeeded after retry"),
        ("cached", BatchState.SUCCEEDED, 0, True, "cache hit"),
        ("skipped", BatchState.SKIPPED, 0, False, "unsupported input"),
        ("changed", BatchState.SOURCE_CHANGED, 0, False, "source changed"),
    ):
        session.record_batch_item(
            event_id=f"batch:{event_id}",
            payload=_batch_item(event_id, state, attempts, cache_hit, detail),
        )
    session.record_message(
        message_id="assistant-1",
        role=MessageRole.ASSISTANT,
        content=(TextContent("The visible evidence is consistent."),),
    )
    completion = session.complete(status=EpisodeStatus.SUCCEEDED, outcome="inspection complete")
    session.record_feedback(
        feedback_id="feedback-1",
        payload=EpisodeFeedback(
            "quality",
            {"score": 0.95, "label": "useful"},
            "human-review",
            completion.record_id,
        ),
    )

    output = tmp_path / "export"
    manifest = TraceExporter(
        cast(TraceReader, store),
        clock=lambda: _NOW,
        export_id_factory=lambda: "export-multimodal",
        media_resolver=WorkspaceExportMediaResolver(workspace.resolve()),
    ).export(
        output.resolve(),
        options=TraceExportOptions(
            query=TraceQuery(),
            media_mode=MediaMode.COPY,
            max_records_per_shard=5,
        ),
    )
    report = validate_trace_export(output.resolve())
    assert report.valid, report.issues
    assert manifest.media_count == 4
    assert manifest.missing_media_count == 0

    records = [
        decode_trace_record(json.loads(line))
        for shard in manifest.shards
        for line in (output / shard.path).read_text(encoding="utf-8").splitlines()
    ]
    grouped = sorted(
        (record for record in records if record.episode_id == "episode-multimodal"),
        key=lambda record: record.sequence,
    )
    assert [record.sequence for record in grouped] == list(range(len(grouped)))
    user_message = next(
        record.payload
        for record in grouped
        if isinstance(record.payload, MessageObserved) and record.payload.role is MessageRole.USER
    )
    assert user_message.content == (
        TextContent("Compare these inputs: "),
        MediaContent("image-1"),
        TextContent(" then "),
        MediaContent("video-1"),
        TextContent(" and finally "),
        MediaContent("audio-1"),
    )
    calls = [record.payload for record in grouped if isinstance(record.payload, ToolCallObserved)]
    results = [
        record.payload for record in grouped if isinstance(record.payload, ToolResultObserved)
    ]
    assert len(calls) == len(results) == 4
    assert cast(dict[str, object], results[0].result)["words"]
    assert cast(dict[str, object], results[1].result)["segments"]
    assert cast(dict[str, object], results[3].result)["warnings"] == ("TIMESTAMP_CLAMPED",)
    artifact = next(
        record.payload for record in grouped if isinstance(record.payload, ArtifactProduced)
    )
    assert artifact.derived_from == ("video-1",)
    batch = [record.payload for record in grouped if isinstance(record.payload, BatchItemObserved)]
    assert {(item.state, item.attempts, item.cache_hit) for item in batch} == {
        (BatchState.SUCCEEDED, 2, False),
        (BatchState.SUCCEEDED, 0, True),
        (BatchState.SKIPPED, 0, False),
        (BatchState.SOURCE_CHANGED, 0, False),
    }
    assert grouped[0].payload.agent_lineage is not None  # type: ignore[union-attr]
    serialized = "\n".join(
        (output / shard.path).read_text(encoding="utf-8") for shard in manifest.shards
    )
    for forbidden in (
        '"chosen"',
        '"rejected"',
        '"reward"',
        '"chain_of_thought"',
        '"reasoning"',
        "<image>",
        "<video>",
        "<audio>",
    ):
        assert forbidden not in serialized


def _write_media(workspace: Path) -> dict[str, MediaReference]:
    values = {
        "image-1": (TraceModality.IMAGE, "image/png", "inputs/image.png", b"image-bytes"),
        "video-1": (TraceModality.VIDEO, "video/mp4", "inputs/video.mp4", b"video-bytes"),
        "audio-1": (TraceModality.AUDIO, "audio/wav", "inputs/audio.wav", b"audio-bytes"),
        "clip-1": (TraceModality.VIDEO, "video/mp4", "derived/clip.mp4", b"clip-bytes"),
    }
    references: dict[str, MediaReference] = {}
    for media_id, (modality, media_type, relative, payload) in values.items():
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        references[media_id] = MediaReference(
            media_id,
            modality,
            media_type,
            hashlib.sha256(payload).hexdigest(),
            len(payload),
            locator=MediaLocator(MediaLocatorKind.WORKSPACE_RELATIVE, relative),
            start_seconds=0.0 if media_id == "clip-1" else None,
            end_seconds=2.0 if media_id == "clip-1" else None,
            frame_indices=(0, 12, 24) if media_id == "clip-1" else (),
            derived_from=("video-1",) if media_id == "clip-1" else (),
        )
    return references


def _record_tool(
    session: object,
    spec: ToolSpec,
    *,
    request_id: str,
    arguments: dict[str, object],
    media: tuple[MediaReference, ...],
    result: dict[str, object],
) -> None:
    from capicove.traces import TraceSession

    assert isinstance(session, TraceSession)
    context = InvocationContext(
        request_id=request_id,
        trace_id=session.trace_id,
        workspace_id="workspace-1",
        actor_id="agent-user",
        environment_id="offline-local",
        deadline=_NOW + timedelta(minutes=5),
        permissions=spec.required_permissions,
        privacy_class=spec.privacy_class,
    )
    session.record_tool_call(
        context=context,
        spec=spec,
        arguments=arguments,
        media=media,
        attempt=1,
    )
    session.record_tool_result(
        result=ToolResult(
            request_id=request_id,
            trace_id=session.trace_id,
            tool_id=spec.tool_id,
            tool_version=spec.contract_version,
            status=ToolStatus.SUCCEEDED,
            result=result,
            evidence_refs=tuple(f"sha256:{item.sha256}" for item in media),
            started_at=_NOW,
            completed_at=_NOW,
        ),
        spec=spec,
        attempt=1,
    )


def _batch_item(
    label: str,
    state: BatchState,
    attempts: int,
    cache_hit: bool,
    detail: str,
) -> BatchItemObserved:
    digest = hashlib.sha256(label.encode()).hexdigest()
    return BatchItemObserved(
        plan_id=f"plan:{'9' * 64}",
        item_id=f"item:{digest}",
        state=state,
        attempts=attempts,
        cache_hit=cache_hit,
        modality=TraceModality.IMAGE,
        media_type="image/png",
        size_bytes=10,
        source_sha256=digest,
        cache_key=f"cache:{digest}",
        detail=detail,
    )
