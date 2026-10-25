"""Additive version-4 native audio retrieval contracts (RFC-0032)."""

from __future__ import annotations

import pytest

from capicove import contracts as public_contracts
from capicove.contracts.retrieval import (
    MAX_INLINE_IMAGE_BYTES,
    RETRIEVAL_CONTRACT_VERSION,
    ArtifactSource,
    AudioClipQuery,
    CharacterRange,
    CompactHit,
    FullHit,
    InlineBytesSource,
    LocalPathSource,
    MatchKind,
    ProcessingFingerprint,
    QueryModality,
    SearchableUnit,
    SearchRequest,
    TargetType,
    TemporalRange,
    TextChunkLocator,
    TextOrigin,
)


def _unit(**overrides: object) -> SearchableUnit:
    values: dict[str, object] = {
        "unit_id": "unit-audio",
        "unit_type": TargetType.AUDIO_SEGMENT,
        "source_hash": "a" * 64,
        "artifact_ref": "artifact:audio",
        "evidence_ref": "ev-audio",
        "representation_refs": ("audio-receipt:1",),
        "processing_run_ref": "run:1",
        "processing_version": "audio-window-v1",
        "processing_fingerprint": ProcessingFingerprint("b" * 64),
        "time_range": TemporalRange(0, 30_000),
    }
    values.update(overrides)
    return SearchableUnit(**values)  # type: ignore[arg-type]


def test_version_four_adds_audio_discriminators_without_renaming_existing_values() -> None:
    assert RETRIEVAL_CONTRACT_VERSION == 4
    assert QueryModality.AUDIO_CLIP.value == "audio_clip"
    assert TargetType.AUDIO_SEGMENT.value == "audio_segment"
    assert MatchKind.AUDIO.value == "audio"
    assert [item.value for item in QueryModality] == ["text", "image", "video_clip", "audio_clip"]
    assert [item.value for item in TextOrigin] == ["document", "ocr", "asr"]
    assert public_contracts.AudioClipQuery is AudioClipQuery


def test_audio_clip_query_accepts_source_range_and_stream() -> None:
    query = AudioClipQuery(LocalPathSource("/media/bark.wav"), TemporalRange(0, 5_000), 1)
    assert query.type is QueryModality.AUDIO_CLIP
    assert query.stream_index == 1
    assert AudioClipQuery(ArtifactSource("artifact:1")).time_range is None
    request = SearchRequest(query, frozenset({TargetType.AUDIO_SEGMENT}))
    assert request.query is query


@pytest.mark.parametrize("stream_index", [-1, True, 1.0, "0", 64])
def test_audio_clip_query_rejects_invalid_stream_ordinals(stream_index: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        AudioClipQuery(LocalPathSource("/media/a.wav"), None, stream_index)  # type: ignore[arg-type]


def test_audio_clip_query_rejects_non_canonical_sources_and_ranges() -> None:
    with pytest.raises(TypeError):
        AudioClipQuery("/media/a.wav")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        AudioClipQuery(LocalPathSource("/media/a.wav"), (0, 10))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        AudioClipQuery(LocalPathSource("/media/a.wav"), TemporalRange(5, 5))


def test_inline_audio_keeps_the_media_ingress_ceiling() -> None:
    inline = AudioClipQuery(InlineBytesSource(b"RIFF", "audio/wav")).source
    assert isinstance(inline, InlineBytesSource) and inline.media_type == "audio/wav"
    with pytest.raises(ValueError):
        InlineBytesSource(b"0" * (MAX_INLINE_IMAGE_BYTES + 1), "audio/wav")


def test_audio_segment_unit_requires_a_nonempty_time_range_only() -> None:
    assert _unit().time_range == TemporalRange(0, 30_000)
    with pytest.raises(ValueError, match="time_range"):
        _unit(time_range=None)
    with pytest.raises(ValueError):
        _unit(timestamp_ms=10)
    with pytest.raises(ValueError):
        _unit(scene_range=TemporalRange(0, 10))
    with pytest.raises(ValueError, match="text_locator"):
        _unit(
            text_locator=TextChunkLocator(
                TextOrigin.ASR, CharacterRange(0, 3), None, TemporalRange(0, 1)
            )
        )


def test_audio_hits_forbid_transcript_text_and_text_locators() -> None:
    hit = CompactHit(
        "h", TargetType.AUDIO_SEGMENT, 0.5, "ev", time_range=TemporalRange(25_000, 31_000)
    )
    assert hit.time_range == TemporalRange(25_000, 31_000)
    with pytest.raises(ValueError, match="time_range"):
        CompactHit("h", TargetType.AUDIO_SEGMENT, 0.5, "ev")
    with pytest.raises(ValueError, match="snippet"):
        CompactHit(
            "h",
            TargetType.AUDIO_SEGMENT,
            0.5,
            "ev",
            time_range=TemporalRange(0, 1),
            snippet="a dog barks",
        )
    full = FullHit(
        "h",
        TargetType.AUDIO_SEGMENT,
        0.5,
        "ev",
        time_range=TemporalRange(0, 1),
        match_kinds=(MatchKind.AUDIO,),
    )
    assert full.match_kinds == (MatchKind.AUDIO,)


def test_legacy_services_reject_audio_variants_before_dispatch() -> None:
    from datetime import UTC, datetime, timedelta

    from capicove.retrieval.text_search import TextSearchError, TextSearchService
    from capicove.retrieval.video_search import VideoSearchError, VideoSearchService
    from capicove.tools.search import decode_search_arguments

    request = SearchRequest(
        AudioClipQuery(LocalPathSource("/media/a.wav")), frozenset({TargetType.AUDIO_SEGMENT})
    )
    with pytest.raises(TextSearchError, match="invalid_query"):
        TextSearchService._validate_request(request)
    video = VideoSearchService.__new__(VideoSearchService)
    with pytest.raises(VideoSearchError, match="unsupported_query_target"):
        video.search_with_deadline(request, deadline=datetime.now(UTC) + timedelta(seconds=5))
    with pytest.raises((TypeError, ValueError, KeyError)):
        decode_search_arguments(
            {
                "query": {"type": "audio_clip", "source": {"type": "local_path", "path": "/a.wav"}},
                "targets": ["audio_segment"],
            }
        )
