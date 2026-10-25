"""Audio-local generation lifecycle, search and evidence with real FFmpeg (RFC-0032).

A deterministic in-process encoder stands in for the model so ranking has an
independent oracle; it establishes orchestration, never semantic quality.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import ClassVar

import pytest

from capicove.contracts import InvocationContext, PrivacyClass
from capicove.contracts.retrieval import (
    AudioClipQuery,
    DetailLevel,
    DistanceMetric,
    EmbeddingSpaceRef,
    LocalPathSource,
    MatchKind,
    QueryModality,
    SearchFilters,
    SearchRequest,
    SearchStatus,
    TargetType,
    TemporalRange,
    TextOrigin,
    TextQuery,
    VectorNormalization,
)
from capicove.model.configuration import ModelCapability, ModelConfiguration, ModelProfile
from capicove.model.factory import ModelFactory
from capicove.model.multimodal_embedding import PcmAudio, TextDocument
from capicove.model.registry import ProviderDefinition, ProviderRegistry
from capicove.runtime import unified_retrieval as runtime
from capicove.runtime.mcp_config import WorkspaceHostConfiguration, parse_mcp_host_config


FFMPEG, FFPROBE = shutil.which("ffmpeg"), shutil.which("ffprobe")
pytestmark = pytest.mark.skipif(
    os.name != "posix" or FFMPEG is None or FFPROBE is None, reason="requires POSIX and FFmpeg"
)
SPACE = EmbeddingSpaceRef(
    space_id="test-shared:fixture",
    provider_id="test-shared",
    model_revision="fixture-1",
    preprocessing_version="fixture-prompts-1",
    dimension=8,
    normalization=VectorNormalization.L2,
    distance_metric=DistanceMetric.COSINE,
    query_modalities=frozenset({QueryModality.TEXT, QueryModality.AUDIO_CLIP}),
    target_modalities=frozenset({TargetType.AUDIO_SEGMENT}),
)


def _unit(seed: bytes) -> tuple[float, ...]:
    digest = hashlib.sha256(seed).digest()
    values = [digest[index] - 127.5 for index in range(8)]
    norm = math.sqrt(sum(value * value for value in values))
    return tuple(value / norm for value in values)


class FixtureEncoder:
    """Maps PCM to a digest-seeded vector; ``match:<sha256>`` text reproduces it."""

    provider_id = "test-shared"
    space = SPACE
    calls: ClassVar[list[str]] = []

    def encode_text_queries(
        self, texts: Sequence[str], *, deadline: datetime
    ) -> tuple[tuple[float, ...], ...]:
        self.calls.append("text")
        return tuple(
            _unit(bytes.fromhex(text.removeprefix("match:")))
            if text.startswith("match:")
            else _unit(text.encode())
            for text in texts
        )

    def encode_text_documents(
        self, documents: Sequence[TextDocument], *, deadline: datetime
    ) -> tuple[tuple[float, ...], ...]:
        raise AssertionError("audio-local profile never encodes text documents")

    def encode_images(
        self, images: Sequence[bytes], *, deadline: datetime
    ) -> tuple[tuple[float, ...], ...]:
        raise AssertionError("audio-local profile never encodes images")

    def encode_audio(
        self, clips: Sequence[PcmAudio], *, deadline: datetime
    ) -> tuple[tuple[float, ...], ...]:
        self.calls.append(f"audio:{len(clips)}")
        return tuple(_unit(hashlib.sha256(clip.pcm).digest()) for clip in clips)

    def close(self) -> None:
        return None


def _models(root: Path) -> ModelFactory:
    registry = ProviderRegistry()
    registry.register(
        ProviderDefinition(
            "test-shared", "1", ModelCapability.MULTIMODAL_EMBEDDING, lambda _: FixtureEncoder()
        )
    )
    profile = ModelProfile(
        "shared", "test-shared", ModelCapability.MULTIMODAL_EMBEDDING, root, "0" * 64
    )
    return ModelFactory(
        ModelConfiguration({"shared": profile}, {"unified": "shared"}), registry=registry
    )


def _tone(path: Path, seconds: float, frequency: float, *, rate: int = 16_000) -> bytes:
    frames = int(seconds * rate)
    pcm = struct.pack(
        f"<{frames}h",
        *(int(8000 * math.sin(2 * math.pi * frequency * i / rate)) for i in range(frames)),
    )
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(pcm)
    return pcm


@dataclass
class Workspace:
    root: Path
    media: Path
    config: Path
    models: ModelFactory

    def host(self, **overrides: object) -> WorkspaceHostConfiguration:
        body = json.loads(self.config.read_text())
        body["capabilities"]["multimodal"].update(overrides)
        parsed = parse_mcp_host_config(json.dumps(body).encode())
        assert isinstance(parsed, WorkspaceHostConfiguration)
        return parsed

    @property
    def context(self) -> InvocationContext:
        return InvocationContext(
            request_id="r",
            trace_id="t",
            deadline=datetime.now(UTC) + timedelta(seconds=60),
            workspace_id="workspace",
            actor_id="owner",
            environment_id="local",
            permissions=frozenset({"evidence:read", "evidence:search"}),
            privacy_class=PrivacyClass.CONFIDENTIAL,
        )


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    root = (tmp_path / "workspace").resolve()
    media = root / "media"
    media.mkdir(parents=True)
    config = tmp_path / "host.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "server_name": "capicove-local",
                "workspace_id": "workspace",
                "actor_id": "owner",
                "environment_id": "local",
                "workspace_root": str(root),
                "profile": "write",
                "permissions": [
                    "evidence:read",
                    "evidence:search",
                    "artifact:write",
                    "audio:read",
                    "subprocess:execute",
                ],
                "enabled_tools": ["capicove.search.multimodal", "capicove.multimodal.get_evidence"],
                "trace_store": None,
                "capabilities": {
                    "multimodal": {
                        "schema_version": 1,
                        "profile": "audio-local-v1",
                        "source_roots": {"media": str(media)},
                        "index_root": str(root / "index"),
                        "staging_root": str(root / "staging"),
                        "corpus_ref": "corpus:recordings",
                        "model_config": None,
                        "model_binding": "unified",
                        "ffmpeg": FFMPEG,
                        "ffprobe": FFPROBE,
                        "max_source_bytes": 64 * 1024 * 1024,
                    }
                },
                "network": {"enabled": False, "source_hosts": []},
            }
        )
    )
    return Workspace(root, media, config, _models(tmp_path))


def _window_hash(pcm: bytes, start: int, end: int) -> str:
    return hashlib.sha256(pcm[start * 2 : end * 2]).hexdigest()


def test_build_search_and_evidence_resolve_exact_windows(workspace: Workspace) -> None:
    pcm = _tone(workspace.media / "dog.wav", 31, 440)
    host = workspace.host()
    imported = runtime.import_media(host, "media", "dog.wav")
    assert imported["imported"] is True and imported["source_ref"] == "media:dog.wav"
    built = runtime.build_generation(host, models=workspace.models)
    assert built["units"] == 2
    runtime.activate_generation(host, built["generation_id"])
    service = runtime.open_unified_retrieval(host, models=workspace.models)
    tail = _window_hash(pcm, 400_000, 496_000)
    result = service.search(
        SearchRequest(
            TextQuery(f"match:{tail}"),
            frozenset({TargetType.AUDIO_SEGMENT}),
            top_k=2,
            detail_level=DetailLevel.FULL,
            retrieval_profile_ref="audio-local-v1",
        )
    )
    assert result.status is SearchStatus.SUCCEEDED
    assert [hit.time_range for hit in result.hits] == [
        TemporalRange(25_000, 31_000),
        TemporalRange(0, 30_000),
    ]
    first = result.hits[0]
    assert first.match_kinds == (MatchKind.AUDIO,)  # type: ignore[union-attr]
    assert first.snippet is None
    evidence = service.read_evidence(workspace.context, first.evidence_id, ("metadata", "lineage"))
    metadata = evidence["metadata"]
    assert metadata["source_ref"] == "media:dog.wav"
    assert metadata["start_frame"] == 400_000 and metadata["end_frame"] == 496_000
    assert metadata["start_ms"] == 25_000 and metadata["end_ms"] == 31_000
    assert metadata["audio_stream_index"] == 0
    assert metadata["pcm_sha256"] == tail
    assert "transcript" not in metadata and "text" not in metadata
    assert str(workspace.root) not in json.dumps(evidence)
    assert evidence["lineage"]["generation_id"] == built["generation_id"]


def test_second_import_preserves_first_and_forget_affects_next_build(workspace: Workspace) -> None:
    _tone(workspace.media / "a.wav", 2, 300)
    _tone(workspace.media / "b.wav", 2, 600)
    host = workspace.host()
    runtime.import_media(host, "media", "a.wav")
    first = runtime.build_generation(host, models=workspace.models)
    runtime.activate_generation(host, first["generation_id"])
    runtime.import_media(host, "media", "b.wav")
    second = runtime.build_generation(host, models=workspace.models)
    assert second["units"] == 2 and second["generation_id"] != first["generation_id"]
    assert runtime.import_media(host, "media", "b.wav")["imported"] is False
    forgotten = runtime.forget_media(host, "media:a.wav")
    assert forgotten["forgotten"] is True
    assert runtime.forget_media(host, "media:a.wav")["forgotten"] is False
    service = runtime.open_unified_retrieval(host, models=workspace.models)
    request = SearchRequest(
        TextQuery("anything"),
        frozenset({TargetType.AUDIO_SEGMENT}),
        retrieval_profile_ref="audio-local-v1",
    )
    assert len(service.search(request).hits) == 1
    third = runtime.build_generation(host, models=workspace.models)
    assert third["units"] == 1
    runtime.activate_generation(host, third["generation_id"])
    assert len(service.search(request).hits) == 1
    hit = service.search(request).hits[0]
    assert (
        service.read_evidence(workspace.context, hit.evidence_id, ("metadata",))["metadata"][
            "source_ref"
        ]
        == "media:b.wav"
    )


def test_identical_inputs_rebuild_to_the_same_generation(workspace: Workspace) -> None:
    _tone(workspace.media / "a.wav", 2, 300)
    host = workspace.host()
    runtime.import_media(host, "media", "a.wav")
    first = runtime.build_generation(host, models=workspace.models)
    again = runtime.build_generation(host, models=workspace.models)
    assert again["generation_id"] == first["generation_id"] and again["reused"] is True


def test_failed_build_and_unactivated_generation_preserve_active_results(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tone(workspace.media / "a.wav", 2, 300)
    host = workspace.host()
    runtime.import_media(host, "media", "a.wav")
    active = runtime.build_generation(host, models=workspace.models)
    runtime.activate_generation(host, active["generation_id"])
    _tone(workspace.media / "b.wav", 2, 700)
    runtime.import_media(host, "media", "b.wav")

    def broken(self: object, clips: Sequence[PcmAudio], *, deadline: datetime) -> object:
        raise RuntimeError("encoder crashed mid-build")

    monkeypatch.setattr(FixtureEncoder, "encode_audio", broken)
    with pytest.raises(Exception, match="encoder"):
        runtime.build_generation(host, models=workspace.models)
    monkeypatch.undo()
    staged = list((workspace.root / "index" / "generations").glob(".staging-*"))
    assert staged == []
    pending = runtime.build_generation(host, models=workspace.models)
    reopened = runtime.open_unified_retrieval(workspace.host(), models=workspace.models)
    request = SearchRequest(
        TextQuery("q"),
        frozenset({TargetType.AUDIO_SEGMENT}),
        retrieval_profile_ref="audio-local-v1",
    )
    assert len(reopened.search(request).hits) == 1
    runtime.activate_generation(host, pending["generation_id"])
    assert len(reopened.search(request).hits) == 2


def test_filters_apply_before_ranking_and_empty_corpus_is_successful(workspace: Workspace) -> None:
    host = workspace.host()
    empty = runtime.build_generation(host, models=workspace.models)
    runtime.activate_generation(host, empty["generation_id"])
    service = runtime.open_unified_retrieval(host, models=workspace.models)
    request = SearchRequest(
        TextQuery("q"),
        frozenset({TargetType.AUDIO_SEGMENT}),
        retrieval_profile_ref="audio-local-v1",
    )
    assert service.search(request).status is SearchStatus.SUCCEEDED
    assert service.search(request).hits == ()
    pcm = _tone(workspace.media / "long.wav", 61, 500)
    runtime.import_media(host, "media", "long.wav")
    runtime.activate_generation(
        host, runtime.build_generation(host, models=workspace.models)["generation_id"]
    )
    nearest_excluded = _window_hash(pcm, 0, 480_000)
    filtered = service.search(
        SearchRequest(
            TextQuery(f"match:{nearest_excluded}"),
            frozenset({TargetType.AUDIO_SEGMENT}),
            top_k=1,
            retrieval_profile_ref="audio-local-v1",
            filters=SearchFilters(time_range=TemporalRange(56_000, 61_000)),
        )
    )
    assert [hit.time_range for hit in filtered.hits] == [TemporalRange(50_000, 61_000)]
    for filters in (
        SearchFilters(text_origins=frozenset({TextOrigin.ASR})),
        SearchFilters(corpus_refs=frozenset({"corpus:other"})),
        SearchFilters(media_types=frozenset({"video/mp4"})),
    ):
        none = service.search(
            SearchRequest(
                TextQuery("q"),
                frozenset({TargetType.AUDIO_SEGMENT}),
                retrieval_profile_ref="audio-local-v1",
                filters=filters,
            )
        )
        assert none.hits == ()
    audio_only = service.search(
        SearchRequest(
            TextQuery("q"),
            frozenset({TargetType.AUDIO_SEGMENT}),
            retrieval_profile_ref="audio-local-v1",
            filters=SearchFilters(media_types=frozenset({"audio/wav"})),
        )
    )
    assert len(audio_only.hits) == 3


def test_audio_clip_query_ranks_the_matching_window_first(workspace: Workspace) -> None:
    pcm = _tone(workspace.media / "a.wav", 31, 300)
    host = workspace.host()
    runtime.import_media(host, "media", "a.wav")
    runtime.activate_generation(
        host, runtime.build_generation(host, models=workspace.models)["generation_id"]
    )
    query = workspace.media / "query.wav"
    with wave.open(str(query), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(pcm[400_000 * 2 : 496_000 * 2])
    service = runtime.open_unified_retrieval(host, models=workspace.models)
    result = service.search(
        SearchRequest(
            AudioClipQuery(LocalPathSource(str(query))),
            frozenset({TargetType.AUDIO_SEGMENT}),
            retrieval_profile_ref="audio-local-v1",
        )
    )
    assert result.hits[0].time_range == TemporalRange(25_000, 31_000)
    ranged = service.search(
        SearchRequest(
            AudioClipQuery(
                LocalPathSource(str(workspace.media / "a.wav")), TemporalRange(25_000, 31_000)
            ),
            frozenset({TargetType.AUDIO_SEGMENT}),
            retrieval_profile_ref="audio-local-v1",
        )
    )
    assert ranged.hits[0].time_range == TemporalRange(25_000, 31_000)


def test_invalid_audio_queries_fail_before_encoding(workspace: Workspace, tmp_path: Path) -> None:
    _tone(workspace.media / "long.wav", 31, 300)
    host = workspace.host()
    runtime.activate_generation(
        host, runtime.build_generation(host, models=workspace.models)["generation_id"]
    )
    service = runtime.open_unified_retrieval(host, models=workspace.models)
    outside = tmp_path / "outside.wav"
    _tone(outside, 1, 300)
    FixtureEncoder.calls.clear()
    for query in (
        AudioClipQuery(LocalPathSource(str(workspace.media / "long.wav"))),
        AudioClipQuery(
            LocalPathSource(str(workspace.media / "long.wav")), TemporalRange(0, 31_000)
        ),
        AudioClipQuery(LocalPathSource(str(outside))),
    ):
        result = service.search(
            SearchRequest(
                query, frozenset({TargetType.AUDIO_SEGMENT}), retrieval_profile_ref="audio-local-v1"
            )
        )
        assert result.status is SearchStatus.FAILED
    assert FixtureEncoder.calls == []
    for request in (
        SearchRequest(
            TextQuery("q"),
            frozenset({TargetType.TEXT_CHUNK}),
            retrieval_profile_ref="audio-local-v1",
        ),
        SearchRequest(
            TextQuery("q"),
            frozenset({TargetType.AUDIO_SEGMENT}),
            cursor="c",
            retrieval_profile_ref="audio-local-v1",
        ),
        SearchRequest(
            TextQuery("q"),
            frozenset({TargetType.AUDIO_SEGMENT}),
            retrieval_profile_ref="unified-local-v1",
        ),
    ):
        assert service.search(request).status is SearchStatus.FAILED


def test_tampered_generation_and_foreign_evidence_fail_closed(workspace: Workspace) -> None:
    _tone(workspace.media / "a.wav", 2, 300)
    host = workspace.host()
    runtime.import_media(host, "media", "a.wav")
    built = runtime.build_generation(host, models=workspace.models)
    runtime.activate_generation(host, built["generation_id"])
    service = runtime.open_unified_retrieval(host, models=workspace.models)
    request = SearchRequest(
        TextQuery("q"),
        frozenset({TargetType.AUDIO_SEGMENT}),
        retrieval_profile_ref="audio-local-v1",
    )
    evidence_id = service.search(request).hits[0].evidence_id
    foreign = InvocationContext(
        request_id="r",
        trace_id="t",
        deadline=datetime.now(UTC) + timedelta(seconds=60),
        workspace_id="other",
        actor_id="owner",
        environment_id="local",
        permissions=frozenset({"evidence:read"}),
        privacy_class=PrivacyClass.CONFIDENTIAL,
    )
    with pytest.raises(PermissionError):
        service.read_evidence(foreign, evidence_id, ("metadata",))
    with pytest.raises(LookupError):
        service.read_evidence(workspace.context, "mev-" + "0" * 64, ("metadata",))
    database = workspace.root / "index" / "generations" / built["generation_id"] / "units.sqlite3"
    database.chmod(0o600)
    with database.open("r+b") as stream:
        stream.seek(200)
        stream.write(b"\xff\xfe")
    assert service.search(request).status is SearchStatus.FAILED
    with pytest.raises(LookupError):
        service.read_evidence(workspace.context, evidence_id, ("metadata",))


def test_read_profile_cannot_import_or_build(workspace: Workspace) -> None:
    _tone(workspace.media / "a.wav", 1, 300)
    body = json.loads(workspace.config.read_text())
    body.update(profile="read", permissions=["evidence:read", "evidence:search"])
    host = parse_mcp_host_config(json.dumps(body).encode())
    assert isinstance(host, WorkspaceHostConfiguration)
    with pytest.raises(PermissionError):
        runtime.import_media(host, "media", "a.wav")
    with pytest.raises(PermissionError):
        runtime.build_generation(host, models=workspace.models)
    assert not (workspace.root / "index").exists()


@pytest.mark.parametrize(
    "overrides",
    [
        {"profile": "unknown-v1"},
        {"schema_version": 2},
        {"extra": True},
        {"index_root": "/tmp/outside-index"},
        {"ffmpeg": "ffmpeg"},
        {"source_roots": {}},
        {"max_source_bytes": 0},
    ],
)
def test_configuration_is_closed_and_validated(
    workspace: Workspace, overrides: dict[str, object]
) -> None:
    with pytest.raises((ValueError, runtime.UnifiedRetrievalError)):
        runtime.unified_settings(workspace.host(**overrides))
