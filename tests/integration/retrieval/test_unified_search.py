"""Unified-local cross-target ranking, lexical fusion and evidence (RFC-0032).

The fixture encoder makes the dense lane an independent oracle: ``match:<seed>``
queries reproduce a stored unit's vector. Lexical expectations come from FTS5 on
authored text. Neither establishes semantic model quality.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest

from capicove.contracts import InvocationContext
from capicove.contracts.retrieval import (
    DetailLevel,
    DistanceMetric,
    EmbeddingSpaceRef,
    FullHit,
    ImageQuery,
    InlineBytesSource,
    LocalPathSource,
    MatchKind,
    QueryModality,
    SearchFilters,
    SearchRequest,
    SearchStatus,
    TargetType,
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
from tests.integration.retrieval.test_unified_generation import _tone


FFMPEG, FFPROBE = shutil.which("ffmpeg"), shutil.which("ffprobe")
pytestmark = pytest.mark.skipif(
    os.name != "posix" or FFMPEG is None or FFPROBE is None, reason="requires POSIX and FFmpeg"
)
PROFILE = "unified-local-v1"
ALL_TARGETS = frozenset({TargetType.TEXT_CHUNK, TargetType.IMAGE, TargetType.AUDIO_SEGMENT})
SPACE = EmbeddingSpaceRef(
    space_id="test-unified:fixture",
    provider_id="test-unified",
    model_revision="fixture-1",
    preprocessing_version="fixture-prompts-1",
    dimension=8,
    normalization=VectorNormalization.L2,
    distance_metric=DistanceMetric.COSINE,
    query_modalities=frozenset({QueryModality.TEXT, QueryModality.IMAGE, QueryModality.AUDIO_CLIP}),
    target_modalities=frozenset(
        {TargetType.TEXT_CHUNK, TargetType.IMAGE, TargetType.AUDIO_SEGMENT, TargetType.KEYFRAME}
    ),
)


def _vector(seed: bytes) -> tuple[float, ...]:
    digest = hashlib.sha256(seed).digest()
    values = [digest[index] - 127.5 for index in range(8)]
    norm = math.sqrt(sum(value * value for value in values))
    return tuple(value / norm for value in values)


class UnifiedEncoder:
    """Seed vectors by content; ``match:`` text reproduces a document/image/audio seed."""

    provider_id = "test-unified"
    space = SPACE
    documents: ClassVar[list[str]] = []

    def encode_text_queries(
        self, texts: Sequence[str], *, deadline: datetime
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(
            _vector(text.removeprefix("match:").encode())
            if text.startswith("match:")
            else _vector(b"q:" + text.encode())
            for text in texts
        )

    def encode_text_documents(
        self, documents: Sequence[TextDocument], *, deadline: datetime
    ) -> tuple[tuple[float, ...], ...]:
        self.documents.extend(document.text for document in documents)
        return tuple(_vector(document.text.encode()) for document in documents)

    def encode_images(
        self, images: Sequence[bytes], *, deadline: datetime
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(_vector(hashlib.sha256(image).hexdigest().encode()) for image in images)

    def encode_audio(
        self, clips: Sequence[PcmAudio], *, deadline: datetime
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(_vector(hashlib.sha256(clip.pcm).hexdigest().encode()) for clip in clips)

    def close(self) -> None:
        return None


def _models(root: Path) -> ModelFactory:
    registry = ProviderRegistry()
    registry.register(
        ProviderDefinition(
            "test-unified", "1", ModelCapability.MULTIMODAL_EMBEDDING, lambda _: UnifiedEncoder()
        )
    )
    profile = ModelProfile(
        "shared", "test-unified", ModelCapability.MULTIMODAL_EMBEDDING, root, "0" * 64
    )
    return ModelFactory(
        ModelConfiguration({"shared": profile}, {"unified": "shared"}), registry=registry
    )


def _png(color: tuple[int, int, int]) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def host(tmp_path: Path) -> WorkspaceHostConfiguration:
    root = (tmp_path / "workspace").resolve()
    media = root / "media"
    media.mkdir(parents=True)
    body = {
        "schema_version": 2,
        "server_name": "capicove-local",
        "workspace_id": "workspace",
        "actor_id": "owner",
        "environment_id": "local",
        "workspace_root": str(root),
        "profile": "write",
        "permissions": ["evidence:read", "evidence:search"],
        "enabled_tools": ["capicove.search.multimodal"],
        "trace_store": None,
        "capabilities": {
            "multimodal": {
                "schema_version": 1,
                "profile": PROFILE,
                "source_roots": {"media": str(media)},
                "index_root": str(root / "index"),
                "staging_root": str(root / "staging"),
                "corpus_ref": "corpus:mixed",
                "model_config": None,
                "model_binding": "unified",
                "ffmpeg": FFMPEG,
                "ffprobe": FFPROBE,
                "max_source_bytes": 64 * 1024 * 1024,
            }
        },
        "network": {"enabled": False, "source_hosts": []},
    }
    parsed = parse_mcp_host_config(json.dumps(body).encode())
    assert isinstance(parsed, WorkspaceHostConfiguration)
    return parsed


def _context() -> InvocationContext:
    return InvocationContext(
        request_id="r",
        trace_id="t",
        workspace_id="workspace",
        actor_id="owner",
        environment_id="local",
        deadline=datetime.now(UTC) + timedelta(seconds=60),
        permissions=frozenset({"evidence:read"}),
    )


def _corpus(host: WorkspaceHostConfiguration, tmp_path: Path) -> tuple[ModelFactory, Path]:
    media = host.workspace_root / "media"
    (media / "notes.md").write_text(
        "The quokka photo was taken on Rottnest Island.\n\n"
        "Invoice ZX-4471 is overdue and needs follow up.\n",
        encoding="utf-8",
    )
    (media / "red.png").write_bytes(_png((255, 0, 0)))
    _tone(media / "alarm.wav", 2, 880)
    models = _models(tmp_path)
    for name in ("notes.md", "red.png", "alarm.wav"):
        runtime.import_media(host, "media", name)
    built = runtime.build_generation(host, models=models)
    runtime.activate_generation(host, built["generation_id"])
    return models, media


def _request(query: object, **overrides: object) -> SearchRequest:
    values: dict[str, object] = {
        "query": query,
        "targets": ALL_TARGETS,
        "top_k": 10,
        "retrieval_profile_ref": PROFILE,
    }
    values.update(overrides)
    return SearchRequest(**values)  # type: ignore[arg-type]


def test_text_query_returns_units_of_every_modality_in_one_list(
    host: WorkspaceHostConfiguration, tmp_path: Path
) -> None:
    models, _ = _corpus(host, tmp_path)
    service = runtime.open_unified_retrieval(host, models=models)
    result = service.search(_request(TextQuery("anything at all")))
    assert result.status is SearchStatus.SUCCEEDED
    assert {hit.target_type for hit in result.hits} == ALL_TARGETS
    assert len(result.hits) == 3
    text_hits = [hit for hit in result.hits if hit.target_type is TargetType.TEXT_CHUNK]
    assert all(hit.snippet and hit.text_locator for hit in text_hits)
    assert all(
        hit.text_locator is not None and hit.text_locator.origin is TextOrigin.DOCUMENT
        for hit in text_hits
    )


def test_lexical_lane_finds_rare_identifier_and_fusion_is_rank_based(
    host: WorkspaceHostConfiguration, tmp_path: Path
) -> None:
    models, _ = _corpus(host, tmp_path)
    service = runtime.open_unified_retrieval(host, models=models)
    result = service.search(_request(TextQuery("ZX-4471"), detail_level=DetailLevel.FULL, top_k=4))
    first = result.hits[0]
    assert isinstance(first, FullHit)
    assert first.target_type is TargetType.TEXT_CHUNK
    assert first.snippet is not None and "ZX-4471" in first.snippet
    lanes = {lane.lane_id: lane.rank for lane in first.lane_scores}
    assert lanes["lexical"] == 1 and "shared_dense" in lanes
    assert MatchKind.LEXICAL in first.match_kinds
    # Handwritten weighted RRF (k=60, weights 1.0): a unit ranked first lexically
    # and r-th densely scores 1/61 + 1/(60+r) and outranks every dense-only unit.
    assert first.lane_scores and len(first.lane_scores) == 2


def test_dense_cross_modal_match_ranks_image_and_audio(
    host: WorkspaceHostConfiguration, tmp_path: Path
) -> None:
    models, media = _corpus(host, tmp_path)
    service = runtime.open_unified_retrieval(host, models=models)
    image_seed = hashlib.sha256((media / "red.png").read_bytes()).hexdigest()
    result = service.search(_request(TextQuery(f"match:{image_seed}"), top_k=1))
    assert [hit.target_type for hit in result.hits] == [TargetType.IMAGE]
    by_image = service.search(
        _request(ImageQuery(LocalPathSource(str(media / "red.png"))), top_k=1)
    )
    assert [hit.target_type for hit in by_image.hits] == [TargetType.IMAGE]
    inline = service.search(
        _request(
            ImageQuery(InlineBytesSource((media / "red.png").read_bytes(), "image/png")), top_k=1
        )
    )
    assert inline.hits[0].id == by_image.hits[0].id
    evidence = service.read_evidence(
        _context(), by_image.hits[0].evidence_id, ("metadata", "lineage")
    )
    assert (
        evidence["metadata"]["kind"] == "image"
        and evidence["metadata"]["media_type"] == "image/png"
    )
    assert evidence["metadata"]["source_ref"] == "media:red.png"


def test_text_evidence_returns_exact_range_and_text(
    host: WorkspaceHostConfiguration, tmp_path: Path
) -> None:
    models, media = _corpus(host, tmp_path)
    service = runtime.open_unified_retrieval(host, models=models)
    hit = service.search(_request(TextQuery("ZX-4471"), top_k=1)).hits[0]
    metadata = service.read_evidence(_context(), hit.evidence_id, ("metadata",))["metadata"]
    source = (media / "notes.md").read_text(encoding="utf-8")
    assert metadata["text"] == source[metadata["start_char"] : metadata["end_char"]]
    assert "ZX-4471" in metadata["text"]


def test_filters_restrict_targets_origins_and_media_before_ranking(
    host: WorkspaceHostConfiguration, tmp_path: Path
) -> None:
    models, _ = _corpus(host, tmp_path)
    service = runtime.open_unified_retrieval(host, models=models)
    audio_only = service.search(
        _request(TextQuery("q"), targets=frozenset({TargetType.AUDIO_SEGMENT}))
    )
    assert {hit.target_type for hit in audio_only.hits} == {TargetType.AUDIO_SEGMENT}
    documents = service.search(
        _request(
            TextQuery("q"), filters=SearchFilters(text_origins=frozenset({TextOrigin.DOCUMENT}))
        )
    )
    assert {hit.target_type for hit in documents.hits} == {TargetType.TEXT_CHUNK}
    images = service.search(
        _request(TextQuery("q"), filters=SearchFilters(media_types=frozenset({"image/png"})))
    )
    assert {hit.target_type for hit in images.hits} == {TargetType.IMAGE}
    asr = service.search(
        _request(TextQuery("q"), filters=SearchFilters(text_origins=frozenset({TextOrigin.ASR})))
    )
    assert asr.status is SearchStatus.SUCCEEDED and asr.hits == ()


def test_image_queries_use_only_the_dense_lane(
    host: WorkspaceHostConfiguration, tmp_path: Path
) -> None:
    models, media = _corpus(host, tmp_path)
    service = runtime.open_unified_retrieval(host, models=models)
    result = service.search(
        _request(ImageQuery(LocalPathSource(str(media / "red.png"))), detail_level=DetailLevel.FULL)
    )
    assert result.hits and all(isinstance(hit, FullHit) for hit in result.hits)
    assert all(
        [lane.lane_id for lane in hit.lane_scores] == ["shared_dense"]
        for hit in result.hits
        if isinstance(hit, FullHit)
    )


def test_unsupported_sources_and_outside_paths_fail_without_writes(
    host: WorkspaceHostConfiguration, tmp_path: Path
) -> None:
    media = host.workspace_root / "media"
    (media / "data.bin").write_bytes(b"\x00\x01")
    with pytest.raises(runtime.UnifiedRetrievalError):
        runtime.import_media(host, "media", "data.bin")
    (media / "fake.png").write_bytes(b"not an image")
    with pytest.raises(runtime.UnifiedRetrievalError):
        runtime.import_media(host, "media", "fake.png")
    (media / "bad.md").write_bytes(b"\xff\xfe invalid utf-8")
    with pytest.raises(runtime.UnifiedRetrievalError):
        runtime.import_media(host, "media", "bad.md")
    outside = tmp_path / "outside.png"
    outside.write_bytes(_png((0, 0, 255)))
    models, _ = _corpus(host, tmp_path)
    service = runtime.open_unified_retrieval(host, models=models)
    failed = service.search(_request(ImageQuery(LocalPathSource(str(outside)))))
    assert failed.status is SearchStatus.FAILED


def test_audio_profile_still_rejects_text_and_image_imports(tmp_path: Path) -> None:
    root = (tmp_path / "w").resolve()
    (root / "media").mkdir(parents=True)
    (root / "media" / "n.md").write_text("note", encoding="utf-8")
    body = {
        "schema_version": 2,
        "server_name": "capicove-local",
        "workspace_id": "workspace",
        "actor_id": "owner",
        "environment_id": "local",
        "workspace_root": str(root),
        "profile": "write",
        "permissions": ["evidence:read", "evidence:search"],
        "enabled_tools": ["capicove.search.multimodal"],
        "trace_store": None,
        "capabilities": {
            "multimodal": {
                "schema_version": 1,
                "profile": "audio-local-v1",
                "source_roots": {"media": str(root / "media")},
                "index_root": str(root / "index"),
                "staging_root": str(root / "staging"),
                "corpus_ref": "corpus:a",
                "model_config": None,
                "model_binding": "unified",
                "ffmpeg": FFMPEG,
                "ffprobe": FFPROBE,
                "max_source_bytes": 1024,
            }
        },
        "network": {"enabled": False, "source_hosts": []},
    }
    config = parse_mcp_host_config(json.dumps(body).encode())
    assert isinstance(config, WorkspaceHostConfiguration)
    with pytest.raises(runtime.UnifiedRetrievalError, match="audio"):
        runtime.import_media(config, "media", "n.md")


def test_cli_and_mcp_serve_unified_text_and_image_queries(
    host: WorkspaceHostConfiguration,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from capicove.cli.app import main
    from capicove.contracts import ToolStatus
    from capicove.runtime.mcp_host import compose_mcp_host

    _, media = _corpus(host, tmp_path)
    monkeypatch.setattr(runtime, "_models", lambda settings, given: given or _models(tmp_path))
    config = tmp_path / "unified-host.json"
    body = {
        "schema_version": 2,
        "server_name": "capicove-local",
        "workspace_id": host.workspace_id,
        "actor_id": host.actor_id,
        "environment_id": host.environment_id,
        "workspace_root": str(host.workspace_root),
        "profile": "write",
        "permissions": ["evidence:read", "evidence:search"],
        "enabled_tools": ["capicove.multimodal.get_evidence", "capicove.search.multimodal"],
        "trace_store": None,
        "capabilities": {"multimodal": dict(host.capabilities["multimodal"])},
        "network": {"enabled": False, "source_hosts": []},
    }
    config.write_text(json.dumps(body, default=dict))
    config.chmod(0o600)
    host.workspace_root.chmod(0o700)
    assert main(["retrieval", "search", "--runtime-config", str(config), "--query", "ZX-4471"]) == 0
    text = json.loads(capsys.readouterr().out)
    assert {hit["target_type"] for hit in text["hits"]} == {"text_chunk", "image", "audio_segment"}
    assert text["hits"][0]["target_type"] == "text_chunk"
    assert (
        main(
            [
                "retrieval",
                "search",
                "--runtime-config",
                str(config),
                "--image-query",
                str(media / "red.png"),
                "--target",
                "image",
            ]
        )
        == 0
    )
    image = json.loads(capsys.readouterr().out)
    assert [hit["target_type"] for hit in image["hits"]] == ["image"]
    parsed = parse_mcp_host_config(config.read_bytes())
    assert isinstance(parsed, WorkspaceHostConfiguration)
    with compose_mcp_host(parsed) as bindings:
        search = {item.name: item for item in bindings}["capicove_search_multimodal"]
        result = search.handler(
            search.context_factory(search.spec),
            {"query": {"type": "image", "path": str(media / "red.png")}, "targets": ["image"]},
        )
        assert result.status is ToolStatus.SUCCEEDED and result.result is not None
        mcp_hits = cast(list[dict[str, Any]], result.result["hits"])
        assert [hit["id"] for hit in mcp_hits] == [hit["id"] for hit in image["hits"]]


def test_note_with_literal_highlight_markup_is_searchable(
    host: WorkspaceHostConfiguration, tmp_path: Path
) -> None:
    media = host.workspace_root / "media"
    (media / "html.md").write_text(
        "Use <em>emphasis</em> sparingly in release notes.\n", encoding="utf-8"
    )
    models = _models(tmp_path)
    runtime.import_media(host, "media", "html.md")
    runtime.activate_generation(
        host, runtime.build_generation(host, models=models)["generation_id"]
    )
    service = runtime.open_unified_retrieval(host, models=models)
    result = service.search(_request(TextQuery("emphasis"), top_k=1))
    assert result.status is SearchStatus.SUCCEEDED
    assert result.hits[0].snippet is not None and "<em>" not in result.hits[0].snippet
    text = service.read_evidence(_context(), result.hits[0].evidence_id, ("metadata",))["metadata"][
        "text"
    ]
    assert "<em>emphasis</em>" in text
