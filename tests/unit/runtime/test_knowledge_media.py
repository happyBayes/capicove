"""Multimodal source boundaries and exact location preservation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from capicove.runtime.native_knowledge import KnowledgeBase
from tests.unit.runtime.test_native_knowledge import configured


@pytest.mark.parametrize("name", ["empty.txt", "unknown.html", "binary.txt"])
def test_unindexable_sources_never_activate(tmp_path: Path, name: str) -> None:
    config = configured(tmp_path)
    (config.source_roots["docs"] / name).write_bytes(b"\xff" if name == "binary.txt" else b"")
    with KnowledgeBase(config) as kb:
        with pytest.raises(ValueError):
            kb.update("docs", name)
        assert kb.status()["sources"] == 0


def test_page_and_time_citations_survive_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = configured(tmp_path)
    for name in ("scan.pdf", "speech.wav"):
        (config.source_roots["docs"] / name).write_bytes(b"authored-source")

    def prepare(config: Any, source: Any, store: Any) -> Any:
        locator = (
            {"kind": "pdf", "page_index": 7}
            if source.original.suffix == ".pdf"
            else {"kind": "audio", "start_seconds": 12.5, "end_seconds": 15.0}
        )
        return [{"text": "Exact original words.", **locator}], {"adapter": "authored"}

    monkeypatch.setattr("capicove.runtime.knowledge_prepare.prepare_source", prepare)
    with KnowledgeBase(config) as kb:
        for name in ("scan.pdf", "speech.wav"):
            kb.update("docs", name)
    with KnowledgeBase(config) as kb:
        hits = kb.search("original")["evidence"]
        assert len(hits) == 2
        for hit in hits:
            resolved = kb.evidence(hit["evidence_id"])
            assert resolved["locator"] == hit["locator"]
            assert resolved["original_path"].endswith(hit["path"])


def test_stale_peer_source_blocks_publication(tmp_path: Path) -> None:
    config = configured(tmp_path)
    first = config.source_roots["docs"] / "first.txt"
    second = config.source_roots["docs"] / "second.txt"
    first.write_text("First evidence")
    second.write_text("Second evidence")
    with KnowledgeBase(config) as kb:
        kb.update("docs", "first.txt")
        first.unlink()
        with pytest.raises(ValueError, match="source_changed"):
            kb.update("docs", "second.txt")
        assert kb.status()["sources"] == 1
        kb.forget("docs", "first.txt")
        kb.update("docs", "second.txt")
        assert kb.search("Second")["evidence"]


def test_no_selected_media_models_fail_explicitly(tmp_path: Path) -> None:
    config = configured(tmp_path)
    for name in ("image.png", "speech.wav"):
        (config.source_roots["docs"] / name).write_bytes(b"authored-source")
        with KnowledgeBase(config) as kb:
            with pytest.raises(ValueError, match="not_configured"):
                kb.update("docs", name)


def test_knowledge_gate_rejects_unknown_evidence_and_changed_revision(tmp_path: Path) -> None:
    from capicove.contracts.graph_build import GraphScope
    from capicove.runtime.knowledge_graph import KnowledgeGate

    config = configured(tmp_path)
    scope = GraphScope("test", "private")
    with KnowledgeBase(config) as kb:
        gate = KnowledgeGate(kb, [], 0, scope, ("allowed",))
        gate.require_access(("allowed",), scope)
        with pytest.raises(PermissionError):
            gate.require_access(("unknown",), scope)
        with pytest.raises(PermissionError):
            gate.require_scope(GraphScope("other", "private"))
        with gate.guard(0):
            assert gate.revision == 0
        with pytest.raises(PermissionError):
            with gate.guard(1):
                pass
        kb.forget("docs", "absent.txt")
        with pytest.raises(ValueError, match="knowledge_changed"):
            gate.require_access(("allowed",), scope)


def test_video_import_requires_visual_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from capicove.runtime.knowledge_prepare import prepare_source
    from capicove.runtime.media_handoff import StagedFile

    calls = []

    def video(*args: Any) -> Any:
        calls.append(True)
        return [{"text": "Frame", "kind": "video_frame", "start_seconds": 0.0}], {
            "coverage": "sampled_frames_and_audio"
        }

    monkeypatch.setattr("capicove.runtime.knowledge_prepare.prepare_video", video)
    source = StagedFile(tmp_path / "video.mp4", tmp_path / "staged.mp4", "a" * 64, 12)
    segments, lineage = prepare_source(configured(tmp_path), source, tmp_path / "store")
    assert calls and segments[0]["kind"] == "video_frame"
    assert lineage["coverage"] == "sampled_frames_and_audio"


def test_real_pdf_rendering_preserves_page_locator_without_model_sdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    from PIL import Image

    from capicove.contracts.graph_build import GraphScope
    from capicove.graph_build.representations import BlockSpan, TextRepresentation
    from capicove.runtime.knowledge_prepare import prepare_source
    from capicove.runtime.media_handoff import MediaIntake

    config = replace(configured(tmp_path), models=tmp_path / "models.json")
    path = config.source_roots["docs"] / "scan.pdf"
    Image.new("RGB", (200, 100), "white").save(path, "PDF")

    class Factory:
        def __init__(self, *args: Any) -> None:
            pass

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

    def image(
        self: Any, source: Any, *, document_ref: str, scope: GraphScope
    ) -> TextRepresentation:
        encoded = source.data if source.data is not None else Path(source.path).read_bytes()
        assert encoded.startswith(b"\x89PNG")
        return TextRepresentation(
            document_ref,
            scope,
            hashlib.sha256(encoded).hexdigest(),
            "Rendered source",
            (BlockSpan("ocr", 0, 15, 0, "paragraph", ()),),
            (),
        )

    monkeypatch.setattr("capicove.model.ModelFactory", Factory)
    monkeypatch.setattr("capicove.model.load_configuration", lambda path: None)
    monkeypatch.setattr(
        "capicove.graph_build.document_preparation.DocumentGraphConfiguration.prepare_image", image
    )
    intake = MediaIntake(
        read_roots=(path.parent,),
        staging_root=tmp_path / "stage",
        max_input_bytes=1000000,
        accepts=lambda p, head: True,
    )
    with intake.stage((path,)) as staged:
        segments, lineage = prepare_source(config, staged[0], tmp_path / "store")
    assert segments == [{"text": "Rendered source", "kind": "pdf", "page_index": 0}]
    assert lineage["pages"][0]["render_sha256"]


def test_native_assets_are_pinned_and_transcript_coordinates_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    import json

    from capicove.processing.audio import (
        AudioBackendLineage,
        AudioTranscript,
        AudioTranscriptionSegment,
    )
    from capicove.runtime.knowledge_prepare import prepare_source
    from capicove.runtime.media_handoff import MediaIntake

    config = configured(tmp_path)
    model = tmp_path / "model.bin"
    model.write_bytes(b"authored test asset")
    executable = Path("/usr/bin/true")
    raw = {
        "backend_version": "test",
        "files": {
            name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in (
                ("model", model),
                ("vad_model", model),
                ("whisper_cli", executable),
                ("whisper_vad_cli", executable),
                ("ffmpeg", executable),
                ("ffprobe", executable),
            )
        },
    }
    audio_config = tmp_path / "audio.json"
    audio_config.write_text(json.dumps(raw))
    audio_config.chmod(0o600)
    config = replace(config, audio=audio_config)
    config.storage_root.mkdir(mode=0o700)
    source = config.source_roots["docs"] / "speech.wav"
    source.write_bytes(b"authored input")

    def transcribe(path: Path, **arguments: Any) -> AudioTranscript:
        assert str(arguments["model"]).startswith("/proc/")
        assert str(arguments["whisper_cli"]).startswith("/proc/")
        assert Path(arguments["model"]).read_bytes() == b"authored test asset"
        return AudioTranscript(
            "Exact speech",
            "en",
            (AudioTranscriptionSegment(0, 1.0, 3.0, "Exact speech"),),
            (),
            AudioBackendLineage("whisper.cpp", "test", "model", "a" * 64, "cpu", "model-defined"),
            3.0,
        )

    monkeypatch.setattr("capicove.processing.audio.local_runtime.transcribe_local", transcribe)
    intake = MediaIntake(
        read_roots=(source.parent,),
        staging_root=tmp_path / "stage",
        max_input_bytes=10000,
        accepts=lambda p, head: True,
    )
    with intake.stage((source,)) as staged:
        segments, lineage = prepare_source(config, staged[0], tmp_path / "store")
        assert segments[0]["start_seconds"] == 1.0
        assert lineage["receipt"]["backend"] == "whisper.cpp"
        model.write_bytes(b"changed")
        with pytest.raises(ValueError):
            prepare_source(config, staged[0], tmp_path / "store")


def test_doctor_and_configuration_rejections(tmp_path: Path) -> None:
    import json

    from capicove.runtime.knowledge_config import KnowledgeConfiguration, read_configuration
    from capicove.runtime.knowledge_doctor import doctor

    config = configured(tmp_path)
    assert doctor(config)["generation"] == "disabled"
    path = tmp_path / "config.json"
    path.write_text('{"schema_version":1,"schema_version":2}')
    path.chmod(0o600)
    with pytest.raises(ValueError, match="duplicate"):
        read_configuration(path)
    change: dict[str, Any]
    for change in (
        {"source_roots": {}},
        {"workspace_id": "bad space"},
        {"storage_root": str(config.source_roots["docs"] / "store")},
        {"source_roots": {"bad.alias": str(tmp_path)}},
        {"graph": "false"},
    ):
        with pytest.raises(ValueError):
            KnowledgeConfiguration.from_mapping({**config.to_mapping(), **change})
    path.write_text(json.dumps(config.to_mapping()))
    path.chmod(0o666)
    with pytest.raises(ValueError, match="unsafe"):
        read_configuration(path)


@pytest.mark.parametrize(
    "documents",
    [(), ("knowledge.ocr",), ("knowledge.parser",), ("knowledge.ocr", "knowledge.parser")],
)
def test_doctor_document_summary_matches_selected_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, documents: tuple[str, ...]
) -> None:
    from contextlib import nullcontext
    from dataclasses import replace
    from types import SimpleNamespace

    from capicove.runtime.knowledge_doctor import doctor

    config = replace(configured(tmp_path), models=tmp_path / "models.json")

    def profile(binding: str, capability: Any) -> object:
        if binding != "knowledge.embedding" and binding not in documents:
            raise ValueError("not_configured")
        return object()

    monkeypatch.setattr("capicove.model.load_configuration", lambda path: object())
    monkeypatch.setattr(
        "capicove.model.ModelFactory",
        lambda config: nullcontext(SimpleNamespace(profile=profile)),
    )
    result = doctor(config)
    assert result["documents"] == ("configured_runtime_unprobed" if documents else "not_configured")
    for binding in ("knowledge.ocr", "knowledge.parser"):
        assert result[binding] == (
            "configured_runtime_unprobed" if binding in documents else "not_configured"
        )
