"""Composition regression tests; the native ES test separately verifies its boundary."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest

from capicove.contracts import EmbeddingSpaceRef, ProcessingFingerprint
from capicove.processing.video.models import VideoSource
from capicove.processing.video.retrieval.models import VideoRetrievalPreparationRequest
from capicove.retrieval.generations import TextIndexGenerationManifest, TextIndexValidationReport
from capicove.retrieval.ports import TextRetrieverBackend
from capicove.runtime.video_search import LocalVideoSearchRuntime
from tests.integration.processing.video.test_retrieval_preparation import authored_video
from tests.unit.retrieval.test_video_models import visual_space
from tests.unit.tools.test_search import context


__all__ = ["authored_video"]


class Encoder:
    def encode_preview_batch(self, previews: tuple[bytes, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0) for _ in previews)

    def encode(
        self, query: object, *, embedding_space: EmbeddingSpaceRef, deadline: datetime
    ) -> tuple[float, ...]:
        return (1.0, 0.0)


def test_publication_failure_retains_previous_snapshot(
    tmp_path: Path, authored_video: bytes
) -> None:
    backend = Mock()
    backend.provider_id = "test.lexical"
    backend.generation_lifecycle.mapping_sha256.return_value = "c" * 64
    backend.lexical_retriever.retrieve.return_value = ()

    def validate(manifest: TextIndexGenerationManifest) -> TextIndexValidationReport:
        return TextIndexValidationReport(
            manifest.generation_id,
            manifest.manifest_sha256,
            manifest.generation_id,
            manifest.mapping_sha256,
            manifest.expected_unit_count,
            manifest.expected_unit_count,
            (),
        )

    backend.generation_lifecycle.validate.side_effect = validate
    fp = ProcessingFingerprint("b" * 64)
    request = VideoRetrievalPreparationRequest(
        VideoSource(data=authored_video),
        "artifact:authored",
        "evidence:authored",
        "representation:asr",
        "run:authored",
        "video-local-exact",
        fp,
        fp,
        fp,
        datetime.now(UTC) + timedelta(seconds=30),
    )
    runtime = LocalVideoSearchRuntime(
        workspace_root=tmp_path,
        workspace_ref="test",
        corpus_refs=("test",),
        embedding_space=visual_space(),
        encoder=Encoder(),
        text_backend=cast(TextRetrieverBackend, backend),
        text_index_format="test-v1",
    )
    arguments = {"query": {"type": "text", "text": "red"}, "targets": ["video_segment"]}
    with runtime:
        first = runtime.build(request)
        assert set(first.warning_codes) == {"asr_not_configured", "ocr_not_configured"}
        original = runtime.search_handler(context(), arguments)
        assert original.result and original.result["hits"]
        backend.index_writer.add.side_effect = OSError("text backend unavailable")
        with pytest.raises(OSError):
            runtime.build(request)
        pin = runtime.components.generation_sets.resolve_active(
            workspace_ref="test", profile_ref="video-local-exact"
        )
        assert pin.set_id == first.generation_set_id
        after = runtime.search_handler(context(), arguments)
        assert after.result and after.result["hits"] == original.result["hits"]
        backend.lexical_retriever.retrieve.side_effect = OSError("offline")
        degraded = runtime.search_handler(context(), arguments)
        assert degraded.result and degraded.result["degraded"] is True
        assert degraded.result["hits"]
        with pytest.raises(ValueError, match="video-local-exact"):
            runtime.build(replace(request, profile_ref="other"))
    runtime.close()
    with pytest.raises(RuntimeError, match="closed"):
        runtime.build(request)


def test_runtime_rejects_relative_workspace() -> None:
    with pytest.raises(ValueError, match="absolute"):
        LocalVideoSearchRuntime(
            workspace_root=Path("relative"),
            workspace_ref="test",
            corpus_refs=("test",),
            embedding_space=visual_space(),
            encoder=Encoder(),
            text_backend=Mock(),
            text_index_format="test-v1",
        )
