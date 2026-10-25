"""Protocol tests for provider-neutral video retrieval ports."""

from __future__ import annotations

from capicove.retrieval.video_ports import (
    VideoGenerationSetCatalog,
    VideoUnitCatalogReader,
    VideoUnitCatalogWriter,
    VisualCandidateRetriever,
    VisualIndexGenerationLifecycle,
    VisualIndexWriter,
    VisualRetrieverBackend,
)


class _Backend:
    provider_id = "capicove.retriever.fake"
    index_writer = None
    candidate_retriever = None
    generation_lifecycle = None


def test_video_ports_are_runtime_checkable() -> None:
    assert isinstance(_Backend(), VisualRetrieverBackend)
    for protocol in (
        VisualIndexWriter,
        VisualCandidateRetriever,
        VisualIndexGenerationLifecycle,
        VideoUnitCatalogReader,
        VideoUnitCatalogWriter,
        VideoGenerationSetCatalog,
    ):
        assert hasattr(protocol, "__protocol_attrs__") or getattr(
            protocol, "_is_runtime_protocol", False
        )
