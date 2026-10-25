"""Structural contracts for reverse-image retrieval ports."""

from __future__ import annotations

from datetime import datetime

from PIL import Image

from capicove.contracts.retrieval import EmbeddingSpaceRef, ExecutionProfile
from capicove.retrieval.search_models import QueryVector, RetrievalLane
from capicove.retrieval.search_ports import QueryEncoder, QueryEncoderResolver


class FakeQueryEncoder:
    def __init__(self, space: EmbeddingSpaceRef) -> None:
        self.embedding_space = space
        self.bundle_manifest_sha256 = "a" * 64

    def encode(
        self,
        image: Image.Image,
        space: EmbeddingSpaceRef,
        deadline: datetime,
    ) -> QueryVector:
        raise NotImplementedError


class FakeResolver:
    def resolve(
        self,
        lane: RetrievalLane,
        space: EmbeddingSpaceRef,
        target_bundle_manifest_sha256: str,
        execution_profile: ExecutionProfile,
    ) -> QueryEncoder:
        return FakeQueryEncoder(space)


def test_query_ports_are_runtime_checkable() -> None:
    assert isinstance(FakeResolver(), QueryEncoderResolver)
