"""Graph consumption of the validated public retrieval service."""

import pytest

from capicove.graph_build._adapters.retrieval_embedding import RetrievalBatchTextEmbeddingProvider
from capicove.graph_build.ports import BatchTextEmbeddingProvider
from capicove.retrieval import TextEmbeddingService
from tests.unit.retrieval.test_text_embedding import Encoder, text_space


def test_bridge_preserves_input_order_roles_and_public_service_validation() -> None:
    encoder = Encoder()
    bridge: BatchTextEmbeddingProvider = RetrievalBatchTextEmbeddingProvider(
        TextEmbeddingService(encoder, space=text_space())
    )
    texts = ("second", "first", "second")
    result = bridge.embed(texts, space=text_space(), role="document")
    assert tuple(item.vector for item in result) == ((1.0, 0.0), (0.0, 1.0), (1.0, 0.0))
    assert encoder.calls == [(texts, text_space(), "document")]
    with pytest.raises(ValueError):
        bridge.embed(texts, space=text_space(), role="target")
    assert len(encoder.calls) == 1
