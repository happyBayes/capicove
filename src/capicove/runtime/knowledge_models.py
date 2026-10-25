"""Reuse the configured model factory for native knowledge retrieval."""

from dataclasses import asdict
from pathlib import Path


def encode_texts(path: Path, texts: list[str], role: str) -> tuple[str, list[list[float]]]:
    """Encode bounded batches in one explicit, verified embedding space."""
    from capicove.model import ModelFactory, load_configuration
    from capicove.retrieval import TextEmbeddingService
    from capicove.runtime.native_knowledge import digest

    with ModelFactory(load_configuration(path)) as factory:
        with factory.text_encoder("knowledge.embedding") as encoder:
            service = TextEmbeddingService(encoder, space=encoder.space)
            identity = asdict(encoder.space)
            for key, value in identity.items():
                if isinstance(value, (set, frozenset)):
                    identity[key] = sorted(value)
            space = digest(identity)
            vectors: list[list[float]] = []
            for start in range(0, len(texts), 16):
                batch = service.encode(
                    tuple(texts[start : start + 16]), space=encoder.space, role=role
                )
                vectors.extend(list(item.vector) for item in batch)
            return space, vectors
