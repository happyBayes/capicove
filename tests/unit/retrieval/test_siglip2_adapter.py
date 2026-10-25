"""Tests for the lazy local-only SigLIP2 retrieval provider."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from capicove.retrieval._adapters.siglip2 import HuggingFaceSiglip2Provider
from tests.unit.processing.image.model_installations import verified_model_installation


class FakeTensor:
    """Expose the tensor surface consumed by the provider."""

    def __init__(self, values: object) -> None:
        self.values = np.asarray(values, dtype=np.float64)

    def to(self, device: str) -> FakeTensor:
        assert device == "mps"
        return self

    def detach(self) -> FakeTensor:
        return self

    def float(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class FakeBatch(dict[str, FakeTensor]):
    """Mimic Transformers BatchFeature device placement."""

    def to(self, device: str) -> FakeBatch:
        assert device == "mps"
        return self


class InferenceMode:
    """Provide a no-op inference context."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


def test_provider_loads_verified_model_locally_and_runs_all_operations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    class Processor:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> Processor:
            loads.append(("processor", args, kwargs))
            return cls()

        def __call__(self, **kwargs: object) -> FakeBatch:
            if "images" in kwargs and "text" in kwargs:
                assert kwargs["padding"] == "max_length"
            return FakeBatch(pixel_values=FakeTensor([[1.0]]))

    class Model:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> Model:
            loads.append(("model", args, kwargs))
            return cls()

        def eval(self) -> Model:
            return self

        def to(self, device: str) -> Model:
            assert device == "mps"
            return self

        def __call__(self, **kwargs: object) -> object:
            del kwargs
            return SimpleNamespace(logits_per_image=FakeTensor([[2.0, -2.0]]))

        def get_image_features(self, **kwargs: object) -> object:
            del kwargs
            return SimpleNamespace(pooler_output=FakeTensor([[3.0, 4.0]]))

        def get_text_features(self, **kwargs: object) -> object:
            del kwargs
            return SimpleNamespace(pooler_output=FakeTensor([[0.0, 2.0], [2.0, 0.0]]))

    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        inference_mode=InferenceMode,
    )
    transformers = SimpleNamespace(
        AutoProcessor=Processor,
        AutoModel=Model,
        __version__="5.0",
    )

    def import_module(name: str) -> object:
        return transformers if name == "transformers" else torch

    monkeypatch.setattr(
        "capicove.retrieval._adapters.siglip2.importlib.import_module",
        import_module,
    )
    provider = HuggingFaceSiglip2Provider(
        verified_model_installation(
            tmp_path,
            adapter="huggingface-siglip2-retrieval",
            preprocess_revision="transformers-siglip2-processor-v1",
        ),
        device="auto",
    )

    matches = provider.rank(Image.new("RGB", (2, 2)), ("cat", "dog"))
    image_vectors = provider.embed_images((Image.new("RGB", (2, 2)),))
    text_vectors = provider.embed_texts(("cat", "dog"))

    assert provider.device == "mps"
    assert loads[0][2] == {"local_files_only": True, "trust_remote_code": False}
    assert loads[1][2] == {"local_files_only": True, "trust_remote_code": False}
    assert matches[0].logit == 2.0
    assert matches[0].match_probability == pytest.approx(1.0 / (1.0 + math.exp(-2.0)))
    assert image_vectors == ((0.6, 0.8),)
    assert text_vectors == ((0.0, 1.0), (1.0, 0.0))


def test_provider_verifies_model_bytes_before_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installation = verified_model_installation(
        tmp_path,
        adapter="huggingface-siglip2-retrieval",
        preprocess_revision="transformers-siglip2-processor-v1",
    )
    imported = False

    def import_module(name: str) -> object:
        nonlocal imported
        del name
        imported = True
        return object()

    monkeypatch.setattr(
        "capicove.retrieval._adapters.siglip2.importlib.import_module",
        import_module,
    )
    provider = HuggingFaceSiglip2Provider(installation)
    (tmp_path / "config.json").write_text('{"tampered":true}', encoding="utf-8")

    with pytest.raises(ValueError, match="model contents"):
        provider.embed_texts(("cat",))

    assert imported is False
