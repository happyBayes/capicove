"""Tests for lazy local-only Hugging Face image scorers."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from capicove.model.manifest import ImageModelSpec
from capicove.model.receipt import (
    build_receipt,
    verify_model_installation,
    write_receipt,
)
from capicove.processing.image._adapters.huggingface import (
    HuggingFaceAestheticsScorer,
    HuggingFaceBlipImageTextMatchingScorer,
    HuggingFaceClipImageTextSimilarityScorer,
    HuggingFaceImageClassificationScorer,
    HuggingFaceImagePairScorer,
)
from tests.unit.processing.image.model_installations import verified_model_installation


class ArrayTensor:
    """Expose the tensor conversion surface consumed by local adapters."""

    def __init__(self, values: object) -> None:
        self._values = values

    def detach(self) -> ArrayTensor:
        return self

    def cpu(self) -> ArrayTensor:
        return self

    def numpy(self) -> object:
        return self._values


def test_huggingface_adapter_reverifies_model_bytes_before_lazy_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ImageModelSpec(
        key="aesthetics-test",
        repository="test-owner/aesthetics-test",
        revision="a" * 40,
        license="apache-2.0",
        license_url="https://example.test/license",
        capability="image_aesthetics_filter",
        adapter="huggingface-aesthetics",
        architecture="TestArchitecture",
        preprocess_revision="test-preprocess-v1",
        default=False,
        restricted=False,
    )
    model_file = tmp_path / "config.json"
    model_file.write_text("{}", encoding="utf-8")
    receipt = build_receipt(model, tmp_path, manifest_sha256="b" * 64, hf_version="hf-test")
    write_receipt(tmp_path, receipt)
    installation = verify_model_installation(
        tmp_path,
        model,
        manifest_sha256="b" * 64,
        expected_model_manifest_sha256=str(receipt["model_manifest_sha256"]),
    )
    imported = False

    def import_module(name: str) -> object:
        nonlocal imported
        del name
        imported = True
        return object()

    monkeypatch.setattr(
        "capicove.processing.image._adapters.huggingface.importlib.import_module",
        import_module,
    )
    scorer = HuggingFaceAestheticsScorer(installation)
    model_file.write_text('{"tampered":true}', encoding="utf-8")

    with pytest.raises(ValueError, match="model contents"):
        scorer.score((Image.new("RGB", (2, 2)),))

    assert imported is False


def test_classification_scorer_resolves_target_label_lazily(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def load_local(path: str, **kwargs: object) -> object:
        assert kwargs == {"local_files_only": True, "trust_remote_code": False}
        return object()

    def pipeline(task: str, **kwargs: object) -> Callable[..., object]:
        calls.append((task, kwargs))
        return lambda images: [
            [{"label": "safe", "score": 0.9}, {"label": "nsfw", "score": 0.1}] for _ in images
        ]

    monkeypatch.setattr(
        "capicove.processing.image._adapters.huggingface.importlib.import_module",
        lambda name: SimpleNamespace(
            pipeline=pipeline,
            AutoModelForImageClassification=SimpleNamespace(from_pretrained=load_local),
            AutoImageProcessor=SimpleNamespace(from_pretrained=load_local),
        ),
    )
    installation = verified_model_installation(
        tmp_path,
        adapter="huggingface-image-classification",
        preprocess_revision="transformers-image-classification-v1",
    )
    scorer = HuggingFaceImageClassificationScorer(
        installation,
        target_label="nsfw",
        device=0,
    )

    assert calls == []
    assert scorer.score((Image.new("RGB", (2, 2)),)) == (0.1,)
    assert "model_kwargs" not in calls[0][1]
    assert not isinstance(calls[0][1]["model"], str)
    assert "image_processor" in calls[0][1]
    assert calls[0][1]["trust_remote_code"] is False
    assert calls[0][1]["device"] == 0
    assert scorer.model_revision == "a" * 40
    assert scorer.model_manifest_sha256 == installation.model_manifest_sha256


def test_classification_scorer_does_not_cache_pipeline_when_post_load_verification_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def load_local(path: str, **kwargs: object) -> object:
        assert kwargs == {"local_files_only": True, "trust_remote_code": False}
        return object()

    factory_calls = 0
    model_file = tmp_path / "config.json"

    def pipeline(task: str, **kwargs: object) -> Callable[..., object]:
        nonlocal factory_calls
        del task, kwargs
        factory_calls += 1
        if factory_calls == 1:
            model_file.write_text('{"changed":true}', encoding="utf-8")
        return lambda images: [[{"label": "nsfw", "score": 0.9}] for _ in images]

    monkeypatch.setattr(
        "capicove.processing.image._adapters.huggingface.importlib.import_module",
        lambda name: SimpleNamespace(
            pipeline=pipeline,
            AutoModelForImageClassification=SimpleNamespace(from_pretrained=load_local),
            AutoImageProcessor=SimpleNamespace(from_pretrained=load_local),
        ),
    )
    installation = verified_model_installation(
        tmp_path,
        adapter="huggingface-image-classification",
        preprocess_revision="transformers-image-classification-v1",
    )
    scorer = HuggingFaceImageClassificationScorer(installation, target_label="nsfw")

    with pytest.raises(ValueError, match="model contents"):
        scorer.score((Image.new("RGB", (2, 2)),))

    model_file.write_text("{}", encoding="utf-8")
    assert scorer.score((Image.new("RGB", (2, 2)),)) == (0.9,)
    assert factory_calls == 2


def test_blip_image_text_scorer_returns_itm_match_probability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Processor:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> Processor:
            return cls()

        def __call__(self, **kwargs: object) -> dict[str, object]:
            assert kwargs["text"] == ["a test image"]
            return {"pixel_values": object(), "input_ids": object()}

    class Model:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> Model:
            return cls()

        def eval(self) -> Model:
            return self

        def __call__(self, **kwargs: object) -> object:
            assert kwargs["use_itm_head"] is True
            return SimpleNamespace(itm_score=ArrayTensor([[0.0, math.log(3.0)]]))

    class NoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: object) -> None:
            return None

    def import_module(name: str) -> object:
        if name == "transformers":
            return SimpleNamespace(
                BlipForImageTextRetrieval=Model,
                BlipProcessor=Processor,
                __version__="5.0",
            )
        return SimpleNamespace(no_grad=NoGrad)

    monkeypatch.setattr(
        "capicove.processing.image._adapters.huggingface.importlib.import_module",
        import_module,
    )
    scorer = HuggingFaceBlipImageTextMatchingScorer(
        verified_model_installation(
            tmp_path,
            adapter="huggingface-blip-itm",
            preprocess_revision="transformers-blip-itm-processor-v1",
        ),
    )

    scores = scorer.score((Image.new("RGB", (2, 2)),), "a test image")

    assert scores == (0.75,)


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize(
    "values, expected",
    [
        ([[1.0, 0.0], [0.0, 1.0]], 0.0),
        ([[1.0, 0.0], [1.0, 0.0]], 1.0),
        ([[0.0, 0.0], [1.0, 0.0]], None),
        ([[float("nan"), 0.0], [1.0, 0.0]], None),
    ],
)
def test_image_pair_scorer_computes_cosine_similarity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    structured: bool,
    values: list[list[float]],
    expected: float | None,
) -> None:
    from contextlib import nullcontext

    class Processor:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> Processor:
            assert kwargs == {"local_files_only": True, "trust_remote_code": False}
            return cls()

        def __call__(self, **kwargs: object) -> dict[str, object]:
            assert len(kwargs["images"]) == 2  # type: ignore[arg-type]
            assert "text" not in kwargs
            return {"pixel_values": object()}

    class Model:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> Model:
            assert kwargs == {"local_files_only": True, "trust_remote_code": False}
            return cls()

        def eval(self) -> Model:
            return self

        def get_image_features(self, **kwargs: object) -> object:
            tensor = ArrayTensor(values)
            return SimpleNamespace(pooler_output=tensor) if structured else tensor

    def import_module(name: str) -> object:
        if name == "transformers":
            return SimpleNamespace(CLIPProcessor=Processor, CLIPModel=Model)
        return SimpleNamespace(no_grad=nullcontext)

    monkeypatch.setattr(
        "capicove.processing.image._adapters.huggingface.importlib.import_module",
        import_module,
    )
    scorer = HuggingFaceImagePairScorer(
        verified_model_installation(
            tmp_path,
            adapter="huggingface-clip-similarity",
            preprocess_revision="transformers-clip-processor-v1",
        )
    )
    if expected is None:
        with pytest.raises(ValueError):
            scorer.score(Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2)))
    else:
        assert scorer.score(Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))) == expected


def test_aesthetics_scorer_normalizes_single_regression_logit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Processor:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> Processor:
            return cls()

        def __call__(self, **kwargs: object) -> dict[str, object]:
            return {"pixel_values": object()}

    class Model:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> Model:
            return cls()

        def eval(self) -> Model:
            return self

        def __call__(self, **kwargs: object) -> object:
            return SimpleNamespace(logits=ArrayTensor([[8.0]]))

    class NoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: object) -> None:
            return None

    def import_module(name: str) -> object:
        if name == "transformers":
            return SimpleNamespace(
                AutoImageProcessor=Processor,
                AutoModelForImageClassification=Model,
                __version__="5.0",
            )
        return SimpleNamespace(no_grad=NoGrad)

    monkeypatch.setattr(
        "capicove.processing.image._adapters.huggingface.importlib.import_module",
        import_module,
    )
    scorer = HuggingFaceAestheticsScorer(
        verified_model_installation(
            tmp_path,
            adapter="huggingface-aesthetics",
            preprocess_revision="transformers-auto-image-processor-v1",
        ),
        score_scale=10.0,
    )

    assert scorer.score((Image.new("RGB", (2, 2)),)) == (0.8,)


def test_aesthetics_scorer_returns_named_class_probability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Processor:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> Processor:
            return cls()

        def __call__(self, **kwargs: object) -> dict[str, object]:
            return {"pixel_values": object()}

    class Model:
        config = SimpleNamespace(id2label={0: "not_aesthetic", 1: "aesthetic"})

        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> Model:
            return cls()

        def eval(self) -> Model:
            return self

        def __call__(self, **kwargs: object) -> object:
            return SimpleNamespace(logits=ArrayTensor([[0.0, math.log(3.0)]]))

    class NoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: object) -> None:
            return None

    def import_module(name: str) -> object:
        if name == "transformers":
            return SimpleNamespace(
                AutoImageProcessor=Processor,
                AutoModelForImageClassification=Model,
                __version__="5.0",
            )
        return SimpleNamespace(no_grad=NoGrad)

    monkeypatch.setattr(
        "capicove.processing.image._adapters.huggingface.importlib.import_module",
        import_module,
    )
    scorer = HuggingFaceAestheticsScorer(
        verified_model_installation(
            tmp_path,
            adapter="huggingface-aesthetics",
            preprocess_revision="transformers-auto-image-processor-v1",
        ),
        target_label="aesthetic",
    )

    assert scorer.score((Image.new("RGB", (2, 2)),)) == pytest.approx((0.75,))


def test_clip_image_text_scorer_returns_cosine_similarity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Processor:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> Processor:
            return cls()

        def __call__(self, **kwargs: object) -> dict[str, object]:
            return {}

    class Model:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> Model:
            return cls()

        def eval(self) -> Model:
            return self

        def __call__(self, **kwargs: object) -> object:
            return SimpleNamespace(
                image_embeds=ArrayTensor([[1.0, 0.0]]),
                text_embeds=ArrayTensor([[0.6, 0.8]]),
            )

    class NoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: object) -> None:
            return None

    def import_module(name: str) -> object:
        if name == "transformers":
            return SimpleNamespace(CLIPModel=Model, CLIPProcessor=Processor, __version__="5.0")
        return SimpleNamespace(no_grad=NoGrad)

    monkeypatch.setattr(
        "capicove.processing.image._adapters.huggingface.importlib.import_module",
        import_module,
    )
    scorer = HuggingFaceClipImageTextSimilarityScorer(
        verified_model_installation(
            tmp_path,
            adapter="huggingface-clip-similarity",
            preprocess_revision="transformers-clip-processor-v1",
        ),
    )

    assert scorer.score((Image.new("RGB", (2, 2)),), "an object") == (0.6,)


def test_clip_does_not_cache_partially_verified_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import nullcontext

    from capicove.model.receipt import VerifiedModelInstallation

    loads = 0
    verification_pending = False
    installation = verified_model_installation(
        tmp_path, adapter="huggingface-clip-similarity", preprocess_revision="test-v1"
    )
    original_verify = VerifiedModelInstallation.verify

    def verify(value: VerifiedModelInstallation) -> None:
        nonlocal verification_pending
        original_verify(value)
        if verification_pending:
            verification_pending = False
            raise ValueError("model contents changed during loading")

    class Model:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> Model:
            nonlocal loads, verification_pending
            loads += 1
            verification_pending = loads == 1
            return cls()

        def eval(self) -> Model:
            return self

        def get_image_features(self, **kwargs: object) -> ArrayTensor:
            return ArrayTensor([[1.0, 0.0], [1.0, 0.0]])

    monkeypatch.setattr(VerifiedModelInstallation, "verify", verify)
    modules = SimpleNamespace(
        CLIPModel=Model,
        CLIPProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: lambda **inputs: {}),
    )
    monkeypatch.setattr(
        "capicove.processing.image._adapters.huggingface.importlib.import_module",
        lambda name: modules if name == "transformers" else SimpleNamespace(no_grad=nullcontext),
    )
    scorer = HuggingFaceImagePairScorer(installation)
    image = Image.new("RGB", (2, 2))
    with pytest.raises(ValueError, match="model contents"):
        scorer.score(image, image)
    assert scorer.score(image, image) == 1.0
    assert loads == 2
