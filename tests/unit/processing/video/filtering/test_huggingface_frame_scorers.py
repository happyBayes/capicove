from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, cast

import numpy as np
import pytest

from capicove.processing.video import AcceptedVideoSource, VideoSource
from capicove.processing.video.filtering._adapters.huggingface_frame_scorers import (
    HuggingFaceClassificationEvaluator,
    HuggingFaceLocalModelResolver,
    ResolvedHuggingFaceModel,
)
from capicove.processing.video.filtering.video_filter_contracts import (
    AestheticsCriterion,
    FrameSamplingConfig,
    LocalModelReference,
    NsfwCriterion,
    Reduction,
    VideoFilterItem,
    VideoFilterMetadata,
    WatermarkCriterion,
)
from capicove.processing.video.filtering.video_filter_ports import (
    BackendUnavailableFilterError,
    ModelIncompatibleFilterError,
    SampledFrame,
)


class FakeFactory:
    calls: ClassVar[list[tuple[str, dict[str, object]]]] = []
    product: ClassVar[object] = object()

    @classmethod
    def from_pretrained(cls, reference: str, **kwargs: object) -> object:
        cls.calls.append((reference, kwargs))
        return cls.product


def _deadline() -> datetime:
    return datetime.now(UTC) + timedelta(minutes=1)


def _frames() -> tuple[SampledFrame, ...]:
    return (
        SampledFrame(0, 0.0, 4, 4, np.zeros((4, 4, 3), dtype=np.uint8)),
        SampledFrame(1, 1.0, 4, 4, np.ones((4, 4, 3), dtype=np.uint8)),
    )


def _source() -> AcceptedVideoSource:
    return AcceptedVideoSource("source-1", "a" * 64, "video/mp4", data=b"video")


def _item() -> VideoFilterItem:
    return VideoFilterItem("item-1", VideoSource(data=b"video"))


def _metadata() -> VideoFilterMetadata:
    return VideoFilterMetadata(1.0, 4, 4, 4, 4, backend="fake")


def test_resolver_forces_local_only_and_disables_remote_code(tmp_path: Path) -> None:
    FakeFactory.calls = []
    model = SimpleNamespace(config=SimpleNamespace(id2label={0: "safe", 1: "nsfw"}))
    processor_factory = cast(
        type[FakeFactory],
        type("ProcessorFactory", (FakeFactory,), {"product": object(), "calls": []}),
    )
    model_factory = cast(
        type[FakeFactory],
        type("ModelFactory", (FakeFactory,), {"product": model, "calls": []}),
    )
    transformers = SimpleNamespace(
        AutoImageProcessor=processor_factory,
        AutoModelForImageClassification=model_factory,
    )
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    reference = LocalModelReference(str(tmp_path), revision="local-v1")

    resolved = HuggingFaceLocalModelResolver(transformers_module=transformers).resolve(
        reference,
        model_kind="classification",
    )

    for _, kwargs in processor_factory.calls + model_factory.calls:
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is False
        assert kwargs["revision"] == "local-v1"
    assert resolved.id2label == {0: "safe", 1: "nsfw"}
    assert len(resolved.manifest_hash) == 64


def test_local_manifest_hash_changes_with_weight_content(tmp_path: Path) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"first")
    transformers = SimpleNamespace(
        AutoImageProcessor=FakeFactory,
        AutoModelForImageClassification=FakeFactory,
    )
    resolver = HuggingFaceLocalModelResolver(transformers_module=transformers)
    reference = LocalModelReference(str(tmp_path))

    first = resolver.resolve(reference, model_kind="classification").manifest_hash
    weights.write_bytes(b"second")
    second = resolver.resolve(reference, model_kind="classification").manifest_hash

    assert first != second


def test_model_id_requires_an_explicit_authorized_cache_directory() -> None:
    transformers = SimpleNamespace(
        AutoImageProcessor=FakeFactory,
        AutoModelForImageClassification=FakeFactory,
    )

    with pytest.raises(BackendUnavailableFilterError, match="cache_directory"):
        HuggingFaceLocalModelResolver(transformers_module=transformers).resolve(
            LocalModelReference("organization/model-id"),
            model_kind="classification",
        )


def _resolved(id2label: dict[int, str]) -> ResolvedHuggingFaceModel:
    return ResolvedHuggingFaceModel(
        reference="local/model",
        revision="v1",
        manifest_hash="b" * 64,
        model=object(),
        processor=object(),
        id2label=id2label,
    )


class FakeResolver:
    def __init__(self, resolved: ResolvedHuggingFaceModel) -> None:
        self.resolved = resolved

    def resolve(
        self,
        reference: LocalModelReference,
        *,
        model_kind: str,
    ) -> ResolvedHuggingFaceModel:
        del reference
        assert model_kind == "classification"
        return self.resolved


def test_aesthetics_reduces_scalar_frame_scores() -> None:
    evaluator = HuggingFaceClassificationEvaluator(
        FakeResolver(_resolved({0: "aesthetic"})),
        inference=lambda resolved, frames: [[0.2], [0.8]],
    )
    criterion = AestheticsCriterion(
        "aesthetics",
        LocalModelReference("local/model"),
        FrameSamplingConfig(),
        min_score=0.5,
        reduction=Reduction.AVG,
    )

    result = evaluator.evaluate(_item(), _source(), _metadata(), _frames(), criterion, _deadline())

    assert result.score == pytest.approx(0.5)
    assert result.passed is True
    assert [metric.raw_value for metric in result.frame_metrics] == [0.2, 0.8]


def test_nsfw_resolves_target_label_without_assuming_index_one() -> None:
    evaluator = HuggingFaceClassificationEvaluator(
        FakeResolver(_resolved({0: "nsfw", 1: "safe"})),
        inference=lambda resolved, frames: [[3.0, 1.0], [1.0, 3.0]],
    )
    criterion = NsfwCriterion(
        "nsfw",
        LocalModelReference("local/model"),
        FrameSamplingConfig(),
        target_label="nsfw",
        max_score=0.9,
        reduction=Reduction.MAX,
    )

    result = evaluator.evaluate(_item(), _source(), _metadata(), _frames(), criterion, _deadline())

    assert result.score == pytest.approx(0.880797, rel=1e-5)
    assert result.passed is True
    assert result.details["target_index"] == 0


def test_watermark_rejects_ambiguous_implicit_target_labels() -> None:
    evaluator = HuggingFaceClassificationEvaluator(
        FakeResolver(_resolved({0: "watermark", 1: "has_watermark", 2: "clean"})),
        inference=lambda resolved, frames: [[1.0, 2.0, 0.0], [1.0, 2.0, 0.0]],
    )
    criterion = WatermarkCriterion(
        "watermark",
        LocalModelReference("local/model"),
        FrameSamplingConfig(),
    )

    with pytest.raises(ModelIncompatibleFilterError, match="target label"):
        evaluator.evaluate(_item(), _source(), _metadata(), _frames(), criterion, _deadline())


@pytest.mark.parametrize(
    "labels", [{0: "aesthetic", 1: "not_aesthetic"}, {1: "aesthetic", 0: "not_aesthetic"}]
)
def test_aesthetics_accepts_named_classifier(labels: dict[int, str]) -> None:
    evaluator = HuggingFaceClassificationEvaluator(
        FakeResolver(_resolved(labels)),
        inference=lambda resolved, frames: [[0.0, 0.0], [0.0, 0.0]],
    )
    criterion = AestheticsCriterion(
        "aesthetics", LocalModelReference("local/model"), FrameSamplingConfig()
    )
    result = evaluator.evaluate(_item(), _source(), _metadata(), _frames(), criterion, _deadline())
    assert result.score == pytest.approx(0.5)
    assert result.details["target_index"] == next(
        i for i, label in labels.items() if label == "aesthetic"
    )


@pytest.mark.parametrize(
    "labels, logits",
    [
        ({0: "LABEL_0", 1: "LABEL_1"}, [[0.0, 1.0], [0.0, 1.0]]),
        ({0: "aesthetic", 1: "AESTHETIC"}, [[0.0, 1.0], [0.0, 1.0]]),
        ({0: "aesthetic"}, [[float("nan")], [1.0]]),
        ({0: "aesthetic"}, [[float("inf")], [1.0]]),
    ],
)
def test_aesthetics_rejects_ambiguous_or_nonfinite_outputs(
    labels: dict[int, str], logits: list[list[float]]
) -> None:
    evaluator = HuggingFaceClassificationEvaluator(
        FakeResolver(_resolved(labels)), inference=lambda resolved, frames: logits
    )
    criterion = AestheticsCriterion(
        "aesthetics", LocalModelReference("local/model"), FrameSamplingConfig()
    )
    with pytest.raises(ModelIncompatibleFilterError):
        evaluator.evaluate(_item(), _source(), _metadata(), _frames(), criterion, _deadline())


def test_classification_processor_receives_contiguous_rgb(monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import nullcontext
    from dataclasses import replace

    from capicove.processing.video.filtering._adapters import huggingface_frame_scorers as adapter

    def processor(*, images: list[np.ndarray], return_tensors: str) -> dict[str, object]:
        assert images[0].flags.c_contiguous
        assert images[0][0, 0].tolist() == [30, 20, 10]
        return {}

    monkeypatch.setattr(
        "capicove.processing.video.filtering._adapters.huggingface_frame_scorers.importlib.import_module",
        lambda name: SimpleNamespace(inference_mode=nullcontext),
    )
    resolved = replace(
        _resolved({}), processor=processor, model=lambda: SimpleNamespace(logits=[[1.0]])
    )
    frame = SampledFrame(0, 0.0, 2, 2, np.full((2, 2, 3), [10, 20, 30], dtype=np.uint8))
    assert adapter._classification_inference(resolved, (frame,)) == [[1.0]]
