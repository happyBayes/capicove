from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from capicove.processing.video import AcceptedVideoSource, VideoSource
from capicove.processing.video.filtering._adapters.ram_plus_video_tagger import (
    RamPlusLocalTagCriterionEvaluator,
    RamPlusTagCriterionEvaluator,
    RamPlusVideoTagger,
    TagEngine,
)
from capicove.processing.video.filtering.video_filter_contracts import (
    Aggregation,
    FrameSamplingConfig,
    LocalModelReference,
    TagCriterion,
    VideoFilterItem,
    VideoFilterMetadata,
)
from capicove.processing.video.filtering.video_filter_ports import (
    BackendUnavailableFilterError,
    FrameTags,
    SampledFrame,
)


def _deadline() -> datetime:
    return datetime.now(UTC) + timedelta(minutes=1)


def _frames() -> tuple[SampledFrame, ...]:
    payload = np.zeros((4, 4, 3), dtype=np.uint8)
    return (
        SampledFrame(0, 0.0, 4, 4, payload),
        SampledFrame(1, 1.0, 4, 4, payload),
    )


def _source() -> AcceptedVideoSource:
    return AcceptedVideoSource("source-1", "a" * 64, "video/mp4", data=b"video")


def _item() -> VideoFilterItem:
    return VideoFilterItem("item-1", VideoSource(data=b"video"))


def _metadata() -> VideoFilterMetadata:
    return VideoFilterMetadata(1.0, 4, 4, 4, 4, backend="fake")


def _tagger() -> RamPlusVideoTagger:
    return RamPlusVideoTagger(
        engine=lambda frames, deadline: (
            FrameTags(("Dog", "Person"), (0.8, 0.9)),
            FrameTags(("person", "Car"), (0.7, 0.6)),
        ),
        weights_hash="b" * 64,
        vocabulary_hash="c" * 64,
        version="1.0",
    )


def test_tag_evaluator_normalizes_and_merges_tags_in_first_seen_order() -> None:
    criterion = TagCriterion(
        "tags",
        LocalModelReference("local/ram"),
        FrameSamplingConfig(),
        tags=("PERSON",),
    )

    result = RamPlusTagCriterionEvaluator(_tagger()).evaluate(
        _item(), _source(), _metadata(), _frames(), criterion, _deadline()
    )

    assert result.tags == ("dog", "person", "car")
    assert result.passed is True
    assert result.details["frame_tags"] == (("dog", "person"), ("person", "car"))


def test_tag_evaluator_supports_all_required_tags() -> None:
    criterion = TagCriterion(
        "tags",
        LocalModelReference("local/ram"),
        FrameSamplingConfig(),
        tags=("person", "car"),
        contain=Aggregation.ALL,
    )

    result = RamPlusTagCriterionEvaluator(_tagger()).evaluate(
        _item(), _source(), _metadata(), _frames(), criterion, _deadline()
    )

    assert result.passed is True


def test_from_local_requires_existing_weights_and_vocabulary(tmp_path: Path) -> None:
    with pytest.raises(BackendUnavailableFilterError, match="weights"):
        RamPlusVideoTagger.from_local(
            tmp_path / "missing.pth",
            tmp_path / "missing-tags.txt",
        )


def test_from_local_hashes_assets_and_uses_injected_local_loader(tmp_path: Path) -> None:
    weights = tmp_path / "ram-plus.pth"
    vocabulary = tmp_path / "tags.txt"
    weights.write_bytes(b"weights")
    vocabulary.write_text("dog\nperson\n", encoding="utf-8")
    received: list[tuple[Path, Path]] = []

    def loader(weights_path: Path, vocabulary_path: Path) -> TagEngine:
        received.append((weights_path, vocabulary_path))
        return lambda frames, deadline: (FrameTags(("dog",)),) * len(frames)

    tagger = RamPlusVideoTagger.from_local(weights, vocabulary, loader=loader)

    assert received == [(weights.resolve(), vocabulary.resolve())]
    assert tagger.identity[0] == "ram-plus"
    assert tagger.weights_hash != tagger.vocabulary_hash


def test_local_evaluator_resolves_assets_from_each_tag_criterion(tmp_path: Path) -> None:
    weights = tmp_path / "ram-plus.pth"
    vocabulary = tmp_path / "tags.txt"
    weights.write_bytes(b"weights")
    vocabulary.write_text("dog\nperson\n", encoding="utf-8")
    received: list[tuple[Path, Path]] = []

    def loader(weights_path: Path, vocabulary_path: Path) -> TagEngine:
        received.append((weights_path, vocabulary_path))
        return lambda frames, deadline: (FrameTags(("person",)),) * len(frames)

    criterion = TagCriterion(
        "tags",
        LocalModelReference(str(weights)),
        FrameSamplingConfig(),
        tags=("person",),
        vocabulary_path=str(vocabulary),
    )
    evaluator = RamPlusLocalTagCriterionEvaluator(loader=loader)

    first = evaluator.evaluate(_item(), _source(), _metadata(), _frames(), criterion, _deadline())
    second = evaluator.evaluate(_item(), _source(), _metadata(), _frames(), criterion, _deadline())

    assert first.passed is True
    assert second.passed is True
    assert received == [(weights.resolve(), vocabulary.resolve())]


def test_default_loader_requires_local_tokenizer_before_importing_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from capicove.processing.video.filtering._adapters import ram_plus_video_tagger as adapter

    def forbidden(name: str) -> Any:
        raise AssertionError("runtime must not load before local tokenizer validation")

    monkeypatch.setattr(
        "capicove.processing.video.filtering._adapters.ram_plus_video_tagger.importlib.import_module",
        forbidden,
    )
    with pytest.raises(BackendUnavailableFilterError, match="tokenizer"):
        adapter._load_ram_engine(tmp_path / "weights.pth", tmp_path / "tags.txt")


def test_default_loader_honors_selected_vocabulary_and_local_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from capicove.processing.video.filtering._adapters import ram_plus_video_tagger as adapter

    (tmp_path / "vocab.txt").write_text("[PAD]\n[UNK]\n")
    options: dict[str, Any] = {}

    def model(**kwargs: Any) -> Any:
        options.update(kwargs)
        return SimpleNamespace(eval=lambda: None)

    modules = {
        "ram": SimpleNamespace(get_transform=lambda **kwargs: None),
        "ram.models": SimpleNamespace(ram_plus=model),
        "torch": SimpleNamespace(),
        "PIL.Image": SimpleNamespace(),
    }
    monkeypatch.setattr(
        "capicove.processing.video.filtering._adapters.ram_plus_video_tagger.importlib.import_module",
        lambda name: modules[name],
    )
    adapter._load_ram_engine(tmp_path / "weights.pth", tmp_path / "tags.txt")
    assert options["tag_list"] == str(tmp_path / "tags.txt")
    assert options["text_encoder_type"] == str(tmp_path)
