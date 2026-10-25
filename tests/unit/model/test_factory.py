"""Bounded leases, typed selection and explicit provider registration."""

import gc
from contextlib import ExitStack
from pathlib import Path
from weakref import ref

import pytest

from capicove.contracts import (
    DistanceMetric,
    EmbeddingSpaceRef,
    QueryModality,
    TargetType,
    VectorNormalization,
)
from capicove.model import (
    ModelCapability,
    ModelCapacityError,
    ModelConfiguration,
    ModelFactory,
    ModelProfile,
    ProviderDefinition,
    ProviderRegistry,
)
from capicove.retrieval.models import TextEmbedding
from tests.unit.model.test_configuration import configuration


class Encoder:
    def __init__(self) -> None:
        self.closed = False

    @property
    def space(self) -> EmbeddingSpaceRef:
        return EmbeddingSpaceRef(
            "fake",
            "fake",
            "1",
            "1",
            2,
            VectorNormalization.L2,
            DistanceMetric.INNER_PRODUCT,
            frozenset({QueryModality.TEXT}),
            frozenset({TargetType.TEXT_CHUNK}),
        )

    def encode(
        self, texts: tuple[str, ...], *, space: EmbeddingSpaceRef, role: str
    ) -> tuple[TextEmbedding, ...]:
        assert not self.closed
        return ()

    def close(self) -> None:
        self.closed = True


def factory(tmp_path: Path) -> tuple[ModelFactory, list[Encoder]]:
    made: list[Encoder] = []

    def create(profile: ModelProfile) -> Encoder:
        encoder = Encoder()
        made.append(encoder)
        return encoder

    registry = ProviderRegistry()
    registry.register(ProviderDefinition("bge-m3", "1", ModelCapability.TEXT_EMBEDDING, create))
    raw = configuration(tmp_path)
    raw["profiles"]["other"] = {**raw["profiles"]["dense"], "manifest_sha256": "b" * 64}
    raw["bindings"]["other.embedding"] = "other"
    return ModelFactory(ModelConfiguration.from_mapping(raw), registry=registry), made


def test_idle_instances_are_reused_and_evicted_at_capacity(tmp_path: Path) -> None:
    runtime, made = factory(tmp_path)
    with runtime.text_encoder("graph.embedding") as first:
        assert first is made[0]
    with runtime.text_encoder("graph.embedding") as same:
        assert same is first
    with runtime.text_encoder("other.embedding"):
        assert made[0].closed
    runtime.close()
    assert len(made) == 2 and made[1].closed
    with pytest.raises(RuntimeError, match="closed"):
        with runtime.text_encoder("graph.embedding"):
            pass


def test_active_instance_cannot_be_evicted_or_closed(tmp_path: Path) -> None:
    runtime, made = factory(tmp_path)
    with runtime.text_encoder("graph.embedding"):
        with pytest.raises(ModelCapacityError):
            with runtime.text_encoder("other.embedding"):
                pass
        with pytest.raises(RuntimeError, match="active"):
            runtime.close()
        assert not made[0].closed
    runtime.close()


def test_failed_request_releases_lease(tmp_path: Path) -> None:
    runtime, made = factory(tmp_path)
    with pytest.raises(ValueError):
        with runtime.text_encoder("graph.embedding"):
            raise ValueError("operation failed")
    runtime.close()
    assert made[0].closed


def test_capability_mismatch_is_rejected_before_creation(tmp_path: Path) -> None:
    runtime, made = factory(tmp_path)
    with pytest.raises(ValueError, match="capability"):
        with runtime.image_ocr("graph.embedding"):
            pass
    assert made == []


def test_registry_rejects_duplicates_and_wrong_result_types(tmp_path: Path) -> None:
    registry = ProviderRegistry()
    provider = ProviderDefinition("bge-m3", "1", ModelCapability.TEXT_EMBEDDING, lambda p: object())
    registry.register(provider)
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(provider)
    runtime = ModelFactory(
        ModelConfiguration.from_mapping(configuration(tmp_path)), registry=registry
    )
    with pytest.raises(TypeError):
        with runtime.text_encoder("graph.embedding"):
            pass


def test_default_registry_rejects_unknown_options_without_loading_models(tmp_path: Path) -> None:
    raw = configuration(tmp_path)
    raw["profiles"]["dense"]["options"]["trust_remote_code"] = True
    with pytest.raises(ValueError, match="options"):
        ModelFactory(ModelConfiguration.from_mapping(raw))


def test_multiple_aliases_share_one_instance(tmp_path: Path) -> None:
    runtime, made = factory(tmp_path)
    with ExitStack() as stack:
        first = stack.enter_context(runtime.text_encoder("graph.embedding"))
        second = stack.enter_context(runtime.text_encoder("graph.embedding"))
        assert first is second
    runtime.close()
    assert len(made) == 1


def test_close_attempts_every_disposer_and_reports_failures(tmp_path: Path) -> None:
    closed: list[str] = []

    def dispose(value: object) -> None:
        closed.append("closed")
        if len(closed) == 1:
            raise RuntimeError("cleanup failed")

    registry = ProviderRegistry()
    registry.register(
        ProviderDefinition(
            "bge-m3", "1", ModelCapability.TEXT_EMBEDDING, lambda p: Encoder(), dispose=dispose
        )
    )
    raw = configuration(tmp_path)
    raw["max_instances"] = 2
    raw["profiles"]["other"] = {**raw["profiles"]["dense"], "manifest_sha256": "b" * 64}
    raw["bindings"]["other.embedding"] = "other"
    runtime = ModelFactory(ModelConfiguration.from_mapping(raw), registry=registry)
    with runtime.text_encoder("graph.embedding"):
        pass
    with runtime.text_encoder("other.embedding"):
        pass
    with pytest.raises(ExceptionGroup, match="cleanup"):
        runtime.close()
    assert len(closed) == 2
    runtime.close()


def test_evicted_provider_is_collectible_before_replacement_is_created(tmp_path: Path) -> None:
    previous: list[ref[Encoder]] = []

    def create(profile: ModelProfile) -> Encoder:
        if previous:
            gc.collect()
            assert previous[0]() is None, "evicted model still occupies a live slot"
        encoder = Encoder()
        previous.append(ref(encoder))
        return encoder

    registry = ProviderRegistry()
    registry.register(ProviderDefinition("bge-m3", "1", ModelCapability.TEXT_EMBEDDING, create))
    raw = configuration(tmp_path)
    raw["profiles"]["other"] = {**raw["profiles"]["dense"], "manifest_sha256": "b" * 64}
    raw["bindings"]["other.embedding"] = "other"
    with ModelFactory(ModelConfiguration.from_mapping(raw), registry=registry) as runtime:
        with runtime.text_encoder("graph.embedding"):
            pass
        with runtime.text_encoder("other.embedding"):
            pass
